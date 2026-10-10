"""JointModel: composition of component models with shared parameters.

A :class:`JointModel` holds multiple
:class:`~harv.models.component.AbstractComponentModel` instances and sums their
log-likelihoods. Shared orbital parameters (period, eccentricity, phase_peri, arg_peri)
are passed once and forwarded to every component. Component-specific nonlinear
parameters (e.g. per-component jitter) are prefixed with the component name
(``"{component_name}.{param_name}"`` becomes ``{param_name}`` when forwarded to the
component).
"""

__all__ = ("JointModel",)

from collections.abc import Callable
from typing import Any, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from unxt import Q

from harv.data.containers import AbstractDatasetContainer
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    PriorDist,
    _is_callable_prior,
    _needs_explicit_sampling,
    _sample_explicit_linear_prior,
    pinned_linear_names,
)
from harv.models.component import (
    AbstractComponentModel,
    _MargBuildingBlocks,
    _sample_nonlinear_params,
)
from harv.models.extensions.base import ParamInfo
from harv.stats import MarginalizedLinear
from harv.stats.marginalized import (
    GeneralizedMarginalizedLinear,
    ResolvedLinearPrior,
    build_marginalized,
)


def _split_nl_values(
    nonlinear_values: dict[str, Any],
    shared_names: frozenset[str],
    component_names: tuple[str, ...],
    per_component_nl_names: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    """Split a flat nonlinear-values dict into per-component dicts.

    Shared parameters are copied to every component. Component-specific
    parameters use the convention ``"component_name.param_name"`` in the
    flat dict and are forwarded as ``"param_name"`` to the component.
    """
    result: dict[str, dict[str, Any]] = {}
    for comp_name in component_names:
        comp_vals: dict[str, Any] = {}
        # Shared params
        for name in shared_names:
            if name in nonlinear_values:
                comp_vals[name] = nonlinear_values[name]
        # Component-specific params
        for param_name in per_component_nl_names.get(comp_name, ()):
            flat_key = f"{comp_name}.{param_name}"
            if flat_key in nonlinear_values:
                comp_vals[param_name] = nonlinear_values[flat_key]
            elif param_name in nonlinear_values:
                # Also accept unqualified name if there's no ambiguity
                comp_vals[param_name] = nonlinear_values[param_name]
        result[comp_name] = comp_vals
    return result


def _priors_equal(a: Any, b: Any) -> bool:
    """Return True if two prior specs are structurally equal.

    Defers to :func:`equinox.tree_equal`, which compares pytree treedef
    (capturing static metadata such as :attr:`QuantityDistribution.unit`)
    *and* per-leaf array values together.  This is more robust than a
    ``__dict__``-based comparison for callable prior factories: many of those
    are :class:`equinox.Module` subclasses whose field values are exposed via
    pytree leaves rather than ``__dict__`` (which can be empty under
    ``__slots__``), and array-shaped fields would also break a raw ``==``
    on ``__dict__``.

    Numpyro distributions, :class:`QuantityDistribution`, and callable
    eqx.Module priors are all proper pytrees, so a single call covers every
    prior shape that flows through ``shared_linear_params`` validation.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    return bool(eqx.tree_equal(a, b))


@final
class JointModel(eqx.Module):
    """Composition of component models that share orbital parameters.

    We recommend using the :class:`~harv.models.joint.JointModel` factory methods like
    :func:`~harv.models.joint.JointModel.for_rv_and_gaia` or
    :func:`~harv.models.joint.JointModel.for_sb2`.

    Parameters
    ----------
    components
        Named component models. Keys are used to namespace component-specific
        parameters (e.g. ``"rv.jitter"``).
    shared_params
        Names of orbital parameters shared across all components (e.g.
        ``("period", "eccentricity", "phase_peri", "arg_peri")``).

    """

    components: dict[str, AbstractComponentModel]
    shared_params: tuple[str, ...]
    shared_linear_params: tuple[str, ...] = ()

    @classmethod
    def for_rv_and_gaia(
        cls,
        components: dict[str, AbstractComponentModel],
        *,
        shared_params: tuple[str, ...] | None = None,
        shared_linear_params: tuple[str, ...] | None = None,
    ) -> "JointModel":
        """Build a JointModel for combined RV + Gaia astrometry.

        TODO: if we want to support more complex shared parameters, look at for_sb2 to
        see what we can generalize

        Parameters
        ----------
        components
            RV and Gaia astrometry component models
            (e.g. ``{"rv": ..., "astro": ...}``).
        shared_params
            Override the default shared orbital parameters. Defaults to
            ``("period", "eccentricity", "phase_peri", "arg_peri")``.
        shared_linear_params
            Linear parameter names shared across components. Defaults to
            ``()`` (no shared linear params for heterogeneous joint models).

        Returns
        -------
            The constructed JointModel.
        """
        if shared_params is None:
            shared_params = ("period", "eccentricity", "phase_peri", "arg_peri")
        if shared_linear_params is None:
            shared_linear_params = ()
        return cls(
            components=components,
            shared_params=shared_params,
            shared_linear_params=shared_linear_params,
        )

    @classmethod
    def for_sb2(  # noqa: C901
        cls,
        prior: "Any",  # TODO: fix type
        *,
        component_names: tuple[str, ...] = ("primary", "secondary"),
        extensions: "tuple[Any, ...] | dict[str, tuple[Any, ...]]" = (),
        shared_params: tuple[str, ...] | None = None,
        shared_linear_params: tuple[str, ...] | None = None,
    ) -> "JointModel":
        """Build an SB2 JointModel from a prior.

        The ``prior`` is expected to follow the convention of
        :func:`~harv.models.priors.default_sb2_prior`: linear-prior keys
        ``name1.rv_semiamp`` and ``name2.rv_semiamp`` map to the per-component
        ``rv_semiamp`` of the components, named "name1" and "name2" in this example, but
        they can be customized. Other linear-prior keys (e.g. ``v_sys``) are
        automatically declared shared across components.

        Data is not bound to the model; pass it at ``sampler.run(data, ...)`` time.

        Parameters
        ----------
        prior
            SB2-style prior.  Keys for named component-specific parameters (e.g.,
            ``rv_semiamp``) must correspond to the component names in
            *component_names*. For example, with the default names "primary" and
            "secondary", the prior must have keys "primary.rv_semiamp" and
            "secondary.rv_semiamp". Other linear parameters (e.g. "v_sys") are
            automatically treated as shared across components.
        component_names
            Names of the two SB2 components. Defaults to ``("primary", "secondary")``.
        extensions
            Extensions to attach to each component.

            - A bare ``tuple`` is applied to **all** components.
            - A ``dict`` is keyed by component name; missing keys yield no extensions
              for that component.
        shared_params
            Defaults to the standard nonlinear shared orbital params. For example,
            "period", "eccentricity", "phase_peri", and "arg_peri".
        shared_linear_params
            Defaults to every key in ``prior.linear_priors`` except the ``rv_semiamp``
            keys.

        Returns
        -------
            The constructed JointModel.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.models.joint import JointModel
        >>> from harv.models.priors import default_sb2_prior
        >>> prior = default_sb2_prior(
        ...     period_min=Q(10., "day"), period_max=Q(1000., "day"),
        ...     sigma_K0=Q(30., "km/s"), sigma_v0=Q(50., "km/s"),
        ... )
        >>> joint = JointModel.for_sb2(prior=prior)
        >>> joint.shared_linear_params
        ('v_sys',)
        """
        # Import here to avoid circular import at module load time.
        from harv.models.rv import RVModel  # noqa: PLC0415

        if len(component_names) != 2:
            raise ValueError(
                f"SB2 expects exactly 2 component names, got {len(component_names)}."
            )

        # Resolve extensions to dict[str, tuple].
        if isinstance(extensions, tuple):
            ext_map: dict[str, tuple[Any, ...]] = dict.fromkeys(
                component_names, extensions
            )
        elif isinstance(extensions, dict):
            ext_map = {n: tuple(extensions.get(n, ())) for n in component_names}
        else:
            raise TypeError("extensions must be a tuple or dict[str, tuple].")

        # Validate that the prior has per-component rv_semiamp keys.
        expected_names = {f"{name}.rv_semiamp" for name in component_names}
        if not all(k in prior.linear_priors for k in expected_names):
            raise ValueError(
                "prior.linear_priors is missing SB2 keys: should contain "
                f"{expected_names}. Use default_sb2_prior(...) or supply a "
                "compatible prior."
            )

        # By default, all keys except the per-component semi-amplitude keys are shared
        # params.  ``None`` means "use defaults"; an explicit empty tuple means "no
        # shared params of this type".
        default_shared_params = tuple(k for k in prior.nonlinear_priors)
        default_linear_shared_params = tuple(
            k for k in prior.linear_priors if k not in expected_names
        )
        if shared_params is None:
            shared_params = default_shared_params
        if shared_linear_params is None:
            shared_linear_params = default_linear_shared_params

        # Validate prior key conventions.  Run the "not shared, must be qualified"
        # checks first so those errors take priority when the prior has multiple issues.

        # Non-shared bare nonlinear keys are ambiguous (must be in shared_params or
        # component-qualified).
        for key in prior.nonlinear_priors:
            if key in shared_params or "." in key:
                continue
            raise ValueError(
                f"Nonlinear param {key!r} is not shared and must be qualified with a "
                f"component name. Add it to shared_params or use a qualified key."
            )
        # Non-shared, non-SB2-component bare linear keys are similarly ambiguous.
        for key in prior.linear_priors:
            if key in shared_linear_params or key in expected_names or "." in key:
                continue
            raise ValueError(
                f"Linear param {key!r} is not shared and must be qualified with a "
                f"component name (e.g. 'primary.{key}'). Add it to "
                f"shared_linear_params or use a qualified key."
            )

        # Shared nonlinear params must be bare and must exist in the prior.
        for name in shared_params:
            if "." in name:
                raise ValueError(
                    f"Shared nonlinear param {name!r} is shared and must not be "
                    f"prefixed with a component name."
                )
            for comp_name in component_names:
                if f"{comp_name}.{name}" in prior.nonlinear_priors:
                    raise ValueError(
                        f"Shared nonlinear param {name!r} is shared and must not be "
                        f"prefixed: found '{comp_name}.{name}' in nonlinear_priors."
                    )
            if name not in prior.nonlinear_priors:
                raise ValueError(
                    f"Shared nonlinear param {name!r} not found in "
                    "prior.nonlinear_priors."
                )
        # Shared linear params must be bare and must exist in the prior.
        for name in shared_linear_params:
            if "." in name:
                raise ValueError(
                    f"Shared linear param {name!r} is shared and must not be "
                    f"prefixed with a component name."
                )
            for comp_name in component_names:
                if f"{comp_name}.{name}" in prior.linear_priors:
                    raise ValueError(
                        f"Shared linear param {name!r} is shared and must not be "
                        f"prefixed: found '{comp_name}.{name}' in linear_priors."
                    )
            if name not in prior.linear_priors:
                raise ValueError(
                    f"Shared linear param {name!r} not found in prior.linear_priors."
                )

        # Now we can build data-less template models (data supplied at run time).
        components = {}
        for name in component_names:
            components[name] = RVModel(extensions=ext_map[name])

        return cls(
            components=components,
            shared_params=shared_params,
            shared_linear_params=shared_linear_params,
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Names of the components in this joint model."""
        return tuple(self.components.keys())

    def _shared_param_names(self) -> frozenset[str]:
        """Names of parameters shared across all components."""
        return frozenset(self.shared_params)

    def _base_nonlinear_names(self) -> frozenset[str]:
        """Base (non-extension) nonlinear parameter names across all components."""
        names: set[str] = set()
        for comp in self.components.values():
            names.update(comp._base_nonlinear_names())
        return frozenset(names)

    def _per_component_nonlinear_names(self) -> dict[str, tuple[str, ...]]:
        """Non-shared nonlinear param names per component."""
        shared = self._shared_param_names()
        result: dict[str, tuple[str, ...]] = {}
        for name, comp in self.components.items():
            comp_nl = comp._all_nonlinear_names()
            result[name] = tuple(n for n in comp_nl if n not in shared)
        return result

    def _per_component_linear_prior(
        self, linear_priors: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None]:
        """Split a flat linear_priors dict into per-component dicts.

        Bare keys that appear in ``shared_linear_params`` are replicated to every
        component.  Qualified keys of the form ``"comp.param"`` are routed to the
        named component as bare ``"param"`` entries.
        """
        shared = set(self.shared_linear_params)
        result: dict[str, dict[str, Any] | None] = {n: {} for n in self.component_names}
        for key, prior in linear_priors.items():
            if key in shared:
                for cname in self.component_names:
                    d = result[cname]
                    if d is not None:
                        d[key] = prior
            elif "." in key:
                cname, base = key.split(".", 1)
                if cname in result:
                    d = result[cname]
                    if d is not None:
                        d[base] = prior
            # Bare non-shared non-qualified keys are not routed to any component
        return result

    def explicit_params(self, linear_priors: dict[str, Any] | None) -> tuple[str, ...]:
        """Names of parameters that must be explicitly sampled.

        Shared nonlinear params use bare names (e.g. ``"period"``).
        Component-specific nonlinear params use ``"comp.param"`` notation
        (e.g. ``"rv.jitter"``).  Explicit-linear params (non-Gaussian priors,
        e.g. ``"parallax"``) are listed flat without namespace prefix,
        matching how they appear in the ``log_prob`` values dict.

        Parameters
        ----------
        linear_priors
            The merged linear-prior dict (with ``"comp.param"`` qualified keys for
            non-shared params).  ``None`` means treat all linear params as
            marginalizable.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )
        shared = self._shared_param_names()
        per_comp = self._per_component_nonlinear_names()

        # Shared nonlinear in a stable order (first component's ordering)
        first_comp = next(iter(self.components.values()))
        shared_names = tuple(
            n for n in first_comp._all_nonlinear_names() if n in shared
        )

        # Component-specific nonlinear, namespaced
        comp_specific: list[str] = []
        for comp_name, names in per_comp.items():
            comp_specific.extend(f"{comp_name}.{n}" for n in names)

        # Explicit-linear (non-Gaussian) — flat, de-duplicated, stable order
        seen: set[str] = set()
        explicit_lin: list[str] = []
        for comp_name, comp in self.components.items():
            comp_lp = per_comp_lp[comp_name]
            marg = set(comp._auto_marginalized_names(comp_lp))
            for name in comp._all_linear_names():
                if name not in marg and name not in seen:
                    explicit_lin.append(name)
                    seen.add(name)

        return shared_names + tuple(comp_specific) + tuple(explicit_lin)

    def marginalized_params(
        self, linear_priors: dict[str, Any] | None
    ) -> tuple[str, ...]:
        """Names of linear parameters analytically marginalized across all components.

        De-duplicated; order follows component iteration order.

        Parameters
        ----------
        linear_priors
            The merged linear-prior dict.  ``None`` means treat all linear params
            as marginalizable.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )
        seen: set[str] = set()
        names: list[str] = []
        for comp_name, comp in self.components.items():
            for name in comp._auto_marginalized_names(per_comp_lp[comp_name]):
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        return tuple(names)

    def _all_param_infos(self) -> tuple[ParamInfo, ...]:
        """All parameter descriptors (shared + per-component).

        Shared params appear once. Component-specific params are prefixed
        with the component name.
        """
        shared = self._shared_param_names()
        infos: list[ParamInfo] = []
        seen_shared: set[str] = set()

        for comp_name, comp in self.components.items():
            for p in comp._param_infos():
                if p.name in shared:
                    if p.name not in seen_shared:
                        infos.append(p)
                        seen_shared.add(p.name)
                elif not p.linear:
                    # Component-specific nonlinear: prefix with component name
                    infos.append(
                        ParamInfo(f"{comp_name}_{p.name}", p.unit, linear=p.linear)
                    )
                # Linear params stay per-component (handled internally)

        return tuple(infos)

    def _route_explicit_linear(
        self,
        nonlinear_values: dict[str, Any],
        comp_nl: dict[str, dict[str, Any]],
        per_comp_lp: dict[str, dict[str, Any] | None],
        marginalized_names: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Copy explicit-linear values from *nonlinear_values* to per-component dicts.

        Explicit linear priors are sampled alongside nonlinear params and
        appear as bare names in *nonlinear_values*. This method routes them to the
        correct component.
        """
        for comp_name, comp in self.components.items():
            comp_lp = per_comp_lp[comp_name]
            if comp_lp:
                if marginalized_names is None:
                    explicit_name_set = set(comp._all_linear_names()) - set(
                        comp._auto_marginalized_names(comp_lp)
                    )
                else:
                    explicit_name_set = set(comp._all_linear_names()) - set(
                        marginalized_names.get(comp_name, ())
                    )
                for name in comp._all_linear_names():
                    if name in explicit_name_set:
                        qualified = f"{comp_name}.{name}"
                        if qualified in nonlinear_values:
                            comp_nl[comp_name][name] = nonlinear_values[qualified]
                        elif name in nonlinear_values:
                            comp_nl[comp_name][name] = nonlinear_values[name]

    def _resolve_component_marginalized_names(
        self,
        marginalized_names: tuple[str, ...] | None,
    ) -> dict[str, tuple[str, ...]] | None:
        """Resolve a flat marginalized-name tuple into per-component subsets."""
        if marginalized_names is None:
            return None

        resolved: dict[str, list[str]] = {name: [] for name in self.component_names}
        for requested_name in marginalized_names:
            if "." in requested_name:
                component_name, linear_name = requested_name.split(".", 1)
                if component_name not in self.components:
                    msg = (
                        "Unknown component in marginalized_names: "
                        f"{component_name!r}. Valid components: {self.component_names}"
                    )
                    raise ValueError(msg)
                resolved[component_name].append(linear_name)
                continue

            matched = False
            for component_name, component in self.components.items():
                if requested_name in component._all_linear_names():
                    resolved[component_name].append(requested_name)
                    matched = True
            if not matched:
                msg = (
                    "Unknown linear parameter in marginalized_names: "
                    f"{requested_name!r}"
                )
                raise ValueError(msg)

        return {
            component_name: tuple(dict.fromkeys(names))
            for component_name, names in resolved.items()
        }

    def _resolve_marginalization(
        self,
        marginalized_names: tuple[str, ...] | None,
        linear_priors: dict[str, Any] | None,
    ) -> tuple[dict[str, tuple[str, ...]], bool]:
        """Resolve per-component marginalized names + decide which path to take.

        Centralizes the dispatch logic shared by ``log_prob``,
        ``sample_conditional_linear``, and the marginalized numpyro model
        builder.

        Parameters
        ----------
        marginalized_names
            User-supplied flat tuple of linear parameter names to marginalize,
            or ``None`` to use each component's auto-marginalized set.
        linear_priors
            The merged linear-prior dict (with ``"comp.param"`` qualified keys for
            non-shared params), or ``None`` to treat all linear params as
            marginalizable.  Used to resolve the auto-marginalized set if
            *marginalized_names* is ``None``.

        Returns
        -------
            ``(per_comp_marg, any_shared_marg)``.

            ``per_comp_marg`` is the per-component marginalized parameter names.
            Always populated (no ``None`` sentinel): callers do not need to
            special-case ``marginalized_names is None``.

            ``any_shared_marg`` is ``True`` iff at least one name in
            ``shared_linear_params`` is being marginalized in any component, in
            which case the joint marginalization path must be used.  ``False``
            means the existing per-component summation gives the correct answer.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )
        per_comp_marg = self._resolve_component_marginalized_names(marginalized_names)
        if per_comp_marg is None:
            # Default: each component marginalizes its auto-classified set.
            # Pre-populating here keeps callers branch-free.
            per_comp_marg = {
                comp_name: comp._auto_marginalized_names(per_comp_lp[comp_name])
                for comp_name, comp in self.components.items()
            }
        shared_lin = set(self.shared_linear_params)
        any_shared_marg = bool(
            shared_lin
            and any(shared_lin & set(names) for names in per_comp_marg.values())
        )
        return per_comp_marg, any_shared_marg

    def _build_joint_marginalized_linear(  # noqa: C901
        self,
        comp_nl: dict[str, dict[str, Any]],
        per_comp_marg: dict[str, tuple[str, ...]],
        data: AbstractDatasetContainer,
        linear_priors: dict[str, Any] | None,
    ) -> tuple[
        MarginalizedLinear | GeneralizedMarginalizedLinear,
        jax.Array,
        list[tuple[str, str | None]],
        dict[str, dict[str, Any]],
    ]:
        """Build one ``MarginalizedLinear`` spanning all components.

        Shared linear columns (from ``shared_linear_params``) appear once in
        the joint design matrix spanning all rows.  Per-component unique
        columns appear in a block-diagonal pattern.

        Parameters
        ----------
        comp_nl
            Per-component nonlinear values, with any explicit linear values
            already routed in by ``_route_explicit_linear``.
        per_comp_marg
            Per-component marginalized parameter names.
        data
            Per-component data, indexed by component name.
        linear_priors
            Flat merged linear-prior dict (``"comp.param"`` qualified for
            non-shared params).

        Returns
        -------
            ``(marg_dist, y_joint, global_cols, explicit_by_comp)``.

            ``marg_dist`` is the single joint marginalized-Gaussian likelihood.

            ``y_joint`` is the concatenated residual-subtracted observations.

            ``global_cols`` is the column ordering used for decomposing joint
            samples back into per-component and shared values.  ``owner`` is
            ``None`` for shared columns, else the component name.

            ``explicit_by_comp`` is the explicit (non-marginalized) linear values
            per component, extracted from each component's building blocks.
        """
        shared_set = set(self.shared_linear_params)
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )

        # --- Step (a): build blocks per component ---
        blocks_by_comp: dict[str, _MargBuildingBlocks] = {}
        for comp_name, comp in self.components.items():
            marg_names = per_comp_marg[comp_name]
            pure_nl, explicit_lin = comp._extract_explicit_linear_values(
                comp_nl[comp_name], marg_names
            )
            blocks_by_comp[comp_name] = comp._build_marg_blocks(
                pure_nl,
                marg_names,
                explicit_lin,
                data[comp_name],
                per_comp_lp[comp_name],
            )

        names_by_comp = {
            comp_name: blocks_by_comp[comp_name].marg_names
            for comp_name in self.components
        }

        # --- Step (b): global column ordering ---
        global_cols: list[tuple[str, str | None]] = []
        seen_shared: set[str] = set()
        for comp_name in self.components:
            for nm in names_by_comp[comp_name]:
                if nm in shared_set:
                    if nm not in seen_shared:
                        global_cols.append((nm, None))
                        seen_shared.add(nm)
                else:
                    global_cols.append((nm, comp_name))

        n_global = len(global_cols)
        col_idx = {(nm, owner): i for i, (nm, owner) in enumerate(global_cols)}

        # --- Step (c): build joint design matrix ---
        total_rows = sum(blocks_by_comp[c].X.shape[0] for c in self.components)
        X_joint = jnp.zeros((total_rows, n_global))
        y_parts: list[jax.Array] = []
        cov_blocks: list[jax.Array] = []
        row_offset = 0
        for comp_name in self.components:
            X_c = blocks_by_comp[comp_name].X
            n_c = X_c.shape[0]
            for local_j, nm in enumerate(names_by_comp[comp_name]):
                owner = None if nm in shared_set else comp_name
                gj = col_idx[(nm, owner)]
                X_joint = X_joint.at[row_offset : row_offset + n_c, gj].set(
                    X_c[:, local_j]
                )
            y_parts.append(blocks_by_comp[comp_name].y)
            cov_blocks.append(blocks_by_comp[comp_name].cov)
            row_offset += n_c
        y_joint = jnp.concatenate(y_parts)

        # --- Step (d): build joint prior ---
        # IMPORTANT: this implementation assumes each component's marginalized
        # linear priors are INDEPENDENT, i.e. every component's resolved prior
        # is diagonal.  Under that assumption the joint prior on the combined
        # linear vector is itself diagonal, and we can read off the joint
        # mean/scale entry-by-entry without ever forming a full covariance
        # matrix or calling ``jnp.linalg.cholesky``.  All current harv
        # parameterizations satisfy this.  If a future component introduces
        # correlated priors, this loop must be rewritten to assemble the full
        # block-structured covariance and Cholesky-factorize it (with a small
        # ridge for PSD safety); see git history before this commit for the
        # previous full-Cholesky construction.
        #
        # Truncation bounds ride along in the same per-slot layout, so a
        # support-constrained prior (e.g. signed SB2 semi-amplitudes) works here
        # unchanged.  Mixtures do not: their components would have to be expanded
        # across the joint slot layout, multiplying between components.
        mu_joint = jnp.zeros(n_global)
        scale_diag_joint = jnp.zeros(n_global)
        low_joint: list[Any] = [-jnp.inf] * n_global
        high_joint: list[Any] = [jnp.inf] * n_global
        low_fin_joint = [False] * n_global
        high_fin_joint = [False] * n_global
        filled: set[int] = set()  # global indices already assigned (for shared cols)
        for comp_name in self.components:
            nms = names_by_comp[comp_name]
            prior_c = blocks_by_comp[comp_name].prior
            if prior_c.n_components > 1:
                msg = (
                    f"Component {comp_name!r} has a Gaussian-mixture linear prior. "
                    "Mixture linear priors are not supported inside a JointModel "
                    "with shared linear parameters; use a single-component model, "
                    "or give the shared parameters non-mixture priors."
                )
                raise NotImplementedError(msg)
            for local_j, nm in enumerate(nms):
                owner = None if nm in shared_set else comp_name
                gj = col_idx[(nm, owner)]
                # Shared global slots are populated by the FIRST component
                # that owns them; validation guarantees later components have
                # an identical prior, so subsequent visits are no-ops.
                if gj in filled:
                    continue
                mu_joint = mu_joint.at[gj].set(prior_c.loc[0, local_j])
                scale_diag_joint = scale_diag_joint.at[gj].set(
                    prior_c.scale[0, local_j]
                )
                low_joint[gj] = prior_c.low[local_j]
                high_joint[gj] = prior_c.high[local_j]
                low_fin_joint[gj] = prior_c.low_finite[local_j]
                high_fin_joint[gj] = prior_c.high_finite[local_j]
                filled.add(gj)

        prior_joint = ResolvedLinearPrior(
            loc=mu_joint[None],
            scale=scale_diag_joint[None],
            ln_weights=jnp.zeros(1),
            low=jnp.stack([jnp.asarray(v) for v in low_joint]),
            high=jnp.stack([jnp.asarray(v) for v in high_joint]),
            low_finite=tuple(low_fin_joint),
            high_finite=tuple(high_fin_joint),
            names=tuple(nm for nm, _ in global_cols),
        )

        # --- Step (e): build joint data distribution ---
        if all(c.ndim == 1 for c in cov_blocks):
            diag_joint = jnp.concatenate(cov_blocks)
            data_dist_joint: dist.Distribution = dist.Normal(
                jnp.zeros(total_rows), jnp.sqrt(diag_joint)
            )
        else:
            full_blocks = [c if c.ndim == 2 else jnp.diag(c) for c in cov_blocks]
            cov_joint = jax.scipy.linalg.block_diag(*full_blocks)
            data_dist_joint = dist.MultivariateNormal(
                loc=jnp.zeros(total_rows), covariance_matrix=cov_joint
            )

        marg_dist = build_marginalized(X_joint, prior_joint, data_dist_joint)

        explicit_by_comp = {
            comp_name: dict(blocks_by_comp[comp_name].explicit_linear)
            for comp_name in self.components
        }
        return marg_dist, y_joint, global_cols, explicit_by_comp

    def log_prob(
        self,
        nonlinear_values: dict[str, Any],
        data: AbstractDatasetContainer,
        *,
        linear_priors: dict[str, Any] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute the joint log-likelihood.

        When ``shared_linear_params`` contains names that are being analytically
        marginalized, a single joint ``MarginalizedLinear`` is built spanning all
        components (the *joint path*).  Otherwise the per-component log-likelihoods are
        summed as before.

        Parameters
        ----------
        nonlinear_values
            Flat dict of parameter values. Shared orbital params use bare names
            (``"period"``, ``"eccentricity"``, etc.). Component-specific nonlinear
            params use ``"component.param"`` convention (e.g. ``"rv.jitter"``).
        data
            Per-component data, indexed by component name.
        linear_priors
            Flat merged linear-prior dict. ``None`` means treat all linear params as
            marginalizable.
        marginalized_names
            Optional linear parameter names to marginalize. Component-qualified names
            are accepted (e.g. ``"rv.parallax"`` or just ``"parallax"`` if unambiguous).

        Returns
        -------
            Scalar log-likelihood.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )
        shared_nl = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marg, any_shared_marg = self._resolve_marginalization(
            marginalized_names, linear_priors
        )

        comp_nl = _split_nl_values(
            nonlinear_values, shared_nl, self.component_names, per_comp_nl
        )
        self._route_explicit_linear(
            nonlinear_values, comp_nl, per_comp_lp, per_comp_marg
        )

        if any_shared_marg:
            # Joint path: a single MarginalizedLinear spanning all components,
            # with shared linear columns merged.  Required when at least one
            # ``shared_linear_params`` entry is being analytically marginalized
            # so its prior is integrated *once* (not once per component).
            marg_dist, y_joint, _, _ = self._build_joint_marginalized_linear(
                comp_nl, per_comp_marg, data, linear_priors
            )
            return marg_dist.log_prob(y_joint)

        # Per-component sum path: each component marginalizes its own linear
        # params independently and we sum the resulting log-likelihoods.  This
        # is the correct behaviour whenever no shared linear param is being
        # marginalized (including the common case of an unshared joint model).
        ln_probs = [
            comp.log_prob(
                comp_nl[name],
                data[name],
                linear_priors=per_comp_lp[name],
                marginalized_names=per_comp_marg[name],
            )
            for name, comp in self.components.items()
        ]
        return jnp.sum(jnp.stack(ln_probs))

    def sample_conditional_linear(
        self,
        nonlinear_values: dict[str, Any],
        data: AbstractDatasetContainer,
        *,
        key: jax.Array,
        linear_priors: dict[str, Any] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
        use_mean: bool = False,
    ) -> "dict[str, Any]":
        """Sample conditional linear params for each component.

        When ``shared_linear_params`` are jointly marginalized, shared
        parameters appear at the top level of the returned dict (with bare
        names) rather than inside per-component sub-dicts.

        Parameters
        ----------
        nonlinear_values
            Flat parameter values dict.
        key
            JAX PRNG key.
        data
            Per-component data, indexed by component name.
        linear_priors
            Flat merged linear-prior dict.
        marginalized_names
            Optional linear parameter names to marginalize.
        use_mean
            When ``True``, return the conditional posterior mean for the
            marginalized linear parameters instead of a random draw. For a
            Gaussian conditional this is also the conditional MAP. Default
            ``False``.

        Returns
        -------
            Sampled parameter values. The structure depends on the path:

            - *Default path* (no shared marginalization): ``dict[comp_name,
              dict[param_name, array]]``.
            - *Joint path* (shared marginalization): mixed dict where shared
              params are top-level and per-component params are in sub-dicts
              keyed by component name.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )
        shared_nl = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marg, any_shared_marg = self._resolve_marginalization(
            marginalized_names, linear_priors
        )

        comp_nl = _split_nl_values(
            nonlinear_values, shared_nl, self.component_names, per_comp_nl
        )
        self._route_explicit_linear(
            nonlinear_values, comp_nl, per_comp_lp, per_comp_marg
        )

        if any_shared_marg:
            # Joint path: sample from one big conditional posterior over all
            # marginalized linear params, then de-multiplex into a dict whose
            # shared entries sit at the top level (bare name) and whose
            # per-component entries sit in sub-dicts keyed by component name.
            marg_dist, y_joint, global_cols, explicit_by_comp = (
                self._build_joint_marginalized_linear(
                    comp_nl, per_comp_marg, data, linear_priors
                )
            )
            cond = marg_dist.conditional(y_joint)
            samples_flat = cond.mean if use_mean else cond.sample(key)

            out: dict[str, Any] = {}
            for (nm, owner), val in zip(global_cols, samples_flat, strict=True):
                if owner is None:
                    out[nm] = val
                else:
                    out.setdefault(owner, {})[nm] = val

            # Explicit (non-marginalized) linear values were already routed
            # into ``comp_nl`` above; surface them in the per-component
            # sub-dicts so consumers see the full per-component linear set.
            for comp_name, explicit_lin in explicit_by_comp.items():
                if explicit_lin:
                    out.setdefault(comp_name, {}).update(explicit_lin)

            return out

        # Per-component path: each component samples its own linear posterior
        # independently.  Result is the standard nested
        # ``dict[comp_name, dict[param_name, array]]``.
        results: dict[str, dict[str, jax.Array]] = {}
        for name, comp in self.components.items():
            key, subkey = jax.random.split(key)
            results[name] = comp.sample_conditional_linear(
                comp_nl[name],
                data[name],
                key=subkey,
                linear_priors=per_comp_lp[name],
                marginalized_names=per_comp_marg[name],
                use_mean=use_mean,
            )
        return results

    def numpyro_model(
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: AbstractDatasetContainer,
        linear_priors: dict[str, Any] | None,
        *,
        marginalized: bool = True,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Build a numpyro model for MCMC sampling of the joint model.

        Parameters
        ----------
        nonlinear_priors
            Prior distributions for all nonlinear parameters. Shared orbital
            params use bare names. Component-specific params use
            ``"component.param"`` convention.
        data
            Per-component data, indexed by component name.
        linear_priors
            Flat merged linear-prior dict.
        marginalized
            If ``True`` (default), linear parameters are marginalized
            per-component. If ``False``, all parameters are sampled
            explicitly.
        marginalized_names
            Optional linear parameter names to marginalize when
            ``marginalized=True``. Component-qualified names are accepted.

        Returns
        -------
            A numpyro model function suitable for use with a sampler.
        """
        if not marginalized and marginalized_names is not None:
            msg = "marginalized_names cannot be set when marginalized=False"
            raise ValueError(msg)
        if marginalized:
            return self._build_marginalized_numpyro(
                nonlinear_priors,
                data,
                linear_priors,
                marginalized_names=marginalized_names,
            )
        return self._build_full_numpyro(nonlinear_priors, data, linear_priors)

    def _build_marginalized_numpyro(  # noqa: C901
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: AbstractDatasetContainer,
        linear_priors: dict[str, Any] | None,
        *,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Build a marginalized numpyro model for the joint model."""
        joint = self
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )
        per_comp_marginalized_names, any_shared_marg = self._resolve_marginalization(
            marginalized_names, linear_priors
        )

        # Identify shared linear params that are explicitly sampled (non-Gaussian
        # prior).  These must be sampled exactly *once* at the top of the model
        # and copied to every component, rather than re-sampled per component.
        shared_lin_set = set(self.shared_linear_params)
        shared_explicit_lin: set[str] = set()
        if shared_lin_set and linear_priors is not None:
            pinned_shared = pinned_linear_names(linear_priors)
            for nm in shared_lin_set:
                if nm in linear_priors and _needs_explicit_sampling(
                    linear_priors[nm], name=nm, pinned_names=pinned_shared
                ):
                    shared_explicit_lin.add(nm)

        # Pre-classify each component's *non-shared* explicit-linear priors into
        # "direct" (plain Distribution / QuantityDistribution) and "callable"
        # (e.g. PeriodDependentKPrior).  Direct priors are sampled first so
        # callable priors can reference their values via ``extra_values``.
        _comp_explicit_direct_lp: dict[str, dict[str, Any]] = {}
        _comp_explicit_callable_lp: dict[str, dict[str, Any]] = {}
        _comp_param_units: dict[str, dict[str, str]] = {}
        for comp_name, comp in self.components.items():
            lp = per_comp_lp[comp_name] or {}
            requested_marg = set(per_comp_marginalized_names[comp_name])
            explicit_lp = {
                name: prior_dist
                for name, prior_dist in lp.items()
                if name not in requested_marg and name not in shared_explicit_lin
            }
            _comp_explicit_direct_lp[comp_name] = {
                n: p for n, p in explicit_lp.items() if not _is_callable_prior(p)
            }
            _comp_explicit_callable_lp[comp_name] = {
                n: p for n, p in explicit_lp.items() if _is_callable_prior(p)
            }
            _comp_param_units[comp_name] = comp._linear_param_units(data[comp_name])

        # Same direct/callable split for the shared explicit-linear priors.
        _shared_explicit_direct_lp: dict[str, Any] = {}
        _shared_explicit_callable_lp: dict[str, Any] = {}
        _shared_param_units: dict[str, str] = {}
        if shared_explicit_lin and linear_priors is not None:
            first_comp_name = next(iter(self.component_names))
            first_comp = self.components[first_comp_name]
            pu = first_comp._linear_param_units(data[first_comp_name])
            for nm in shared_explicit_lin:
                p = linear_priors[nm]
                _shared_param_units[nm] = pu.get(nm, "")
                if not _is_callable_prior(p):
                    _shared_explicit_direct_lp[nm] = p
                else:
                    _shared_explicit_callable_lp[nm] = p

        def model_fn() -> None:
            # 1. Sample nonlinear params.
            values = _sample_nonlinear_params(nonlinear_priors)

            # Re-attach units to shared QD priors so callable linear priors can
            # consume them (downstream resolvers expect Q-wrapped values).
            nonlinear_values = dict(values)
            for name, d in nonlinear_priors.items():
                if isinstance(d, QuantityDistribution) and name in shared:
                    nonlinear_values[name] = Q(values[name], cast("str", d.unit))

            # 2. Sample shared explicit-linear priors ONCE.  Direct priors first
            #    so callable shared priors that depend on them can read the
            #    sampled values out of ``nonlinear_values``.
            for name, p in _shared_explicit_direct_lp.items():
                nonlinear_values[name] = _sample_explicit_linear_prior(
                    name, p, _shared_param_units.get(name, ""), nonlinear_values
                )
            for name, p in _shared_explicit_callable_lp.items():
                nonlinear_values[name] = _sample_explicit_linear_prior(
                    name, p, _shared_param_units.get(name, ""), nonlinear_values
                )

            # 3. Sample per-component explicit-linear priors.  Direct first,
            #    then callable; the per-component ``explicit_linear_q``
            #    feeds Q-wrapped values to callable resolvers that need units.
            for comp_name in joint.component_names:
                pu = _comp_param_units[comp_name]
                explicit_linear_q: dict[str, Any] = {}
                for name, p in _comp_explicit_direct_lp[comp_name].items():
                    target_u = pu.get(name, "")
                    raw = _sample_explicit_linear_prior(
                        name,
                        p,
                        target_u,
                        nonlinear_values,
                        site_name=f"{comp_name}.{name}",
                        parameterization=joint.components[comp_name].parameterization,
                    )
                    nonlinear_values[f"{comp_name}.{name}"] = raw
                    explicit_linear_q[name] = Q(raw, target_u) if target_u else raw
                for name, p in _comp_explicit_callable_lp[comp_name].items():
                    target_u = pu.get(name, "")
                    raw = _sample_explicit_linear_prior(
                        name,
                        p,
                        target_u,
                        nonlinear_values,
                        extra_values=explicit_linear_q,
                        parameterization=joint.components[comp_name].parameterization,
                    )
                    nonlinear_values[f"{comp_name}.{name}"] = raw
                    explicit_linear_q[name] = Q(raw, target_u) if target_u else raw

            # 4. Split nonlinear_values per component and route explicit-linear values.
            comp_nl = _split_nl_values(
                nonlinear_values, shared, joint.component_names, per_comp_nl
            )
            joint._route_explicit_linear(
                nonlinear_values, comp_nl, per_comp_lp, per_comp_marginalized_names
            )

            # 5. Compute the marginalized log-likelihood.
            if any_shared_marg:
                # Joint path: a single MarginalizedLinear spanning all
                # components — required so each shared linear prior is
                # integrated once, not once per component.
                marg_dist, y_joint, _, _ = joint._build_joint_marginalized_linear(
                    comp_nl, per_comp_marginalized_names, data, linear_priors
                )
                ln_lik = marg_dist.log_prob(y_joint)
            else:
                # Per-component sum: each component's marginalization is
                # independent, so the log-likelihoods simply add.
                ln_lik = jnp.zeros(())
                for comp_name, comp in joint.components.items():
                    ln_lik = ln_lik + comp.log_prob(
                        comp_nl[comp_name],
                        data[comp_name],
                        linear_priors=per_comp_lp[comp_name],
                        marginalized_names=per_comp_marginalized_names[comp_name],
                    )
            numpyro.factor("ln_lik", ln_lik)

        return model_fn

    def _build_full_numpyro(  # noqa: C901
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: AbstractDatasetContainer,
        linear_priors: dict[str, Any] | None,
    ) -> Callable[[], None]:
        """Build a full (non-marginalized) numpyro model for the joint model.

        Both nonlinear and linear parameters are sampled, each linear parameter
        as its own site. Per-component priors are independent, so a single joint
        site would carry a diagonal covariance and be the same distribution; one
        site each is that distribution without the machinery, and it accepts
        every prior family :func:`_sample_explicit_linear_prior` handles rather
        than only the untruncated single Gaussians one ``MultivariateNormal``
        can express.
        """
        joint = self
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_priors)
            if linear_priors is not None
            else dict.fromkeys(self.component_names)
        )

        _comp_lp: dict[str, dict[str, Any]] = {}
        _comp_param_units: dict[str, dict[str, str]] = {}
        for comp_name, comp in self.components.items():
            lp = per_comp_lp[comp_name]
            if lp is None:
                msg = (
                    f"Cannot build full numpyro model: component {comp_name!r} "
                    "has no linear_priors"
                )
                raise ValueError(msg)
            _comp_lp[comp_name] = lp
            _comp_param_units[comp_name] = comp._linear_param_units(data[comp_name])

        # Build the ordered slot list.  Each slot is
        # ``(site_name, comp_name, base_name)``:
        #
        # * Names in ``shared_linear_params`` appear ONCE with ``site_name ==
        #   base_name``, owned by the first component that holds them.
        #   Validation in ``__check_init__`` guarantees identical priors
        #   across components, so picking the first owner is well-defined.
        # * Names that are NOT shared appear once per owning component with
        #   ``site_name == f"{comp_name}.{base_name}"``.  This is essential
        #   when the same bare linear name (e.g. ``rv_semiamp`` for SB2) is
        #   present in multiple components: without qualified site names
        #   numpyro raises a duplicate-site error and the per-component
        #   posteriors collapse onto whichever owner happened to be visited
        #   last.
        shared_lin_set = set(self.shared_linear_params)

        def _build_slots(
            per_comp_lp_: dict[str, dict[str, Any]],
        ) -> list[tuple[str, str, str]]:
            slots: list[tuple[str, str, str]] = []
            seen_shared: set[str] = set()
            for comp_name in joint.component_names:
                for base in per_comp_lp_[comp_name]:
                    if base in shared_lin_set:
                        if base in seen_shared:
                            continue
                        seen_shared.add(base)
                        slots.append((base, comp_name, base))
                    else:
                        slots.append((f"{comp_name}.{base}", comp_name, base))
            return slots

        # Plain priors before callables, so a callable sees the values it
        # declares in ``requires``. ``sorted`` is stable, so plain comes first.
        _slots = sorted(
            _build_slots(_comp_lp),
            key=lambda slot: _is_callable_prior(_comp_lp[slot[1]][slot[2]]),
        )

        def model_fn() -> None:
            # Sample all nonlinear params
            values = _sample_nonlinear_params(nonlinear_priors)

            # Wrap shared QD priors in Quantity
            nonlinear_values: dict[str, Any] = dict(values)
            for name, d in nonlinear_priors.items():
                if isinstance(d, QuantityDistribution) and name in shared:
                    nonlinear_values[name] = Q(values[name], cast("str", d.unit))

            # Split per component
            comp_nl = _split_nl_values(
                nonlinear_values, shared, joint.component_names, per_comp_nl
            )

            # Per-component view of linear values, used both to feed callable
            # Gaussian priors and to assemble each component's log-prob input.
            # Shared linear values are mirrored into every component's entry so
            # callable priors can read them under their bare parameter name.
            linear_by_comp: dict[str, dict[str, jax.Array]] = {
                c: {} for c in joint.component_names
            }
            # The same draws, ``Q``-wrapped, for callable priors to read: they
            # are written against unit-aware values (see the
            # ``LinearPriorCallable`` contract).
            linear_q_by_comp: dict[str, dict[str, Any]] = {
                c: {} for c in joint.component_names
            }

            def _record(
                cname: str, base: str, value: jax.Array, *, is_shared: bool
            ) -> None:
                unit = _comp_param_units[cname].get(base, "")
                as_q = Q(value, unit) if unit else value
                targets = joint.component_names if is_shared else (cname,)
                for c in targets:
                    linear_by_comp[c][base] = value
                    linear_q_by_comp[c][base] = as_q

            for site_name, cname, base in _slots:
                if base in shared_lin_set and base in linear_by_comp[cname]:
                    continue  # a shared parameter is drawn once, by its owner
                raw = _sample_explicit_linear_prior(
                    base,
                    _comp_lp[cname][base],
                    _comp_param_units[cname].get(base, ""),
                    comp_nl[cname],
                    extra_values=linear_q_by_comp[cname],
                    site_name=site_name,
                    parameterization=joint.components[cname].parameterization,
                )
                _record(cname, base, jnp.asarray(raw), is_shared=base in shared_lin_set)

            # Evaluate explicit log-likelihood per component
            ln_lik = jnp.zeros(())
            for comp_name, comp in joint.components.items():
                comp_linear = {
                    n: linear_by_comp[comp_name][n]
                    for n in comp._all_linear_names()
                    if n in linear_by_comp[comp_name]
                }
                ln_lik = ln_lik + comp._log_prob_explicit(
                    comp_nl[comp_name], comp_linear, data[comp_name]
                )

            numpyro.factor("ln_lik", ln_lik)

        return model_fn
