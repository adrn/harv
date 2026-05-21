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

import types
from collections.abc import Callable
from typing import Any, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Q
from unxt.quantity import ustrip

from harv.distributions import QuantityDistribution
from harv.models._helpers import PriorDist, _needs_explicit_sampling, _unwrap_dist
from harv.models.component import (
    AbstractComponentModel,
    _MargBuildingBlocks,
    _resolve_prior_to_mvn,
    _sample_nonlinear_params,
)
from harv.models.extensions.base import ParamInfo


def _split_nl_values(
    nl_values: dict[str, Any],
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
            if name in nl_values:
                comp_vals[name] = nl_values[name]
        # Component-specific params
        for param_name in per_component_nl_names.get(comp_name, ()):
            flat_key = f"{comp_name}.{param_name}"
            if flat_key in nl_values:
                comp_vals[param_name] = nl_values[flat_key]
            elif param_name in nl_values:
                # Also accept unqualified name if there's no ambiguity
                comp_vals[param_name] = nl_values[param_name]
        result[comp_name] = comp_vals
    return result


_DEFAULT_SHARED_PARAMS: tuple[str, ...] = (
    "period",
    "eccentricity",
    "phase_peri",
    "arg_peri",
)


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


def _is_callable_prior(p: Any) -> bool:
    """True iff ``p`` is a callable prior factory (e.g. ``PeriodDependentKPrior``).

    Plain ``dist.Distribution`` and :class:`QuantityDistribution` instances are
    *not* considered callable priors here even if their classes happen to be
    callable: they are sampled directly without being resolved against
    nonlinear parameter values first.
    """
    return callable(p) and not isinstance(p, dist.Distribution | QuantityDistribution)


def _sample_explicit_linear_prior(
    name: str,
    prior_dist: Any,
    target_unit: str,
    nl_values: dict[str, Any],
    extra_values: dict[str, Any] | None = None,
    *,
    site_name: str | None = None,
) -> jax.Array:
    """Sample one explicit (non-marginalized) linear prior in a numpyro model.

    Unifies two cases that previously needed separate code paths:

    * Plain ``dist.Distribution`` or :class:`QuantityDistribution` priors are sampled
      directly via ``numpyro.sample``; if a ``QuantityDistribution`` is provided the
      result is unit-stripped to ``target_unit``.
    * Callable priors (e.g. :class:`PeriodDependentKPrior`) are resolved to a
      :class:`numpyro.distributions.distributions.Normal` at the current ``nl_values`` /
      ``extra_values`` and then sampled.  The resolver returns values already expressed
      in ``target_unit``, so no further unit-strip is performed.

    Parameters
    ----------
    name
        Site name passed to ``numpyro.sample``.
    prior_dist
        The prior specification.
    target_unit
        Unit string the returned value must be expressed in.  ``""`` for
        dimensionless.
    nl_values
        Already-sampled nonlinear (and previously-sampled explicit-linear)
        values, keyed by bare parameter name.  Used by callable priors.
    extra_values
        Optional ``Q``-wrapped versions of values that callable priors may
        consume to evaluate unit-aware dependencies; ``None`` when no such
        proxy is needed (e.g. for shared explicit-linear priors).
    site_name
        Optional site name to use within numpyro.

    Returns
    -------
        The sampled value, unit-stripped to ``target_unit``.
    """
    _site = site_name if site_name is not None else name
    if _is_callable_prior(prior_dist):
        resolved = _resolve_prior_to_mvn(
            {name: prior_dist},
            nl_values,
            {name: target_unit},
            extra_values=extra_values,
        )
        return cast(
            "jax.Array",
            numpyro.sample(
                _site,
                dist.Normal(resolved.loc[0], resolved.scale_tril[0, 0]),
            ),
        )

    # Direct sampling for plain Distribution / QuantityDistribution priors.
    raw = cast("jax.Array", numpyro.sample(_site, _unwrap_dist(prior_dist)))
    if isinstance(prior_dist, QuantityDistribution) and target_unit:
        raw = jnp.asarray(ustrip(target_unit, Q(raw, cast("str", prior_dist.unit))))
    return raw


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
            shared_params = _DEFAULT_SHARED_PARAMS
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
            Defaults to every key in ``prior.linear_prior`` except the ``rv_semiamp``
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
        if not all(k in prior.linear_prior for k in expected_names):
            raise ValueError(
                "prior.linear_prior is missing SB2 keys: should contain "
                f"{expected_names}. Use default_sb2_prior(...) or supply a "
                "compatible prior."
            )

        # By default, all keys except the per-component semi-amplitude keys are shared
        # params.  ``None`` means "use defaults"; an explicit empty tuple means "no
        # shared params of this type".
        default_shared_params = tuple(k for k in prior.nonlinear_priors)
        default_linear_shared_params = tuple(
            k for k in prior.linear_prior if k not in expected_names
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
        for key in prior.linear_prior:
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
                if f"{comp_name}.{name}" in prior.linear_prior:
                    raise ValueError(
                        f"Shared linear param {name!r} is shared and must not be "
                        f"prefixed: found '{comp_name}.{name}' in linear_prior."
                    )
            if name not in prior.linear_prior:
                raise ValueError(
                    f"Shared linear param {name!r} not found in prior.linear_prior."
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
        self, linear_prior: dict[str, Any]
    ) -> dict[str, dict[str, Any] | None]:
        """Split a flat linear_prior dict into per-component dicts.

        Bare keys that appear in ``shared_linear_params`` are replicated to every
        component.  Qualified keys of the form ``"comp.param"`` are routed to the
        named component as bare ``"param"`` entries.
        """
        shared = set(self.shared_linear_params)
        result: dict[str, dict[str, Any] | None] = {n: {} for n in self.component_names}
        for key, prior in linear_prior.items():
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

    def params_explicit(self, linear_prior: dict[str, Any] | None) -> tuple[str, ...]:
        """Names of parameters that must be explicitly sampled.

        Shared nonlinear params use bare names (e.g. ``"period"``).
        Component-specific nonlinear params use ``"comp.param"`` notation
        (e.g. ``"rv.jitter"``).  Explicit-linear params (non-Gaussian priors,
        e.g. ``"parallax"``) are listed flat without namespace prefix,
        matching how they appear in the ``log_prob`` values dict.

        Parameters
        ----------
        linear_prior
            The merged linear-prior dict (with ``"comp.param"`` qualified keys for
            non-shared params).  ``None`` means treat all linear params as
            marginalizable.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
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

    def params_marginalized(
        self, linear_prior: dict[str, Any] | None
    ) -> tuple[str, ...]:
        """Names of linear parameters analytically marginalized across all components.

        De-duplicated; order follows component iteration order.

        Parameters
        ----------
        linear_prior
            The merged linear-prior dict.  ``None`` means treat all linear params
            as marginalizable.
        """
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
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
        nl_values: dict[str, Any],
        comp_nl: dict[str, dict[str, Any]],
        per_comp_lp: dict[str, dict[str, Any] | None],
        marginalized_names: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Copy explicit-linear values from *nl_values* to per-component dicts.

        Explicit linear priors are sampled alongside nonlinear params and
        appear as bare names in *nl_values*. This method routes them to the
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
                        if qualified in nl_values:
                            comp_nl[comp_name][name] = nl_values[qualified]
                        elif name in nl_values:
                            comp_nl[comp_name][name] = nl_values[name]

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
        linear_prior: dict[str, Any] | None,
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
        linear_prior
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
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
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
        data: Any,
        linear_prior: dict[str, Any] | None,
    ) -> tuple[
        MarginalizedLinear,
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
        linear_prior
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
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
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
        # linear priors are INDEPENDENT, i.e. ``prior_scale_tril`` is diagonal
        # for every component.  Under that assumption the joint prior on the
        # combined linear vector is itself diagonal, and we can read off the
        # joint mean/scale entry-by-entry without ever forming a full
        # covariance matrix or calling ``jnp.linalg.cholesky``.  All current
        # harv parameterizations satisfy this.  If a future component
        # introduces correlated priors (off-diagonal ``prior_scale_tril``),
        # this loop must be rewritten to assemble the full block-structured
        # covariance and Cholesky-factorize it (with a small ridge for PSD
        # safety); see git history before this commit for the previous
        # full-Cholesky construction.
        mu_joint = jnp.zeros(n_global)
        scale_diag_joint = jnp.zeros(n_global)
        filled: set[int] = set()  # global indices already assigned (for shared cols)
        for comp_name in self.components:
            nms = names_by_comp[comp_name]
            L_c = blocks_by_comp[comp_name].prior_scale_tril
            mu_c = blocks_by_comp[comp_name].prior_mu
            for local_j, nm in enumerate(nms):
                owner = None if nm in shared_set else comp_name
                gj = col_idx[(nm, owner)]
                # Shared global slots are populated by the FIRST component
                # that owns them; validation guarantees later components have
                # an identical prior, so subsequent visits are no-ops.
                if gj in filled:
                    continue
                mu_joint = mu_joint.at[gj].set(mu_c[local_j])
                # Diagonal-only assumption: take the (j, j) entry of L_c.
                scale_diag_joint = scale_diag_joint.at[gj].set(L_c[local_j, local_j])
                filled.add(gj)

        prior_joint = dist.MultivariateNormal(
            loc=mu_joint, scale_tril=jnp.diag(scale_diag_joint)
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

        marg_dist = MarginalizedLinear(
            design_matrix=X_joint,
            prior_distribution=prior_joint,
            data_distribution=data_dist_joint,
        )

        explicit_by_comp = {
            comp_name: dict(blocks_by_comp[comp_name].explicit_linear)
            for comp_name in self.components
        }
        return marg_dist, y_joint, global_cols, explicit_by_comp

    def log_prob(
        self,
        nl_values: dict[str, Any],
        data: Any,
        *,
        linear_prior: dict[str, Any] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute the joint log-likelihood.

        When ``shared_linear_params`` contains names that are being analytically
        marginalized, a single joint ``numpyro_ext.distributions.MarginalizedLinear`` is
        built spanning all components (the *joint path*).  Otherwise the per-component
        log-likelihoods are summed as before.

        Parameters
        ----------
        nl_values
            Flat dict of parameter values. Shared orbital params use bare names
            (``"period"``, ``"eccentricity"``, etc.). Component-specific nonlinear
            params use ``"component.param"`` convention (e.g. ``"rv.jitter"``).
        data
            Per-component data, indexed by component name.
        linear_prior
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
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
            else dict.fromkeys(self.component_names)
        )
        shared_nl = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marg, any_shared_marg = self._resolve_marginalization(
            marginalized_names, linear_prior
        )

        comp_nl = _split_nl_values(
            nl_values, shared_nl, self.component_names, per_comp_nl
        )
        self._route_explicit_linear(nl_values, comp_nl, per_comp_lp, per_comp_marg)

        if any_shared_marg:
            # Joint path: a single MarginalizedLinear spanning all components,
            # with shared linear columns merged.  Required when at least one
            # ``shared_linear_params`` entry is being analytically marginalized
            # so its prior is integrated *once* (not once per component).
            marg_dist, y_joint, _, _ = self._build_joint_marginalized_linear(
                comp_nl, per_comp_marg, data, linear_prior
            )
            return marg_dist.log_prob(y_joint)

        # Per-component sum path: each component marginalizes its own linear
        # params independently and we sum the resulting log-likelihoods.  This
        # is the correct behaviour whenever no shared linear param is being
        # marginalized (including the common case of an unshared joint model).
        log_probs = [
            comp.log_prob(
                comp_nl[name],
                data[name],
                linear_prior=per_comp_lp[name],
                marginalized_names=per_comp_marg[name],
            )
            for name, comp in self.components.items()
        ]
        return jnp.sum(jnp.stack(log_probs))

    def sample_conditional_linear(
        self,
        nl_values: dict[str, Any],
        key: jax.Array,
        data: Any,
        *,
        linear_prior: dict[str, Any] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
        use_mean: bool = False,
    ) -> "dict[str, Any]":
        """Sample conditional linear params for each component.

        When ``shared_linear_params`` are jointly marginalized, shared
        parameters appear at the top level of the returned dict (with bare
        names) rather than inside per-component sub-dicts.

        Parameters
        ----------
        nl_values
            Flat parameter values dict.
        key
            JAX PRNG key.
        data
            Per-component data, indexed by component name.
        linear_prior
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
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
            else dict.fromkeys(self.component_names)
        )
        shared_nl = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marg, any_shared_marg = self._resolve_marginalization(
            marginalized_names, linear_prior
        )

        comp_nl = _split_nl_values(
            nl_values, shared_nl, self.component_names, per_comp_nl
        )
        self._route_explicit_linear(nl_values, comp_nl, per_comp_lp, per_comp_marg)

        if any_shared_marg:
            # Joint path: sample from one big conditional posterior over all
            # marginalized linear params, then de-multiplex into a dict whose
            # shared entries sit at the top level (bare name) and whose
            # per-component entries sit in sub-dicts keyed by component name.
            marg_dist, y_joint, global_cols, explicit_by_comp = (
                self._build_joint_marginalized_linear(
                    comp_nl, per_comp_marg, data, linear_prior
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
                subkey,
                data[name],
                linear_prior=per_comp_lp[name],
                marginalized_names=per_comp_marg[name],
                use_mean=use_mean,
            )
        return results

    def numpyro_model(
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: Any,
        linear_prior: dict[str, Any] | None,
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
        linear_prior
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
                linear_prior,
                marginalized_names=marginalized_names,
            )
        return self._build_full_numpyro(nonlinear_priors, data, linear_prior)

    def _build_marginalized_numpyro(  # noqa: C901
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: Any,
        linear_prior: dict[str, Any] | None,
        *,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Build a marginalized numpyro model for the joint model."""
        joint = self
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
            else dict.fromkeys(self.component_names)
        )
        per_comp_marginalized_names, any_shared_marg = self._resolve_marginalization(
            marginalized_names, linear_prior
        )

        # Identify shared linear params that are explicitly sampled (non-Gaussian
        # prior).  These must be sampled exactly *once* at the top of the model
        # and copied to every component, rather than re-sampled per component.
        shared_lin_set = set(self.shared_linear_params)
        shared_explicit_lin: set[str] = set()
        if shared_lin_set and linear_prior is not None:
            for nm in shared_lin_set:
                if nm in linear_prior and _needs_explicit_sampling(linear_prior[nm]):
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
        if shared_explicit_lin and linear_prior is not None:
            first_comp_name = next(iter(self.component_names))
            first_comp = self.components[first_comp_name]
            pu = first_comp._linear_param_units(data[first_comp_name])
            for nm in shared_explicit_lin:
                p = linear_prior[nm]
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
            nl_values = dict(values)
            for name, d in nonlinear_priors.items():
                if isinstance(d, QuantityDistribution) and name in shared:
                    nl_values[name] = Q(values[name], cast("str", d.unit))

            # 2. Sample shared explicit-linear priors ONCE.  Direct priors first
            #    so callable shared priors that depend on them can read the
            #    sampled values out of ``nl_values``.
            for name, p in _shared_explicit_direct_lp.items():
                nl_values[name] = _sample_explicit_linear_prior(
                    name, p, _shared_param_units.get(name, ""), nl_values
                )
            for name, p in _shared_explicit_callable_lp.items():
                nl_values[name] = _sample_explicit_linear_prior(
                    name, p, _shared_param_units.get(name, ""), nl_values
                )

            # 3. Sample per-component explicit-linear priors.  Direct first,
            #    then callable; the per-component ``explicit_linear_proxy``
            #    feeds Q-wrapped values to callable resolvers that need units.
            for comp_name in joint.component_names:
                pu = _comp_param_units[comp_name]
                explicit_linear_proxy: dict[str, Any] = {}
                for name, p in _comp_explicit_direct_lp[comp_name].items():
                    target_u = pu.get(name, "")
                    raw = _sample_explicit_linear_prior(
                        name,
                        p,
                        target_u,
                        nl_values,
                        site_name=f"{comp_name}.{name}",
                    )
                    nl_values[f"{comp_name}.{name}"] = raw
                    explicit_linear_proxy[name] = Q(raw, target_u) if target_u else raw
                for name, p in _comp_explicit_callable_lp[comp_name].items():
                    target_u = pu.get(name, "")
                    raw = _sample_explicit_linear_prior(
                        name, p, target_u, nl_values, extra_values=explicit_linear_proxy
                    )
                    nl_values[f"{comp_name}.{name}"] = raw
                    explicit_linear_proxy[name] = Q(raw, target_u) if target_u else raw

            # 4. Split nl_values per component and route explicit-linear values.
            comp_nl = _split_nl_values(
                nl_values, shared, joint.component_names, per_comp_nl
            )
            joint._route_explicit_linear(
                nl_values, comp_nl, per_comp_lp, per_comp_marginalized_names
            )

            # 5. Compute the marginalized log-likelihood.
            if any_shared_marg:
                # Joint path: a single MarginalizedLinear spanning all
                # components — required so each shared linear prior is
                # integrated once, not once per component.
                marg_dist, y_joint, _, _ = joint._build_joint_marginalized_linear(
                    comp_nl, per_comp_marginalized_names, data, linear_prior
                )
                log_lik = marg_dist.log_prob(y_joint)
            else:
                # Per-component sum: each component's marginalization is
                # independent, so the log-likelihoods simply add.
                log_lik = jnp.zeros(())
                for comp_name, comp in joint.components.items():
                    log_lik = log_lik + comp.log_prob(
                        comp_nl[comp_name],
                        data[comp_name],
                        linear_prior=per_comp_lp[comp_name],
                        marginalized_names=per_comp_marginalized_names[comp_name],
                    )
            numpyro.factor("log_lik", log_lik)

        return model_fn

    def _build_full_numpyro(  # noqa: C901
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: Any,
        linear_prior: dict[str, Any] | None,
    ) -> Callable[[], None]:
        """Build a full (non-marginalized) numpyro model for the joint model.

        Both nonlinear and linear parameters are sampled. Gaussian linear
        priors are sampled jointly across all components; non-Gaussian linear
        priors are sampled individually.
        """
        joint = self
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_lp: dict[str, dict[str, Any] | None] = (
            self._per_component_linear_prior(linear_prior)
            if linear_prior is not None
            else dict.fromkeys(self.component_names)
        )

        # Pre-classify each component's linear priors.
        _comp_gaussian_lp: dict[str, dict[str, Any]] = {}
        _comp_explicit_lp: dict[str, dict[str, Any]] = {}
        _comp_param_units: dict[str, dict[str, str]] = {}
        for comp_name, comp in self.components.items():
            lp = per_comp_lp[comp_name]
            if lp is None:
                msg = (
                    f"Cannot build full numpyro model: component {comp_name!r} "
                    "has no linear_prior"
                )
                raise ValueError(msg)
            gaussian: dict[str, Any] = {}
            explicit: dict[str, Any] = {}
            for n, d in lp.items():
                if _needs_explicit_sampling(d):
                    explicit[n] = d
                else:
                    gaussian[n] = d
            _comp_gaussian_lp[comp_name] = gaussian
            _comp_explicit_lp[comp_name] = explicit
            _comp_param_units[comp_name] = comp._linear_param_units(data[comp_name])

        # Build ordered slot lists for both Gaussian and explicit (non-Gaussian)
        # linear parameters.  Each slot is ``(site_name, comp_name, base_name)``:
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

        _gaussian_slots = _build_slots(_comp_gaussian_lp)
        _explicit_slots = _build_slots(_comp_explicit_lp)

        def model_fn() -> None:  # noqa: C901
            # Sample all nonlinear params
            values = _sample_nonlinear_params(nonlinear_priors)

            # Wrap shared QD priors in Quantity
            nl_values: dict[str, Any] = dict(values)
            for name, d in nonlinear_priors.items():
                if isinstance(d, QuantityDistribution) and name in shared:
                    nl_values[name] = Q(values[name], cast("str", d.unit))

            # Split per component
            comp_nl = _split_nl_values(
                nl_values, shared, joint.component_names, per_comp_nl
            )

            # Per-component view of linear values, used both to feed callable
            # Gaussian priors (proxy) and to assemble each component's log-prob
            # input.  Shared linear values are mirrored into every component's
            # entry so callable priors can read them as bare attributes.
            linear_by_comp: dict[str, dict[str, jax.Array]] = {
                c: {} for c in joint.component_names
            }

            def _record(
                cname: str, base: str, value: jax.Array, *, is_shared: bool
            ) -> None:
                if is_shared:
                    for c in joint.component_names:
                        linear_by_comp[c][base] = value
                else:
                    linear_by_comp[cname][base] = value

            # Sample explicit (non-Gaussian) linear priors using the resolved
            # site names (qualified for per-component, bare for shared).
            for site_name, cname, base in _explicit_slots:
                d = _comp_explicit_lp[cname][base]
                pu = _comp_param_units[cname]
                raw = numpyro.sample(site_name, _unwrap_dist(d))
                target_u = pu.get(base, "")
                if isinstance(d, QuantityDistribution) and target_u:
                    raw = ustrip(target_u, Q(raw, str(d.unit)))
                _record(cname, base, jnp.asarray(raw), is_shared=base in shared_lin_set)

            # Sample all Gaussian linear params jointly as a single _linear site.
            if _gaussian_slots:
                # Resolve per-component priors and combine into one MVN.
                all_locs: list[Any] = []
                all_scales: list[Any] = []
                for _, cname, base in _gaussian_slots:
                    d = _comp_gaussian_lp[cname][base]
                    pu = _comp_param_units[cname]
                    target_u = pu.get(base, "")

                    # Resolve callable priors against the owning component's
                    # nonlinear + already-sampled-explicit values.
                    proxy_values = dict(comp_nl[cname])
                    proxy_values.update(linear_by_comp[cname])
                    params_proxy = types.SimpleNamespace(**proxy_values)
                    if callable(d) and not isinstance(
                        d, dist.Distribution | QuantityDistribution
                    ):
                        resolved = d(params_proxy)
                    else:
                        resolved = d

                    if isinstance(resolved, QuantityDistribution):
                        prior_unit = cast("str", resolved.unit)
                        inner = resolved.distribution
                        loc = ustrip(target_u, Q(inner.loc, prior_unit))
                        scale = ustrip(target_u, Q(inner.scale, prior_unit))
                    elif isinstance(resolved, dist.Normal):
                        loc = resolved.loc
                        scale = resolved.scale
                    else:
                        msg = f"Expected Normal for Gaussian linear prior {base}"
                        raise TypeError(msg)
                    all_locs.append(loc)
                    all_scales.append(scale)

                mvn = dist.MultivariateNormal(
                    loc=jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in all_locs]),
                    scale_tril=jnp.diag(
                        jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in all_scales])
                    ),
                )
                linear_vec = cast("jax.Array", numpyro.sample("_linear", mvn))
                for i, (site_name, cname, base) in enumerate(_gaussian_slots):
                    v = linear_vec[i]
                    numpyro.deterministic(site_name, v)
                    _record(cname, base, v, is_shared=base in shared_lin_set)

            # Evaluate explicit log-likelihood per component
            log_lik = jnp.zeros(())
            for comp_name, comp in joint.components.items():
                comp_linear = {
                    n: linear_by_comp[comp_name][n]
                    for n in comp._all_linear_names()
                    if n in linear_by_comp[comp_name]
                }
                log_lik = log_lik + comp._log_prob_explicit(
                    comp_nl[comp_name], comp_linear, data[comp_name]
                )

            numpyro.factor("log_lik", log_lik)

        return model_fn
