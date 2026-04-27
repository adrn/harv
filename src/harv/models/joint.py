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
import numpy as np
import numpyro
import numpyro.distributions as dist
from unxt import Q
from unxt.quantity import ustrip

from harv.distributions import QuantityDistribution
from harv.extensions.base import ParamInfo
from harv.models._helpers import PriorDist, _needs_explicit_sampling, _unwrap_dist
from harv.models.component import (
    AbstractComponentModel,
    _resolve_prior_to_mvn,
    _sample_nonlinear_params,
)


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


def _synchronize_component_t_refs(
    components: dict[str, AbstractComponentModel],
) -> dict[str, AbstractComponentModel]:
    """Return components with all component data sharing a common t_ref.

    All component models share ``phase_peri``, which is interpreted as a
    fraction of the orbit elapsed since ``t_ref``.  When component datasets
    have different reference epochs the same ``phase_peri`` value maps to
    different absolute periastron times, corrupting a joint fit.  This
    function computes the global mean observation time across all components
    and rebuilds each component with that shared epoch.
    """
    if len(components) <= 1:
        return dict(components)

    first = next(iter(components.values()))
    time_unit = str(first.data.time.unit)

    all_times = np.concatenate(
        [np.asarray(ustrip(time_unit, comp.data.time)) for comp in components.values()]
    )
    shared_t_ref = Q(float(np.mean(all_times)), time_unit)

    return {
        name: eqx.tree_at(lambda m: m.data.t_ref, comp, shared_t_ref)
        for name, comp in components.items()
    }


@final
class JointModel(eqx.Module):
    """Composition of component models that share orbital parameters.

    Parameters
    ----------
    components : dict[str, AbstractComponentModel]
        Named component models. Keys are used to namespace component-specific
        parameters (e.g. ``"rv.jitter"``).
    shared_params : tuple of str
        Names of orbital parameters shared across all components (e.g.
        ``("period", "eccentricity", "phase_peri", "arg_peri")``).

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.data import RVData
    >>> from harv.models import RVModel
    >>> from harv.models.joint import JointModel
    >>> data1 = RVData(
    ...     time=Q([0.0, 50.0], "day"),
    ...     rv=Q([1.0, -2.0], "km/s"),
    ...     rv_err=Q([0.5, 0.5], "km/s"),
    ... )
    >>> data2 = RVData(
    ...     time=Q([10.0, 60.0], "day"),
    ...     rv=Q([-0.5, 1.5], "km/s"),
    ...     rv_err=Q([0.3, 0.3], "km/s"),
    ... )
    >>> joint = JointModel.for_sb2(
    ...     components={
    ...                     "primary": RVModel(data=data1),
    ...                     "secondary": RVModel(data=data2)
    ...     },
    ... )
    >>> sorted(joint.component_names)
    ['primary', 'secondary']
    """

    components: dict[str, AbstractComponentModel]
    shared_params: tuple[str, ...]

    @classmethod
    def for_sb2(
        cls,
        components: dict[str, AbstractComponentModel],
        *,
        shared_params: tuple[str, ...] | None = None,
    ) -> "JointModel":
        """Build a JointModel for an SB2 (two RV components).

        Parameters
        ----------
        components : dict[str, AbstractComponentModel]
            Two RV component models (e.g. ``{"primary": ..., "secondary": ...}``).
        shared_params : tuple of str, optional
            Override the default shared orbital parameters. Defaults to
            ``("period", "eccentricity", "phase_peri", "arg_peri")``.

        Returns
        -------
        JointModel
        """
        if shared_params is None:
            shared_params = _DEFAULT_SHARED_PARAMS
        return cls(components=components, shared_params=shared_params)

    @classmethod
    def for_rv_and_gaia(
        cls,
        components: dict[str, AbstractComponentModel],
        *,
        shared_params: tuple[str, ...] | None = None,
    ) -> "JointModel":
        """Build a JointModel for combined RV + Gaia astrometry.

        Parameters
        ----------
        components : dict[str, AbstractComponentModel]
            RV and Gaia astrometry component models
            (e.g. ``{"rv": ..., "astro": ...}``).
        shared_params : tuple of str, optional
            Override the default shared orbital parameters. Defaults to
            ``("period", "eccentricity", "phase_peri", "arg_peri")``.

        Returns
        -------
        JointModel
        """
        if shared_params is None:
            shared_params = _DEFAULT_SHARED_PARAMS
        components = _synchronize_component_t_refs(components)
        return cls(components=components, shared_params=shared_params)

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

    @property
    def params_explicit(self) -> tuple[str, ...]:
        """Names of parameters that must be explicitly sampled.

        Shared nonlinear params use bare names (e.g. ``"period"``).
        Component-specific nonlinear params use ``"comp.param"`` notation
        (e.g. ``"rv.jitter"``).  Explicit-linear params (non-Gaussian priors,
        e.g. ``"parallax"``) are listed flat without namespace prefix,
        matching how they appear in the ``log_prob`` values dict.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.data import RVData
        >>> from harv.models import RVModel
        >>> from harv.models.joint import JointModel
        >>> d = RVData(time=Q([0., 50.], "day"), rv=Q([1., -1.], "km/s"),
        ...           rv_err=Q([0.5, 0.5], "km/s"))
        >>> joint = JointModel.for_sb2(
        ...     {"primary": RVModel(data=d), "secondary": RVModel(data=d)}
        ... )
        >>> set(joint.params_explicit) >= {"period", "eccentricity"}
        True
        """
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
        for comp in self.components.values():
            marg = set(comp._auto_marginalized_names())
            for name in comp._all_linear_names():
                if name not in marg and name not in seen:
                    explicit_lin.append(name)
                    seen.add(name)

        return shared_names + tuple(comp_specific) + tuple(explicit_lin)

    @property
    def params_marginalized(self) -> tuple[str, ...]:
        """Names of linear parameters analytically marginalized across all components.

        De-duplicated; order follows component iteration order.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.data import RVData
        >>> from harv.models import RVModel
        >>> from harv.models.joint import JointModel
        >>> d = RVData(time=Q([0., 50.], "day"), rv=Q([1., -1.], "km/s"),
        ...           rv_err=Q([0.5, 0.5], "km/s"))
        >>> joint = JointModel.for_sb2(
        ...     {"primary": RVModel(data=d), "secondary": RVModel(data=d)}
        ... )
        >>> joint.params_marginalized
        ('rv_semiamp', 'v_sys')
        """
        seen: set[str] = set()
        names: list[str] = []
        for comp in self.components.values():
            for name in comp._auto_marginalized_names():
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
        marginalized_names: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Copy explicit-linear values from *nl_values* to per-component dicts.

        Explicit linear priors are sampled alongside nonlinear params and
        appear as bare names in *nl_values*. This method routes them to the
        correct component.
        """
        for comp_name, comp in self.components.items():
            if comp.linear_prior:
                if marginalized_names is None:
                    explicit_name_set = set(comp._all_linear_names()) - set(
                        comp._auto_marginalized_names()
                    )
                else:
                    explicit_name_set = set(comp._all_linear_names()) - set(
                        marginalized_names.get(comp_name, ())
                    )
                for name in comp._all_linear_names():
                    if name in explicit_name_set and name in nl_values:
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

    def log_prob(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute the joint (summed) log-likelihood.

        Parameters
        ----------
        nl_values : dict
            Flat dict of parameter values. Shared orbital params use bare
            names (``"period"``, ``"eccentricity"``, etc.). Component-specific
            nonlinear params use ``"component.param"`` convention (e.g.
            ``"rv.jitter"``).

        Returns
        -------
        jax.Array
            Scalar log-likelihood (sum over components).
        """
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marginalized_names = self._resolve_component_marginalized_names(
            marginalized_names
        )

        comp_nl = _split_nl_values(nl_values, shared, self.component_names, per_comp_nl)
        self._route_explicit_linear(nl_values, comp_nl, per_comp_marginalized_names)

        log_probs = []
        for name, comp in self.components.items():
            log_probs.append(
                comp.log_prob(
                    comp_nl[name],
                    marginalized_names=(
                        None
                        if per_comp_marginalized_names is None
                        else per_comp_marginalized_names[name]
                    ),
                )
            )
        return jnp.sum(jnp.stack(log_probs))

    def sample_conditional_linear(
        self,
        nl_values: dict[str, Any],
        key: jax.Array,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, jax.Array]]:
        """Sample conditional linear params for each component.

        Returns a dict keyed by component name, each containing the
        component's sampled linear parameter values.
        """
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marginalized_names = self._resolve_component_marginalized_names(
            marginalized_names
        )

        comp_nl = _split_nl_values(nl_values, shared, self.component_names, per_comp_nl)
        self._route_explicit_linear(nl_values, comp_nl, per_comp_marginalized_names)

        results: dict[str, dict[str, jax.Array]] = {}
        for name, comp in self.components.items():
            key, subkey = jax.random.split(key)
            results[name] = comp.sample_conditional_linear(
                comp_nl[name],
                subkey,
                marginalized_names=(
                    None
                    if per_comp_marginalized_names is None
                    else per_comp_marginalized_names[name]
                ),
            )
        return results

    def numpyro_model(
        self,
        nonlinear_priors: dict[str, PriorDist],
        *,
        marginalized: bool = True,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Build a numpyro model for MCMC sampling of the joint model.

        Parameters
        ----------
        nonlinear_priors : dict[str, PriorDist]
            Prior distributions for all nonlinear parameters. Shared orbital
            params use bare names. Component-specific params use
            ``"component.param"`` convention.
        marginalized : bool
            If ``True`` (default), linear parameters are marginalized
            per-component. If ``False``, all parameters are sampled
            explicitly.
        marginalized_names : tuple of str or None
            Optional linear parameter names to marginalize when
            ``marginalized=True``. Component-qualified names are accepted.

        Returns
        -------
        model_fn : callable
        """
        if not marginalized and marginalized_names is not None:
            msg = "marginalized_names cannot be set when marginalized=False"
            raise ValueError(msg)
        if marginalized:
            return self._build_marginalized_numpyro(
                nonlinear_priors,
                marginalized_names=marginalized_names,
            )
        return self._build_full_numpyro(nonlinear_priors)

    def _build_marginalized_numpyro(
        self,
        nonlinear_priors: dict[str, PriorDist],
        *,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Build a marginalized numpyro model for the joint model."""
        joint = self
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()
        per_comp_marginalized_names = self._resolve_component_marginalized_names(
            marginalized_names
        ) or {
            comp_name: comp._auto_marginalized_names()
            for comp_name, comp in self.components.items()
        }

        # Pre-compute explicitly sampled linear priors per component.
        _comp_explicit_direct_lp: dict[str, dict[str, Any]] = {}
        _comp_explicit_callable_lp: dict[str, dict[str, Any]] = {}
        _comp_param_units: dict[str, dict[str, str]] = {}
        for comp_name, comp in self.components.items():
            lp = comp.linear_prior or {}
            requested_marginalized_names = per_comp_marginalized_names[comp_name]
            explicit_linear_prior = {
                name: prior_dist
                for name, prior_dist in lp.items()
                if name not in set(requested_marginalized_names)
            }
            _comp_explicit_direct_lp[comp_name] = {
                name: prior_dist
                for name, prior_dist in explicit_linear_prior.items()
                if not callable(prior_dist)
                or isinstance(prior_dist, (dist.Distribution, QuantityDistribution))
            }
            _comp_explicit_callable_lp[comp_name] = {
                name: prior_dist
                for name, prior_dist in explicit_linear_prior.items()
                if callable(prior_dist)
                and not isinstance(
                    prior_dist, (dist.Distribution, QuantityDistribution)
                )
            }
            _comp_param_units[comp_name] = comp._linear_param_units()

        def model_fn() -> None:
            # Sample all nonlinear params
            values = _sample_nonlinear_params(nonlinear_priors)

            # Wrap shared QD priors in Quantity
            nl_values = dict(values)
            for name, d in nonlinear_priors.items():
                if isinstance(d, QuantityDistribution) and name in shared:
                    nl_values[name] = Q(values[name], cast("str", d.unit))

            # Sample explicit linear params per component and merge
            for comp_name in joint.component_names:
                pu = _comp_param_units[comp_name]
                explicit_linear_proxy: dict[str, Any] = {}
                for name, prior_dist in _comp_explicit_direct_lp[comp_name].items():
                    raw = numpyro.sample(name, _unwrap_dist(prior_dist))
                    target_u = pu.get(name, "")
                    if isinstance(prior_dist, QuantityDistribution) and target_u:
                        raw = ustrip(target_u, Q(raw, cast("str", prior_dist.unit)))
                    nl_values[name] = raw
                    explicit_linear_proxy[name] = Q(raw, target_u) if target_u else raw

                for name, prior_dist in _comp_explicit_callable_lp[comp_name].items():
                    target_u = pu.get(name, "")
                    resolved_prior = _resolve_prior_to_mvn(
                        {name: prior_dist},
                        nl_values,
                        {name: target_u},
                        extra_values=explicit_linear_proxy,
                    )
                    raw = numpyro.sample(
                        name,
                        dist.Normal(
                            resolved_prior.loc[0],
                            resolved_prior.scale_tril[0, 0],
                        ),
                    )
                    nl_values[name] = raw
                    explicit_linear_proxy[name] = Q(raw, target_u) if target_u else raw

            # Split per component and route explicit linear values
            comp_nl = _split_nl_values(
                nl_values, shared, joint.component_names, per_comp_nl
            )
            joint._route_explicit_linear(
                nl_values,
                comp_nl,
                per_comp_marginalized_names,
            )

            log_lik = jnp.zeros(())
            for comp_name, comp in joint.components.items():
                log_lik = log_lik + comp.log_prob(
                    comp_nl[comp_name],
                    marginalized_names=per_comp_marginalized_names[comp_name],
                )
            numpyro.factor("log_lik", log_lik)

        return model_fn

    def _build_full_numpyro(  # noqa: C901
        self,
        nonlinear_priors: dict[str, PriorDist],
    ) -> Callable[[], None]:
        """Build a full (non-marginalized) numpyro model for the joint model.

        Both nonlinear and linear parameters are sampled. Gaussian linear
        priors are sampled jointly across all components; non-Gaussian linear
        priors are sampled individually.
        """
        joint = self
        shared = self._shared_param_names()
        per_comp_nl = self._per_component_nonlinear_names()

        # Pre-classify each component's linear priors.
        _comp_gaussian_lp: dict[str, dict[str, Any]] = {}
        _comp_explicit_lp: dict[str, dict[str, Any]] = {}
        _comp_param_units: dict[str, dict[str, str]] = {}
        for comp_name, comp in self.components.items():
            lp = comp.linear_prior
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
            _comp_param_units[comp_name] = comp._linear_param_units()

        # Build ordered list of all Gaussian linear names across components.
        _all_gaussian_names: list[str] = []
        _name_to_comp: dict[str, str] = {}
        for comp_name in joint.component_names:
            for n in _comp_gaussian_lp[comp_name]:
                _all_gaussian_names.append(n)
                _name_to_comp[n] = comp_name

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

            # Collect all explicit (non-Gaussian) linear params across components
            all_explicit: dict[str, jax.Array] = {}
            for comp_name in joint.component_names:
                pu = _comp_param_units[comp_name]
                for name, d in _comp_explicit_lp[comp_name].items():
                    raw = numpyro.sample(name, _unwrap_dist(d))
                    target_u = pu.get(name, "")
                    if isinstance(d, QuantityDistribution) and target_u:
                        raw = ustrip(target_u, Q(raw, cast("str", d.unit)))
                    all_explicit[name] = raw

            # Sample all Gaussian linear params jointly as a single _linear site
            all_linear: dict[str, jax.Array] = {}
            all_linear.update(all_explicit)
            if _all_gaussian_names:
                # Resolve per-component priors and combine into one MVN
                all_locs: list[Any] = []
                all_scales: list[Any] = []
                for gname in _all_gaussian_names:
                    cname = _name_to_comp[gname]
                    d = _comp_gaussian_lp[cname][gname]
                    pu = _comp_param_units[cname]
                    target_u = pu.get(gname, "")

                    # Resolve callable priors
                    proxy_values = dict(comp_nl[cname])
                    proxy_values.update(all_explicit)
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
                        msg = f"Expected Normal for Gaussian linear prior {gname}"
                        raise TypeError(msg)
                    all_locs.append(loc)
                    all_scales.append(scale)

                mvn = dist.MultivariateNormal(
                    loc=jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in all_locs]),
                    scale_tril=jnp.diag(
                        jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in all_scales])
                    ),
                )
                linear_vec = numpyro.sample("_linear", mvn)
                for i, lname in enumerate(_all_gaussian_names):
                    numpyro.deterministic(lname, linear_vec[i])
                    all_linear[lname] = linear_vec[i]

            # Evaluate explicit log-likelihood per component
            log_lik = jnp.zeros(())
            for comp_name, comp in joint.components.items():
                # Build this component's linear values
                comp_linear: dict[str, jax.Array] = {}
                for n in comp._all_linear_names():
                    if n in all_linear:
                        comp_linear[n] = all_linear[n]
                log_lik = log_lik + comp._log_prob_explicit(
                    comp_nl[comp_name], comp_linear, comp.data
                )

            numpyro.factor("log_lik", log_lik)

        return model_fn
