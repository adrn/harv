"""Numpyro MCMC sampler for harv.

Provides the :class:`NumpyroSampler` class for warm-started MCMC using the
new component model API. The component models' ``numpyro_model()`` method
builds the numpyro model closure directly.
"""

import uuid
from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import infer as _numpyro_infer
from numpyro.distributions import biject_to
from unxt import AbstractQuantity, Q
from unxt.quantity import ustrip

from harv.distributions import QuantityDistribution
from harv.extensions.base import AbstractExtension
from harv.models._helpers import PriorDist, _needs_explicit_sampling, _unwrap_dist
from harv.models.component import (
    AbstractComponentModel,
    _apply_unit_conversions,
    _resolve_prior_to_mvn,
    _sample_nonlinear_params,
)
from harv.models.joint import JointModel
from harv.models.parameterizations import AbstractParameterization
from harv.samplers.rejection import (
    _prepare_sampler_model,
    _wrap_unit_values,
)
from harv.samplers.rejection_prior import RejectionPrior
from harv.samplers.samples import Samples

__all__ = ("NumpyroSampler",)


def _build_all_priors(
    prior: RejectionPrior,
    nonlinear_extension_priors: dict[str, Any],
) -> dict[str, PriorDist]:
    """Merge base nonlinear_priors and extension nonlinear priors.

    Extension nonlinear priors are already normalized to model-key convention.
    """
    all_priors: dict[str, PriorDist] = dict(prior.nonlinear_priors)
    all_priors.update(nonlinear_extension_priors)
    return all_priors


def _unconstrain_init_params(
    init_params: dict[str, Any],
    prior: RejectionPrior,
    effective_linear_prior: dict[str, Any] | None = None,
    nonlinear_extension_priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transform init_params from constrained to unconstrained space.

    Numpyro's HMC/NUTS operates in unconstrained space and applies
    ``biject_to(constraint)`` as the forward transform. The init values
    we build from the rejection-sampler posterior are in *constrained*
    (natural) space, so we must apply the inverse transform before
    passing them to the MCMC kernel.
    """
    site_dists: dict[str, dist.Distribution] = {}
    for name, d in prior.nonlinear_priors.items():
        site_dists[name] = _unwrap_dist(d)
    if isinstance(effective_linear_prior, dict):
        for name, d in effective_linear_prior.items():
            if isinstance(d, dist.Distribution | QuantityDistribution):
                site_dists[name] = _unwrap_dist(d)
    if nonlinear_extension_priors:
        for model_key, d in nonlinear_extension_priors.items():
            site_dists[model_key] = _unwrap_dist(d)

    out: dict[str, Any] = {}
    for name, val in init_params.items():
        d = site_dists.get(name)
        if d is not None:
            transform = biject_to(d.support)
            out[name] = transform.inv(jnp.asarray(val))
        else:
            out[name] = val
    return out


def _build_extra_numpyro_model(  # noqa: C901
    model: AbstractComponentModel | JointModel,
    all_priors: dict[str, PriorDist],
    extra_model_fn: Callable[[dict[str, Any]], dict[str, Any]],
    marginalized: bool,
    marginalized_names: tuple[str, ...] | None,
) -> Callable[[], None]:
    """Build a numpyro model with an ``extra_model`` reparameterization.

    Allows users to replace specific linear parameters (e.g. ``rv_semiamp``)
    with deterministic functions of additional physically-motivated parameters
    (e.g. stellar masses and inclination).
    """
    if isinstance(model, JointModel):
        msg = "extra_model is not yet supported with JointModel"
        raise NotImplementedError(msg)

    component = model
    all_linear_names = component._all_linear_names()
    linear_prior = component.linear_prior or {}
    param_units = component._linear_param_units()

    def model_fn() -> None:
        values = _sample_nonlinear_params(all_priors)
        nl_values = _apply_unit_conversions(values, all_priors, component)

        fixed_linear: dict[str, Any] = extra_model_fn(values)

        unknown = set(fixed_linear.keys()) - set(all_linear_names)
        if unknown:
            msg = (
                f"extra_model returned unknown linear parameter name(s): {unknown}. "
                f"Valid names: {all_linear_names}"
            )
            raise ValueError(msg)

        for name, val in fixed_linear.items():
            numpyro.deterministic(name, val)

        free_names = tuple(n for n in all_linear_names if n not in fixed_linear)
        if marginalized:
            requested_marginalized_names = (
                free_names
                if marginalized_names is None
                else tuple(name for name in marginalized_names if name in free_names)
            )
        else:
            requested_marginalized_names = ()

        explicit_names = tuple(
            name for name in free_names if name not in set(requested_marginalized_names)
        )
        explicit_direct_prior = {
            name: linear_prior[name]
            for name in explicit_names
            if name in linear_prior
            and (
                not callable(linear_prior[name])
                or isinstance(
                    linear_prior[name],
                    (dist.Distribution, QuantityDistribution),
                )
            )
        }
        explicit_callable_prior = {
            name: linear_prior[name]
            for name in explicit_names
            if name in linear_prior
            and callable(linear_prior[name])
            and not isinstance(
                linear_prior[name],
                (dist.Distribution, QuantityDistribution),
            )
        }

        explicit_linear_values = dict(fixed_linear)
        explicit_linear_proxy = {
            name: Q(value, param_units.get(name, ""))
            if param_units.get(name, "")
            else value
            for name, value in fixed_linear.items()
        }

        for name, prior_dist in explicit_direct_prior.items():
            raw = numpyro.sample(name, _unwrap_dist(prior_dist))
            target_unit = param_units.get(name, "")
            if isinstance(prior_dist, QuantityDistribution) and target_unit:
                raw = ustrip(target_unit, Q(raw, str(prior_dist.unit)))
            explicit_linear_values[name] = raw
            explicit_linear_proxy[name] = Q(raw, target_unit) if target_unit else raw

        for name, prior_dist in explicit_callable_prior.items():
            target_unit = param_units.get(name, "")
            resolved_prior = _resolve_prior_to_mvn(
                {name: prior_dist},
                nl_values,
                {name: target_unit},
                extra_values=explicit_linear_proxy,
            )
            raw = numpyro.sample(
                name,
                dist.Normal(resolved_prior.loc[0], resolved_prior.scale_tril[0, 0]),
            )
            explicit_linear_values[name] = raw
            explicit_linear_proxy[name] = Q(raw, target_unit) if target_unit else raw

        if marginalized and requested_marginalized_names:
            numpyro.factor(
                "log_lik",
                component.log_prob(
                    nl_values,
                    linear_values=explicit_linear_values,
                    marginalized_names=requested_marginalized_names,
                ),
            )
        elif explicit_linear_values:
            numpyro.factor(
                "log_lik",
                component.log_prob(
                    nl_values,
                    linear_values=explicit_linear_values,
                    marginalized_names=(),
                ),
            )
        else:
            numpyro.factor(
                "log_lik",
                component.log_prob(
                    nl_values,
                    linear_values=explicit_linear_values,
                    marginalized_names=(),
                ),
            )

    return model_fn


class NumpyroSampler(eqx.Module):
    """MCMC sampler for Keplerian orbital parameters using numpyro.

    Builds a numpyro model from a component model (or joint model) and runs
    warm-started MCMC from rejection-sampler output, returning a
    :class:`Samples` container. Data is passed to :meth:`run` rather than at
    construction, so the same configured sampler can be applied to multiple datasets.

    Parameters
    ----------
    prior : RejectionPrior
        Prior distributions for nonlinear (and optionally linear) parameters.
    parameterization : AbstractParameterization or None, optional
        Orbital parameterization. For RV data defaults to
        :class:`~harv.models.parameterizations.rv.StandardRV`. Ignored for Gaia
        astrometry data.
    extensions : tuple of AbstractExtension, optional
        Model extensions (jitter, trends, offsets, GP).

    See Also
    --------
    NumpyroSampler.from_model : Expert path for pre-built models.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers import RejectionPrior, NumpyroSampler
    >>> prior = RejectionPrior.default_rv(
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> mcmc = NumpyroSampler(prior)  # doctest: +SKIP
    >>> mcmc_samples = mcmc.run(
    ...     data, init_samples=rej_samples, num_warmup=500, num_samples=1000
    ... )  # doctest: +SKIP
    """

    prior: RejectionPrior
    parameterization: AbstractParameterization | None = None
    extensions: tuple[AbstractExtension, ...] = ()
    marginalized_names: tuple[str, ...] | None = None
    model: AbstractComponentModel | JointModel | None = None

    def __check_init__(self) -> None:
        if self.model is not None and (
            self.parameterization is not None or self.extensions
        ):
            msg = (
                "Cannot specify parameterization or extensions when model is provided. "
                "Use NumpyroSampler.from_model(model, prior) and configure the "
                "model directly."
            )
            raise ValueError(msg)

    @classmethod
    def from_model(
        cls,
        model: AbstractComponentModel | JointModel,
        prior: RejectionPrior,
    ) -> "NumpyroSampler":
        """Construct from a pre-built model (expert bypass path).

        Use this when you need full control over model construction,
        parameterization, or extensions and want to bypass the automatic
        model-building in :meth:`run`. With this constructor, omit ``data``
        when calling :meth:`run`.

        Parameters
        ----------
        model : AbstractComponentModel or JointModel
            A fully constructed model with data and extensions attached.
        prior : RejectionPrior
            Prior distributions compatible with the model's parameters.

        Returns
        -------
        NumpyroSampler
        """
        return cls(prior, model=model)

    def run(
        self,
        data: Any = None,
        *,
        init_samples: "Samples | None" = None,
        seed: int | None = None,
        marginalized: bool = True,
        extra_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        extra_init_params: dict[str, Any] | None = None,
        kernel: type | None = None,
        num_chains: int = 4,
        num_warmup: int = 500,
        num_samples: int = 1000,
        chain_method: str = "parallel",
    ) -> "Samples":
        """Run MCMC warm-started from rejection-sampler output.

        Parameters
        ----------
        data : RVData or GaiaAstrometryData, optional
            Observed data to condition on. Required unless the sampler was
            constructed via :meth:`from_model`.
        init_samples : Samples, optional
            Posterior samples produced by rejection sampling, used to set the
            initial positions for each MCMC chain.
        seed : int, optional
            Random number seed. If not specified, picks a seed based on the
            current time.
        marginalized : bool, optional
            If ``True`` (default), linear parameters are analytically
            marginalized in the likelihood and conditionally sampled
            afterward. If ``False``, all parameters are sampled jointly.
        extra_model : callable, optional
            A function ``(pars: dict) -> dict`` that replaces specific linear
            parameters with deterministic functions of new physical parameters.
        extra_init_params : dict, optional
            Initial values for parameters introduced by ``extra_model``.
        kernel : type, optional
            A numpyro MCMC kernel class. Defaults to ``numpyro.infer.NUTS``.
        num_chains : int, optional
            Number of independent MCMC chains. Default: 4.
        num_warmup : int, optional
            Number of warmup (burn-in) steps per chain. Default: 500.
        num_samples : int, optional
            Number of posterior samples per chain. Default: 1000.
        chain_method : str, optional
            How to run chains: ``"parallel"``, ``"sequential"``, or
            ``"vectorized"``. Default: ``"parallel"``.

        Returns
        -------
        samples : Samples
            Posterior samples container with nonlinear and linear parameters.
        """
        samples = init_samples
        if samples is None:
            msg = (
                "init_samples is required: provide rejection-sampler output to "
                "initialise MCMC chains. Got init_samples=None."
            )
            raise ValueError(msg)
        if samples.n_samples == 0:
            msg = "Cannot initialise MCMC: no posterior samples available."
            raise ValueError(msg)
        if extra_model is not None and extra_init_params is None:
            msg = "extra_init_params is required when extra_model is provided."
            raise ValueError(msg)
        if not marginalized and self.marginalized_names is not None:
            msg = "marginalized_names cannot be set when marginalized=False"
            raise ValueError(msg)

        if kernel is None:
            kernel = _numpyro_infer.NUTS

        try:
            prepared = _prepare_sampler_model(
                self.prior,
                self.model,
                data,
                self.extensions,
                self.parameterization,
                self.marginalized_names if marginalized else None,
            )
        except ValueError as err:
            if self.model is None and data is None:
                msg = (
                    "data must be provided unless the sampler was constructed via "
                    "NumpyroSampler.from_model(). Got data=None and no pre-built model."
                )
                raise ValueError(msg) from err
            raise

        model = prepared.model
        nonlinear_extension_priors = prepared.nonlinear_extension_priors
        effective_linear_prior = prepared.effective_linear_prior
        effective_marginalized_names = prepared.effective_marginalized_names
        linear_extension_names = prepared.linear_extension_names

        prior = self.prior
        all_priors = _build_all_priors(prior, nonlinear_extension_priors)

        # Build numpyro model
        if extra_model is not None:
            numpyro_model = _build_extra_numpyro_model(
                model,
                all_priors,
                extra_model,
                marginalized,
                effective_marginalized_names if marginalized else None,
            )
        else:
            numpyro_model = model.numpyro_model(
                all_priors,
                marginalized=marginalized,
                marginalized_names=(
                    effective_marginalized_names if marginalized else None
                ),
            )

        # Build init_params from rejection-sampler posterior
        init_params = self._build_init_params(
            model,
            samples,
            effective_linear_prior=effective_linear_prior,
            effective_marginalized_names=effective_marginalized_names,
            marginalized=marginalized,
            num_chains=num_chains,
            extra_model=extra_model,
            extra_init_params=extra_init_params,
        )
        init_params = _unconstrain_init_params(
            init_params,
            prior,
            effective_linear_prior=effective_linear_prior,
            nonlinear_extension_priors=nonlinear_extension_priors,
        )

        # Create and run MCMC
        seed = uuid.uuid4().int >> 96 if seed is None else seed
        rng_key = jr.key(seed)

        kernel_instance = kernel(numpyro_model)
        mcmc = _numpyro_infer.MCMC(
            kernel_instance,
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
            chain_method=chain_method,
        )
        mcmc.run(rng_key, init_params=init_params)
        posterior = mcmc.get_samples()

        # Convert numpyro posterior dict -> Samples
        return self._posterior_to_samples(
            model,
            posterior,
            rng_key,
            marginalized=marginalized,
            nonlinear_extension_priors=nonlinear_extension_priors,
            effective_marginalized_names=effective_marginalized_names,
            linear_extension_names=linear_extension_names,
            num_chains=num_chains,
        )

    def _build_init_params(  # noqa: C901
        self,
        _model: AbstractComponentModel | JointModel,
        samples: "Samples",
        *,
        effective_linear_prior: dict[str, Any] | None,
        effective_marginalized_names: tuple[str, ...] | None,
        marginalized: bool,
        num_chains: int,
        extra_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        extra_init_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build init_params dict from rejection-sampler posterior."""
        _scalar_init = num_chains == 1
        # Cycle through available samples to fill num_chains slots.
        # When n_samples >= num_chains this picks the first num_chains distinct
        # samples; when n_samples < num_chains (e.g. only 1 rejection sample
        # accepted) the available samples are repeated so that every chain has
        # a valid starting point.
        indices = [i % samples.n_samples for i in range(num_chains)]

        if _scalar_init:
            init_params: dict[str, Any] = {
                key_name: jnp.asarray(ustrip(str(qty.unit), qty)[0])
                for key_name, qty in samples.nonlinear.items()
            }
        else:
            init_params = {
                key_name: jnp.stack([ustrip(str(qty.unit), qty)[i] for i in indices])
                for key_name, qty in samples.nonlinear.items()
            }

        # Include init values for explicit (non-marginalized) linear params.
        # For both the marginalized and non-marginalized cases, any linear param
        # with a non-Gaussian prior (needs_explicit_sampling) must be included in
        # init_params because the numpyro model samples it explicitly rather than
        # analytically marginalizing it.
        if isinstance(effective_linear_prior, dict):
            explicit_linear_name_set = (
                {
                    name
                    for name in effective_linear_prior
                    if name not in set(effective_marginalized_names or ())
                }
                if marginalized
                else {
                    name
                    for name, prior_dist in effective_linear_prior.items()
                    if _needs_explicit_sampling(prior_dist)
                }
            )
            for name, d in effective_linear_prior.items():
                if name not in explicit_linear_name_set or name not in samples.linear:
                    continue
                qty = samples.linear[name]
                if isinstance(d, QuantityDistribution):
                    prior_unit = str(d.unit)
                    vals = ustrip(prior_unit, qty)
                else:
                    vals = np.asarray(qty.value)
                if _scalar_init:
                    init_params[name] = jnp.asarray(vals[0])
                else:
                    init_params[name] = jnp.stack(
                        [jnp.asarray(vals[i]) for i in indices]
                    )

        # Remap nonlinear keys that use model-key convention.
        # Extension params in Samples.nonlinear may use model keys
        # (e.g. "rv.jitter") and are already correctly named for numpyro.

        if extra_model is not None:
            init_params.update(extra_init_params)  # type: ignore[arg-type]

        if (
            not marginalized
            and extra_model is None
            and isinstance(effective_linear_prior, dict)
        ):
            # Full model: include init values for _linear site.
            gaussian_names = [
                n
                for n in effective_linear_prior
                if not _needs_explicit_sampling(effective_linear_prior[n])
                and n in samples.linear
            ]
            if gaussian_names:
                lin_arr = np.column_stack(
                    [np.asarray(samples.linear[n].value) for n in gaussian_names]
                )
                if _scalar_init:
                    init_params["_linear"] = jnp.asarray(lin_arr[0])
                else:
                    init_params["_linear"] = jnp.stack(
                        [jnp.asarray(lin_arr[i]) for i in indices]
                    )

        return init_params

    def _posterior_to_samples(
        self,
        model: AbstractComponentModel | JointModel,
        posterior: dict[str, Any],
        rng_key: jax.Array,
        *,
        marginalized: bool,
        nonlinear_extension_priors: dict[str, Any],
        effective_marginalized_names: tuple[str, ...] | None,
        linear_extension_names: tuple[str, ...],
        num_chains: int = 1,
    ) -> Samples:
        """Convert a numpyro posterior dict into a :class:`Samples` container."""
        prior = self.prior

        # Build nonlinear dict with units from the prior.
        _all_nl_priors: dict[str, Any] = dict(prior.nonlinear_priors)
        _all_nl_priors.update(nonlinear_extension_priors)

        nonlinear_q: dict[str, AbstractQuantity] = {}
        for key, d in _all_nl_priors.items():
            if key in posterior:
                unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
                nonlinear_q[key] = Q(posterior[key], unit)

        # Build linear dict.
        if marginalized:
            # Linear params were analytically marginalized -- sample them
            # conditionally from the posterior nonlinear values.
            linear_q = self._sample_conditional_linear(
                model,
                posterior,
                rng_key,
                nonlinear_extension_priors,
                effective_marginalized_names,
            )
        else:
            # Non-marginalized: linear params are in the posterior as named
            # deterministic sites.
            linear_q = self._extract_linear_from_posterior(model, posterior)

        # Build metadata
        t_ref: Any = None
        if isinstance(model, JointModel):
            first_comp = next(iter(model.components.values()))
            if hasattr(first_comp, "data") and hasattr(first_comp.data, "t_ref"):
                t_ref = first_comp.data.t_ref
        elif hasattr(model, "data") and hasattr(model.data, "t_ref"):
            t_ref = model.data.t_ref

        metadata: dict[str, Any] = {"num_chains": num_chains}
        if t_ref is not None:
            # Strip to a plain Python float so a JAX-traced array never lands in a
            # static metadata dict (which would trigger an equinox UserWarning).
            _t_unit = str(t_ref.unit) if hasattr(t_ref, "unit") else ""
            metadata["t_ref"] = (
                float(ustrip(_t_unit, t_ref)) if _t_unit else float(t_ref)
            )
            metadata["t_ref_unit"] = _t_unit

        return Samples(
            nonlinear=cast("dict[str, Q]", nonlinear_q),
            linear=cast("dict[str, Q]", linear_q),
            data_type=type(model).__name__,
            metadata=metadata,
            linear_extension_names=linear_extension_names,
        )

    def _sample_conditional_linear(  # noqa: C901
        self,
        model: AbstractComponentModel | JointModel,
        posterior: dict[str, Any],
        rng_key: jax.Array,
        nonlinear_extension_priors: dict[str, Any],
        effective_marginalized_names: tuple[str, ...] | None,
    ) -> dict[str, AbstractQuantity]:
        """Conditionally sample linear params given MCMC nonlinear posterior."""
        prior = self.prior
        base_names = model._base_nonlinear_names()
        if isinstance(model, JointModel):
            linear_units: dict[str, str] = {}
            if effective_marginalized_names is None:
                explicit_linear_names = {
                    name
                    for comp in model.components.values()
                    for name in set(comp._all_linear_names())
                    - set(comp._auto_marginalized_names())
                }
            else:
                per_comp_marginalized_names = (
                    model._resolve_component_marginalized_names(
                        effective_marginalized_names
                    )
                )
                assert per_comp_marginalized_names is not None
                explicit_linear_names = {
                    name
                    for comp_name, comp in model.components.items()
                    for name in set(comp._all_linear_names())
                    - set(per_comp_marginalized_names[comp_name])
                }
            for comp in model.components.values():
                linear_units.update(comp._linear_param_units())
        else:
            linear_units = model._linear_param_units()
            marginalized_name_set = set(
                model._auto_marginalized_names()
                if effective_marginalized_names is None
                else effective_marginalized_names
            )
            explicit_linear_names = (
                set(model._all_linear_names()) - marginalized_name_set
            )

        # Collect all keys the model needs: base nonlinear, extension nonlinear
        # (using model key convention), and explicitly-sampled linear params.
        nl_keys = list(prior.nonlinear_priors.keys())

        for model_key in nonlinear_extension_priors:
            if model_key in posterior and model_key not in nl_keys:
                nl_keys.append(model_key)

        for name in explicit_linear_names:
            if name in posterior and name not in nl_keys:
                nl_keys.append(name)

        n_samples = len(posterior[nl_keys[0]])
        keys = jr.split(jr.fold_in(rng_key, 3), n_samples)

        filtered = {k: posterior[k] for k in nl_keys if k in posterior}

        def _sample_one(
            key: jax.Array,
            sample: dict[str, jax.Array],
        ) -> dict[str, Any]:
            wrapped = _wrap_unit_values(sample, prior.nonlinear_priors, base_names)
            for name in explicit_linear_names:
                if name in sample and name not in wrapped:
                    unit = linear_units.get(name, "")
                    wrapped[name] = Q(sample[name], unit) if unit else sample[name]
            return model.sample_conditional_linear(
                wrapped,
                key,
                marginalized_names=effective_marginalized_names,
            )

        result = jax.vmap(_sample_one)(keys, filtered)

        # Attach units
        if isinstance(model, JointModel):
            final: dict[str, AbstractQuantity] = {}
            for comp_name, comp in model.components.items():
                units = comp._linear_param_units()
                comp_result = result[comp_name]
                for name, arr in comp_result.items():
                    final[name] = Q(arr, units.get(name, ""))
            return final
        units = model._linear_param_units()
        return {name: Q(arr, units.get(name, "")) for name, arr in result.items()}

    def _extract_linear_from_posterior(
        self,
        model: AbstractComponentModel | JointModel,
        posterior: dict[str, Any],
    ) -> dict[str, AbstractQuantity]:
        """Extract linear params from a non-marginalized posterior."""
        if isinstance(model, JointModel):
            linear_q: dict[str, AbstractQuantity] = {}
            for comp in model.components.values():
                units = comp._linear_param_units()
                for name in comp._all_linear_names():
                    if name in posterior:
                        linear_q[name] = Q(posterior[name], units.get(name, ""))
            return linear_q
        units = model._linear_param_units()
        linear_q = {}
        for name in model._all_linear_names():
            if name in posterior:
                linear_q[name] = Q(posterior[name], units.get(name, ""))
        return linear_q
