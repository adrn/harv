"""Rejection sampler for orbital parameter inference."""

import uuid
import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from unxt import AbstractQuantity, Q
from unxt.quantity import ustrip

from harv.data.containers import InputData
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    _evaluate_nonlinear_log_prior,
    _needs_explicit_sampling,
    _unwrap_dist,
)
from harv.models.component import AbstractComponentModel
from harv.models.joint import JointModel
from harv.models.priors import HarvPrior
from harv.samplers.base import AbstractSampler, _validate_data
from harv.samplers.samples import Samples

__all__ = ("RejectionSampler",)


class _PreparedSamplerModel(NamedTuple):
    """Normalized model-preparation bundle shared across sampler entry paths."""

    model: AbstractComponentModel | JointModel
    nonlinear_extension_priors: dict[str, Any]
    effective_linear_prior: dict[str, Any] | None
    effective_marginalized_names: tuple[str, ...] | None
    linear_extension_names: tuple[str, ...]


def _lookup_extension_prior(
    extension_priors: Mapping[str, Any],
    param_name: str,
    *,
    component_name: str = "",
) -> Any | None:
    """Get an extension prior by component-qualified or bare parameter name."""
    if component_name:
        qualified_name = f"{component_name}.{param_name}"
        if qualified_name in extension_priors:
            return extension_priors[qualified_name]
    return extension_priors.get(param_name)


def _iter_component_extensions(
    model: AbstractComponentModel | JointModel,
) -> list[tuple[str, Any]]:
    """Return ``(component_name, extension)`` pairs for a sampler model."""
    if isinstance(model, JointModel):
        return [
            (comp_name, ext)
            for comp_name, comp in model.components.items()
            for ext in comp.extensions
        ]
    return [("", ext) for ext in model.extensions]


def _extension_model_key(component_name: str, param_name: str) -> str:
    """Return the flattened sampler/model key for an extension parameter."""
    return f"{component_name}.{param_name}" if component_name else param_name


def _resolve_effective_marginalized_names(
    effective_linear_prior: dict[str, Any] | None,
    marginalized_names: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    """Resolve and validate the effective marginalized linear parameter subset."""
    if effective_linear_prior is None:
        return marginalized_names

    if marginalized_names is not None:
        unknown = set(marginalized_names) - set(effective_linear_prior)
        if unknown:
            msg = (
                "marginalized_names contains unknown linear parameter(s): "
                f"{unknown}. Valid names: {tuple(effective_linear_prior.keys())}"
            )
            raise ValueError(msg)

    # Only check names the user actually wants to marginalize
    names_to_check = (
        set(effective_linear_prior)
        if marginalized_names is None
        else set(marginalized_names)
    )

    explicit = {
        name
        for name in names_to_check
        if _needs_explicit_sampling(effective_linear_prior[name])
    }
    if not explicit:
        return marginalized_names

    if marginalized_names is None:
        resolved_names = tuple(
            name for name in effective_linear_prior if name not in explicit
        )
    else:
        resolved_names = tuple(
            name for name in marginalized_names if name not in explicit
        )

    warnings.warn(
        f"Non-Gaussian linear prior(s) {sorted(explicit)} cannot be analytically "
        f"marginalized and will be sampled explicitly. Marginalized parameters: "
        f"{resolved_names}",
        stacklevel=3,
    )
    return resolved_names


def _effective_linear_prior_from_prior(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
) -> dict[str, Any] | None:
    """Build the effective linear prior from prior.linear_priors + extensions.

    Models are now templates and carry no ``linear_prior`` themselves.  The
    sampler computes it at run-time by merging ``prior.linear_prior`` with any
    linear-extension parameters declared on the model's extensions.
    """
    effective: dict[str, Any] | None = (
        dict(prior.linear_priors)
        if isinstance(prior.linear_priors, dict)
        else prior.linear_priors
    )
    if effective is None:
        return None
    # Merge linear extension params from the model.
    for comp_name, ext in _iter_component_extensions(model):
        for p in ext.extra_params():
            if not p.linear:
                continue
            extension_prior = _lookup_extension_prior(
                prior.extension_priors,
                p.name,
                component_name=comp_name,
            )
            if extension_prior is not None:
                effective[_extension_model_key(comp_name, p.name)] = extension_prior
    return effective


def _validate_extension_priors(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
    effective_linear_prior: dict[str, Any] | None,
) -> None:
    """Ensure every extension-declared parameter has a prior before sampling."""
    linear_names = (
        set(effective_linear_prior)
        if isinstance(effective_linear_prior, dict)
        else set()
    )
    missing: list[str] = []

    for comp_name, ext in _iter_component_extensions(model):
        for p in ext.extra_params():
            model_key = _extension_model_key(comp_name, p.name)
            if p.linear:
                if model_key not in linear_names:
                    missing.append(model_key)
                continue

            extension_prior = _lookup_extension_prior(
                prior.extension_priors,
                p.name,
                component_name=comp_name,
            )
            if extension_prior is None:
                missing.append(model_key)

    if missing:
        msg = (
            "Missing required prior(s) for extension parameter(s): "
            f"{tuple(missing)}. Add priors for every parameter declared by "
            "model.extensions."
        )
        raise ValueError(msg)


def _nonlinear_extension_priors_from_model(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Derive nonlinear extension priors and linear extension names from a model.

    Walks the model's extensions, looks up matching priors in
    ``prior.extension_priors``, and routes them by the ``linear`` flag on
    each param. Component-qualified names (for example ``"rv.jitter"``)
    are preferred when the model is a :class:`JointModel`, but bare names are
    accepted as a fallback for backward compatibility.

    Used for the ``from_model`` expert path, where ``_resolve_extension_priors``
    is not called.

    Returns
    -------
    nonlinear_extension_priors : dict[str, PriorDist]
        Extension nonlinear params, keyed by model-key convention.
    linear_extension_names : tuple[str, ...]
        Names of extension linear (offset) params.
    """
    nonlinear_extension_priors: dict[str, Any] = {}
    linear_extension_names: list[str] = []

    for comp_name, ext in _iter_component_extensions(model):
        for p in ext.extra_params():
            extension_prior = _lookup_extension_prior(
                prior.extension_priors,
                p.name,
                component_name=comp_name,
            )
            if extension_prior is None:
                continue
            model_key = _extension_model_key(comp_name, p.name)
            if p.linear:
                linear_extension_names.append(model_key)
            else:
                nonlinear_extension_priors[model_key] = extension_prior

    return nonlinear_extension_priors, tuple(linear_extension_names)


def _ext_nonlinear_from_model(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Backward-compatible alias for nonlinear extension prior extraction."""
    return _nonlinear_extension_priors_from_model(prior, model)


def _ext_nonlinear_model_keys(
    nonlinear_extension_priors: dict[str, Any],
    model: AbstractComponentModel | JointModel,
) -> dict[str, str]:
    """Map bare nonlinear extension names to the keys used in model.log_prob."""
    if not isinstance(model, JointModel):
        return {name: name for name in nonlinear_extension_priors}

    per_component_names = model._per_component_nonlinear_names()
    name_to_model_key: dict[str, str] = {}
    for component_name, nonlinear_names in per_component_names.items():
        for param_name in nonlinear_names:
            if param_name in nonlinear_extension_priors:
                name_to_model_key[param_name] = f"{component_name}.{param_name}"

    for name in nonlinear_extension_priors:
        if name not in name_to_model_key:
            name_to_model_key[name] = name
    return name_to_model_key


def _prepare_sampler_model(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
    marginalized_names: tuple[str, ...] | None,
) -> _PreparedSamplerModel:
    """Prepare a normalized model/prior bundle for rejection or MCMC sampling.

    Walks the attached ``model`` to extract nonlinear extension priors and
    computes the effective linear prior at run-time from ``prior.linear_prior``
    plus any linear-extension parameters declared on the model's extensions.
    """
    nonlinear_extension_priors, linear_extension_names = (
        _nonlinear_extension_priors_from_model(prior, model)
    )
    effective_linear_prior = _effective_linear_prior_from_prior(prior, model)
    _validate_extension_priors(prior, model, effective_linear_prior)
    effective_marginalized_names = _resolve_effective_marginalized_names(
        effective_linear_prior,
        marginalized_names,
    )
    return _PreparedSamplerModel(
        model=model,
        nonlinear_extension_priors=nonlinear_extension_priors,
        effective_linear_prior=effective_linear_prior,
        effective_marginalized_names=effective_marginalized_names,
        linear_extension_names=linear_extension_names,
    )


def _wrap_unit_values(
    values: dict[str, Any],
    nonlinear_priors: dict[str, Any],
    base_names: frozenset[str],
) -> dict[str, Any]:
    """Wrap QuantityDistribution-sampled base params in Q objects.

    Extension params (jitter, etc.) are left as raw scalars.
    TODO: why are the extension params left as raw scalars?
    """
    result = dict(values)
    for name, d in nonlinear_priors.items():
        if isinstance(d, QuantityDistribution) and name in base_names:
            result[name] = Q(result[name], str(d.unit))
    return result


@final
class RejectionSampler(AbstractSampler):
    """Rejection sampler for Keplerian orbital parameters.

    Implements rejection sampling with analytic marginalization over linear
    parameters. Configure once, then call :meth:`run` with each dataset.

    Parameters
    ----------
    prior
        Prior distributions for nonlinear (and optionally linear) parameters.
    model
        Fully constructed model template (no data, no linear_priors).
    marginalized_names
        Linear parameter names to analytically marginalize. If None, all
        Gaussian linear parameters are auto-classified for marginalization.
    batch_size
        Number of samples to process per batch. Smaller values use less memory
        but may be slower. Default: 100_000.

    Examples
    --------
    >>> from unxt import Q
    >>> import jax.numpy as jnp
    >>> from harv import HarvPrior, RejectionSampler, RVData
    >>> from harv.models.rv import RVModel
    >>> data = RVData(  # doctest: +SKIP
    ...     time=Q(jnp.linspace(0, 100, 5), "day"),
    ...     rv=Q(jnp.zeros(5), "km/s"),
    ...     rv_err=Q(jnp.full(5, 1.0), "km/s"),
    ... )
    >>> prior = HarvPrior.default_rv(  # doctest: +SKIP
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> sampler = RejectionSampler(prior, RVModel())  # doctest: +SKIP
    >>> samples = sampler.run(data, n_prior_samples=100_000)  # doctest: +SKIP
    """

    batch_size: int = eqx.field(static=True, default=100_000)

    def run(
        self,
        data: InputData,
        *,
        n_prior_samples: int,
        max_posterior_samples: int | None = None,
        seed: int | None = None,
        ignore_non_finite: bool = False,
        return_logprobs: bool = False,
    ) -> Samples:
        """Run rejection sampling.

        Parameters
        ----------
        data
            Observed data: an :class:`~harv.data.AbstractData` subclass
            (e.g. :class:`~harv.data.RVData`,
            :class:`~harv.data.GaiaAstrometryData`) for single-component
            models, or an :class:`~harv.data.AbstractDatasetContainer`
            (e.g. :class:`~harv.data.SystemData`,
            :class:`~harv.data.SourceData`) for :class:`~harv.JointModel`.
        n_prior_samples
            Number of samples to draw from the prior.
        max_posterior_samples
            Maximum number of posterior samples to return. If None, returns all
            accepted samples.
        seed
            Random number seed. If not specified, picks a seed based on the
            current time.
        ignore_non_finite
            If ``True``, any ``NaN`` or infinite log-likelihood values are
            treated as rejected samples by replacing them with ``-inf`` before
            the rejection step. If ``False`` (default), non-finite values are
            left unchanged.
        return_logprobs
            If ``True``, store per-sample log-probabilities on the returned
            :class:`~harv.samplers.samples.Samples`: ``ln_likelihood`` (the
            marginal log-likelihood) and ``ln_prior`` (the summed nonlinear
            prior log-density).  These enable :meth:`Samples.map_sample` and
            the :attr:`Samples.ln_posterior` property.  Default ``False``.

        Returns
        -------
            Posterior samples container.
        """
        _validate_data(data, self.model)

        prepared = _prepare_sampler_model(
            self.prior,
            self.model,
            self.marginalized_names,
        )

        model = prepared.model
        nonlinear_extension_priors = prepared.nonlinear_extension_priors
        effective_linear_prior = prepared.effective_linear_prior or {}
        effective_marginalized_names = prepared.effective_marginalized_names
        linear_extension_names = prepared.linear_extension_names

        # if not specified, pick a different random seed each run:
        _seed: int = uuid.uuid4().int >> 96 if seed is None else seed

        key = jr.key(_seed)
        sample_key, rej_key = jr.split(key)

        # generate prior samples and evaluate (marginalized) log likelihoods in batches
        # TODO: only return accepted samples and acceptance rate to conserve memory?

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            model,
            sample_key,
            n_prior_samples,
            nonlinear_extension_priors,
            effective_linear_prior,
            effective_marginalized_names,
            data,
        )

        if ignore_non_finite:
            log_likelihoods = jnp.where(
                jnp.isfinite(log_likelihoods), log_likelihoods, -jnp.inf
            )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)
        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}
        accepted_log_likelihood = log_likelihoods[accepted_mask]

        linear_key = jr.fold_in(key, 2)
        # TODO: support oversampling of linear parameters?
        linear_samples = self._sample_linear_parameters(
            model,
            linear_key,
            accepted_nonlinear,
            effective_marginalized_names,
            data,
            effective_linear_prior,
        )

        if max_posterior_samples is not None:
            n_accepted = len(next(iter(accepted_nonlinear.values())))
            if n_accepted > max_posterior_samples:
                idx_key = jr.fold_in(key, 3)
                idx = jr.choice(
                    idx_key,
                    n_accepted,
                    shape=(max_posterior_samples,),
                    replace=False,
                )
                accepted_nonlinear = {k: v[idx] for k, v in accepted_nonlinear.items()}
                linear_samples = {k: v[idx] for k, v in linear_samples.items()}
                accepted_log_likelihood = accepted_log_likelihood[idx]

        # Build nonlinear dict as Quantities with units from the prior.
        # Base orbital params come from prior.nonlinear_priors.
        # Extension nonlinear params (e.g. jitter) come from the prepared model-key map.
        _nl_keys = set(self.prior.nonlinear_priors)
        _all_nl_priors: dict[str, Any] = dict(self.prior.nonlinear_priors)
        _all_nl_priors.update(nonlinear_extension_priors)

        nonlinear_q: dict[str, AbstractQuantity] = {}
        for k, v in accepted_nonlinear.items():
            if k not in _all_nl_priors:
                continue
            d = _all_nl_priors[k]
            unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
            nonlinear_q[k] = Q(v, unit)

        # t_ref is uniformly exposed by both AbstractData and
        # AbstractDatasetContainer; no branching needed.
        t_ref = data.t_ref

        metadata: dict[str, Any] = {}
        if t_ref is not None:
            # Strip to a plain Python float so a JAX-traced array never lands in a
            # static metadata dict (which would trigger an equinox UserWarning).
            _t_unit = str(t_ref.unit)
            metadata["t_ref"] = float(ustrip(_t_unit, t_ref))
            metadata["t_ref_unit"] = _t_unit

        ln_likelihood_arr: jax.Array | None = None
        ln_prior_arr: jax.Array | None = None
        if return_logprobs:
            ln_likelihood_arr = accepted_log_likelihood
            ln_prior_arr = _evaluate_nonlinear_log_prior(
                _all_nl_priors, accepted_nonlinear
            )

        return Samples(
            nonlinear=cast("dict[str, Q]", nonlinear_q),
            linear=cast("dict[str, Q]", linear_samples),
            data_type=type(model).__name__,
            metadata=metadata,
            linear_extension_names=linear_extension_names,
            ln_likelihood=ln_likelihood_arr,
            ln_prior=ln_prior_arr,
        )

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(  # noqa: C901
        self,
        model: AbstractComponentModel | JointModel,
        key: jax.Array,
        n_prior_samples: int,
        ext_nl_priors: dict[str, Any],
        eff_linear: dict[str, Any],
        marginalize_names: "tuple[str, ...] | None",
        # data is correlated with the (polymorphic) model and is dispatched
        # through model.log_prob; the static type cannot be narrowed here.
        data: Any,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        The model's ``log_prob`` is called either in auto mode or with the
        sampler-resolved ``marginalized_names`` override.
        """
        prior = self.prior

        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        n_total = n_batches * self.batch_size

        key, nl_key = jr.split(key)
        prior_samples = prior.sample_nonlinear(nl_key, n_total)

        base_names = model._base_nonlinear_names()

        # Sample explicit linear params (non-Gaussian, not analytically marginalized).
        # Use the effective marginalize_names computed by _resolve_extension_priors
        # (which auto-classified non-Gaussian entries at run-time).
        if isinstance(eff_linear, dict):
            if marginalize_names is not None:
                marg_set = set(marginalize_names)
                for name, d in eff_linear.items():
                    if name not in marg_set:
                        key, k = jr.split(key)
                        prior_samples[name] = _unwrap_dist(d).sample(k, (n_total,))
            else:
                # marginalize_names is None => auto-marginalize Gaussians, sample rest
                for name, d in eff_linear.items():
                    if _needs_explicit_sampling(d):
                        key, k = jr.split(key)
                        prior_samples[name] = _unwrap_dist(d).sample(k, (n_total,))

        # Sample extension nonlinear parameters (jitter, GP hypers, etc.).
        if ext_nl_priors:
            key, ext_key = jr.split(key)
            ext_keys = jr.split(ext_key, len(ext_nl_priors))
            for (model_key, d), k in zip(ext_nl_priors.items(), ext_keys, strict=True):
                prior_samples[model_key] = _unwrap_dist(d).sample(k, (n_total,))

        model_keys = tuple(prior_samples.keys())

        # Reshape into (n_batches, batch_size).
        batched: dict[str, jax.Array] = {
            k: prior_samples[k].reshape(n_batches, self.batch_size) for k in model_keys
        }

        def body_fn(i: int, acc: jax.Array) -> jax.Array:
            raw = {k: batched[k][i] for k in model_keys}
            wrapped = _wrap_unit_values(raw, prior.nonlinear_priors, base_names)
            if marginalize_names is None:
                return acc.at[i].set(
                    jax.vmap(
                        lambda s: model.log_prob(
                            s, data, linear_priors=eff_linear or None
                        )
                    )(wrapped)
                )
            return acc.at[i].set(
                jax.vmap(
                    lambda sample: model.log_prob(
                        sample,
                        data,
                        linear_priors=eff_linear or None,
                        marginalized_names=marginalize_names,
                    )
                )(wrapped)
            )

        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )

        trimmed = {k: prior_samples[k][:n_prior_samples] for k in model_keys}
        return trimmed, log_liks_batched.flatten()[:n_prior_samples]

    @staticmethod
    @jax.jit
    def _rejection_step(key: jax.Array, log_likelihoods: jax.Array) -> jax.Array:
        """Compute rejection mask."""
        max_log_likelihood = jnp.max(log_likelihoods)
        weights = jnp.where(
            jnp.isfinite(max_log_likelihood),
            jnp.exp(log_likelihoods - max_log_likelihood),
            jnp.zeros_like(log_likelihoods),
        )
        uniform_draws = jr.uniform(key, shape=log_likelihoods.shape)
        return uniform_draws < weights

    def _sample_linear_parameters(  # noqa: C901
        self,
        model: AbstractComponentModel | JointModel,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
        marginalized_names: tuple[str, ...] | None,
        # data is correlated with the (polymorphic) model and is dispatched
        # through model.sample_conditional_linear; cannot be narrowed here.
        data: Any,
        linear_priors: dict[str, Any] | None,
    ) -> dict[str, AbstractQuantity]:
        """Sample linear parameters from conditional posterior using vmap.

        The model's ``sample_conditional_linear`` uses the sampler-resolved
        ``marginalized_names`` override when one is provided.
        """
        prior = self.prior

        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            if isinstance(model, JointModel):
                names: list[str] = []
                for comp in model.components.values():
                    names.extend(comp._all_linear_names())
            else:
                names = list(model._all_linear_names())
            return {name: Q(jnp.zeros(0), "") for name in names}

        keys = jr.split(key, n_samples)
        base_names = model._base_nonlinear_names()
        model_keys = tuple(nonlinear_samples.keys())

        def _sample_one(key: jax.Array, sample: dict[str, jax.Array]) -> dict[str, Any]:
            raw = {k: sample[k] for k in model_keys}
            wrapped = _wrap_unit_values(raw, prior.nonlinear_priors, base_names)
            return model.sample_conditional_linear(
                wrapped,
                key,
                data,
                linear_priors=linear_priors,
                marginalized_names=marginalized_names,
            )

        filtered = {k: nonlinear_samples[k] for k in model_keys}
        result = jax.vmap(_sample_one)(keys, filtered)

        # Attach units from the model's linear_param_units
        if isinstance(model, JointModel):
            # Detect which per-component param names appear in more than one
            # component.  Colliding names (e.g. both "rv_semiamp" in an SB2)
            # are namespaced as "comp_name.param_name" to avoid silent overwrites.
            name_counts: dict[str, int] = {}
            for comp in model.components.values():
                for name in comp._all_linear_names():
                    name_counts[name] = name_counts.get(name, 0) + 1

            # Shared linear params that appear at the top level (not in
            # per-component sub-dicts) should use bare names.
            final: dict[str, AbstractQuantity] = {}
            first_comp_name = next(iter(model.components))
            first_comp = model.components[first_comp_name]
            shared_units = first_comp._linear_param_units(data[first_comp_name])

            for k, value in result.items():
                if isinstance(value, dict):
                    # Per-component sub-dict.
                    comp_name = k
                    comp = model.components[comp_name]
                    units = comp._linear_param_units(data[comp_name])
                    for nm, arr in value.items():
                        final_name = (
                            f"{comp_name}.{nm}" if name_counts.get(nm, 1) > 1 else nm
                        )
                        final[final_name] = Q(arr, units.get(nm, ""))
                else:
                    # Shared top-level param (joint path).
                    final[k] = Q(value, shared_units.get(k, ""))
            return final
        units = model._linear_param_units(data)
        return {name: Q(arr, units.get(name, "")) for name, arr in result.items()}
