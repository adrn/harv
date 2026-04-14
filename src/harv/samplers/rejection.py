"""Rejection sampler for orbital parameter inference.

This module implements rejection sampling with analytical marginalization over
linear parameters. The sampler draws samples from the prior distribution over
nonlinear parameters, evaluates the marginalized likelihood, and performs
rejection sampling to obtain posterior samples.

The sampler accepts a :class:`~harv.model.Model` that pre-computes the
likelihood and provides parameter-building methods.  Numpyro model builder
helpers live in ``_numpyro.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from unxt import AbstractQuantity, Q, ustrip

from harv.distributions import QuantityDistribution
from harv.likelihood.helpers import _unwrap_dist
from harv.samplers.samples import Samples

if TYPE_CHECKING:
    from harv.model import Model

__all__ = ["RejectionSampler"]


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class RejectionSampler(eqx.Module):
    """Rejection sampler for Keplerian orbital parameters.

    This class implements rejection sampling with analytical marginalization
    over linear parameters. It supports both astrometric and radial velocity
    data.

    Parameters
    ----------
    model : Model
        A pre-built model combining prior and data.
    batch_size : int, optional
        Number of samples to process per batch. Smaller values use less memory
        but may be slower. Default: 100_000.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv import Model
    >>> from harv.samplers import RejectionPrior, RejectionSampler
    >>> prior = RejectionPrior.default_rv(
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> sampler = RejectionSampler(Model(prior, rv_data))  # doctest: +SKIP
    >>> sampler.batch_size
    100000

    Run rejection sampling (expensive):

    >>> samples = sampler.run(n_prior_samples=100_000)  # doctest: +SKIP
    """

    model: Model
    batch_size: int = eqx.field(static=True, default=100_000)

    def run(
        self,
        n_prior_samples: int,
        *,
        max_posterior_samples: int | None = None,
        seed: int = 0,
    ) -> Samples:
        """Run rejection sampling.

        Parameters
        ----------
        n_prior_samples
            Number of samples to draw from the prior.
        max_posterior_samples
            Maximum number of posterior samples to return. If None, returns all
            accepted samples.
        seed
            Random seed for reproducibility. Default: 0.

        Returns
        -------
        samples
            Posterior samples container.

        Examples
        --------
        >>> from unxt import Q  # doctest: +SKIP
        >>> from harv import Model  # doctest: +SKIP
        >>> from harv.samplers import RejectionPrior, RejectionSampler  # doctest: +SKIP
        >>> from harv.simulate.rv import simulate_rv_sb1_data  # doctest: +SKIP
        >>> rv_data, _ = simulate_rv_sb1_data()  # doctest: +SKIP
        >>> prior = RejectionPrior.default_rv(  # doctest: +SKIP
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> sampler = RejectionSampler(Model(prior, rv_data))  # doctest: +SKIP
        >>> samples = sampler.run(n_prior_samples=100_000)  # doctest: +SKIP
        >>> samples.n_samples  # doctest: +SKIP
        42
        """
        model = self.model

        key = jr.key(seed)
        sample_key, rej_key = jr.split(key)

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            sample_key,
            n_prior_samples,
        )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)
        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}

        linear_key = jr.fold_in(key, 2)
        linear_samples = self._sample_linear_parameters(
            linear_key,
            accepted_nonlinear,
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

        time_unit = model.time_unit

        extra_linear_names: tuple[str, ...] = ()
        if model.prior.offsets is not None:
            extra_linear_names = tuple(
                k for k, v in model.prior.offsets.items() if v is not None
            )

        # Build nonlinear dict as Quantities with units baked in.
        # Only include actual nonlinear parameters (not explicit-linear ones
        # that may also be in accepted_nonlinear due to partial marginalization).
        _nl_units: dict[str, str] = {
            "period": time_unit,
            "eccentricity": "",
            "phase_peri": "",
            "arg_peri": "rad",
            "cos_i": "",
            "lon_asc_node": "rad",
        }
        _nl_keys = set(model.prior.nonlinear_priors)
        nonlinear_q: dict[str, AbstractQuantity] = {
            k: Q(v, _nl_units.get(k, ""))
            for k, v in accepted_nonlinear.items()
            if k in _nl_keys
        }

        # Include jitter samples (keyed by user-friendly names like
        # "jitter_rv", "jitter_astrometry") in the nonlinear dict.
        if model.prior.jitter_priors is not None:
            for dt_label, d in model.prior.jitter_priors.items():
                values_key = f"_jitter_{dt_label}"
                user_key = f"jitter_{dt_label}"
                if values_key in accepted_nonlinear:
                    unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
                    nonlinear_q[user_key] = Q(accepted_nonlinear[values_key], unit)

        return Samples(
            nonlinear=nonlinear_q,
            linear=linear_samples,
            orbit_cls=model.full_cls[0],
            full_cls=model.full_cls,
            data_type=model.data_type,
            metadata={"t_ref": model.t_ref},
            extra_linear_names=extra_linear_names,
        )

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(
        self,
        key: jax.Array,
        n_prior_samples: int,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        The pre-built likelihood from ``self.model`` is evaluated with
        ``jax.vmap`` inside a ``fori_loop`` over batches of ``batch_size``
        samples.

        Instead of zero-padding the last batch, we oversample so that every
        evaluation uses a real prior draw.  The returned arrays are trimmed to
        ``n_prior_samples``.
        """
        model = self.model
        prior = model.prior
        lik = model.likelihood

        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        n_total = n_batches * self.batch_size

        key, nl_key = jr.split(key)
        prior_samples = prior.sample_nonlinear(nl_key, n_total)

        time_unit = model.time_unit

        # Convert period samples from the prior's unit to the data's time
        # unit.  When the period prior is a bare distribution (not wrapped in
        # QDistribution), assume values are already in time_unit.
        _p_prior = prior.nonlinear_priors.get("period")
        _p_unit = (
            str(_p_prior.unit) if isinstance(_p_prior, QuantityDistribution) else ""
        )
        if _p_unit:
            period_converted = ustrip(time_unit, Q(prior_samples["period"], _p_unit))
            prior_samples["period"] = period_converted

        # Sample explicit linear params (those not being marginalized).
        marg_names = prior.marginalize_names
        if isinstance(prior.linear_prior, dict) and marg_names is not None:
            marg_set = set(marg_names)
            explicit_linear = {
                name: d
                for name, d in prior.linear_prior.items()
                if name not in marg_set
            }
            if explicit_linear:
                key, lin_key = jr.split(key)
                lin_keys = jr.split(lin_key, len(explicit_linear))
                lp_units = lik.linear_param_units
                for (name, d), k in zip(explicit_linear.items(), lin_keys, strict=True):
                    raw = _unwrap_dist(d).sample(k, (n_total,))
                    # Convert to data units if the prior carries a unit.
                    target_u = lp_units.get(name, "")
                    if isinstance(d, QuantityDistribution) and target_u:
                        raw = ustrip(target_u, Q(raw, str(d.unit)))
                    prior_samples[name] = raw

        # Sample jitter parameters from jitter_priors (keyed by data type).
        # Each jitter sample is stored with a namespaced key (e.g.
        # "_jitter_rv") that the model maps to the "jitter" param field.
        _jitter_keys: list[str] = []
        if prior.jitter_priors is not None:
            key, jit_key = jr.split(key)
            jit_keys = jr.split(jit_key, len(prior.jitter_priors))
            for (dt_label, d), k in zip(
                prior.jitter_priors.items(), jit_keys, strict=True
            ):
                values_key = f"_jitter_{dt_label}"
                _jitter_keys.append(values_key)
                prior_samples[values_key] = _unwrap_dist(d).sample(k, (n_total,))

        # Reshape all parameter arrays into (n_batches, batch_size).
        _zeros = jnp.zeros(n_total)
        _required_keys = list(model.required_prior_params)
        _required_keys.extend(_jitter_keys)
        batched: dict[str, jax.Array] = {
            k: prior_samples.get(k, _zeros).reshape(n_batches, self.batch_size)
            for k in _required_keys
        }

        # Static list of keys for dict reconstruction inside the fori_loop.
        _keys = tuple(batched.keys())

        def body_fn(i: int, acc: jax.Array) -> jax.Array:
            values = {k: batched[k][i] for k in _keys}
            params = model._build_params_raw(values)
            return acc.at[i].set(jax.vmap(lik.log_prob)(params))

        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )

        # Trim oversampled entries to match the requested count.
        trimmed = {k: v[:n_prior_samples] for k, v in prior_samples.items()}
        return trimmed, log_liks_batched.flatten()[:n_prior_samples]

    @staticmethod
    @jax.jit
    def _rejection_step(key: jax.Array, log_likelihoods: jax.Array) -> jax.Array:
        """Compute rejection mask."""
        weights = jnp.exp(log_likelihoods - jnp.max(log_likelihoods))
        uniform_draws = jr.uniform(key, shape=log_likelihoods.shape)
        return uniform_draws < weights

    def _sample_linear_parameters(
        self,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
    ) -> dict[str, AbstractQuantity]:
        """Sample linear parameters from conditional posterior using vmap.

        For each accepted nonlinear sample, draws from the conditional posterior
        of the linear parameters given the nonlinear parameters and data.
        """
        model = self.model

        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            names = model.all_linear_names
            return {name: Q(jnp.zeros(0), "") for name in names}

        keys = jr.split(key, n_samples)

        def _sample_one(key: jax.Array, sample: dict[str, jax.Array]) -> dict[str, Q]:
            return model._sample_conditional_linear_raw(sample, key)

        return jax.vmap(_sample_one)(keys, nonlinear_samples)
