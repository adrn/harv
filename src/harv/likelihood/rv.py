"""Likelihood functions for radial velocity data.

This module implements the unified :class:`RVLikelihood` for radial velocity
observations.  The class supports three evaluation modes via the same
``log_prob`` interface, determined by the type of ``params`` and the
presence of an optional ``indicator_matrix``:

1. **Marginalized** (``params`` is :class:`MarginalizedParameters`,
   ``linear_prior`` provided): analytically integrates over the linear RV
   parameters (K, v₀) given a Gaussian prior.  Supports partial
   marginalization via ``params.marginalized_names``.

2. **Multi-survey marginalized** (``indicator_matrix`` is not ``None``):
   appends instrument-offset columns to the design matrix and marginalizes
   ``[K, v₀, δ₁, …, δₖ]`` jointly.

3. **Explicit** (``params`` is :class:`RVParameters`, ``linear_prior`` is
   ``None``): evaluates the Gaussian data log-likelihood directly at the
   provided linear parameter values.

For the SB1 model the RV model is:

    RV(t) = K·[cos(ω + f(t)) + e·cos(ω)] + v₀

For the SB2 case, see :func:`_get_design_matrix_sb2`.
"""

from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.data import RadialVelocityData
from harv.kepler._orbit_math import rv_shape as _rv_shape
from harv.likelihood._params import (
    AbstractParameters,
    MarginalizedParameters,
    RVParameters,
)
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import (
    LinearPriorCallable,
    _resolve_linear_prior,
    _solve_kepler,
)

__all__ = ("RVLikelihood",)

# Column order in the RV design matrix.
_RV_LINEAR_NAMES: tuple[str, ...] = RVParameters.linear_param_names  # ("K", "v0")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_design_matrix(
    params: AbstractParameters | MarginalizedParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build (n_obs, 2) design matrix for SB1: columns [K, v₀]."""
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    rv_amplitude = _rv_shape(sin_f, cos_f, params.eccentricity, _arg_peri)
    return jnp.column_stack([rv_amplitude, jnp.ones_like(rv_amplitude)])


def _get_design_matrix_sb2(
    params: AbstractParameters | MarginalizedParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
    primary: bool,
) -> jax.Array:
    """Build (n_obs, 3) design matrix for SB2: columns [K₁, K₂, v₀].

    For primary: [X(t), 0, 1].  For secondary: [0, -X(t), 1].
    """
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    rv_amplitude = _rv_shape(sin_f, cos_f, params.eccentricity, _arg_peri)

    if primary:
        return jnp.column_stack(
            [rv_amplitude, jnp.zeros_like(rv_amplitude), jnp.ones_like(rv_amplitude)]
        )
    return jnp.column_stack(
        [jnp.zeros_like(rv_amplitude), -rv_amplitude, jnp.ones_like(rv_amplitude)]
    )


# ---------------------------------------------------------------------------
# Unified RV likelihood
# ---------------------------------------------------------------------------


class RVLikelihood(AbstractLikelihood[MarginalizedParameters | RVParameters]):
    """Unified RV likelihood supporting marginalized and explicit evaluation.

    When ``linear_prior`` is provided and ``params`` is a
    :class:`MarginalizedParameters` instance, the likelihood analytically
    marginalizes over the linear parameters (K, v₀) — and optionally
    per-instrument offsets when ``indicator_matrix`` is supplied.

    When ``linear_prior`` is ``None``, ``params`` must be a full
    :class:`RVParameters` and the likelihood is evaluated explicitly.

    Parameters
    ----------
    data : RadialVelocityData
        Radial velocity observations.
    linear_prior : dist.MultivariateNormal or LinearPriorCallable or None
        Gaussian prior over the marginalized linear parameters.  ``None``
        for explicit evaluation.
    indicator_matrix : jax.Array or None
        For multi-survey RV: boolean/float indicator matrix of shape
        ``(n_obs_total, n_non_ref)`` selecting which observations belong to
        each non-reference instrument.  ``None`` for single-instrument data.

    Examples
    --------
    Marginalized (single instrument)::

        lik = RVLikelihood(data=rv_data, linear_prior=prior)
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    Multi-survey marginalized::

        lik = RVLikelihood(data=stacked, linear_prior=prior, indicator_matrix=ind)
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    Explicit evaluation::

        lik = RVLikelihood(data=rv_data)
        log_liks = jax.jit(jax.vmap(lik.log_prob))(full_params_batch)
    """

    data: RadialVelocityData
    linear_prior: dist.MultivariateNormal | LinearPriorCallable | None = None
    indicator_matrix: jax.Array | None = None

    param_names = ("period", "eccentricity", "phase_peri", "arg_peri")

    def design_matrix(self, params: MarginalizedParameters | RVParameters) -> jax.Array:
        """Build the full design matrix for the given parameters.

        Returns shape ``(n_obs, n_cols)`` where ``n_cols`` is 2 for single-
        instrument data or 2 + n_non_ref for multi-survey data.
        """
        sin_f, cos_f = _solve_kepler(self.data, params)
        dm = _get_design_matrix(params, sin_f, cos_f)
        if self.indicator_matrix is not None:
            dm = jnp.concatenate([dm, self.indicator_matrix], axis=-1)
        return dm

    def log_prob(self, params: MarginalizedParameters | RVParameters) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample.

        Dispatches to marginalized or explicit evaluation based on the
        presence of ``linear_prior``.
        """
        if self.linear_prior is None:
            return self._log_prob_explicit(cast("RVParameters", params))
        return self._log_prob_marginalized(cast("MarginalizedParameters", params))

    def sample_conditional_linear(
        self, params: MarginalizedParameters, key: jax.Array
    ) -> jax.Array:
        """Sample linear parameters from the conditional posterior.

        Builds a ``MarginalizedLinear`` from the design matrix, the resolved
        linear prior, and the data errors, then draws one sample from the
        posterior conditioned on the observed data.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear orbital parameters (period, eccentricity, etc.).
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        jax.Array
            Sampled linear parameter vector of length ``n_cols``.
        """
        dm = self.design_matrix(params)
        lp = _resolve_linear_prior(self.linear_prior, params)
        rv_unit = self.data.rv.unit
        marg = MarginalizedLinear(
            design_matrix=dm,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, ustrip(rv_unit, self.data.rv_err)),
        )
        return marg.conditional(ustrip(rv_unit, self.data.rv)).sample(key)

    # -- private helpers ----------------------------------------------------

    def _log_prob_marginalized(self, params: MarginalizedParameters) -> jax.Array:
        """Marginalized log-likelihood (single or multi-survey)."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(params, sin_f, cos_f)  # (n_obs, 2)

        rv_unit = self.data.rv.unit
        rv_obs = ustrip(rv_unit, self.data.rv)
        rv_err = ustrip(rv_unit, self.data.rv_err)

        if self.indicator_matrix is not None:
            # Multi-survey: append indicator columns and marginalize everything.
            dm = jnp.concatenate([design_matrix, self.indicator_matrix], axis=-1)
            lp = _resolve_linear_prior(self.linear_prior, params)
            marg_dist = MarginalizedLinear(
                design_matrix=dm,
                prior_distribution=lp,
                data_distribution=dist.Normal(0.0, rv_err),
            )
            return marg_dist.log_prob(rv_obs)

        # Single-instrument: partial marginalization.
        my_marg = tuple(n for n in _RV_LINEAR_NAMES if n in params.marginalized_names)
        my_explicit = tuple(
            n for n in _RV_LINEAR_NAMES if n not in params.marginalized_names
        )
        marg_idx = [_RV_LINEAR_NAMES.index(n) for n in my_marg]
        explicit_idx = [_RV_LINEAR_NAMES.index(n) for n in my_explicit]

        if explicit_idx:
            explicit_vals = jnp.array(
                [ustrip(rv_unit, getattr(params, n)) for n in my_explicit]
            )
            rv_obs = rv_obs - design_matrix[:, jnp.array(explicit_idx)] @ explicit_vals

        dm_marg = design_matrix[:, jnp.array(marg_idx)]
        lp = _resolve_linear_prior(self.linear_prior, params)
        marg_dist = MarginalizedLinear(
            design_matrix=dm_marg,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, rv_err),
        )
        return marg_dist.log_prob(rv_obs)

    def _log_prob_explicit(self, params: RVParameters) -> jax.Array:
        """Explicit log-likelihood with all parameters specified."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(params, sin_f, cos_f)

        linear_params = jnp.array([params.K, params.v0])
        rv_pred = design_matrix @ linear_params
        rv_unit = self.data.rv.unit
        rv_obs = ustrip(rv_unit, self.data.rv)
        rv_err = ustrip(rv_unit, self.data.rv_err)

        return dist.Normal(rv_pred, rv_err).log_prob(rv_obs).sum()


# Backward-compatibility aliases (deprecated)
MarginalizedRVLikelihood = RVLikelihood
MarginalizedMultiSurveyRVLikelihood = RVLikelihood
