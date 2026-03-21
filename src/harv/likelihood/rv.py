"""Likelihood functions for radial velocity data.

This module implements likelihood evaluations for radial velocity observations.
Two variants are provided:

- :class:`MarginalizedRVLikelihood`: analytically marginalizes over the linear
  RV parameters (K, v₀) given a Gaussian prior. Requires a
  ``dist.MultivariateNormal`` prior.

- :class:`RVLikelihood`: full likelihood with all parameters specified
  explicitly.

For the marginalized SB1 model, the RV model is:

    RV(t) = K·[cos(ω + f(t)) + e·cos(ω)] + v₀

For the SB2 case, see :func:`_get_design_matrix_sb2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.likelihood._params import (
    AbstractRVParameters,
    RVFullParameters,
    RVOrbitParameters,
)
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import _solve_kepler

if TYPE_CHECKING:
    from harv.data import RadialVelocityData

__all__ = [
    "MarginalizedMultiSurveyRVLikelihood",
    "MarginalizedRVLikelihood",
    "RVLikelihood",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_design_matrix(
    params: AbstractRVParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build (n_obs, 2) design matrix for SB1: columns [K, v₀]."""
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    cos_omega_plus_f = jnp.cos(_arg_peri) * cos_f - jnp.sin(_arg_peri) * sin_f
    rv_amplitude = cos_omega_plus_f + params.eccentricity * jnp.cos(_arg_peri)
    return jnp.column_stack([rv_amplitude, jnp.ones_like(rv_amplitude)])


def _get_design_matrix_sb2(
    params: AbstractRVParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
    primary: bool,
) -> jax.Array:
    """Build (n_obs, 3) design matrix for SB2: columns [K₁, K₂, v₀].

    For primary: [X(t), 0, 1].  For secondary: [0, -X(t), 1].
    """
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    cos_omega_plus_f = jnp.cos(_arg_peri) * cos_f - jnp.sin(_arg_peri) * sin_f
    rv_amplitude = cos_omega_plus_f + params.eccentricity * jnp.cos(_arg_peri)

    if primary:
        return jnp.column_stack(
            [rv_amplitude, jnp.zeros_like(rv_amplitude), jnp.ones_like(rv_amplitude)]
        )
    return jnp.column_stack(
        [jnp.zeros_like(rv_amplitude), -rv_amplitude, jnp.ones_like(rv_amplitude)]
    )


# ---------------------------------------------------------------------------
# Marginalized likelihood
# ---------------------------------------------------------------------------


class MarginalizedRVLikelihood(AbstractLikelihood[RVOrbitParameters]):
    """RV likelihood with linear parameters (K, v₀) analytically marginalized.

    Analytically integrates over K and v₀ given a Gaussian prior, using the
    ``MarginalizedLinear`` distribution from numpyro-ext.

    Parameters
    ----------
    data : RadialVelocityData
        Radial velocity observations.
    linear_prior : dist.MultivariateNormal
        Gaussian prior over [K, v₀]. Must be multivariate normal — this is
        required for analytic marginalization.

    Examples
    --------
    >>> lik = MarginalizedRVLikelihood(data=rv_data, linear_prior=prior)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
    """

    data: RadialVelocityData
    linear_prior: dist.MultivariateNormal

    param_names = ("period", "eccentricity", "phase_peri", "arg_peri")

    def __check_init__(self) -> None:
        if not isinstance(self.linear_prior, dist.MultivariateNormal):
            msg = (
                "MarginalizedRVLikelihood requires a dist.MultivariateNormal "
                "prior for analytic marginalization; "
                f"got {type(self.linear_prior)}"
            )
            raise TypeError(msg)

    def log_prob(self, params: RVOrbitParameters) -> jax.Array:
        """Compute the marginalized log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(params, sin_f, cos_f)

        marg_dist = MarginalizedLinear(
            design_matrix=design_matrix,
            prior_distribution=self.linear_prior,
            data_distribution=dist.Normal(0.0, ustrip("km/s", self.data.rv_err)),
        )
        return marg_dist.log_prob(ustrip("km/s", self.data.rv))


# ---------------------------------------------------------------------------
# Multi-survey marginalized likelihood
# ---------------------------------------------------------------------------


class MarginalizedMultiSurveyRVLikelihood(AbstractLikelihood[RVOrbitParameters]):
    """RV likelihood for multiple instruments with offset parameters marginalized.

    Linear parameters are ``[K, v₀, δ₁, …, δₖ]`` where δᵢ is the zero-point
    offset for the i-th non-reference instrument.  The ``indicator_matrix``
    (shape ``n_obs_total × n_non_ref``) selects which observations belong to
    each non-reference instrument; it is constant across parameter samples and
    is closed over at construction time.

    Parameters
    ----------
    data : RadialVelocityData
        All RV observations stacked in instrument dict order.
    indicator_matrix : jax.Array
        Boolean/float indicator matrix, shape ``(n_obs_total, n_non_ref)``.
        ``indicator_matrix[i, j] == 1`` when observation *i* comes from
        non-reference instrument *j*.
    linear_prior : dist.MultivariateNormal
        Joint Gaussian prior over ``[K, v₀, δ₁, …, δₖ]``.
    """

    data: RadialVelocityData
    indicator_matrix: jax.Array
    linear_prior: dist.MultivariateNormal

    param_names = ("period", "eccentricity", "phase_peri", "arg_peri")

    def __check_init__(self) -> None:
        if not isinstance(self.linear_prior, dist.MultivariateNormal):
            msg = (
                "MarginalizedMultiSurveyRVLikelihood requires a "
                "dist.MultivariateNormal prior; "
                f"got {type(self.linear_prior)}"
            )
            raise TypeError(msg)

    def log_prob(self, params: RVOrbitParameters) -> jax.Array:
        """Compute the marginalized log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        dm_base = _get_design_matrix(params, sin_f, cos_f)  # (n_obs, 2)
        dm = jnp.concatenate([dm_base, self.indicator_matrix], axis=-1)

        marg_dist = MarginalizedLinear(
            design_matrix=dm,
            prior_distribution=self.linear_prior,
            data_distribution=dist.Normal(0.0, ustrip("km/s", self.data.rv_err)),
        )
        return marg_dist.log_prob(ustrip("km/s", self.data.rv))


# ---------------------------------------------------------------------------
# Full likelihood
# ---------------------------------------------------------------------------


class RVLikelihood(AbstractLikelihood[RVFullParameters]):
    """Full RV likelihood with all parameters (including K and v₀) specified.

    Parameters
    ----------
    data : RadialVelocityData
        Radial velocity observations.

    Examples
    --------
    >>> lik = RVLikelihood(data=rv_data)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
    """

    data: RadialVelocityData

    param_names = ("period", "eccentricity", "phase_peri", "arg_peri", "K", "v0")

    def log_prob(self, params: RVFullParameters) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(params, sin_f, cos_f)

        linear_params = jnp.array([params.K, params.v0])
        rv_pred = design_matrix @ linear_params
        rv_obs = ustrip("km/s", self.data.rv)
        rv_err = ustrip("km/s", self.data.rv_err)

        return dist.Normal(rv_pred, rv_err).log_prob(rv_obs).sum()
