"""Likelihood functions for radial velocity data.

This module implements likelihood evaluations for radial velocity observations.
Two variants are provided:

- :class:`MarginalizedRVLikelihood`: analytically marginalizes over some or all
  of the linear RV parameters (K, v₀) given a Gaussian prior.  Supports
  partial marginalization: parameters listed in
  ``params.marginalized_names`` are integrated out; any remaining linear
  parameters must be present as explicit fields on the params object.

- :class:`RVLikelihood`: full likelihood with all parameters specified
  explicitly.

For the marginalized SB1 model, the RV model is:

    RV(t) = K·[cos(ω + f(t)) + e·cos(ω)] + v₀

For the SB2 case, see :func:`_get_design_matrix_sb2`.
"""

import equinox as eqx
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
from harv.likelihood.helpers import _resolve_linear_prior, _solve_kepler

__all__ = [
    "MarginalizedMultiSurveyRVLikelihood",
    "MarginalizedRVLikelihood",
    "RVLikelihood",
]

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
# Marginalized likelihood
# ---------------------------------------------------------------------------


class MarginalizedRVLikelihood(AbstractLikelihood[MarginalizedParameters]):
    """RV likelihood with linear parameters (K, v₀) analytically marginalized.

    Analytically integrates over some or all of the linear RV parameters
    given a Gaussian prior, using the ``MarginalizedLinear`` distribution
    from numpyro-ext.  Partial marginalization is supported: parameters in
    ``params.marginalized_names`` are integrated out, while any remaining
    linear parameters must be present on the ``MarginalizedParameters``
    wrapper as explicit fields.

    Parameters
    ----------
    data : RadialVelocityData
        Radial velocity observations.
    linear_prior : dist.MultivariateNormal or eqx.Module
        Gaussian prior over the marginalized linear parameters.  For full
        marginalization this covers [K, v₀]; for partial marginalization it
        covers only the marginalized subset.  May be a fixed
        ``dist.MultivariateNormal`` or a callable ``eqx.Module``.

    Examples
    --------
    >>> lik = MarginalizedRVLikelihood(data=rv_data, linear_prior=prior)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
    """

    data: RadialVelocityData
    linear_prior: dist.MultivariateNormal | eqx.Module

    param_names = ("period", "eccentricity", "phase_peri", "arg_peri")

    def log_prob(self, params: MarginalizedParameters) -> jax.Array:
        """Compute the marginalized log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(params, sin_f, cos_f)  # (n_obs, 2)

        rv_unit = self.data.rv.unit
        rv_obs = ustrip(rv_unit, self.data.rv)
        rv_err = ustrip(rv_unit, self.data.rv_err)

        # Determine which RV linear params are marginalized vs explicit.
        my_marg = tuple(n for n in _RV_LINEAR_NAMES if n in params.marginalized_names)
        my_explicit = tuple(
            n for n in _RV_LINEAR_NAMES if n not in params.marginalized_names
        )

        # Indices into the design matrix columns (column order = _RV_LINEAR_NAMES).
        marg_idx = [_RV_LINEAR_NAMES.index(n) for n in my_marg]
        explicit_idx = [_RV_LINEAR_NAMES.index(n) for n in my_explicit]

        # Adjust observations for explicit parameters.
        if explicit_idx:
            explicit_vals = jnp.array(
                [ustrip(rv_unit, getattr(params, n)) for n in my_explicit]
            )
            rv_obs = rv_obs - design_matrix[:, jnp.array(explicit_idx)] @ explicit_vals

        # Marginalize over the remaining columns.
        dm_marg = design_matrix[:, jnp.array(marg_idx)]
        lp = _resolve_linear_prior(self.linear_prior, params)
        marg_dist = MarginalizedLinear(
            design_matrix=dm_marg,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, rv_err),
        )
        return marg_dist.log_prob(rv_obs)


# ---------------------------------------------------------------------------
# Multi-survey marginalized likelihood
# ---------------------------------------------------------------------------


class MarginalizedMultiSurveyRVLikelihood(AbstractLikelihood[MarginalizedParameters]):
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
    linear_prior: dist.MultivariateNormal | eqx.Module

    param_names = ("period", "eccentricity", "phase_peri", "arg_peri")

    def log_prob(self, params: MarginalizedParameters) -> jax.Array:
        """Compute the marginalized log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        dm_base = _get_design_matrix(params, sin_f, cos_f)  # (n_obs, 2)
        dm = jnp.concatenate([dm_base, self.indicator_matrix], axis=-1)
        lp = _resolve_linear_prior(self.linear_prior, params)

        rv_unit = self.data.rv.unit
        marg_dist = MarginalizedLinear(
            design_matrix=dm,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, ustrip(rv_unit, self.data.rv_err)),
        )
        return marg_dist.log_prob(ustrip(rv_unit, self.data.rv))


# ---------------------------------------------------------------------------
# Full likelihood
# ---------------------------------------------------------------------------


class RVLikelihood(AbstractLikelihood[RVParameters]):
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

    def log_prob(self, params: RVParameters) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(params, sin_f, cos_f)

        linear_params = jnp.array([params.K, params.v0])
        rv_pred = design_matrix @ linear_params
        rv_unit = self.data.rv.unit
        rv_obs = ustrip(rv_unit, self.data.rv)
        rv_err = ustrip(rv_unit, self.data.rv_err)

        return dist.Normal(rv_pred, rv_err).log_prob(rv_obs).sum()
