"""Likelihood functions for Gaia epoch astrometry data.

This module implements likelihood evaluations for Gaia along-scan astrometry.
Two variants are provided:

- :class:`MarginalizedGaiaAstrometryLikelihood`: analytically marginalizes over
  some or all of the 6 linear astrometric parameters (α₀, δ₀, μ_α, μ_δ, ϖ, a)
  given a Gaussian prior.  Supports partial marginalization via
  ``params.marginalized_names``.

- :class:`GaiaAstrometryLikelihood`: full likelihood with all parameters
  specified explicitly.

For the marginalized model, the astrometric model is:

    y_AL = α₀·cos(ψ) + δ₀·sin(ψ)
         + (μ_α·cos(ψ) + μ_δ·sin(ψ))·dt
         + ϖ·H_ϖ(t)
         + a·[(A·sin(ψ) + B·cos(ψ))·cos(f) + (F·sin(ψ) + G·cos(ψ))·sin(f)]

where A, B, F, G are Thiele-Innes constants and f is the true anomaly.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.data import GaiaAstrometryData
from harv.kepler._orbit_math import thiele_innes_ABFG
from harv.likelihood._params import (
    GaiaAstrometryParameters,
    MarginalizedParameters,
)
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import _resolve_linear_prior, _solve_kepler

__all__ = [
    "MarginalizedGaiaAstrometryLikelihood",
    "GaiaAstrometryLikelihood",
]

# Column order in the astrometry design matrix.
_ASTRO_LINEAR_NAMES: tuple[str, ...] = GaiaAstrometryParameters.linear_param_names

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_design_matrix(
    data: GaiaAstrometryData,
    params: MarginalizedParameters | GaiaAstrometryParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build the (n_obs, 6) Gaia along-scan design matrix.

    Columns: [α₀, δ₀, μ_α, μ_δ, ϖ, a].
    See Appendix A of https://arxiv.org/abs/2206.05726.
    """
    dt_yr = ustrip("yr", data.time - data.t_ref)
    scan_angle_rad = ustrip("rad", data.scan_angle)
    cos_psi = jnp.cos(scan_angle_rad)
    sin_psi = jnp.sin(scan_angle_rad)

    _parallax_factor = ustrip(AllowValue, "", data.parallax_factor)
    _cos_i = ustrip(AllowValue, "", params.cos_i)
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    _lon_asc_node = ustrip(AllowValue, "", params.lon_asc_node)

    # Thiele-Innes constants (unit, i.e. semi-major axis = 1)
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(_arg_peri),
        jnp.sin(_arg_peri),
        jnp.cos(_lon_asc_node),
        jnp.sin(_lon_asc_node),
        _cos_i,
    )

    semimaj_term = (A * sin_psi + B * cos_psi) * cos_f + (
        F * sin_psi + G * cos_psi
    ) * sin_f

    return jnp.stack(
        [
            cos_psi,
            sin_psi,
            cos_psi * dt_yr,
            sin_psi * dt_yr,
            _parallax_factor,
            semimaj_term,
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Marginalized likelihood
# ---------------------------------------------------------------------------


class MarginalizedGaiaAstrometryLikelihood(
    AbstractLikelihood[MarginalizedParameters]
):
    """Gaia astrometry likelihood with linear parameters analytically marginalized.

    Analytically integrates over some or all of the 6 linear astrometric
    parameters (α₀, δ₀, μ_α, μ_δ, ϖ, a) given a Gaussian prior, using the
    ``MarginalizedLinear`` distribution from numpyro-ext.  Partial
    marginalization is supported: parameters in ``params.marginalized_names``
    are integrated out, while any remaining linear parameters must be present
    on the ``MarginalizedParameters`` wrapper as explicit fields.

    Parameters
    ----------
    data : GaiaAstrometryData
        Gaia epoch astrometry observations.
    linear_prior : dist.MultivariateNormal or eqx.Module
        Gaussian prior over the marginalized linear parameters.

    Examples
    --------
    >>> lik = MarginalizedGaiaAstrometryLikelihood(data=gaia_data, linear_prior=prior)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
    """

    data: GaiaAstrometryData
    linear_prior: dist.MultivariateNormal | eqx.Module

    param_names = (
        "period",
        "eccentricity",
        "phase_peri",
        "cos_i",
        "arg_peri",
        "lon_asc_node",
    )

    def log_prob(self, params: MarginalizedParameters) -> jax.Array:
        """Compute the marginalized log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(self.data, params, sin_f, cos_f)

        y_obs = ustrip("mas", self.data.al_position)
        y_err = ustrip("mas", self.data.al_position_err)

        # Determine which astro linear params are marginalized vs explicit.
        my_marg = tuple(
            n for n in _ASTRO_LINEAR_NAMES if n in params.marginalized_names
        )
        my_explicit = tuple(
            n for n in _ASTRO_LINEAR_NAMES if n not in params.marginalized_names
        )
        marg_idx = [_ASTRO_LINEAR_NAMES.index(n) for n in my_marg]
        explicit_idx = [_ASTRO_LINEAR_NAMES.index(n) for n in my_explicit]

        # Adjust observations for explicit parameters.
        if explicit_idx:
            explicit_vals = jnp.array(
                [ustrip("mas", getattr(params, n)) for n in my_explicit]
            )
            y_obs = y_obs - design_matrix[:, jnp.array(explicit_idx)] @ explicit_vals

        # Marginalize over the remaining columns.
        dm_marg = design_matrix[:, jnp.array(marg_idx)]
        lp = _resolve_linear_prior(self.linear_prior, params)
        marg_dist = MarginalizedLinear(
            design_matrix=dm_marg,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, y_err),
        )
        return marg_dist.log_prob(y_obs)


# ---------------------------------------------------------------------------
# Full likelihood
# ---------------------------------------------------------------------------


class GaiaAstrometryLikelihood(AbstractLikelihood[GaiaAstrometryParameters]):
    """Full Gaia astrometry likelihood with all parameters specified explicitly.

    Parameters
    ----------
    data : GaiaAstrometryData
        Gaia epoch astrometry observations.

    Examples
    --------
    >>> lik = GaiaAstrometryLikelihood(data=gaia_data)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
    """

    data: GaiaAstrometryData

    param_names = (
        "period",
        "eccentricity",
        "phase_peri",
        "cos_i",
        "arg_peri",
        "lon_asc_node",
        "ra0",
        "dec0",
        "pmra",
        "pmdec",
        "parallax",
        "semi_major_axis",
    )

    def log_prob(self, params: GaiaAstrometryParameters) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(self.data, params, sin_f, cos_f)

        linear_params = jnp.array(
            [
                params.ra0,
                params.dec0,
                params.pmra,
                params.pmdec,
                params.parallax,
                params.semi_major_axis,
            ]
        )
        y_pred = design_matrix @ linear_params
        y_obs = ustrip("mas", self.data.al_position)
        y_err = ustrip("mas", self.data.al_position_err)

        return dist.Normal(y_pred, y_err).log_prob(y_obs).sum()
