"""Likelihood functions for Gaia epoch astrometry data.

This module implements likelihood evaluations for Gaia along-scan astrometry.
Two variants are provided:

- :class:`MarginalizedGaiaAstrometryLikelihood`: analytically marginalizes over
  the 6 linear astrometric parameters (α₀, δ₀, μ_α, μ_δ, ϖ, a) given a
  Gaussian prior. Requires a ``dist.MultivariateNormal`` prior.

- :class:`GaiaAstrometryLikelihood`: full likelihood with all parameters
  specified explicitly.

For the marginalized model, the astrometric model is:

    y_AL = α₀·cos(ψ) + δ₀·sin(ψ)
         + (μ_α·cos(ψ) + μ_δ·sin(ψ))·dt
         + ϖ·H_ϖ(t)
         + a·[(A·sin(ψ) + B·cos(ψ))·cos(f) + (F·sin(ψ) + G·cos(ψ))·sin(f)]

where A, B, F, G are Thiele-Innes constants and f is the true anomaly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.likelihood._params import (
    AbstractGaiaAstrometryParameters,
    GaiaAstrometryFullParameters,
    GaiaAstrometryOrbitParameters,
)
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import _resolve_linear_prior, _solve_kepler

if TYPE_CHECKING:
    from harv.data import GaiaAstrometryData

__all__ = [
    "MarginalizedGaiaAstrometryLikelihood",
    "GaiaAstrometryLikelihood",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_design_matrix(
    data: GaiaAstrometryData,
    params: AbstractGaiaAstrometryParameters,
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

    # Thiele-Innes constants
    A = (
        jnp.cos(_arg_peri) * jnp.cos(_lon_asc_node)
        - jnp.sin(_arg_peri) * jnp.sin(_lon_asc_node) * _cos_i
    )
    B = (
        jnp.cos(_arg_peri) * jnp.sin(_lon_asc_node)
        + jnp.sin(_arg_peri) * jnp.cos(_lon_asc_node) * _cos_i
    )
    F = (
        -jnp.sin(_arg_peri) * jnp.cos(_lon_asc_node)
        - jnp.cos(_arg_peri) * jnp.sin(_lon_asc_node) * _cos_i
    )
    G = (
        -jnp.sin(_arg_peri) * jnp.sin(_lon_asc_node)
        + jnp.cos(_arg_peri) * jnp.cos(_lon_asc_node) * _cos_i
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
    AbstractLikelihood[GaiaAstrometryOrbitParameters]
):
    """Gaia astrometry likelihood with linear parameters analytically marginalized.

    Analytically integrates over the 6 linear astrometric parameters
    (α₀, δ₀, μ_α, μ_δ, ϖ, a) given a Gaussian prior, using the
    ``MarginalizedLinear`` distribution from numpyro-ext.

    Parameters
    ----------
    data : GaiaAstrometryData
        Gaia epoch astrometry observations.
    linear_prior : dist.MultivariateNormal
        Gaussian prior over the 6 linear parameters. Must be multivariate
        normal — this is required for analytic marginalization.

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

    def log_prob(self, params: GaiaAstrometryOrbitParameters) -> jax.Array:
        """Compute the marginalized log-likelihood for a single parameter sample."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix(self.data, params, sin_f, cos_f)
        lp = _resolve_linear_prior(self.linear_prior, params)

        marg_dist = MarginalizedLinear(
            design_matrix=design_matrix,
            prior_distribution=lp,
            data_distribution=dist.Normal(
                0.0, ustrip("mas", self.data.al_position_err)
            ),
        )
        return marg_dist.log_prob(ustrip("mas", self.data.al_position))


# ---------------------------------------------------------------------------
# Full likelihood
# ---------------------------------------------------------------------------


class GaiaAstrometryLikelihood(AbstractLikelihood[GaiaAstrometryFullParameters]):
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

    def log_prob(self, params: GaiaAstrometryFullParameters) -> jax.Array:
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
