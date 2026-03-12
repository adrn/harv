"""Likelihood functions for Gaia epoch astrometry data.

This module implements marginalized likelihood computation for astrometric data,
where linear parameters (ra, dec, pmra, pmdec, plx) are analytically marginalized
over while evaluating the likelihood for nonlinear parameters (P, e, phase_peri, cos(i),
ω, Ω).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxoplanet.core.kepler import kepler
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Quantity, ustrip
from unxt.quantity import AllowValue

from harv.custom_types import DimlessOrArray

__all__ = [
    "get_astrometry_design_matrix",
    "compute_marginal_log_likelihood_astrometry",
    "compute_marginal_log_likelihood_astrometry_batch",
]


def get_astrometry_design_matrix(
    times: Quantity["time"],
    scan_angle: Quantity["angle"],
    parallax_factor: DimlessOrArray,
    sin_f: DimlessOrArray,
    cos_f: DimlessOrArray,
    t_ref: Quantity["time"],
    cos_i: DimlessOrArray,
    arg_peri: DimlessOrArray,
    lon_asc_node: DimlessOrArray,
) -> jax.Array:
    """Build design matrix for Gaia along-scan astrometry.

    The astrometric model is:
        y_AL = α₀·cos(ψ) + δ₀·sin(ψ)
             + (μ_α·cos(ψ) + μ_δ·sin(ψ))·dt
             + ϖ·H_ϖ(t)
             + a·[(A·sin(ψ) + B·cos(ψ))·cos(f) + (F·sin(ψ) + G·cos(ψ))·sin(f)]

    where A, B, F, G are Thiele-Innes constants computed from the orbital
    orientation angles (ω, Ω, i).

    Parameters
    ----------
    times : Quantity["time"]
        Observation times.
    scan_angle : Quantity["angle"]
        Gaia scan angle ψ for each observation.
    parallax_factor : DimlessOrArray
        Along-scan parallax factor H_ϖ(t) for each observation.
    sin_f : DimlessOrArray
        sin(true anomaly) at each observation time.
    cos_f : DimlessOrArray
        cos(true anomaly) at each observation time.
    t_ref : Quantity["time"]
        Reference epoch for proper motion.
    cos_i : DimlessOrArray
        cos(inclination) of the orbit.
    arg_peri : DimlessOrArray
        Argument of pericenter ω (radians).
    lon_asc_node : DimlessOrArray
        Longitude of ascending node Ω (radians).

    Returns
    -------
    design_matrix : jax.Array
        Design matrix of shape (n_obs, 6) for linear parameters:
        [α₀, δ₀, μ_α, μ_δ, ϖ, a]
    """
    # Convert to dimensionless for computation
    dt_yr = ustrip("yr", times - t_ref)
    scan_angle_rad = ustrip("rad", scan_angle)
    cos_psi = jnp.cos(scan_angle_rad)
    sin_psi = jnp.sin(scan_angle_rad)

    _sin_f = ustrip(AllowValue, "", sin_f)
    _cos_f = ustrip(AllowValue, "", cos_f)
    _parallax_factor = ustrip(AllowValue, "", parallax_factor)

    _cos_i = ustrip(AllowValue, "", cos_i)
    _arg_peri = ustrip(AllowValue, "", arg_peri)
    _lon_asc_node = ustrip(AllowValue, "", lon_asc_node)

    # Compute Thiele-Innes constants from orientation angles
    # See Appendix A of https://arxiv.org/abs/2206.05726
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

    # Compute semi-major axis term (not scaled by a yet)
    # This is the column that will be multiplied by the semi-major axis
    semimaj_term = (A * sin_psi + B * cos_psi) * _cos_f + (
        F * sin_psi + G * cos_psi
    ) * _sin_f

    # Stack into design matrix
    # Columns: [α₀, δ₀, μ_α, μ_δ, ϖ, a]
    design_matrix = jnp.stack(
        [
            cos_psi,  # α₀
            sin_psi,  # δ₀
            cos_psi * dt_yr,  # μ_α
            sin_psi * dt_yr,  # μ_δ
            _parallax_factor,  # ϖ
            semimaj_term,  # a (semi-major axis)
        ],
        axis=-1,
    )

    return design_matrix


@jax.jit
def compute_marginal_log_likelihood_astrometry(
    log_period: float,
    eccentricity: float,
    phase_peri: float,
    cos_i: float,
    arg_peri: float,
    lon_asc_node: float,
    times: Quantity["time"],
    scan_angle: Quantity["angle"],
    parallax_factor: DimlessOrArray,
    y_al: Quantity["angle"],
    y_al_error: Quantity["angle"],
    t_ref: Quantity["time"],
    linear_prior: dist.Distribution,
) -> float:
    """Compute marginalized log-likelihood for Gaia astrometry.

    This function analytically marginalizes over the 6 linear parameters
    (α₀, δ₀, μ_α, μ_δ, ϖ, a) while evaluating the likelihood for the
    6 nonlinear parameters (log(P), e, phase_peri, cos(i), ω, Ω).

    Parameters
    ----------
    log_period : float
        log₁₀(period/day).
    eccentricity : float
        Orbital eccentricity (0 ≤ e < 1).
    phase_peri : float
        Phase at pericenter (t_peri / period), range [0, 1).
    cos_i : float
        cos(inclination), range [-1, 1].
    arg_peri : float
        Argument of pericenter ω (radians), range [0, 2π].
    lon_asc_node : float
        Longitude of ascending node Ω (radians), range [0, 2π].
    times : Quantity["time"]
        Observation times.
    scan_angle : Quantity["angle"]
        Gaia scan angle ψ for each observation.
    parallax_factor : DimlessOrArray
        Along-scan parallax factor H_ϖ(t) for each observation.
    y_al : Quantity["angle"]
        Observed along-scan positions (mas).
    y_al_error : Quantity["angle"]
        Along-scan position uncertainties (mas).
    t_ref : Quantity["time"]
        Reference epoch for proper motion.
    linear_prior : dist.Distribution
        Prior distribution for linear parameters. Typically Normal(0, σ).

    Returns
    -------
    log_likelihood : float
        Marginalized log-likelihood value.

    Notes
    -----
    Uses the MarginalizedLinear distribution from numpyro-ext to analytically
    integrate over linear parameters, avoiding expensive MCMC or numerical
    integration.
    """
    # Convert period and compute time of pericenter
    period = 10.0**log_period  # days
    t_peri = phase_peri * period  # days

    # Compute mean anomaly
    dt = ustrip("day", times) - t_peri
    M = 2 * jnp.pi * dt / period

    # Solve Kepler's equation for true anomaly
    sin_f, cos_f = kepler(M, eccentricity)

    # Build design matrix
    design_matrix = get_astrometry_design_matrix(
        times,
        scan_angle,
        parallax_factor,
        sin_f,
        cos_f,
        t_ref,
        cos_i,
        arg_peri,
        lon_asc_node,
    )

    # Compute marginalized likelihood using MarginalizedLinear
    marg_dist = MarginalizedLinear(
        design_matrix=design_matrix,
        prior_distribution=linear_prior,
        data_distribution=dist.Normal(0.0, ustrip("mas", y_al_error)),
    )

    return marg_dist.log_prob(ustrip("mas", y_al))


# Vectorized version for batch processing
compute_marginal_log_likelihood_astrometry_batch = jax.jit(
    jax.vmap(
        compute_marginal_log_likelihood_astrometry,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    )
)
