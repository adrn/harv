"""Likelihood functions for combined astrometry + RV data.

This module implements marginalized likelihood computation for joint astrometry
and radial velocity data. The likelihoods are independent and simply added in
log space.
"""

from __future__ import annotations

import jax
import numpyro.distributions as dist
from unxt import Quantity

from harv.custom_types import DimlessOrArray
from harv.likelihood.astrometry import compute_marginal_log_likelihood_astrometry
from harv.likelihood.rv import compute_marginal_log_likelihood_rv

__all__ = [
    "compute_marginal_log_likelihood_combined",
    "compute_marginal_log_likelihood_combined_batch",
]


def compute_marginal_log_likelihood_combined(
    log_period: float,
    eccentricity: float,
    phase_peri: float,
    cos_i: float,
    arg_peri: float,
    lon_asc_node: float,
    # Astrometry data
    astro_times: Quantity["time"],
    scan_angle: Quantity["angle"],
    parallax_factor: DimlessOrArray,
    al_position: Quantity["angle"],
    al_position_err: Quantity["angle"],
    astro_t_ref: Quantity["time"],
    astro_linear_prior: dist.Distribution,
    # RV data
    rv_times: Quantity["time"],
    rv: Quantity["speed"],
    rv_err: Quantity["speed"],
    rv_t_ref: Quantity["time"],
    rv_linear_prior: dist.Distribution,
) -> float:
    """Compute marginalized log-likelihood for combined astrometry + RV data.

    The combined likelihood is simply the sum of independent log-likelihoods:
        log L_combined = log L_astrometry + log L_rv

    Parameters
    ----------
    log_period, eccentricity, phase_peri, cos_i, arg_peri, lon_asc_node
        Nonlinear orbital parameters (shared between astrometry and RV).
    astro_times, scan_angle, parallax_factor, al_position, al_position_err, astro_t_ref, astro_linear_prior
        Astrometry data and prior.
    rv_times, rv, rv_err, rv_t_ref, rv_linear_prior
        RV data and prior.

    Returns
    -------
    log_likelihood : float
        Combined marginalized log-likelihood.
    """
    log_lik_astro = compute_marginal_log_likelihood_astrometry(
        log_period,
        eccentricity,
        phase_peri,
        cos_i,
        arg_peri,
        lon_asc_node,
        astro_times,
        scan_angle,
        parallax_factor,
        al_position,
        al_position_err,
        astro_t_ref,
        astro_linear_prior,
    )

    log_lik_rv = compute_marginal_log_likelihood_rv(
        log_period,
        eccentricity,
        phase_peri,
        arg_peri,
        rv_times,
        rv,
        rv_err,
        rv_t_ref,
        rv_linear_prior,
    )

    return log_lik_astro + log_lik_rv


@jax.jit
def compute_marginal_log_likelihood_combined_batch(
    log_period: jax.Array,
    eccentricity: jax.Array,
    phase_peri: jax.Array,
    cos_i: jax.Array,
    arg_peri: jax.Array,
    lon_asc_node: jax.Array,
    astro_times: Quantity["time"],
    scan_angle: Quantity["angle"],
    parallax_factor: DimlessOrArray,
    al_position: Quantity["angle"],
    al_position_err: Quantity["angle"],
    astro_t_ref: Quantity["time"],
    astro_linear_prior: dist.Distribution,
    rv_times: Quantity["time"],
    rv: Quantity["speed"],
    rv_err: Quantity["speed"],
    rv_t_ref: Quantity["time"],
    rv_linear_prior: dist.Distribution,
) -> jax.Array:
    """Vectorized combined likelihood for batch of samples."""
    batched_likelihood = jax.vmap(
        compute_marginal_log_likelihood_combined,
        in_axes=(
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )

    return batched_likelihood(
        log_period,
        eccentricity,
        phase_peri,
        cos_i,
        arg_peri,
        lon_asc_node,
        astro_times,
        scan_angle,
        parallax_factor,
        al_position,
        al_position_err,
        astro_t_ref,
        astro_linear_prior,
        rv_times,
        rv,
        rv_err,
        rv_t_ref,
        rv_linear_prior,
    )
