"""Likelihood functions for combined astrometry + RV data.

This module implements marginalized likelihood computation for joint astrometry
and radial velocity data. The likelihoods are independent and simply added in
log space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from quaxed import vmap

from harv.likelihood.astrometry import compute_marginal_log_likelihood_astrometry
from harv.likelihood.rv import compute_marginal_log_likelihood_rv

if TYPE_CHECKING:
    import numpyro.distributions as dist
    from unxt import Quantity

    from harv.custom_types import Angle, DimlessValue, Speed, Time

__all__ = [
    "compute_marginal_log_likelihood_combined",
    "compute_marginal_log_likelihood_combined_batch",
]


def compute_marginal_log_likelihood_combined(
    log_period: DimlessValue,
    eccentricity: DimlessValue,
    phase_peri: DimlessValue,
    cos_i: DimlessValue,
    arg_peri: DimlessValue,
    lon_asc_node: DimlessValue,
    # Astrometry data
    astro_times: Quantity[Time],
    scan_angle: Quantity[Angle],
    parallax_factor: DimlessValue,
    al_position: Quantity[Angle],
    al_position_err: Quantity[Angle],
    astro_t_ref: Quantity[Time],
    astro_linear_prior: dist.Distribution,
    # RV data
    rv_times: Quantity[Time],
    rv: Quantity[Speed],
    rv_err: Quantity[Speed],
    rv_t_ref: Quantity[Time],
    rv_linear_prior: dist.Distribution,
) -> DimlessValue:
    """Compute marginalized log-likelihood for combined astrometry + RV data.

    The combined likelihood is simply the sum of independent log-likelihoods:
        log L_combined = log L_astrometry + log L_rv

    Parameters
    ----------
    log_period, eccentricity, phase_peri, cos_i, arg_peri, lon_asc_node
        Nonlinear orbital parameters (shared between astrometry and RV).
    astro_times, scan_angle, parallax_factor
        Astrometry metadata.
    al_position, al_position_err, astro_t_ref,astro_linear_prior
        Astrometry data and prior.
    rv_times, rv, rv_err, rv_t_ref, rv_linear_prior
        RV data and prior.

    Returns
    -------
    log_likelihood
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
    log_period: DimlessValue,
    eccentricity: DimlessValue,
    phase_peri: DimlessValue,
    cos_i: DimlessValue,
    arg_peri: DimlessValue,
    lon_asc_node: DimlessValue,
    astro_times: Quantity[Time],
    scan_angle: Quantity[Angle],
    parallax_factor: DimlessValue,
    al_position: Quantity[Angle],
    al_position_err: Quantity[Angle],
    astro_t_ref: Quantity[Time],
    astro_linear_prior: dist.Distribution,
    rv_times: Quantity[Time],
    rv: Quantity[Speed],
    rv_err: Quantity[Speed],
    rv_t_ref: Quantity[Time],
    rv_linear_prior: dist.Distribution,
) -> DimlessValue:
    """Vectorized combined likelihood for batch of samples."""
    batched_likelihood = vmap(
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
