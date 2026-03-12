"""Likelihood functions for radial velocity data.

This module implements marginalized likelihood computation for RV data, where
linear parameters (K, v₀) are analytically marginalized over while evaluating
the likelihood for nonlinear parameters (P, e, ω, phase_peri).

For single-lined spectroscopic binaries (SB1), the RV model is:
    RV(t) = K·[cos(ω + f(t)) + e·cos(ω)] + v₀

For double-lined spectroscopic binaries (SB2), we model both components:
    RV₁(t) = K₁·[cos(ω + f(t)) + e·cos(ω)] + v₀
    RV₂(t) = K₂·[cos(ω + f(t) + π) + e·cos(ω + π)] + v₀
           = -K₂·[cos(ω + f(t)) + e·cos(ω)] + v₀

where f(t) is the true anomaly computed via Kepler's equation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxoplanet.core.kepler import kepler
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Quantity, ustrip

from harv.custom_types import DimlessOrArray

__all__ = [
    "get_rv_design_matrix",
    "get_rv_design_matrix_sb2",
    "compute_marginal_log_likelihood_rv",
    "compute_marginal_log_likelihood_rv_batch",
]


def get_rv_design_matrix(
    sin_f: DimlessOrArray,
    cos_f: DimlessOrArray,
    eccentricity: DimlessOrArray,
    arg_peri: DimlessOrArray,
) -> jax.Array:
    """Build design matrix for single-lined RV observations (SB1).

    The RV model is:
        RV(t) = K·[cos(ω + f(t)) + e·cos(ω)] + v₀

    This can be written as a linear model:
        RV(t) = K·X(t) + v₀·1

    where X(t) = cos(ω + f(t)) + e·cos(ω).

    Parameters
    ----------
    sin_f : array_like
        sin(f), where f is the true anomaly.
    cos_f : array_like
        cos(f), where f is the true anomaly.
    eccentricity : float or array_like
        Orbital eccentricity, 0 ≤ e < 1.
    arg_peri : float or array_like
        Argument of periastron ω (radians).

    Returns
    -------
    design_matrix : jax.Array
        Design matrix of shape (n_obs, 2) for linear parameters [K, v₀].
    """
    # Compute cos(ω + f) using angle addition formula
    cos_omega_plus_f = jnp.cos(arg_peri) * cos_f - jnp.sin(arg_peri) * sin_f

    # RV amplitude term: K·[cos(ω + f) + e·cos(ω)]
    rv_amplitude = cos_omega_plus_f + eccentricity * jnp.cos(arg_peri)

    # Design matrix: [K, v₀]
    # Column 0: coefficient for K
    # Column 1: coefficient for v₀ (constant 1)
    return jnp.column_stack([rv_amplitude, jnp.ones_like(rv_amplitude)])


def get_rv_design_matrix_sb2(
    sin_f: DimlessOrArray,
    cos_f: DimlessOrArray,
    eccentricity: DimlessOrArray,
    arg_peri: DimlessOrArray,
    primary: bool = True,
) -> jax.Array:
    """Build design matrix for double-lined RV observations (SB2).

    For SB2 systems, we observe both stellar components. The design matrix
    differs for primary and secondary:
        RV₁(t) = K₁·X(t) + v₀
        RV₂(t) = -K₂·X(t) + v₀

    where X(t) = cos(ω + f(t)) + e·cos(ω).

    Parameters
    ----------
    sin_f : array_like
        sin(f), where f is the true anomaly.
    cos_f : array_like
        cos(f), where f is the true anomaly.
    eccentricity : float or array_like
        Orbital eccentricity, 0 ≤ e < 1.
    arg_peri : float or array_like
        Argument of periastron ω (radians).
    primary : bool, optional
        If True, build design matrix for primary star (RV₁).
        If False, build for secondary star (RV₂). Default: True.

    Returns
    -------
    design_matrix : jax.Array
        Design matrix of shape (n_obs, 3) for linear parameters [K₁, K₂, v₀].
        For primary: [X(t), 0, 1]
        For secondary: [0, -X(t), 1]
    """
    # Compute cos(ω + f)
    cos_omega_plus_f = jnp.cos(arg_peri) * cos_f - jnp.sin(arg_peri) * sin_f

    # RV amplitude term
    rv_amplitude = cos_omega_plus_f + eccentricity * jnp.cos(arg_peri)

    if primary:
        # Primary: [K₁·X, 0, v₀]
        return jnp.column_stack(
            [
                rv_amplitude,  # K₁ coefficient
                jnp.zeros_like(rv_amplitude),  # K₂ coefficient
                jnp.ones_like(rv_amplitude),  # v₀ coefficient
            ]
        )
    # Secondary: [0, -K₂·X, v₀]
    return jnp.column_stack(
        [
            jnp.zeros_like(rv_amplitude),  # K₁ coefficient
            -rv_amplitude,  # K₂ coefficient (negative!)
            jnp.ones_like(rv_amplitude),  # v₀ coefficient
        ]
    )


def compute_marginal_log_likelihood_rv(
    log_period: float,
    eccentricity: float,
    phase_peri: float,
    arg_peri: float,
    times: Quantity["time"],
    rv: Quantity["speed"],
    rv_err: Quantity["speed"],
    t_ref: Quantity["time"],
    linear_prior: dist.Distribution,
) -> float:
    """Compute marginalized log-likelihood for single RV sample.

    This function analytically marginalizes over linear parameters (K, v₀)
    and returns the log-likelihood for a single set of nonlinear parameters.

    Parameters
    ----------
    log_period : float
        log₁₀(Period) in days.
    eccentricity : float
        Orbital eccentricity, 0 ≤ e < 1.
    phase_peri : float
        Phase at periastron, t_peri/P, dimensionless in [0, 1).
    arg_peri : float
        Argument of periastron ω (radians).
    times : Quantity["time"]
        Observation times.
    rv : Quantity["speed"]
        Observed radial velocities.
    rv_err : Quantity["speed"]
        RV measurement uncertainties (1σ).
    t_ref : Quantity["time"]
        Reference time for orbital phase.
    linear_prior : dist.Distribution
        Prior distribution for linear parameters (K, v₀).
        Should be a 1D distribution applied independently to each parameter.

    Returns
    -------
    log_likelihood : float
        Marginalized log-likelihood value.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from unxt import Quantity
    >>> times = Quantity(jnp.array([0., 10., 20., 30.]), "day")
    >>> rv = Quantity(jnp.array([5.0, -3.0, -5.0, 3.0]), "km/s")
    >>> rv_err = Quantity(jnp.ones(4) * 0.1, "km/s")
    >>> t_ref = Quantity(0.0, "day")
    >>> linear_prior = dist.Normal(0.0, 100.0)
    >>> log_lik = compute_marginal_log_likelihood_rv(
    ...     log_period=1.5,  # ~31.6 days
    ...     eccentricity=0.3,
    ...     phase_peri=0.0,
    ...     arg_peri=1.0,
    ...     times=times,
    ...     rv=rv,
    ...     rv_err=rv_err,
    ...     t_ref=t_ref,
    ...     linear_prior=linear_prior,
    ... )
    """
    # Convert period from log to linear (days)
    period_day = 10.0**log_period

    # Compute time of periastron passage
    t_peri = phase_peri * period_day

    # Compute mean anomaly
    dt = ustrip("day", times) - t_peri
    M = 2 * jnp.pi * dt / period_day

    # Solve Kepler's equation for true anomaly
    sin_f, cos_f = kepler(M, eccentricity)

    # Build design matrix for linear parameters [K, v₀]
    design_matrix = get_rv_design_matrix(sin_f, cos_f, eccentricity, arg_peri)

    # Create marginalized linear distribution
    # MarginalizedLinear handles analytical marginalization over [K, v₀]
    marg_dist = MarginalizedLinear(
        design_matrix=design_matrix,
        prior_distribution=linear_prior,
        data_distribution=dist.Normal(0.0, ustrip("km/s", rv_err)),
    )

    # Evaluate log-likelihood (marginalized over linear parameters)
    return marg_dist.log_prob(ustrip("km/s", rv))


@jax.jit
def compute_marginal_log_likelihood_rv_batch(
    log_period: jax.Array,
    eccentricity: jax.Array,
    phase_peri: jax.Array,
    arg_peri: jax.Array,
    times: Quantity["time"],
    rv: Quantity["speed"],
    rv_err: Quantity["speed"],
    t_ref: Quantity["time"],
    linear_prior: dist.Distribution,
) -> jax.Array:
    """Compute marginalized log-likelihood for batch of RV samples.

    This is a vectorized version of `compute_marginal_log_likelihood_rv` that
    processes multiple parameter sets in parallel using vmap.

    Parameters
    ----------
    log_period : jax.Array
        log₁₀(Period) in days, shape (n_samples,).
    eccentricity : jax.Array
        Orbital eccentricity, shape (n_samples,).
    phase_peri : jax.Array
        Phase at periastron, shape (n_samples,).
    arg_peri : jax.Array
        Argument of periastron ω (radians), shape (n_samples,).
    times : Quantity["time"]
        Observation times, shape (n_obs,).
    rv : Quantity["speed"]
        Observed radial velocities, shape (n_obs,).
    rv_err : Quantity["speed"]
        RV measurement uncertainties, shape (n_obs,).
    t_ref : Quantity["time"]
        Reference time for orbital phase.
    linear_prior : dist.Distribution
        Prior distribution for linear parameters.

    Returns
    -------
    log_likelihoods : jax.Array
        Marginalized log-likelihood values, shape (n_samples,).
    """
    # Vectorize over first 4 arguments (nonlinear parameters)
    batched_likelihood = jax.vmap(
        compute_marginal_log_likelihood_rv,
        in_axes=(0, 0, 0, 0, None, None, None, None, None),
    )

    return batched_likelihood(
        log_period,
        eccentricity,
        phase_peri,
        arg_peri,
        times,
        rv,
        rv_err,
        t_ref,
        linear_prior,
    )
