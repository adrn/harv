"""Simulate Gaia-like epoch astrometry data.

This module provides utilities for generating synthetic along-scan astrometry
measurements similar to Gaia DR3/DR4 epoch data. The simulated data includes:
- Along-scan positions with realistic noise
- Scan angles
- Parallax factors
- Measurement uncertainties

The astrometric model includes:
- 5-parameter astrometry (a0, d0, mu_a, mu_d, parallax)
- Keplerian orbital motion parameterized by Thiele-Innes constants
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import quaxed.numpy as jnp
from unxt import AbstractQuantity, Q, uconvert, ustrip

from harv.data import GaiaAstrometryData

if TYPE_CHECKING:
    from harv.custom_types import (
        BatchQAngle,
        BatchQTime,
        ScalarQAngle,
        ScalarQAngularSpeed,
        ScalarQTime,
    )
from harv.kepler.orbits import astrometric_orbit_at_times, thiele_innes_ABFG

__all__ = ["simulate_gaia_epoch_astrometry", "fake_parallax_factor"]


def fake_parallax_factor(
    time: BatchQTime,
    ra: ScalarQAngle,
    dec: ScalarQAngle,
    scan_angle: BatchQAngle,
) -> jnp.ndarray:
    """Mock, super simplified parallax factor for a star at (ra, dec).

    This is a simplified analytical model for the parallax factor, assuming
    Earth's orbit is circular and using a sinusoidal approximation. For real
    Gaia data, use the actual parallax factors from the epoch data.

    Parameters
    ----------
    time : Q["time"]
        Observation times.
    ra : Q["angle"]
        Right ascension of the source.
    dec : Q["angle"]
        Declination of the source.
    scan_angle : Q["angle"]
        Gaia scan angle at each observation.

    Returns
    -------
    parallax_factor : jax.Array
        Dimensionless parallax factor for each observation.
    """
    # Simple sinusoidal model assuming 1-year period
    ang: AbstractQuantity = Q(2 * jnp.pi * ustrip("yr", time), "rad")
    P_alpha = jnp.sin(ang - ra)
    P_delta = -jnp.sin(dec) * jnp.cos(ang - ra)
    return P_alpha * jnp.cos(scan_angle) + P_delta * jnp.sin(scan_angle)


def simulate_gaia_epoch_astrometry(
    seed: int = 42,
    n_obs: int = 100,
    baseline: ScalarQTime | None = None,
    # Orbital parameters
    period: ScalarQTime | None = None,
    eccentricity: float | None = None,
    t_peri: ScalarQTime | None = None,
    arg_peri: ScalarQAngle | None = None,
    lon_asc_node: ScalarQAngle | None = None,
    inclination: ScalarQAngle | None = None,
    semi_major_axis: ScalarQAngle | None = None,
    # Sky position (for parallax factor calculation)
    ra: ScalarQAngle | None = None,
    dec: ScalarQAngle | None = None,
    # Astrometric parameters (small offsets in mas)
    alpha0: ScalarQAngle | None = None,
    delta0: ScalarQAngle | None = None,
    mu_alpha: ScalarQAngularSpeed | None = None,
    mu_delta: ScalarQAngularSpeed | None = None,
    parallax: ScalarQAngle | None = None,
    # Uncertainty
    al_error: ScalarQAngle | None = None,
    # Reference time
    t_ref: ScalarQTime | None = None,
) -> tuple[GaiaAstrometryData, dict[str, Any]]:
    """Simulate Gaia-like along-scan epoch astrometry.

    This function generates synthetic Gaia astrometry data including both
    5-parameter astrometry and Keplerian orbital motion. Random values are
    drawn for any parameters not specified.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    n_obs : int, optional
        Number of observations. Default: 100.
    baseline : Q["time"], optional
        Time baseline for observations. Default: 5 years.
    period : Q["time"], optional
        Orbital period. If None, randomly drawn from [0, 3] years.
    eccentricity : float, optional
        Orbital eccentricity. If None, randomly drawn from [0, 0.9].
    t_peri : Q["time"], optional
        Time of periastron passage. If None, randomly drawn from [0, period].
    arg_peri : Q["angle"], optional
        Argument of periastron omega. If None, randomly drawn from [0, 2pi].
    lon_asc_node : Q["angle"], optional
        Longitude of ascending node Omega. If None, randomly drawn from [0, 2pi].
    inclination : Q["angle"], optional
        Orbital inclination. If None, randomly drawn from cos(i) ~ U(-1, 1).
    semi_major_axis : Q["angle"], optional
        Semi-major axis in angular units. If None, randomly drawn from [0.5, 50] mas.
    ra : Q["angle"], optional
        Right ascension of the source (for parallax factor). Default: 180 deg.
    dec : Q["angle"], optional
        Declination of the source (for parallax factor). Default: 45 deg.
    alpha0 : Q["angle"], optional
        Small RA offset from reference position at t_ref. Default: 0 mas.
        This is a linear parameter, not the absolute RA.
    delta0 : Q["angle"], optional
        Small Dec offset from reference position at t_ref. Default: 0 mas.
        This is a linear parameter, not the absolute Dec.
    mu_alpha : Q["angular speed"], optional
        Proper motion in RA. If None, randomly drawn ~ N(0, 10 mas/yr).
    mu_delta : Q["angular speed"], optional
        Proper motion in Dec. If None, randomly drawn ~ N(0, 10 mas/yr).
    parallax : Q["angle"], optional
        Parallax. If None, randomly drawn from Exp(10 mas).
    al_error : Q["angle"], optional
        Along-scan measurement errors (1-sigma). If None, randomly drawn from
        U(0.02, 0.1) mas for each observation.
    t_ref : Q["time"], optional
        Reference time for astrometry. If None, randomly chosen.

    Returns
    -------
    data : GaiaAstrometryData
        Simulated Gaia astrometry data container.
    true_params : dict
        Dictionary of true parameter values used in simulation, including:
        period, eccentricity, semi_major_axis, t_peri, alpha0, delta0, mu_alpha,
        mu_delta, parallax, A, B, F, G (Thiele-Innes), arg_peri, lon_asc_node,
        inclination.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.simulate import simulate_gaia_epoch_astrometry
    >>> data, true_params = simulate_gaia_epoch_astrometry(
    ...     seed=42,
    ...     n_obs=50,
    ...     period=Q(100.0, "day"),
    ...     eccentricity=0.3,
    ...     semi_major_axis=Q(2.0, "mas"),
    ... )
    >>> data.time.shape
    (50,)
    >>> true_params["period"]
    Q['time'](Array(100., dtype=float64), unit='d')
    """
    ss = np.random.SeedSequence(seed)
    # One RNG per parameter that needs a default value
    rngs = [np.random.default_rng(s) for s in ss.spawn(14)]
    rng = rngs[0]

    if baseline is None:
        baseline = Q(5.0, "yr")

    if period is None:
        period = Q(rngs[1].uniform(0.3, 3.0), "yr")

    if eccentricity is None:
        eccentricity = rngs[2].uniform(0.0, 0.9)

    if t_peri is None:
        t_peri = Q(rngs[3].uniform(0.0, ustrip(period.unit, period)), period.unit)

    if arg_peri is None:
        arg_peri = Q(rngs[4].uniform(0, 2 * np.pi), "rad")

    if lon_asc_node is None:
        lon_asc_node = Q(rngs[5].uniform(0, 2 * np.pi), "rad")

    if inclination is None:
        inclination = Q(np.arccos(rngs[6].uniform(-1.0, 1.0)), "rad")

    if semi_major_axis is None:
        semi_major_axis = Q(rngs[7].uniform(0.5, 50.0), "mas")

    # Sky position for parallax factor calculation (absolute coordinates)
    ra = Q(180.0, "deg") if ra is None else ra
    dec = Q(45.0, "deg") if dec is None else dec

    # Astrometric offsets - these are small mas-scale deviations from reference
    # These are the LINEAR parameters in the astrometric model
    alpha0 = Q(0.0, "mas") if alpha0 is None else uconvert("mas", alpha0)
    delta0 = Q(0.0, "mas") if delta0 is None else uconvert("mas", delta0)

    mu_alpha = Q(rngs[8].normal(0, 10), "mas/yr") if mu_alpha is None else mu_alpha
    mu_delta = Q(rngs[9].normal(0, 10), "mas/yr") if mu_delta is None else mu_delta
    parallax = Q(rngs[10].exponential(10.0), "mas") if parallax is None else parallax
    al_error = (
        Q(rngs[11].uniform(0.02, 0.1, n_obs), "mas") if al_error is None else al_error
    )

    if t_ref is None:
        t_ref = Q(rng.uniform(0, ustrip(baseline.unit, baseline)), baseline.unit)

    # Observation times over baseline
    dt: AbstractQuantity = Q(
        jnp.sort(rng.uniform(0.0, ustrip(baseline.unit, baseline), n_obs)),
        baseline.unit,
    )
    times = dt + t_ref

    # Random scan angles
    scan_angle: AbstractQuantity = Q(rng.uniform(0, 2 * np.pi, n_obs), "rad")

    # Fudged parallax factor (uses absolute sky position)
    parallax_factor = fake_parallax_factor(times, ra, dec, scan_angle)

    # Compute true along-scan positions
    cos_psi = jnp.cos(scan_angle)
    sin_psi = jnp.sin(scan_angle)

    # 5-parameter astrometry contribution (all in mas)
    # Follows the Gaia LPC convention (Lindegren & Bastian,
    # GAIA-C3-TN-LU-LL-061-08, Eqs. 4 & 6):
    #   w = a*sin theta + d*cos theta   where a ~= Deltaalpha*, d ~= Deltadelta
    y_astro = (
        sin_psi * alpha0
        + cos_psi * delta0
        + sin_psi * mu_alpha * dt
        + cos_psi * mu_delta * dt
        + parallax * parallax_factor
    )
    y_astro = uconvert("mas", y_astro)

    # Orbital contribution: along-scan projection of (Deltara, Deltadec)
    # w_orbit = Deltara*sin theta + Deltadec*cos theta  (LPC convention, Eq. 6)
    cos_i = jnp.cos(inclination)
    delta_ra, delta_dec = astrometric_orbit_at_times(
        times,
        period,
        eccentricity,
        t_peri,
        arg_peri,
        cos_i,
        lon_asc_node,
        semi_major_axis,
    )
    y_orbit = uconvert("mas", sin_psi * delta_ra + cos_psi * delta_dec)

    # Thiele-Innes constants for true_params output
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(arg_peri),
        jnp.sin(arg_peri),
        jnp.cos(lon_asc_node),
        jnp.sin(lon_asc_node),
        cos_i,
    )
    A = A * semi_major_axis
    B = B * semi_major_axis
    F = F * semi_major_axis
    G = G * semi_major_axis

    noise: AbstractQuantity = Q.from_(rng.normal(size=n_obs), "")
    y_al = y_astro + y_orbit + al_error * noise

    # Store true parameters with units
    true_params = {
        "period": period,
        "eccentricity": eccentricity,
        "semi_major_axis": semi_major_axis,
        "t_peri": t_peri,
        "alpha0": alpha0,
        "delta0": delta0,
        "mu_alpha": mu_alpha,
        "mu_delta": mu_delta,
        "parallax": parallax,
        "A": A,
        "B": B,
        "F": F,
        "G": G,
        "arg_peri": arg_peri,
        "lon_asc_node": lon_asc_node,
        "inclination": inclination,
    }

    data = GaiaAstrometryData(
        time=times,
        al_position=y_al,
        al_position_err=al_error,
        scan_angle=scan_angle,
        parallax_factor=parallax_factor,
        t_ref=t_ref,
    )

    return data, true_params
