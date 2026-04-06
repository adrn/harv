"""Simulate Gaia-like epoch astrometry data.

This module provides utilities for generating synthetic along-scan astrometry
measurements similar to Gaia DR3/DR4 epoch data. The simulated data includes:
- Along-scan positions with realistic noise
- Scan angles
- Parallax factors
- Measurement uncertainties

The astrometric model includes:
- 5-parameter astrometry (α₀, δ₀, μ_α, μ_δ, ϖ)
- Keplerian orbital motion parameterized by Thiele-Innes constants
"""

from __future__ import annotations

from typing import Any

import numpy as np
import quaxed.numpy as jnp
from unxt import Quantity, uconvert, ustrip

from harv.custom_types import AngularSpeed
from harv.data import GaiaAstrometryData
from harv.kepler import KeplerianOrientation, compute_true_anomaly_components

__all__ = ["simulate_gaia_epoch_astrometry", "fake_parallax_factor"]


def fake_parallax_factor(
    time: Quantity["time"],
    ra: Quantity["angle"],
    dec: Quantity["angle"],
    scan_angle: Quantity["angle"],
) -> jnp.ndarray:
    """Mock, super simplified parallax factor for a star at (ra, dec).

    This is a simplified analytical model for the parallax factor, assuming
    Earth's orbit is circular and using a sinusoidal approximation. For real
    Gaia data, use the actual parallax factors from the epoch data.

    Parameters
    ----------
    time : Quantity["time"]
        Observation times.
    ra : Quantity["angle"]
        Right ascension of the source.
    dec : Quantity["angle"]
        Declination of the source.
    scan_angle : Quantity["angle"]
        Gaia scan angle at each observation.

    Returns
    -------
    parallax_factor : jax.Array
        Dimensionless parallax factor for each observation.
    """
    # Simple sinusoidal model assuming 1-year period
    ang: Quantity[Any] = Quantity(2 * jnp.pi * ustrip("yr", time), "rad")
    P_alpha = jnp.sin(ang - ra)
    P_delta = -jnp.sin(dec) * jnp.cos(ang - ra)
    return P_alpha * jnp.cos(scan_angle) + P_delta * jnp.sin(scan_angle)


def simulate_gaia_epoch_astrometry(
    seed: int = 42,
    n_obs: int = 100,
    baseline: Quantity["time"] | None = None,
    # Orbital parameters
    period: Quantity["time"] | None = None,
    eccentricity: float | None = None,
    t_peri: Quantity["time"] | None = None,
    arg_peri: Quantity["angle"] | None = None,
    lon_asc_node: Quantity["angle"] | None = None,
    inclination: Quantity["angle"] | None = None,
    semimajor_axis: Quantity["angle"] | None = None,
    # Sky position (for parallax factor calculation)
    ra: Quantity["angle"] | None = None,
    dec: Quantity["angle"] | None = None,
    # Astrometric parameters (small offsets in mas)
    alpha0: Quantity["angle"] | None = None,
    delta0: Quantity["angle"] | None = None,
    mu_alpha: Quantity[AngularSpeed] | None = None,
    mu_delta: Quantity[AngularSpeed] | None = None,
    parallax: Quantity["angle"] | None = None,
    # Uncertainty
    al_error: Quantity["angle"] | None = None,
    # Reference time
    t_ref: Quantity["time"] | None = None,
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
    baseline : Quantity["time"], optional
        Time baseline for observations. Default: 5 years.
    period : Quantity["time"], optional
        Orbital period. If None, randomly drawn from [0, 3] years.
    eccentricity : float, optional
        Orbital eccentricity. If None, randomly drawn from [0, 0.9].
    t_peri : Quantity["time"], optional
        Time of periastron passage. If None, randomly drawn from [0, period].
    arg_peri : Quantity["angle"], optional
        Argument of periastron ω. If None, randomly drawn from [0, 2π].
    lon_asc_node : Quantity["angle"], optional
        Longitude of ascending node Ω. If None, randomly drawn from [0, 2π].
    inclination : Quantity["angle"], optional
        Orbital inclination. If None, randomly drawn from cos(i) ~ U(-1, 1).
    semimajor_axis : Quantity["angle"], optional
        Semi-major axis in angular units. If None, randomly drawn from [0.5, 50] mas.
    ra : Quantity["angle"], optional
        Right ascension of the source (for parallax factor). Default: 180 deg.
    dec : Quantity["angle"], optional
        Declination of the source (for parallax factor). Default: 45 deg.
    alpha0 : Quantity["angle"], optional
        Small RA offset from reference position at t_ref. Default: 0 mas.
        This is a linear parameter, not the absolute RA.
    delta0 : Quantity["angle"], optional
        Small Dec offset from reference position at t_ref. Default: 0 mas.
        This is a linear parameter, not the absolute Dec.
    mu_alpha : Quantity["angular speed"], optional
        Proper motion in RA. If None, randomly drawn ~ N(0, 10 mas/yr).
    mu_delta : Quantity["angular speed"], optional
        Proper motion in Dec. If None, randomly drawn ~ N(0, 10 mas/yr).
    parallax : Quantity["angle"], optional
        Parallax. If None, randomly drawn from Exp(10 mas).
    al_error : Quantity["angle"], optional
        Along-scan measurement errors (1σ). If None, randomly drawn from
        U(0.02, 0.1) mas for each observation.
    t_ref : Quantity["time"], optional
        Reference time for astrometry. If None, randomly chosen.

    Returns
    -------
    data : GaiaAstrometryData
        Simulated Gaia astrometry data container.
    true_params : dict
        Dictionary of true parameter values used in simulation, including:
        period, eccentricity, semimajor_axis, t_peri, alpha0, delta0, mu_alpha,
        mu_delta, parallax, A, B, F, G (Thiele-Innes), arg_peri, lon_asc_node,
        inclination.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.simulate import simulate_gaia_epoch_astrometry
    >>> data, true_params = simulate_gaia_epoch_astrometry(
    ...     seed=42,
    ...     n_obs=50,
    ...     period=Quantity(100.0, "day"),
    ...     eccentricity=0.3,
    ...     semimajor_axis=Quantity(2.0, "mas"),
    ... )
    >>> data.time.shape
    (50,)
    >>> true_params["period"]
    Quantity['time'](Array(100., dtype=float64), unit='d')
    """
    ss = np.random.SeedSequence(seed)
    # One RNG per parameter that needs a default value
    rngs = [np.random.default_rng(s) for s in ss.spawn(14)]
    rng = rngs[0]

    if baseline is None:
        baseline = Quantity(5.0, "yr")

    if period is None:
        period = Quantity(rngs[1].uniform(0.3, 3.0), "yr")

    if eccentricity is None:
        eccentricity = rngs[2].uniform(0.0, 0.9)

    if t_peri is None:
        t_peri = Quantity(
            rngs[3].uniform(0.0, ustrip(period.unit, period)), period.unit
        )

    if arg_peri is None:
        arg_peri = Quantity(rngs[4].uniform(0, 2 * np.pi), "rad")

    if lon_asc_node is None:
        lon_asc_node = Quantity(rngs[5].uniform(0, 2 * np.pi), "rad")

    if inclination is None:
        inclination = Quantity(np.arccos(rngs[6].uniform(-1.0, 1.0)), "rad")

    if semimajor_axis is None:
        semimajor_axis = Quantity(rngs[7].uniform(0.5, 50.0), "mas")

    # Sky position for parallax factor calculation (absolute coordinates)
    ra = Quantity(180.0, "deg") if ra is None else ra
    dec = Quantity(45.0, "deg") if dec is None else dec

    # Astrometric offsets - these are small mas-scale deviations from reference
    # These are the LINEAR parameters in the astrometric model
    alpha0 = Quantity(0.0, "mas") if alpha0 is None else uconvert("mas", alpha0)
    delta0 = Quantity(0.0, "mas") if delta0 is None else uconvert("mas", delta0)

    mu_alpha = (
        Quantity(rngs[8].normal(0, 10), "mas/yr") if mu_alpha is None else mu_alpha
    )
    mu_delta = (
        Quantity(rngs[9].normal(0, 10), "mas/yr") if mu_delta is None else mu_delta
    )
    parallax = (
        Quantity(rngs[10].exponential(10.0), "mas") if parallax is None else parallax
    )
    al_error = (
        Quantity(rngs[11].uniform(0.02, 0.1, n_obs), "mas")
        if al_error is None
        else al_error
    )

    if t_ref is None:
        t_ref = Quantity(rng.uniform(0, ustrip(baseline.unit, baseline)), baseline.unit)

    # Observation times over baseline
    dt: Quantity[Any] = Quantity(
        jnp.sort(rng.uniform(0.0, ustrip(baseline.unit, baseline), n_obs)),
        baseline.unit,
    )
    times = dt + t_ref

    # Random scan angles
    scan_angle: Quantity[Any] = Quantity(rng.uniform(0, 2 * np.pi, n_obs), "rad")

    # Fudged parallax factor (uses absolute sky position)
    parallax_factor = fake_parallax_factor(times, ra, dec, scan_angle)

    # True anomaly
    sin_f, cos_f = compute_true_anomaly_components(times, period, eccentricity, t_peri)

    # Compute true along-scan positions
    cos_psi = jnp.cos(scan_angle)
    sin_psi = jnp.sin(scan_angle)

    # 5-parameter astrometry contribution (all in mas)
    # Note: alpha0 and delta0 are small offsets, not absolute coordinates
    y_astro = (
        cos_psi * alpha0
        + sin_psi * delta0
        + cos_psi * mu_alpha * dt
        + sin_psi * mu_delta * dt
        + parallax * parallax_factor
    )
    y_astro = uconvert("mas", y_astro)

    # Orbital contribution using Thiele-Innes
    orientation = KeplerianOrientation.from_angles(
        arg_peri=arg_peri, lon_asc_node=lon_asc_node, inclination=inclination
    )
    A, B, F, G = orientation.thiele_innes_constants(semi_major_axis=semimajor_axis)
    y_orbit = uconvert(
        "mas", sin_psi * (A * cos_f + F * sin_f) + cos_psi * (B * cos_f + G * sin_f)
    )

    noise: Quantity[Any] = Quantity.from_(rng.normal(size=n_obs), "")
    y_al = y_astro + y_orbit + al_error * noise

    # Store true parameters with units
    true_params = {
        "period": period,
        "eccentricity": eccentricity,
        "semimajor_axis": semimajor_axis,
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
