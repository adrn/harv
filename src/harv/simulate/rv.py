"""Simulate radial velocity data.

This module provides utilities for generating synthetic RV measurements
for single-lined (SB1) and double-lined (SB2) spectroscopic binaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import Quantity, ustrip

from harv.data import RadialVelocityData, SourceData

__all__ = ["simulate_rv_multisurv_data", "simulate_rv_sb1_data"]


def simulate_rv_sb1_data(
    seed: int = 42,
    n_obs: int = 50,
    baseline: Quantity["time"] | None = None,
    # Orbital parameters
    period: Quantity["time"] | None = None,
    eccentricity: float | None = None,
    t_peri: Quantity["time"] | None = None,
    arg_peri: Quantity["angle"] | None = None,
    # RV parameters
    K: Quantity["speed"] | None = None,
    v0: Quantity["speed"] | None = None,
    # Uncertainty
    rv_err: Quantity["speed"] | None = None,
    # Reference time
    t_ref: Quantity["time"] | None = None,
    # Instrument
    instrument: str = "default",
) -> tuple[RadialVelocityData, dict[str, Any]]:
    """Simulate radial velocity data for a single-lined binary (SB1).

    This function generates synthetic RV measurements following the model:
        RV(t) = K·[cos(ω + f(t)) + e·cos(ω)] + v₀

    where f(t) is the true anomaly computed via Kepler's equation.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    n_obs : int, optional
        Number of observations. Default: 50.
    baseline : Quantity["time"], optional
        Time baseline for observations. Default: 5 years.
    period : Quantity["time"], optional
        Orbital period. If None, randomly drawn from [10, 1000] days.
    eccentricity : float, optional
        Orbital eccentricity. If None, randomly drawn from [0, 0.7].
    t_peri : Quantity["time"], optional
        Time of periastron passage. If None, randomly drawn from [0, period].
    arg_peri : Quantity["angle"], optional
        Argument of periastron ω. If None, randomly drawn from [0, 2π].
    K : Quantity["speed"], optional
        RV semi-amplitude. If None, randomly drawn from [1, 50] km/s.
    v0 : Quantity["speed"], optional
        Systemic velocity. If None, randomly drawn ~ N(0, 20) km/s.
    rv_err : Quantity["speed"], optional
        RV measurement uncertainties (1-sigma). If None, randomly drawn from
        U(0.01, 0.5) km/s for each observation.
    t_ref : Quantity["time"], optional
        Reference time. If None, randomly chosen.
    instrument : str, optional
        Instrument name for the observations. Default: "default".

    Returns
    -------
    data : RadialVelocityData
        Simulated RV data container.
    true_params : dict
        Dictionary of true parameter values used in simulation.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.simulate import simulate_rv_sb1_data
    >>> data, true_params = simulate_rv_sb1_data(
    ...     seed=42,
    ...     n_obs=30,
    ...     period=Quantity(100.0, "day"),
    ...     eccentricity=0.3,
    ...     K=Quantity(10.0, "km/s"),
    ... )
    >>> data.time.shape
    (30,)
    >>> true_params["K"]
    Quantity['speed'](Array(10., dtype=float64), unit='km / s')
    """
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(10)]
    rng = rngs[0]

    if baseline is None:
        baseline = Quantity(5.0, "yr")

    if period is None:
        period = Quantity(rngs[1].uniform(10, 1000), "day")

    if eccentricity is None:
        eccentricity = rngs[2].uniform(0.0, 0.7)

    if t_peri is None:
        t_peri = Quantity(rngs[3].uniform(0.0, ustrip("day", period)), "day")

    if arg_peri is None:
        arg_peri = Quantity(rngs[4].uniform(0, 2 * np.pi), "rad")

    if K is None:
        K = Quantity(rngs[5].uniform(1.0, 50.0), "km/s")

    if v0 is None:
        v0 = Quantity(rngs[6].normal(0, 20), "km/s")

    if rv_err is None:
        rv_err = Quantity(rngs[7].uniform(0.01, 0.5, n_obs), "km/s")

    if t_ref is None:
        t_ref = Quantity(rng.uniform(0, 1000.0), "day")

    # Observation times over baseline
    dt: Quantity[Any] = Quantity(
        jnp.sort(rng.uniform(0.0, ustrip("day", baseline), n_obs)), "day"
    )
    times = dt + t_ref

    # Compute mean anomaly
    M = 2 * jnp.pi * ustrip("day", dt) / ustrip("day", period)

    # Solve Kepler's equation
    sin_f, cos_f = kepler(M, eccentricity)

    # Compute RV
    # RV = K·[cos(ω + f) + e·cos(ω)] + v₀
    arg_peri_rad = ustrip("rad", arg_peri)
    cos_omega_plus_f = jnp.cos(arg_peri_rad) * cos_f - jnp.sin(arg_peri_rad) * sin_f
    rv_amplitude = cos_omega_plus_f + eccentricity * jnp.cos(arg_peri_rad)

    rv_true = K * rv_amplitude + v0

    # Add noise
    noise: Quantity[Any] = Quantity.from_(rng.normal(size=n_obs), "")
    rv = rv_true + rv_err * noise

    # Store true parameters
    true_params = {
        "period": period,
        "eccentricity": eccentricity,
        "t_peri": t_peri,
        "arg_peri": arg_peri,
        "K": K,
        "v0": v0,
    }

    data = RadialVelocityData(
        time=times,
        rv=rv,
        rv_err=rv_err,
    )

    return data, true_params


def simulate_rv_multisurv_data(
    instruments: dict[str, Quantity["speed"] | None],
    seed: int = 42,
    n_obs_per_instrument: int = 30,
    baseline: Quantity["time"] | None = None,
    # Shared orbital parameters
    period: Quantity["time"] | None = None,
    eccentricity: float | None = None,
    t_peri: Quantity["time"] | None = None,
    arg_peri: Quantity["angle"] | None = None,
    # Shared RV amplitude and systemic velocity
    K: Quantity["speed"] | None = None,
    v0: Quantity["speed"] | None = None,
    rv_err: Quantity["speed"] | None = None,
    t_ref: Quantity["time"] | None = None,
) -> tuple[SourceData, dict[str, Any]]:
    """Simulate multi-survey RV data with per-instrument zero-point offsets.

    Generates a :class:`~harv.data.SourceData` containing one
    :class:`~harv.data.RadialVelocityData` per instrument, all sharing the
    same orbital solution but with different per-instrument zero-point offsets.

    Parameters
    ----------
    instruments : dict[str, Quantity["speed"] | None]
        Mapping from instrument name to its zero-point offset.  Pass ``None``
        for the reference instrument (offset = 0).  Example::

            {"keck": None, "espresso": Quantity(3.5, "km/s")}

    seed : int, optional
        Random seed. Default: 42.
    n_obs_per_instrument : int, optional
        Number of observations per instrument. Default: 30.
    baseline : Quantity["time"], optional
        Total time baseline. Default: 5 years.
    period : Quantity["time"], optional
        Orbital period. Randomly drawn if None.
    eccentricity : float, optional
        Orbital eccentricity. Randomly drawn if None.
    t_peri : Quantity["time"], optional
        Time of periastron passage. Randomly drawn if None.
    arg_peri : Quantity["angle"], optional
        Argument of periastron. Randomly drawn if None.
    K : Quantity["speed"], optional
        RV semi-amplitude. Randomly drawn if None.
    v0 : Quantity["speed"], optional
        Systemic velocity of the reference instrument. Randomly drawn if None.
    rv_err : Quantity["speed"], optional
        Measurement uncertainty (same for all instruments and observations).
        Randomly drawn if None.
    t_ref : Quantity["time"], optional
        Reference epoch. Randomly drawn if None.

    Returns
    -------
    source_data : SourceData
        Multi-instrument RV data container.
    true_params : dict
        True parameter values: ``period``, ``eccentricity``, ``t_peri``,
        ``arg_peri``, ``K``, ``v0``, and one entry per non-reference
        instrument named ``"offset_{name}"``.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.simulate import simulate_rv_multisurv_data
    >>> source_data, true_params = simulate_rv_multisurv_data(
    ...     instruments={"keck": None, "espresso": Quantity(3.5, "km/s")},
    ...     seed=0,
    ...     n_obs_per_instrument=20,
    ...     period=Quantity(200.0, "day"),
    ...     K=Quantity(15.0, "km/s"),
    ... )
    >>> list(source_data.keys())
    ['keck', 'espresso']
    >>> source_data["keck"].time.shape
    (20,)
    """
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(10 + len(instruments))]
    rng = rngs[0]

    if baseline is None:
        baseline = Quantity(5.0, "yr")
    if period is None:
        period = Quantity(rngs[1].uniform(10, 1000), "day")
    if eccentricity is None:
        eccentricity = rngs[2].uniform(0.0, 0.7)
    if t_peri is None:
        t_peri = Quantity(rngs[3].uniform(0.0, ustrip("day", period)), "day")
    if arg_peri is None:
        arg_peri = Quantity(rngs[4].uniform(0, 2 * np.pi), "rad")
    if K is None:
        K = Quantity(rngs[5].uniform(1.0, 50.0), "km/s")
    if v0 is None:
        v0 = Quantity(rngs[6].normal(0, 20), "km/s")
    if rv_err is None:
        rv_err = Quantity(rngs[7].uniform(0.01, 0.5), "km/s")
    if t_ref is None:
        t_ref = Quantity(rng.uniform(0, 1000.0), "day")

    # Pre-compute RV model constants
    arg_peri_rad = ustrip("rad", arg_peri)

    datasets: dict[str, RadialVelocityData] = {}
    true_params: dict[str, Any] = {
        "period": period,
        "eccentricity": eccentricity,
        "t_peri": t_peri,
        "arg_peri": arg_peri,
        "K": K,
        "v0": v0,
    }

    for i, (name, offset) in enumerate(instruments.items()):
        inst_rng = rngs[8 + i]
        eff_offset = offset if offset is not None else Quantity(0.0, "km/s")
        if offset is not None:
            true_params[f"offset_{name}"] = offset

        dt: Quantity[Any] = Quantity(
            jnp.sort(
                inst_rng.uniform(0.0, ustrip("day", baseline), n_obs_per_instrument)
            ),
            "day",
        )
        times = dt + t_ref

        M = 2 * jnp.pi * ustrip("day", dt) / ustrip("day", period)
        from jaxoplanet.core.kepler import kepler

        sin_f, cos_f = kepler(M, eccentricity)
        cos_omega_plus_f = jnp.cos(arg_peri_rad) * cos_f - jnp.sin(arg_peri_rad) * sin_f
        rv_amplitude = cos_omega_plus_f + eccentricity * jnp.cos(arg_peri_rad)
        rv_true = K * rv_amplitude + v0 + eff_offset

        n_obs = n_obs_per_instrument
        noise: Quantity[Any] = Quantity.from_(inst_rng.normal(size=n_obs), "")
        rv_obs = rv_true + rv_err * noise

        datasets[name] = RadialVelocityData(
            time=times,
            rv=rv_obs,
            rv_err=Quantity(jnp.full(n_obs, ustrip("km/s", rv_err)), "km/s"),
        )

    return SourceData(**datasets), true_params
