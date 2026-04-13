"""Simulate radial velocity data.

This module provides utilities for generating synthetic RV measurements
for single-lined (SB1) and double-lined (SB2) spectroscopic binaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import quaxed.numpy as jnp
from unxt import AbstractQuantity, Q, ustrip

from harv.custom_types import ScalarQAngle, ScalarQSpeed, ScalarQTime
from harv.data import RVData, SourceData
from harv.kepler.orbits import rv_at_times

__all__ = ["simulate_rv_multisurv_data", "simulate_rv_sb1_data"]


def simulate_rv_sb1_data(
    seed: int = 42,
    n_obs: int = 50,
    baseline: ScalarQTime | None = None,
    # Orbital parameters
    period: ScalarQTime | None = None,
    eccentricity: float | None = None,
    t_peri: ScalarQTime | None = None,
    arg_peri: ScalarQAngle | None = None,
    # RV parameters
    rv_semiamp: ScalarQSpeed | None = None,
    v_sys: ScalarQSpeed | None = None,
    # Uncertainty
    rv_err: ScalarQSpeed | None = None,
    # Reference time
    t_ref: ScalarQTime | None = None,
    # Instrument
    instrument: str = "default",
) -> tuple[RVData, dict[str, Any]]:
    """Simulate radial velocity data for a single-lined binary (SB1).

    This function generates synthetic RV measurements following the model:
        RV(t) = K*[cos(omega + f(t)) + e*cos(omega)] + v_0

    where f(t) is the true anomaly computed via Kepler's equation.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. Default: 42.
    n_obs : int, optional
        Number of observations. Default: 50.
    baseline : Q["time"], optional
        Time baseline for observations. Default: 5 years.
    period : Q["time"], optional
        Orbital period. If None, randomly drawn from [10, 1000] days.
    eccentricity : float, optional
        Orbital eccentricity. If None, randomly drawn from [0, 0.7].
    t_peri : Q["time"], optional
        Time of periastron passage. If None, randomly drawn from [0, period].
    arg_peri : Q["angle"], optional
        Argument of periastron omega. If None, randomly drawn from [0, 2pi].
    rv_semiamp : Q["speed"], optional
        RV semi-amplitude. If None, randomly drawn from [1, 50] km/s.
    v_sys : Q["speed"], optional
        Systemic velocity. If None, randomly drawn ~ N(0, 20) km/s.
    rv_err : Q["speed"], optional
        RV measurement uncertainties (1-sigma). If None, randomly drawn from
        U(0.01, 0.5) km/s for each observation.
    t_ref : Q["time"], optional
        Reference time. If None, randomly chosen.
    instrument : str, optional
        Instrument name for the observations. Default: "default".

    Returns
    -------
    data : RVData
        Simulated RV data container.
    true_params : dict
        Dictionary of true parameter values used in simulation.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.simulate import simulate_rv_sb1_data
    >>> data, true_params = simulate_rv_sb1_data(
    ...     seed=42,
    ...     n_obs=30,
    ...     period=Q(100.0, "day"),
    ...     eccentricity=0.3,
    ...     rv_semiamp=Q(10.0, "km/s"),
    ... )
    >>> data.time.shape
    (30,)
    >>> true_params["rv_semiamp"]
    Q['speed'](Array(10., dtype=float64), unit='km / s')
    """
    ss = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in ss.spawn(10)]
    rng = rngs[0]

    if baseline is None:
        baseline = Q(5.0, "yr")

    if period is None:
        period = Q(rngs[1].uniform(10, 1000), "day")

    if eccentricity is None:
        eccentricity = rngs[2].uniform(0.0, 0.7)

    if t_peri is None:
        t_peri = Q(rngs[3].uniform(0.0, ustrip(period.unit, period)), period.unit)

    if arg_peri is None:
        arg_peri = Q(rngs[4].uniform(0, 2 * np.pi), "rad")

    if rv_semiamp is None:
        rv_semiamp = Q(rngs[5].uniform(1.0, 50.0), "km/s")

    if v_sys is None:
        v_sys = Q(rngs[6].normal(0, 20), "km/s")

    if rv_err is None:
        rv_err = Q(rngs[7].uniform(0.01, 0.5, n_obs), "km/s")
    else:
        # Broadcast scalar rv_err to per-observation array
        rv_err = Q(jnp.broadcast_to(ustrip(rv_err.unit, rv_err), (n_obs,)), rv_err.unit)

    if t_ref is None:
        t_ref = Q(rng.uniform(0, ustrip(baseline.unit, baseline)), baseline.unit)

    # Observation times over baseline
    dt: AbstractQuantity = Q(
        jnp.sort(rng.uniform(0.0, ustrip(baseline.unit, baseline), n_obs)),
        baseline.unit,
    )
    times = dt + t_ref

    # Compute RV model: RV(t) = K*[cos(omega + f) + e*cos(omega)] + v_0
    rv_true = rv_at_times(
        times, period, eccentricity, t_peri, arg_peri, rv_semiamp, v_sys
    )

    # Add noise
    noise: AbstractQuantity = Q.from_(rng.normal(size=n_obs), "")
    rv = rv_true + rv_err * noise

    # Store true parameters
    true_params = {
        "period": period,
        "eccentricity": eccentricity,
        "t_peri": t_peri,
        "arg_peri": arg_peri,
        "rv_semiamp": rv_semiamp,
        "v_sys": v_sys,
    }

    data = RVData(
        time=times,
        rv=rv,
        rv_err=rv_err,
    )

    return data, true_params


def simulate_rv_multisurv_data(  # noqa: C901
    instruments: dict[str, ScalarQSpeed | None],
    seed: int = 42,
    n_obs_per_instrument: int = 30,
    baseline: ScalarQTime | None = None,
    # Shared orbital parameters
    period: ScalarQTime | None = None,
    eccentricity: float | None = None,
    t_peri: ScalarQTime | None = None,
    arg_peri: ScalarQAngle | None = None,
    # Shared RV amplitude and systemic velocity
    rv_semiamp: ScalarQSpeed | None = None,
    v_sys: ScalarQSpeed | None = None,
    rv_err: ScalarQSpeed | None = None,
    t_ref: ScalarQTime | None = None,
) -> tuple[SourceData, dict[str, Any]]:
    """Simulate multi-survey RV data with per-instrument zero-point offsets.

    Generates a :class:`~harv.data.SourceData` containing one
    :class:`~harv.data.RVData` per instrument, all sharing the
    same orbital solution but with different per-instrument zero-point offsets.

    Parameters
    ----------
    instruments : dict[str, Q["speed"] | None]
        Mapping from instrument name to its zero-point offset.  Pass ``None``
        for the reference instrument (offset = 0).  Example::

            {"keck": None, "espresso": Q(3.5, "km/s")}

    seed : int, optional
        Random seed. Default: 42.
    n_obs_per_instrument : int, optional
        Number of observations per instrument. Default: 30.
    baseline : Q["time"], optional
        Total time baseline. Default: 5 years.
    period : Q["time"], optional
        Orbital period. Randomly drawn if None.
    eccentricity : float, optional
        Orbital eccentricity. Randomly drawn if None.
    t_peri : Q["time"], optional
        Time of periastron passage. Randomly drawn if None.
    arg_peri : Q["angle"], optional
        Argument of periastron. Randomly drawn if None.
    rv_semiamp : Q["speed"], optional
        RV semi-amplitude. Randomly drawn if None.
    v_sys : Q["speed"], optional
        Systemic velocity of the reference instrument. Randomly drawn if None.
    rv_err : Q["speed"], optional
        Measurement uncertainty (same for all instruments and observations).
        Randomly drawn if None.
    t_ref : Q["time"], optional
        Reference epoch. Randomly drawn if None.

    Returns
    -------
    source_data : SourceData
        Multi-instrument RV data container.
    true_params : dict
        True parameter values: ``period``, ``eccentricity``, ``t_peri``,
        ``arg_peri``, ``rv_semiamp``, ``v_sys``, and one entry per non-reference
        instrument named ``"offset_{name}"``.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.simulate import simulate_rv_multisurv_data
    >>> source_data, true_params = simulate_rv_multisurv_data(
    ...     instruments={"keck": None, "espresso": Q(3.5, "km/s")},
    ...     seed=0,
    ...     n_obs_per_instrument=20,
    ...     period=Q(200.0, "day"),
    ...     rv_semiamp=Q(15.0, "km/s"),
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
        baseline = Q(5.0, "yr")
    if period is None:
        period = Q(rngs[1].uniform(10, 1000), "day")
    if eccentricity is None:
        eccentricity = rngs[2].uniform(0.0, 0.7)
    if t_peri is None:
        t_peri = Q(rngs[3].uniform(0.0, ustrip(period.unit, period)), period.unit)
    if arg_peri is None:
        arg_peri = Q(rngs[4].uniform(0, 2 * np.pi), "rad")
    if rv_semiamp is None:
        rv_semiamp = Q(rngs[5].uniform(1.0, 50.0), "km/s")
    if v_sys is None:
        v_sys = Q(rngs[6].normal(0, 20), "km/s")
    if rv_err is None:
        rv_err = Q(rngs[7].uniform(0.01, 0.5), "km/s")
    if t_ref is None:
        t_ref = Q(rng.uniform(0, ustrip(baseline.unit, baseline)), baseline.unit)

    datasets: dict[str, RVData] = {}
    true_params: dict[str, Any] = {
        "period": period,
        "eccentricity": eccentricity,
        "t_peri": t_peri,
        "arg_peri": arg_peri,
        "rv_semiamp": rv_semiamp,
        "v_sys": v_sys,
    }

    for i, (name, offset) in enumerate(instruments.items()):
        inst_rng = rngs[8 + i]
        eff_offset = offset if offset is not None else Q(0.0, v_sys.unit)
        if offset is not None:
            true_params[f"offset_{name}"] = offset

        dt: AbstractQuantity = Q(
            jnp.sort(
                inst_rng.uniform(
                    0.0, ustrip(baseline.unit, baseline), n_obs_per_instrument
                )
            ),
            baseline.unit,
        )
        times = dt + t_ref

        rv_true = (
            rv_at_times(
                times, period, eccentricity, t_peri, arg_peri, rv_semiamp, v_sys
            )
            + eff_offset
        )

        n_obs = n_obs_per_instrument
        noise: AbstractQuantity = Q.from_(inst_rng.normal(size=n_obs), "")
        rv_obs = rv_true + rv_err * noise

        datasets[name] = RVData(
            time=times,
            rv=rv_obs,
            rv_err=Q(jnp.full(n_obs, ustrip(rv_err.unit, rv_err)), rv_err.unit),
        )

    return SourceData(**datasets), true_params
