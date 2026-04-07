"""Helper for Keplerian orbits."""

import quaxed.numpy as jnp
from jaxtyping import Array, Float
from unxt.quantity import ustrip

from harv.custom_types import (
    BatchableQAngle,
    BatchableQSpeed,
    BatchableQTime,
    ScalarFloat,
    ScalarQAngle,
    ScalarQSpeed,
    ScalarQTime,
)
from harv.kepler._orbit_math import (
    mean_anomaly,
    rv_shape,
    thiele_innes_ABFG,
    true_anomaly_from_mean,
)


def compute_true_anomaly_components(
    time: BatchableQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
) -> tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
    """Compute true anomaly at given times.

    Parameters
    ----------
    time
        Observation times, shape (n,)
    period
        Orbital period
    eccentricity
        Orbital eccentricity
    t_peri
        Time of pericenter passage

    Returns
    -------
    sin_f, cos_f
        True anomaly components, each shape (n,)
    """
    M = mean_anomaly(time - t_peri, period)
    return true_anomaly_from_mean(M, eccentricity)


def rv_at_times(
    times: BatchableQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
    arg_peri: ScalarQAngle,
    K: ScalarQSpeed,
    v0: ScalarQSpeed,
) -> BatchableQSpeed:
    """Compute the RV model: K·[cos(ω + f(t)) + e·cos(ω)] + v₀.

    Parameters
    ----------
    times
        Observation times.
    period
        Orbital period.
    eccentricity
        Orbital eccentricity.
    t_peri
        Time of periastron passage.
    arg_peri
        Argument of periastron ω.
    K
        RV semi-amplitude.
    v0
        Systemic velocity.

    Returns
    -------
    rv
        Radial velocities in the same unit as ``K`` and ``v0``.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.kepler.helpers import rv_at_times
    >>> times = Quantity([0.0, 50.0, 100.0], "day")
    >>> rv = rv_at_times(
    ...     times,
    ...     period=Quantity(200.0, "day"),
    ...     eccentricity=0.3,
    ...     t_peri=Quantity(50.0, "day"),
    ...     arg_peri=Quantity(1.2, "rad"),
    ...     K=Quantity(8.0, "km/s"),
    ...     v0=Quantity(-5.0, "km/s"),
    ... )
    >>> rv.unit
    Unit("km / s")
    """
    sin_f, cos_f = compute_true_anomaly_components(times, period, eccentricity, t_peri)
    amplitude = rv_shape(sin_f, cos_f, eccentricity, ustrip("rad", arg_peri))
    return K * amplitude + v0


def astrometric_orbit_at_times(
    times: BatchableQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
    arg_peri: ScalarQAngle,
    cos_i: ScalarFloat,
    lon_asc_node: ScalarQAngle,
    semi_major_axis: ScalarQAngle,
) -> tuple[BatchableQAngle, BatchableQAngle]:
    """Compute sky-plane astrometric orbit (Δra, Δdec) at given times.

    Uses the Thiele-Innes parameterization:

        Δra  = (A·cos f + F·sin f) · a
        Δdec = (B·cos f + G·sin f) · a

    where A, B, F, G are the unit Thiele-Innes constants and a is the
    photocentric semi-major axis.

    Parameters
    ----------
    times
        Observation times.
    period
        Orbital period.
    eccentricity
        Orbital eccentricity.
    t_peri
        Time of periastron passage.
    arg_peri
        Argument of periastron ω.
    cos_i
        Cosine of orbital inclination.
    lon_asc_node
        Longitude of the ascending node Ω.
    semi_major_axis
        Photocentric semi-major axis.

    Returns
    -------
    delta_ra, delta_dec
        Sky-plane offsets in the same unit as ``semi_major_axis``.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.kepler.helpers import astrometric_orbit_at_times
    >>> times = Quantity([0.0, 100.0, 200.0], "day")
    >>> dra, ddec = astrometric_orbit_at_times(
    ...     times,
    ...     period=Quantity(300.0, "day"),
    ...     eccentricity=0.3,
    ...     t_peri=Quantity(0.0, "day"),
    ...     arg_peri=Quantity(1.2, "rad"),
    ...     cos_i=0.5,
    ...     lon_asc_node=Quantity(0.8, "rad"),
    ...     semi_major_axis=Quantity(3.0, "mas"),
    ... )
    >>> dra.unit
    Unit("mas")
    """
    sin_f, cos_f = compute_true_anomaly_components(times, period, eccentricity, t_peri)
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(ustrip("rad", arg_peri)),
        jnp.sin(ustrip("rad", arg_peri)),
        jnp.cos(ustrip("rad", lon_asc_node)),
        jnp.sin(ustrip("rad", lon_asc_node)),
        cos_i,
    )
    delta_ra = (A * cos_f + F * sin_f) * semi_major_axis
    delta_dec = (B * cos_f + G * sin_f) * semi_major_axis
    return delta_ra, delta_dec
