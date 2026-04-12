"""Helper functions for Keplerian orbits.

Implementations of core orbit computations used by ``harv.kepler``, ``harv.likelihood``,
and ``harv.simulate``.

All functions accept :class:`~unxt.Quantity` objects (including dimensionless ones)
as well as plain JAX arrays and Python scalars.
"""

__all__ = (
    "mean_anomaly",
    "rv_shape",
    "thiele_innes_ABFG",
    "true_anomaly_from_mean",
    "compute_true_anomaly_components",
    "rv_at_times",
    "astrometric_orbit_at_times",
)

from typing import cast

import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import Quantity
from unxt.quantity import ustrip

from harv.custom_types import (
    BatchFloat,
    BatchQAngle,
    BatchQSpeed,
    BatchQTime,
    ScalarFloat,
    ScalarQAngle,
    ScalarQSpeed,
    ScalarQTime,
)


def mean_anomaly(dt: BatchQTime, period: ScalarQTime) -> BatchQAngle:
    """Compute mean anomaly from elapsed time and period.

    ``M = 2pi * dt / period``, returned as a :class:`~unxt.Quantity` with angle
    units (radians).
    """
    return Quantity.from_(ustrip("", 2 * jnp.pi * dt / period), "rad")


def true_anomaly_from_mean(
    M: BatchQAngle, eccentricity: ScalarFloat
) -> tuple[BatchFloat, BatchFloat]:
    """Solve Kepler's equation: mean anomaly -> (sin f, cos f).

    Wraps ``jaxoplanet.core.kepler.kepler``. The mean anomaly is stripped to
    radians internally.
    """
    return kepler(ustrip("rad", M), eccentricity)


def rv_shape(
    sin_f: BatchFloat,
    cos_f: BatchFloat,
    eccentricity: ScalarFloat,
    arg_peri: ScalarQAngle | ScalarFloat,
) -> BatchFloat:
    """RV shape function: cos(omega + f) + e*cos(omega).

    Returns the dimensionless RV amplitude factor for each observation.
    ``arg_peri`` may be a plain float/array in radians or a
    :class:`~unxt.Quantity` with angle units; ``jnp.cos`` handles both via
    quax dispatch.  ``eccentricity`` may likewise be a dimensionless
    :class:`~unxt.Quantity` or a plain scalar.
    """
    cos_wf = jnp.cos(arg_peri) * cos_f - jnp.sin(arg_peri) * sin_f
    # cast: jnp ops on Quantity inputs return AbstractQuantity (quax dispatch),
    # which mypy cannot verify is a subtype of BatchFloat without the hint.
    return cast("BatchFloat", cos_wf + eccentricity * jnp.cos(arg_peri))


def thiele_innes_ABFG(
    cos_arg_peri: ScalarFloat,
    sin_arg_peri: ScalarFloat,
    cos_lon_asc_node: ScalarFloat,
    sin_lon_asc_node: ScalarFloat,
    cos_i: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
    """Compute unit Thiele-Innes constants (A, B, F, G).

    Returns the constants with an implicit semi-major axis of 1. Multiply each
    by ``a`` to recover the physical Thiele-Innes constants.

    In the Gaia LPC convention (Lindegren & Bastian, GAIA-C3-TN-LU-LL-061-08,
    Eq. 4), B and G project into the RA (``a``) direction, while A and F
    project into the Dec (``d``) direction.

    See also Eq. A.1 of https://arxiv.org/abs/2206.05726.
    """
    A = cos_arg_peri * cos_lon_asc_node - sin_arg_peri * sin_lon_asc_node * cos_i
    B = cos_arg_peri * sin_lon_asc_node + sin_arg_peri * cos_lon_asc_node * cos_i
    F = -(sin_arg_peri * cos_lon_asc_node + cos_arg_peri * sin_lon_asc_node * cos_i)
    G = -(sin_arg_peri * sin_lon_asc_node - cos_arg_peri * cos_lon_asc_node * cos_i)
    # cast: arithmetic on ScalarFloat inputs may return AbstractQuantity via quax
    # dispatch; mypy cannot verify that AbstractQuantity satisfies ScalarFloat.
    return cast(
        "tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]", (A, B, F, G)
    )


def compute_true_anomaly_components(
    time: BatchQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
) -> tuple[BatchFloat, BatchFloat]:
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
    times: BatchQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
    arg_peri: ScalarQAngle,
    rv_semiamp: ScalarQSpeed,
    v_sys: ScalarQSpeed,
) -> BatchQSpeed:
    """Compute the RV model: K*[cos(omega + f(t)) + e*cos(omega)] + v_0.

    Parameters
    ----------
    times
        Observation times.
    period
        Orbital period.
    eccentricity
        Orbital eccentricity.
    t_peri
        Time of periastron passage.  In the likelihood layer this is
        derived from the dimensionless ``phase_peri`` as
        ``t_peri = phase_peri * period`` (see ``_solve_kepler``).
    arg_peri
        Argument of periastron omega.
    rv_semiamp
        RV semi-amplitude.
    v_sys
        Systemic velocity.

    Returns
    -------
    rv
        Radial velocities in the same unit as ``rv_semiamp`` and ``v_sys``.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.kepler.orbits import rv_at_times
    >>> times = Quantity([0.0, 50.0, 100.0], "day")
    >>> rv = rv_at_times(
    ...     times,
    ...     period=Quantity(200.0, "day"),
    ...     eccentricity=0.3,
    ...     t_peri=Quantity(50.0, "day"),
    ...     arg_peri=Quantity(1.2, "rad"),
    ...     rv_semiamp=Quantity(8.0, "km/s"),
    ...     v_sys=Quantity(-5.0, "km/s"),
    ... )
    >>> rv.unit
    Unit("km / s")
    """
    sin_f, cos_f = compute_true_anomaly_components(times, period, eccentricity, t_peri)
    amplitude = rv_shape(sin_f, cos_f, eccentricity, arg_peri)
    return cast("BatchQSpeed", rv_semiamp * amplitude + v_sys)


def astrometric_orbit_at_times(
    times: BatchQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
    arg_peri: ScalarQAngle,
    cos_i: ScalarFloat,
    lon_asc_node: ScalarQAngle,
    semi_major_axis: ScalarQAngle,
) -> tuple[BatchQAngle, BatchQAngle]:
    """Compute sky-plane astrometric orbit (Deltara, Deltadec) at given times.

    Uses the Thiele-Innes parameterization following the Gaia local plane
    coordinate (LPC) convention (Lindegren & Bastian, GAIA-C3-TN-LU-LL-061-08,
    Eq. 4)::

        Deltara  = (B*cos f + G*sin f) * a      (RA / ``a`` direction)
        Deltadec = (A*cos f + F*sin f) * a      (Dec / ``d`` direction)

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
        Time of periastron passage.  In the likelihood layer this is
        derived from the dimensionless ``phase_peri`` as
        ``t_peri = phase_peri * period`` (see ``_solve_kepler``).
    arg_peri
        Argument of periastron omega.
    cos_i
        Cosine of orbital inclination.
    lon_asc_node
        Longitude of the ascending node Omega.
    semi_major_axis
        Photocentric semi-major axis.

    Returns
    -------
    delta_ra, delta_dec
        Sky-plane offsets in the same unit as ``semi_major_axis``.

    Examples
    --------
    >>> from unxt import Quantity
    >>> from harv.kepler.orbits import astrometric_orbit_at_times
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
    # LPC convention: B,G -> RA (a); A,F -> Dec (d)
    delta_ra = (B * cos_f + G * sin_f) * semi_major_axis
    delta_dec = (A * cos_f + F * sin_f) * semi_major_axis
    # cast: multiplication by a Quantity returns AbstractQuantity via quax dispatch;
    # mypy cannot verify that AbstractQuantity satisfies BatchQAngle.
    return cast("tuple[BatchQAngle, BatchQAngle]", (delta_ra, delta_dec))
