"""Helper functions for Keplerian orbits.

Implementations of core orbit computations used by ``harv.kepler``, ``harv.likelihood``,
and ``harv.simulate``.

All functions accept :class:`~unxt.Q` objects (including dimensionless ones)
as well as plain JAX arrays and Python scalars.
"""

__all__ = (
    "mean_anomaly",
    "rv_shape",
    "thiele_innes_ABFG",
    "true_anomaly_from_mean",
    "campbell_from_thiele_innes",
    "thiele_innes_from_campbell",
    "ecc_omega_from_ecosw_esinw",
    "ecosw_esinw_from_ecc_omega",
    "compute_true_anomaly_components",
    "rv_at_times",
    "astrometric_orbit_at_times",
)

from typing import cast

import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.custom_types import (
    BatchFloat,
    BatchFloatLike,
    BatchQAngle,
    BatchQDimless,
    BatchQSpeed,
    BatchQTime,
    ScalarFloat,
    ScalarQAngle,
    ScalarQSpeed,
    ScalarQTime,
)


def mean_anomaly(dt: BatchQTime, period: ScalarQTime) -> BatchQAngle:
    """Compute mean anomaly from elapsed time and period.

    ``M = 2pi * dt / period``, returned as a :class:`~unxt.Q` with angle
    units (radians).

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import mean_anomaly
    >>> M = mean_anomaly(Q(50.0, "day"), Q(100.0, "day"))
    >>> M.unit
    Unit("rad")
    """
    return Q.from_(ustrip("", 2 * jnp.pi * dt / period), "rad")


def true_anomaly_from_mean(
    M: BatchQAngle, eccentricity: ScalarFloat
) -> tuple[BatchFloat, BatchFloat]:
    """Solve Kepler's equation: mean anomaly -> (sin f, cos f).

    Wraps ``jaxoplanet.core.kepler.kepler``. The mean anomaly is stripped to
    radians internally.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import true_anomaly_from_mean
    >>> sin_f, cos_f = true_anomaly_from_mean(Q(1.0, "rad"), 0.3)
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
    :class:`~unxt.Q` with angle units; ``jnp.cos`` handles both via
    quax dispatch.  ``eccentricity`` may likewise be a dimensionless
    :class:`~unxt.Q` or a plain scalar.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import true_anomaly_from_mean, rv_shape
    >>> sin_f, cos_f = true_anomaly_from_mean(Q(1.0, "rad"), 0.3)
    >>> shape = rv_shape(sin_f, cos_f, 0.3, Q(0.5, "rad"))
    """
    cos_wf = jnp.cos(arg_peri) * cos_f - jnp.sin(arg_peri) * sin_f
    ecc = ustrip(AllowValue, "", eccentricity)

    # cast: jnp ops on Quantity inputs return AbstractQuantity (quax dispatch),
    # which mypy cannot verify is a subtype of BatchFloat without the hint.
    return cast("BatchFloat", cos_wf + ecc * jnp.cos(arg_peri))


def thiele_innes_ABFG(
    cos_arg_peri: BatchFloatLike,
    sin_arg_peri: BatchFloatLike,
    cos_lon_asc_node: BatchFloatLike,
    sin_lon_asc_node: BatchFloatLike,
    cos_i: BatchFloatLike,
) -> tuple[BatchFloatLike, BatchFloatLike, BatchFloatLike, BatchFloatLike]:
    """Compute unit Thiele-Innes constants (A, B, F, G).

    Returns the constants with an implicit semi-major axis of 1. Multiply each
    by ``a`` to recover the physical Thiele-Innes constants.

    Inputs are dimensionless and may be scalar or batched (plain scalars, JAX
    arrays, or dimensionless :class:`~unxt.Q`); the computation broadcasts
    elementwise.

    The sky-plane orbital displacement uses the Thiele-Innes coordinates
    ``X = (cos E - e)`` and ``Y = sqrt(1-e^2) sin E``, or equivalently in
    terms of true anomaly: ``X = r/a cos f``, ``Y = r/a sin f`` where
    ``r/a = (1-e^2)/(1+e cos f)``.

    In the Gaia LPC convention (Lindegren & Bastian, GAIA-C3-TN-LU-LL-061-08,
    Eq. 4), B and G project into the RA (``a``) direction, while A and F
    project into the Dec (``d``) direction.

    See also Eq. A.1 of https://arxiv.org/abs/2206.05726.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> from unxt import Q
    >>> from harv.kepler.orbits import thiele_innes_ABFG
    >>> A, B, F, G = thiele_innes_ABFG(
    ...     cos_arg_peri=jnp.cos(Q(0.5, "rad")),
    ...     sin_arg_peri=jnp.sin(Q(0.5, "rad")),
    ...     cos_lon_asc_node=jnp.cos(Q(1.0, "rad")),
    ...     sin_lon_asc_node=jnp.sin(Q(1.0, "rad")),
    ...     cos_i=jnp.cos(Q(0.3, "rad")),
    ... )
    """
    A = cos_arg_peri * cos_lon_asc_node - sin_arg_peri * sin_lon_asc_node * cos_i
    B = cos_arg_peri * sin_lon_asc_node + sin_arg_peri * cos_lon_asc_node * cos_i
    F = -(sin_arg_peri * cos_lon_asc_node + cos_arg_peri * sin_lon_asc_node * cos_i)
    G = -(sin_arg_peri * sin_lon_asc_node - cos_arg_peri * cos_lon_asc_node * cos_i)
    # cast: arithmetic on BatchFloatLike inputs may return AbstractQuantity via
    # quax dispatch; mypy cannot verify AbstractQuantity satisfies BatchFloatLike.
    return cast(
        "tuple[BatchFloatLike, BatchFloatLike, BatchFloatLike, BatchFloatLike]",
        (A, B, F, G),
    )


def campbell_from_thiele_innes(
    A: BatchQAngle,
    B: BatchQAngle,
    F: BatchQAngle,
    G: BatchQAngle,
) -> dict[str, Q]:
    r"""Invert Thiele-Innes constants to Campbell orbital elements.

    This follows Halbwachs, Pourbaix, et al. 2023 (see the appendix):

    .. math::

        u &= \frac{1}{2}(A^2+B^2+F^2+G^2) \\
        v &= A\,G - B\,F \\
        a_0 &= \sqrt{u + \sqrt{\max(u^2 - v^2, 0)}} \\
        \omega + \Omega &= \mathrm{atan2}(B - F, A + G) \\
        \omega - \Omega &= \mathrm{atan2}(-B - F, A - G) \\
        \cos i &= v / a_0^2

    The returned ``cos_i`` is **signed**: it preserves the sign of
    :math:`v = AG - BF`, which determines whether the orbit is prograde
    (``cos_i > 0``) or retrograde (``cos_i < 0``) under the LPC convention.
    This is required for the round-trip
    ``thiele_innes_from_campbell(*campbell_from_thiele_innes(...))`` to recover
    the original TI constants -- a positive-only convention would silently flip
    the sky orbit for retrograde fits.
    """
    u = 0.5 * (A**2 + B**2 + F**2 + G**2)
    v = A * G - B * F
    # cast: jnp.sqrt on Quantity inputs returns AbstractQuantity via quax
    # dispatch; arg_peri, lon_asc_node, and cos_i below are re-wrapped with
    # Q(...), so the returned dict is uniformly typed as concrete Quantity.
    a0 = cast(
        "BatchQAngle",
        jnp.sqrt(u + jnp.sqrt(jnp.maximum(u * u - v * v, Q(0.0, (u * u).unit)))),
    )
    # arctan2 of same-unit Quantities returns Q[rad]; strip before mod
    wPO = ustrip(AllowValue, "rad", jnp.arctan2(B - F, A + G))  # ω + Ω
    wMO = ustrip(AllowValue, "rad", jnp.arctan2(-B - F, A - G))  # ω - Ω
    arg_peri = Q(jnp.mod(0.5 * (wPO + wMO), 2.0 * jnp.pi), "rad")
    lon_asc_node = Q(jnp.mod(0.5 * (wPO - wMO), 2.0 * jnp.pi), "rad")
    # v/a0² is dimensionless and signed (the sign carries the cos_i sign,
    # which is required for the inverse map to round-trip correctly).
    cos_i = Q(
        ustrip(AllowValue, "", v / jnp.maximum(a0**2, Q(1e-30, (a0**2).unit))),
        "",
    )
    return {
        "semi_major_axis": a0,
        "arg_peri": arg_peri,
        "lon_asc_node": lon_asc_node,
        "cos_i": cos_i,
    }


def thiele_innes_from_campbell(
    semi_major_axis: BatchQAngle,
    arg_peri: BatchQAngle,
    lon_asc_node: BatchQAngle,
    cos_i: BatchQDimless,
) -> tuple[BatchQAngle, BatchQAngle, BatchQAngle, BatchQAngle]:
    r"""Convert Campbell orbital elements to physical Thiele-Innes constants.

    The forward direction of the change of variables inverted by
    :func:`campbell_from_thiele_innes`.  The unit Thiele-Innes constants from
    :func:`thiele_innes_ABFG` are scaled by the semi-major axis:

    .. math::

        (A, B, F, G) = a_0 \cdot
            \mathrm{thiele\_innes\_ABFG}(\cos\omega, \sin\omega,
                                        \cos\Omega, \sin\Omega, \cos i)

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import thiele_innes_from_campbell
    >>> A, B, F, G = thiele_innes_from_campbell(
    ...     Q(2.0, "mas"), Q(0.5, "rad"), Q(1.0, "rad"), Q(0.3, ""),
    ... )
    >>> A.unit
    Unit("mas")
    """
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(arg_peri),
        jnp.sin(arg_peri),
        jnp.cos(lon_asc_node),
        jnp.sin(lon_asc_node),
        cos_i,
    )
    # cast: multiplying a Quantity returns AbstractQuantity via quax dispatch,
    # which ty cannot verify is a subtype of BatchQAngle.
    return cast(
        "tuple[BatchQAngle, BatchQAngle, BatchQAngle, BatchQAngle]",
        (
            semi_major_axis * A,
            semi_major_axis * B,
            semi_major_axis * F,
            semi_major_axis * G,
        ),
    )


def ecc_omega_from_ecosw_esinw(
    ecosw: BatchQDimless,
    esinw: BatchQDimless,
) -> tuple[BatchQDimless, BatchQAngle]:
    r"""Convert ``(e cos omega, e sin omega)`` to eccentricity and arg. of periastron.

    The inverse of :func:`ecosw_esinw_from_ecc_omega`:

    .. math::

        e &= \sqrt{\mathrm{ecosw}^2 + \mathrm{esinw}^2} \\
        \omega &= \mathrm{atan2}(\mathrm{esinw}, \mathrm{ecosw})

    The returned ``arg_peri`` lies in ``(-pi, pi]`` (the range of ``atan2``); use
    :meth:`harv.samplers.Samples.wrap_angles` to wrap it into ``[0, 2*pi)`` if a
    prior requires that range.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import ecc_omega_from_ecosw_esinw
    >>> ecc, arg_peri = ecc_omega_from_ecosw_esinw(Q(0.6, ""), Q(0.8, ""))
    >>> float(ecc.value)
    1.0
    >>> arg_peri.unit
    Unit("rad")
    """
    eccentricity = jnp.sqrt(ecosw**2 + esinw**2)
    arg_peri = jnp.arctan2(esinw, ecosw)
    # cast: jnp ops on Quantity inputs return AbstractQuantity via quax dispatch,
    # which ty cannot verify is a subtype of the declared return types.
    return cast("tuple[BatchQDimless, BatchQAngle]", (eccentricity, arg_peri))


def ecosw_esinw_from_ecc_omega(
    eccentricity: BatchQDimless,
    arg_peri: BatchQAngle,
) -> tuple[BatchQDimless, BatchQDimless]:
    r"""Convert eccentricity and arg. of periastron to ``(e cos omega, e sin omega)``.

    The inverse of :func:`ecc_omega_from_ecosw_esinw`:

    .. math::

        \mathrm{ecosw} &= e \cos\omega \\
        \mathrm{esinw} &= e \sin\omega

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import ecosw_esinw_from_ecc_omega
    >>> ecosw, esinw = ecosw_esinw_from_ecc_omega(Q(0.5, ""), Q(0.0, "rad"))
    >>> float(ecosw.value)
    0.5
    >>> float(esinw.value)
    0.0
    """
    ecosw = eccentricity * jnp.cos(arg_peri)
    esinw = eccentricity * jnp.sin(arg_peri)
    # cast: arithmetic on Quantity inputs returns AbstractQuantity via quax
    # dispatch, which ty cannot verify is a subtype of BatchQDimless.
    return cast("tuple[BatchQDimless, BatchQDimless]", (ecosw, esinw))


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
        ``(sin_f, cos_f)`` — true anomaly components, each shape ``(n,)``.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import compute_true_anomaly_components
    >>> sin_f, cos_f = compute_true_anomaly_components(
    ...     time=Q([0.0, 25.0, 50.0], "day"),
    ...     period=Q(100.0, "day"),
    ...     eccentricity=0.3,
    ...     t_peri=Q(0.0, "day"),
    ... )
    """
    M = mean_anomaly(time - t_peri, period)
    return true_anomaly_from_mean(M, ustrip(AllowValue, "", eccentricity))


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
        Radial velocities in the same unit as ``rv_semiamp`` and ``v_sys``.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import rv_at_times
    >>> times = Q([0.0, 50.0, 100.0], "day")
    >>> rv = rv_at_times(
    ...     times,
    ...     period=Q(200.0, "day"),
    ...     eccentricity=0.3,
    ...     t_peri=Q(50.0, "day"),
    ...     arg_peri=Q(1.2, "rad"),
    ...     rv_semiamp=Q(8.0, "km/s"),
    ...     v_sys=Q(-5.0, "km/s"),
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
        ``(delta_ra, delta_dec)`` — sky-plane offsets in the same unit as
        ``semi_major_axis``.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.orbits import astrometric_orbit_at_times
    >>> times = Q([0.0, 100.0, 200.0], "day")
    >>> dra, ddec = astrometric_orbit_at_times(
    ...     times,
    ...     period=Q(300.0, "day"),
    ...     eccentricity=0.3,
    ...     t_peri=Q(0.0, "day"),
    ...     arg_peri=Q(1.2, "rad"),
    ...     cos_i=0.5,
    ...     lon_asc_node=Q(0.8, "rad"),
    ...     semi_major_axis=Q(3.0, "mas"),
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
    # Thiele-Innes orbital coordinates: X = (cos E - e), Y = sqrt(1-e^2) sin E,
    # expressed in terms of true anomaly via r/a = (1-e^2)/(1+e cos f).
    r_over_a = (1 - eccentricity**2) / (1 + eccentricity * cos_f)
    X = r_over_a * cos_f
    Y = r_over_a * sin_f
    # LPC convention: B,G -> RA (a); A,F -> Dec (d)
    delta_ra = (B * X + G * Y) * semi_major_axis
    delta_dec = (A * X + F * Y) * semi_major_axis
    # cast: multiplication by a Q returns AbstractQuantity via quax dispatch;
    # mypy cannot verify that AbstractQuantity satisfies BatchQAngle.
    return cast("tuple[BatchQAngle, BatchQAngle]", (delta_ra, delta_dec))
