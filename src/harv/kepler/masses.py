"""Mass functions and physical-quantity helpers for Keplerian orbits.

These functions turn posterior orbital elements into physical masses and
physical (AU) orbit sizes. They are pure, unit-aware, and broadcast over
batched inputs (one value per posterior sample), so callers can apply them
directly to arrays of samples. See docs/spec.md ("Mass functions").
"""

__all__ = (
    "binary_mass_function",
    "astrometric_mass_function",
    "companion_mass_from_mass_function",
    "semi_major_axis_physical",
)

import jax
import quaxed.numpy as jnp
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.custom_types import (
    BatchFloatLike,
    BatchQAngle,
    BatchQLength,
    BatchQMass,
    BatchQSpeed,
    BatchQTime,
    QMass,
)
from harv.kepler.constants import G


def binary_mass_function(
    period: BatchQTime,
    rv_semiamp: BatchQSpeed,
    eccentricity: BatchFloatLike,
) -> BatchQMass:
    r"""Binary mass function from radial-velocity orbital elements.

    .. math::

        f(m) = \frac{m_2^3 \sin^3 i}{(m_1 + m_2)^2}
             = \frac{P\,K^3\,(1 - e^2)^{3/2}}{2\pi G}

    Parameters
    ----------
    period
        Orbital period.
    rv_semiamp
        Radial-velocity semi-amplitude ``K``.
    eccentricity
        Orbital eccentricity (dimensionless).

    Returns
    -------
        The binary mass function, in solar masses.

    Examples
    --------
    >>> from unxt import Q, ustrip
    >>> from harv.kepler.masses import binary_mass_function
    >>> mf = binary_mass_function(Q(100.0, "day"), Q(10.0, "km/s"), 0.1)
    >>> round(float(ustrip("Msun", mf)), 4)
    0.0102
    """
    ecc = ustrip(AllowValue, "", eccentricity)
    mf = period * rv_semiamp**3 * (1.0 - ecc**2) ** 1.5 / (2.0 * jnp.pi * G)
    return Q(ustrip("Msun", mf), "Msun")


def astrometric_mass_function(
    a_physical: BatchQLength,
    period: BatchQTime,
) -> BatchQMass:
    r"""Astrometric mass function from a physical orbit size and period.

    .. math::

        f(m) = \frac{4\pi^2\,a^3}{G\,P^2}

    With ``a`` the semi-major axis of the *primary's* barycentric orbit -- the
    photocentre orbit when the companion is dark or faint -- this equals
    :math:`m_2^3 / (m_1 + m_2)^2`.  See docs/spec.md ("Mass functions").

    Parameters
    ----------
    a_physical
        Physical semi-major axis (a length), e.g. from
        :func:`semi_major_axis_physical`.
    period
        Orbital period.

    Returns
    -------
        The astrometric mass function, in solar masses.

    Examples
    --------
    >>> from unxt import Q, ustrip
    >>> from harv.kepler.masses import astrometric_mass_function
    >>> mf = astrometric_mass_function(Q(1.0, "AU"), Q(1.0, "yr"))
    >>> round(float(ustrip("Msun", mf)), 4)
    1.0
    """
    mf = 4.0 * jnp.pi**2 * a_physical**3 / (G * period**2)
    return Q(ustrip("Msun", mf), "Msun")


def companion_mass_from_mass_function(
    mass_function: BatchQMass,
    m1: QMass,
    sini: BatchFloatLike = 1.0,
) -> BatchQMass:
    r"""Solve the mass function for the companion mass :math:`m_2`.

    Inverts :math:`m_2^3 \sin^3 i / (m_1 + m_2)^2 = f` for :math:`m_2`, given
    the mass function ``f``, primary mass ``m1``, and ``sin i``.  ``sini = 1``
    (the default) yields the *minimum* companion mass.

    The cubic has a single positive root, so it is solved by bisection -- which
    is robust and ``jax.jit`` / ``jax.vmap`` friendly.  The root is bracketed
    by ``[0, max(4 f_eff, (4 f_eff m_1^2)^{1/3})]`` with
    :math:`f_\mathrm{eff} = f / \sin^3 i`.

    Parameters
    ----------
    mass_function
        Binary or astrometric mass function (a mass).
    m1
        Primary mass.
    sini
        Sine of the orbital inclination.  Default 1 (edge-on -> minimum mass).

    Returns
    -------
        The companion mass :math:`m_2`, in solar masses.

    Examples
    --------
    >>> from unxt import Q, ustrip
    >>> from harv.kepler.masses import companion_mass_from_mass_function
    >>> m2 = companion_mass_from_mass_function(Q(0.25, "Msun"), Q(1.0, "Msun"))
    >>> round(float(ustrip("Msun", m2)), 3)
    1.0
    """
    f = ustrip("Msun", mass_function)
    m1v = ustrip("Msun", m1)
    s = ustrip(AllowValue, "", sini)
    mf_eff = f / s**3

    hi0 = jnp.maximum(4.0 * mf_eff, jnp.cbrt(4.0 * mf_eff * m1v**2))
    lo0 = jnp.zeros_like(hi0)

    def _step(
        _: jax.Array, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        # g is negative below the root and positive above it.
        g = mid**3 - mf_eff * (m1v + mid) ** 2
        return jnp.where(g < 0.0, mid, lo), jnp.where(g < 0.0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 80, _step, (lo0, hi0))
    return Q(0.5 * (lo + hi), "Msun")


def semi_major_axis_physical(
    a_angular: BatchQAngle,
    parallax: BatchQAngle,
) -> BatchQLength:
    r"""Physical semi-major axis (AU) from an angular size and parallax.

    For an orbit at heliocentric distance :math:`d`, the angular semi-major
    axis is :math:`\alpha = a / d` and the parallax is
    :math:`\varpi = 1\,\mathrm{AU} / d`, so :math:`a = (\alpha / \varpi)\;
    \mathrm{AU}`.

    Parameters
    ----------
    a_angular
        Angular semi-major axis (an angle, e.g. ``mas``).
    parallax
        Parallax (an angle, e.g. ``mas``).

    Returns
    -------
        The physical semi-major axis, in AU.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.kepler.masses import semi_major_axis_physical
    >>> a = semi_major_axis_physical(Q(10.0, "mas"), Q(5.0, "mas"))
    >>> float(a.value)
    2.0
    """
    return Q(ustrip("", a_angular / parallax), "AU")
