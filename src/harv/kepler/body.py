"""Keplerian orbit implementation with units support and JAX compatibility."""

from dataclasses import KW_ONLY
from typing import Any, cast

import astropy.units as apyu
import equinox as eqx
import jax
import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import Quantity, ustrip

from harv.custom_types import (
    ScalarFloat,
    ScalarQLength,
    ScalarQMass,
    ScalarQTime,
    Vec3QLength,
    Vec3QSpeed,
    float_converter,
)
from harv.kepler.constants import G
from harv.kepler.orientation import KeplerianOrientation


class KeplerianBody(eqx.Module):
    """Orbital parameters of a Keplerian body (companion).

    This class represents the orbital parameters of a companion or second body (relative
    to some barycenter). So, all parameters represent the orbital elements of a specific
    body relative to the barycenter.

    The primary parameterization uses:
    - Period (P)
    - Eccentricity (e)
    - Semi-major axis (a) or ``semi_major_axis``
    - Time of pericenter or ``t_peri``

    Alternative constructors support different parameterizations.

    Parameters
    ----------
    period
        Orbital period.
    eccentricity
        Orbital eccentricity.
    semi_major_axis
        The semi-major axis of the body relative to its system barycenter.
    t_peri
        Time of pericenter passage.
    orientation
        Optional: Orientation of the orbit.
    ecc_zero_tol
        Optional: Tolerance for treating eccentricity as zero in circular orbit
        shortcuts. This is used to avoid numerical issues in the Kepler solver when
        eccentricity is very small, and can be set to a small multiple of machine
        epsilon for the data type used.

    """

    # Note: the annotations below are the _input_ types, which pass through the
    # converters and get stored internally with stricter type rules.
    period: ScalarQTime
    eccentricity: ScalarFloat = eqx.field(converter=float_converter)
    semi_major_axis: ScalarQLength
    t_peri: ScalarQTime
    orientation: KeplerianOrientation = KeplerianOrientation()
    _: KW_ONLY
    ecc_zero_tol: ScalarFloat = jnp.finfo(float).eps * 10.0  # type: ignore[no-untyped-call]

    def __check_init__(self) -> None:
        # Trace-friendly eccentricity bounds check (works inside jit/vmap)
        eqx.error_if(
            self.eccentricity,
            (self.eccentricity < 0.0) | (self.eccentricity >= 1.0),
            "Eccentricity must be in the range [0, 1) for bound orbits",
        )

        # Check that either all dimensional inputs are specified with units, or are all
        # dimensionless - no mixing of dimensionless and dimensional
        checks = [
            x.unit.decompose().is_equivalent(apyu.one) if hasattr(x, "unit") else True
            for x in [self.period, self.semi_major_axis, self.t_peri]
        ]
        if any(checks) and not all(checks):
            raise ValueError(
                "Either all or none of period, semi_major_axis, and t_peri must have "
                "units"
            )

    # ========================================================================
    # Alternative constructors
    #

    @classmethod
    def from_masses(
        cls,
        period: ScalarQTime,
        eccentricity: ScalarFloat,
        m_companion: ScalarQMass,
        m_primary: ScalarQMass,
        t_peri: ScalarQTime,
        **kwargs: Any,
    ) -> "KeplerianBody":
        r"""Construct companion's barycentric orbit from masses and period.

        Computes the companion's barycentric semi-major axis from Kepler's 3rd law:
        1. Compute relative orbit: a_rel = (G (m_1 + m_2) P^2 / 4 \pi^2)^(1/3)
        2. Convert to barycentric: a_body = a_rel * m_1 / (m_1 + m_2)

        Parameters
        ----------
        period
            Orbital period.
        eccentricity
            Orbital eccentricity.
        m_companion
            Companion mass (this body).
        m_primary
            Primary (central body) mass.
        t_peri
            Time of pericenter passage.
        orientation
            Optional: Orientation of the orbit.
        kwargs
            Additional keyword arguments to pass to the main constructor.

        Returns
        -------
        orbit: KeplerianBody
            The companion's orbit about the barycenter
        """
        period = Quantity["time"].from_(period)
        m_tot = m_primary + m_companion
        a_rel = jnp.cbrt((G * m_tot * period**2) / (4 * jnp.pi**2))
        a_body = a_rel * (m_primary / m_tot)

        return cls(
            period=period,
            eccentricity=eccentricity,
            semi_major_axis=cast("ScalarQLength", a_body),
            t_peri=t_peri,
            **kwargs,
        )

    # ========================================================================
    # Other methods
    #
    def get_mass(self, m_primary: ScalarQMass) -> ScalarQMass:
        """Compute companion mass given primary mass and barycentric semi-major axis."""
        num = G * m_primary**3 * self.period**2
        den = 4 * jnp.pi**2 * self.semi_major_axis**3
        m_tot = jnp.sqrt(num / den)
        return cast("ScalarQMass", m_tot - m_primary)

    def get_position(
        self, time: ScalarQTime, orientation: KeplerianOrientation | None = None
    ) -> Vec3QLength:
        """Get 3D position of the body in its orbit at given time(s).

        By definition and convention of this class, this is the position of the body
        relative to the system barycenter, accounting for the orbit orientation.
        """
        # Mean anomaly
        M = ustrip("", 2 * jnp.pi * (time - self.t_peri) / self.period)

        # Eccentric anomaly using jaxoplanet kepler solver
        sin_cos_f = jax.lax.cond(
            jnp.isclose(self.eccentricity, 0.0, atol=self.ecc_zero_tol),
            lambda: (jnp.sin(M), jnp.cos(M)),
            lambda: kepler(M, self.eccentricity),
        )

        # Distance from focus
        r = (
            self.semi_major_axis
            * (1 - self.eccentricity**2)
            / (1 + self.eccentricity * sin_cos_f[1])
        )

        # Position in orbital plane
        x_orb = r * sin_cos_f[1]
        y_orb = r * sin_cos_f[0]
        xyz_orb = jnp.stack([x_orb, y_orb, jnp.zeros_like(x_orb)], axis=0)

        orientation = self.orientation if orientation is None else orientation

        # Rotate to observer frame
        # TODO: identify if rotation is close to identity and skip, for performance
        return cast(
            "Vec3QLength",
            jnp.einsum("ij,j...->i...", orientation.rotation_matrix, xyz_orb),
        )

    def get_velocity(
        self, time: ScalarQTime, orientation: KeplerianOrientation | None = None
    ) -> Vec3QSpeed:
        """Get 3D velocity of the body relative to the system barycenter."""
        # Mean anomaly (dimensionless)
        M = ustrip("", 2 * jnp.pi * (time - self.t_peri) / self.period)

        # True anomaly (sin f, cos f); circular shortcut consistent with get_position
        sin_f, cos_f = jax.lax.cond(
            jnp.isclose(self.eccentricity, 0.0, atol=self.ecc_zero_tol),
            lambda: (jnp.sin(M), jnp.cos(M)),
            lambda: kepler(M, self.eccentricity),
        )

        a = self.semi_major_axis
        e = self.eccentricity
        n = (2 * jnp.pi) / self.period

        # Radius and kinematic rates
        r = a * (1 - e**2) / (1 + e * cos_f)

        def _vel_circular():  # type: ignore[no-untyped-def] # noqa: ANN202
            vx = (-n * a) * sin_f
            vy = (n * a) * cos_f
            return vx, vy

        def _vel_eccentric():  # type: ignore[no-untyped-def] # noqa: ANN202
            rdot = n * a * e * sin_f / jnp.sqrt(1 - e**2)
            fdot = n * (1 + e * cos_f) ** 2 / (1 - e**2) ** 1.5
            vx = rdot * cos_f - r * fdot * sin_f
            vy = rdot * sin_f + r * fdot * cos_f
            return vx, vy

        vx_orb, vy_orb = jax.lax.cond(
            jnp.isclose(e, 0.0, atol=self.ecc_zero_tol), _vel_circular, _vel_eccentric
        )
        vz_orb = jnp.zeros_like(vx_orb)
        vel_orb = jnp.stack([vx_orb, vy_orb, vz_orb], axis=0)

        orientation = self.orientation if orientation is None else orientation
        return cast(
            "Vec3QSpeed",
            jnp.einsum("ij,j...->i...", orientation.rotation_matrix, vel_orb),
        )
