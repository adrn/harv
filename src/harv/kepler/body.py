"""Keplerian orbit implementation with units support and JAX compatibility."""

from dataclasses import KW_ONLY
from typing import Any, cast

import astropy.units as apyu
import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import Quantity, ustrip

from harv.custom_types import (
    BatchQTime,
    BatchVec3QLength,
    BatchVec3QSpeed,
    ScalarFloat,
    ScalarQLength,
    ScalarQMass,
    ScalarQTime,
    Vec3QLength,
    Vec3QSpeed,
    float_converter,
)
from harv.kepler.constants import G
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.kepler.orientation import KeplerianOrientation


class KeplerianBody(eqx.Module):
    """Orbital parameters of a Keplerian body (companion, i.e. body 2).

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
        m_total: ScalarQMass,
        m_body: ScalarQMass,
        t_peri: ScalarQTime,
        **kwargs: Any,
    ) -> "KeplerianBody":
        r"""Construct body's barycentric orbit from masses and period.

        Computes this body's barycentric semi-major axis from Kepler's 3rd law:
        1. Compute relative orbit: a_rel = (G m_total P^2 / 4 \pi^2)^(1/3)
        2. Convert to barycentric: a_body = a_rel * (1 - m_body / m_total)

        Parameters
        ----------
        period
            Orbital period.
        eccentricity
            Orbital eccentricity.
        m_total
            Total system mass.
        m_body
            Mass of this body.
        t_peri
            Time of pericenter passage.
        orientation
            Optional: Orientation of the orbit.
        kwargs
            Additional keyword arguments to pass to the main constructor.

        Returns
        -------
        orbit: KeplerianBody
            The body's orbit about the system barycenter.
        """
        period = Quantity["time"].from_(period)
        a_rel = jnp.cbrt((G * m_total * period**2) / (4 * jnp.pi**2))
        a_body = a_rel * (1 - m_body / m_total)

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

    def get_mass(self, m_total: ScalarQMass) -> ScalarQMass:
        r"""Recover this body's mass from the total system mass.

        This inverts `from_masses`: given the total system mass and the stored
        barycentric semi-major axis, it recovers the body mass via

        .. math::

            a_\mathrm{rel} = \left(\frac{G\, m_\mathrm{total}\, P^2}{4\pi^2}\right)^{1/3}

            m_\mathrm{body} = m_\mathrm{total}\left(1 - \frac{a_\mathrm{body}}{a_\mathrm{rel}}\right)

        Parameters
        ----------
        m_total
            Total system mass (sum of all bodies).

        Returns
        -------
        m_body : ScalarQMass
            The mass of this body.
        """
        a_rel = jnp.cbrt((G * m_total * self.period**2) / (4 * jnp.pi**2))
        return cast("ScalarQMass", m_total * (1 - self.semi_major_axis / a_rel))

    def get_position(
        self, time: BatchQTime, orientation: KeplerianOrientation | None = None
    ) -> BatchVec3QLength:
        """Get 3D position of the body in its orbit at given time(s).

        By definition and convention of this class, this is the position of the body
        relative to the system barycenter, accounting for the orbit orientation.
        """
        # Mean anomaly
        M = mean_anomaly(time - self.t_peri, self.period)
        M_raw = ustrip("rad", M)

        # True anomaly; circular shortcut avoids Kepler solver for e ≈ 0
        sin_cos_f = jax.lax.cond(
            jnp.isclose(self.eccentricity, 0.0, atol=self.ecc_zero_tol),
            lambda: (jnp.sin(M_raw), jnp.cos(M_raw)),
            lambda: true_anomaly_from_mean(M, self.eccentricity),
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
        self, time: BatchQTime, orientation: KeplerianOrientation | None = None
    ) -> BatchVec3QSpeed:
        """Get 3D velocity of the body relative to the system barycenter."""
        # Mean anomaly
        M = mean_anomaly(time - self.t_peri, self.period)
        M_raw = ustrip("rad", M)

        # True anomaly (sin f, cos f); circular shortcut consistent with get_position
        sin_f, cos_f = jax.lax.cond(
            jnp.isclose(self.eccentricity, 0.0, atol=self.ecc_zero_tol),
            lambda: (jnp.sin(M_raw), jnp.cos(M_raw)),
            lambda: true_anomaly_from_mean(M, self.eccentricity),
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
