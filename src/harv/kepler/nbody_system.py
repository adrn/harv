"""Keplerian orbit implementation with units support and JAX compatibility."""

from typing import cast

import equinox as eqx
import quaxed.numpy as jnp

from harv.custom_types import (
    ScalarQMass,
    ScalarQTime,
    Vec3QLength,
    Vec3QSpeed,
)
from harv.kepler.body import KeplerianBody
from harv.kepler.constants import G


class AbstractNBodySystem(eqx.Module):
    """Abstract base class for Keplerian N-body systems."""

    @property
    def n_bodies(self) -> int:
        """Total number of bodies (primary + companions)."""
        raise NotImplementedError

    @property
    def m_total(self) -> ScalarQMass:
        """Total system mass."""
        raise NotImplementedError


class TwoBodySystem(AbstractNBodySystem):
    """A system with a primary body and one companion.

    Bodies are indexed as:
    - 0: Primary body
    - 1: Companion
    """

    m_primary: ScalarQMass
    companion: KeplerianBody

    # ========================================================================
    # Properties
    #

    @property
    def n_bodies(self) -> int:
        """Total number of bodies (primary + companions)."""
        return 2

    @property
    def m_total(self) -> ScalarQMass:
        """Total system mass, derived from Kepler's 3rd law.

        The companion's semi-major axis is defined relative to the system barycenter, so
        we can use Kepler's 3rd law to derive the total mass from the companion's
        orbital parameters and the primary mass:
        a_body = a_rel * m_primary / m_total
        a_rel^3 = G * m_total * P^2 / 4 π^2
        a_body^3 = G * m_primary^3 * P^2 / (4 π^2 * m_total^2)
        (solve for m_total)
        """
        a_body = self.companion.semi_major_axis
        P = self.companion.period
        return cast(
            "ScalarQMass",
            jnp.sqrt(G * self.m_primary**3 * P**2 / (4 * jnp.pi**2 * a_body**3)),
        )

    @property
    def m_companion(self) -> ScalarQMass:
        """Companion mass."""
        return self.m_total - self.m_primary

    # ========================================================================
    # Methods
    #

    def position_barycentric(self, time: ScalarQTime, body_idx: int) -> Vec3QLength:
        """Get barycentric position of specified body at given time(s).

        Parameters
        ----------
        time
            Time(s) to evaluate
        body_idx
            Index of body (0=primary, 1=companion)

        Returns
        -------
        pos
            3D position vector(s) of specified body relative to barycenter
        """
        r2 = self.companion.get_position(time)

        if body_idx == 1:
            return r2  # companion about barycenter

        if body_idx == 0:
            return cast("Vec3QLength", -(self.m_companion / self.m_primary) * r2)

        raise IndexError("body_idx must be 0 (primary) or 1 (companion)")

    def position_relative(self, time: ScalarQTime) -> Vec3QLength:
        """Position of companion relative to primary.

        Parameters
        ----------
        time
            Time(s) to evaluate

        Returns
        -------
        pos
            3D position vector(s) of companion relative to primary
        """
        r2 = self.position_barycentric(time, 1)
        return cast("Vec3QLength", r2 * (1 + self.m_companion / self.m_primary))

    def velocity_barycentric(self, time: ScalarQTime, body_idx: int) -> Vec3QSpeed:
        """Get barycentric velocity of specified body at given time(s).

        Parameters
        ----------
        time
            Time(s) to evaluate
        body_idx
            Index of body (0=primary, 1=companion)

        Returns
        -------
        vel
            3D velocity vector(s) of specified body relative to barycenter
        """
        v2 = self.companion.get_velocity(time)  # companion barycentric velocity

        if body_idx == 1:
            return v2

        if body_idx == 0:
            return cast("Vec3QSpeed", -(self.m_companion / self.m_primary) * v2)

        raise IndexError("body_idx must be 0 (primary) or 1 (companion)")

    def velocity_relative(self, time: ScalarQTime) -> Vec3QSpeed:
        """Velocity of companion relative to primary.

        Parameters
        ----------
        time
            Time(s) to evaluate

        Returns
        -------
        vel
            3D velocity vector(s) of companion relative to primary
        """
        v2 = self.velocity_barycentric(time, 1)
        return cast("Vec3QSpeed", v2 * (1 + self.m_companion / self.m_primary))
