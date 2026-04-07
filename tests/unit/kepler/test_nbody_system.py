"""Unit tests for :mod:`harv.kepler.nbody_system`."""

import jax
import pytest
import quaxed.numpy as jnp
from unxt import Quantity, ustrip

from harv.kepler.body import KeplerianBody
from harv.kepler.nbody_system import TwoBodySystem

# =============================================================================
# Helpers
# =============================================================================


def _make_system(
    m_primary: float = 1.0,
    m_companion: float = 1e-3,
    period: float = 1.0,
    eccentricity: float = 0.0,
) -> TwoBodySystem:
    """Create a TwoBodySystem from masses."""
    companion = KeplerianBody.from_masses(
        period=Quantity(period, "yr"),
        eccentricity=eccentricity,
        m_total=Quantity(m_primary + m_companion, "Msun"),
        m_body=Quantity(m_companion, "Msun"),
        t_peri=Quantity(0.0, "yr"),
    )
    return TwoBodySystem(
        m_primary=Quantity(m_primary, "Msun"),
        companion=companion,
    )


# =============================================================================
# Properties
# =============================================================================


class TestProperties:
    def test_n_bodies(self) -> None:
        sys = _make_system()
        assert sys.n_bodies == 2

    def test_m_total(self) -> None:
        sys = _make_system(m_primary=1.0, m_companion=1e-3)
        m_tot = ustrip("Msun", sys.m_total)
        assert jnp.allclose(m_tot, 1.001, rtol=1e-6)

    def test_m_companion_matches_get_mass(self) -> None:
        """m_companion property matches KeplerianBody.get_mass."""
        sys = _make_system(m_companion=1e-3)
        m_from_prop = ustrip("Msun", sys.m_companion)
        m_from_method = ustrip("Msun", sys.companion.get_mass(sys.m_primary))
        assert jnp.allclose(m_from_prop, m_from_method, rtol=1e-8)


# =============================================================================
# Physics
# =============================================================================


class TestPhysics:
    def test_barycentric_momentum_conservation(self) -> None:
        """m1*v1 + m2*v2 ≈ 0 in the barycentric frame."""
        sys = _make_system(m_primary=1.0, m_companion=5e-3, eccentricity=0.3)
        t = Quantity(0.37, "yr")

        v0 = sys.velocity_barycentric(t, 0)
        v1 = sys.velocity_barycentric(t, 1)
        p_total = sys.m_primary * v0 + sys.m_companion * v1

        # Momentum should be zero (to numerical precision)
        assert jnp.allclose(
            ustrip("Msun AU / yr", p_total),
            jnp.zeros(3),
            atol=1e-10,
        )

    def test_relative_position_relation(self) -> None:
        """position_relative = pos(1) - pos(0)."""
        sys = _make_system(eccentricity=0.2)
        t = Quantity(0.25, "yr")

        r_rel = sys.position_relative(t)
        r0 = sys.position_barycentric(t, 0)
        r1 = sys.position_barycentric(t, 1)

        assert jnp.allclose(ustrip("AU", r_rel), ustrip("AU", r1 - r0), rtol=1e-8)

    def test_relative_velocity_relation(self) -> None:
        """velocity_relative = vel(1) - vel(0)."""
        sys = _make_system(eccentricity=0.2)
        t = Quantity(0.25, "yr")

        v_rel = sys.velocity_relative(t)
        v0 = sys.velocity_barycentric(t, 0)
        v1 = sys.velocity_barycentric(t, 1)

        assert jnp.allclose(ustrip("AU/yr", v_rel), ustrip("AU/yr", v1 - v0), rtol=1e-8)

    def test_body_idx_out_of_range(self) -> None:
        sys = _make_system()
        with pytest.raises(IndexError, match="body_idx"):
            sys.position_barycentric(Quantity(0.0, "yr"), 2)

    def test_body_idx_out_of_range_velocity(self) -> None:
        sys = _make_system()
        with pytest.raises(IndexError, match="body_idx"):
            sys.velocity_barycentric(Quantity(0.0, "yr"), 2)


# =============================================================================
# JAX compatibility
# =============================================================================


class TestJAXCompat:
    def test_pytree_flatten_unflatten(self) -> None:
        sys = _make_system()
        leaves, treedef = jax.tree.flatten(sys)
        sys2 = treedef.unflatten(leaves)
        t = Quantity(0.1, "yr")
        r1 = sys.position_barycentric(t, 1)
        r2 = sys2.position_barycentric(t, 1)
        assert jnp.allclose(ustrip("AU", r1), ustrip("AU", r2))

    def test_jit_position_barycentric(self) -> None:
        sys = _make_system()
        t = Quantity(0.25, "yr")

        @jax.jit
        def f(sys, t):
            return sys.position_barycentric(t, 1)

        r = f(sys, t)
        assert r.shape == (3,)

    def test_vmap_over_period_rv(self) -> None:
        """Vmap over TwoBodySystem with different orbital periods.

        Demonstrates the scalar-design pattern: define get_rv for a single
        system at a single time, then vmap over a batch of systems.
        """
        period_values = [0.5, 1.0, 2.0]
        t = Quantity(0.25, "yr")

        systems = [_make_system(period=P, eccentricity=0.1) for P in period_values]
        systems_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *systems)

        def get_rv(system):
            """Radial (z) component of primary velocity."""
            v = system.velocity_barycentric(t, 0)
            return v[2]

        result = jax.vmap(get_rv)(systems_batched)
        assert result.shape == (3,)

        # Compute reference values; use their unit to strip consistently
        rv_directs = [get_rv(system) for system in systems]
        ref_unit = rv_directs[0].unit  # same for all (same input unit types)
        for i, rv_ref in enumerate(rv_directs):
            assert jnp.allclose(
                ustrip(ref_unit, result[i]),
                ustrip(ref_unit, rv_ref),
                atol=1e-7,
            )

    def test_vmap_over_eccentricity_momentum(self) -> None:
        """Vmap over TwoBodySystem with different eccentricities.

        Momentum conservation must hold for each batched system independently.
        """
        ecc_values = [0.0, 0.2, 0.5]
        t = Quantity(0.3, "yr")

        systems = [_make_system(eccentricity=e) for e in ecc_values]
        systems_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *systems)

        def total_momentum(system):
            v0 = system.velocity_barycentric(t, 0)
            v1 = system.velocity_barycentric(t, 1)
            return system.m_primary * v0 + system.m_companion * v1

        result = jax.vmap(total_momentum)(systems_batched)
        assert result.shape == (3, 3)  # (batch, xyz)

        assert jnp.allclose(
            ustrip("Msun AU / yr", result),
            jnp.zeros((3, 3)),
            atol=1e-10,
        )
