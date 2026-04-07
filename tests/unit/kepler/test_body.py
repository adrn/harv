"""Unit tests for :mod:`harv.kepler.body`."""

import jax
import pytest
import quaxed.numpy as jnp
from unxt import Quantity, ustrip

from harv.kepler.body import KeplerianBody
from harv.kepler.orientation import KeplerianOrientation

# =============================================================================
# Helpers
# =============================================================================


def _make_circular_body(
    period: float = 1.0,
    a: float = 1.0,
    t_peri: float = 0.0,
) -> KeplerianBody:
    """Create a circular orbit KeplerianBody with units."""
    return KeplerianBody(
        period=Quantity(period, "yr"),
        eccentricity=0.0,
        semi_major_axis=Quantity(a, "AU"),
        t_peri=Quantity(t_peri, "yr"),
    )


def _make_eccentric_body(
    period: float = 1.0,
    eccentricity: float = 0.5,
    a: float = 1.0,
    t_peri: float = 0.0,
) -> KeplerianBody:
    """Create an eccentric orbit KeplerianBody with units."""
    return KeplerianBody(
        period=Quantity(period, "yr"),
        eccentricity=eccentricity,
        semi_major_axis=Quantity(a, "AU"),
        t_peri=Quantity(t_peri, "yr"),
    )


# =============================================================================
# Construction & validation
# =============================================================================


class TestConstruction:
    def test_basic_construction(self) -> None:
        body = _make_circular_body()
        assert jnp.allclose(ustrip("yr", body.period), 1.0)
        assert jnp.allclose(body.eccentricity, 0.0)
        assert jnp.allclose(ustrip("AU", body.semi_major_axis), 1.0)

    def test_eccentricity_negative_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Eccentricity"):
            KeplerianBody(
                period=Quantity(1.0, "yr"),
                eccentricity=-0.1,
                semi_major_axis=Quantity(1.0, "AU"),
                t_peri=Quantity(0.0, "yr"),
            )

    def test_eccentricity_one_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Eccentricity"):
            KeplerianBody(
                period=Quantity(1.0, "yr"),
                eccentricity=1.0,
                semi_major_axis=Quantity(1.0, "AU"),
                t_peri=Quantity(0.0, "yr"),
            )

    def test_eccentricity_converter_accepts_quantity(self) -> None:
        body = KeplerianBody(
            period=Quantity(1.0, "yr"),
            eccentricity=Quantity(0.3, ""),
            semi_major_axis=Quantity(1.0, "AU"),
            t_peri=Quantity(0.0, "yr"),
        )
        assert jnp.allclose(body.eccentricity, 0.3)

    def test_eccentricity_converter_accepts_jnp_scalar(self) -> None:
        body = KeplerianBody(
            period=Quantity(1.0, "yr"),
            eccentricity=jnp.float32(0.2),
            semi_major_axis=Quantity(1.0, "AU"),
            t_peri=Quantity(0.0, "yr"),
        )
        assert jnp.allclose(body.eccentricity, 0.2, atol=1e-6)

    def test_mixed_units_rejected(self) -> None:
        """Mixing units and dimensionless for period/a/t_peri raises."""
        with pytest.raises((ValueError, TypeError, Exception)):
            KeplerianBody(
                period=Quantity(1.0, "yr"),
                eccentricity=0.0,
                semi_major_axis=Quantity(1.0, ""),  # dimensionless
                t_peri=Quantity(0.0, "yr"),
            )


# =============================================================================
# from_masses & get_mass round-trip
# =============================================================================


class TestFromMasses:
    def test_kepler_third_law_round_trip(self) -> None:
        """from_masses → get_mass recovers the companion mass."""
        m_comp = Quantity(1.0, "Mjup")
        m_prim = Quantity(1.0, "Msun")
        body = KeplerianBody.from_masses(
            period=Quantity(1.0, "yr"),
            eccentricity=0.1,
            m_total=m_prim + m_comp,
            m_body=m_comp,
            t_peri=Quantity(0.0, "yr"),
        )
        recovered = body.get_mass(m_prim)
        assert jnp.allclose(
            ustrip("Mjup", recovered), ustrip("Mjup", m_comp), rtol=5e-4
        )

    def test_from_masses_with_orientation(self) -> None:
        """from_masses accepts orientation kwarg."""
        o = KeplerianOrientation.from_angles(
            arg_peri=Quantity(0.5, "rad"),
            lon_asc_node=Quantity(1.0, "rad"),
            inclination=Quantity(0.3, "rad"),
        )
        body = KeplerianBody.from_masses(
            period=Quantity(1.0, "yr"),
            eccentricity=0.0,
            m_total=Quantity(1.0, "Msun") + Quantity(1.0, "Mjup"),
            m_body=Quantity(1.0, "Mjup"),
            t_peri=Quantity(0.0, "yr"),
            orientation=o,
        )
        assert jnp.allclose(
            body.orientation.rotation_matrix, o.rotation_matrix, atol=1e-12
        )


# =============================================================================
# Physics: circular orbit
# =============================================================================


class TestCircularOrbit:
    def test_constant_radius(self) -> None:
        """For e=0, |r| = a at all times."""
        body = _make_circular_body(a=2.0)
        for t_val in [0.0, 0.25, 0.5, 0.75]:
            time = Quantity(t_val, "yr")
            r = body.get_position(time)
            r_mag = jnp.sqrt(jnp.sum(ustrip("AU", r) ** 2))
            assert jnp.allclose(r_mag, 2.0, rtol=1e-6)

    def test_constant_speed(self) -> None:
        """For e=0, |v| = 2πa/P at all times."""
        a, P = 2.0, 1.0
        body = _make_circular_body(period=P, a=a)
        expected_speed = 2 * jnp.pi * a / P  # AU/yr

        for t_val in [0.0, 0.25, 0.5, 0.75]:
            time = Quantity(t_val, "yr")
            v = body.get_velocity(time)
            v_mag = jnp.sqrt(jnp.sum(ustrip("AU/yr", v) ** 2))
            assert jnp.allclose(v_mag, expected_speed, rtol=1e-5)

    def test_periodic(self) -> None:
        """Position at t and t+P are equal."""
        body = _make_circular_body()
        t0 = Quantity(0.3, "yr")
        t1 = Quantity(1.3, "yr")  # t0 + P
        r0 = body.get_position(t0)
        r1 = body.get_position(t1)
        assert jnp.allclose(ustrip("AU", r0), ustrip("AU", r1), atol=1e-6)


# =============================================================================
# Physics: eccentric orbit
# =============================================================================


class TestEccentricOrbit:
    def test_pericenter_distance(self) -> None:
        """At t_peri, |r| = a(1-e)."""
        e, a = 0.5, 3.0
        body = _make_eccentric_body(eccentricity=e, a=a)
        r = body.get_position(Quantity(0.0, "yr"))  # t = t_peri
        r_mag = jnp.sqrt(jnp.sum(ustrip("AU", r) ** 2))
        assert jnp.allclose(r_mag, a * (1 - e), rtol=1e-5)

    def test_velocity_pericenter_gt_apocenter(self) -> None:
        """Velocity at pericenter is greater than at half-period (near apocenter)."""
        body = _make_eccentric_body(eccentricity=0.5)
        v_peri = body.get_velocity(Quantity(0.0, "yr"))
        v_apo = body.get_velocity(Quantity(0.5, "yr"))
        speed_peri = jnp.sqrt(jnp.sum(ustrip("AU/yr", v_peri) ** 2))
        speed_apo = jnp.sqrt(jnp.sum(ustrip("AU/yr", v_apo) ** 2))
        assert speed_peri > speed_apo

    def test_eccentric_orbit_periodic(self) -> None:
        """Position at t and t+P are equal for eccentric orbit."""
        body = _make_eccentric_body(eccentricity=0.3)
        t0 = Quantity(0.2, "yr")
        t1 = Quantity(1.2, "yr")
        r0 = body.get_position(t0)
        r1 = body.get_position(t1)
        assert jnp.allclose(ustrip("AU", r0), ustrip("AU", r1), atol=1e-5)


# =============================================================================
# JAX compatibility
# =============================================================================


class TestJAXCompat:
    def test_pytree_flatten_unflatten(self) -> None:
        body = _make_circular_body()
        leaves, treedef = jax.tree.flatten(body)
        body2 = treedef.unflatten(leaves)
        r = body.get_position(Quantity(0.1, "yr"))
        r2 = body2.get_position(Quantity(0.1, "yr"))
        assert jnp.allclose(ustrip("AU", r), ustrip("AU", r2))

    def test_jit_get_position(self) -> None:
        body = _make_circular_body()
        t = Quantity(0.25, "yr")

        @jax.jit
        def f(body, t):
            return body.get_position(t)

        r = f(body, t)
        assert r.shape == (3,)

    def test_jit_get_velocity(self) -> None:
        body = _make_circular_body()
        t = Quantity(0.25, "yr")

        @jax.jit
        def f(body, t):
            return body.get_velocity(t)

        v = f(body, t)
        assert v.shape == (3,)

    def test_vmap_over_time(self) -> None:
        """Vmap over scalar time inputs matches direct calls."""
        body = _make_circular_body()
        times = Quantity(jnp.array([0.0, 0.25, 0.5, 0.75]), "yr")

        @jax.vmap
        def pos_vmap(t):
            return body.get_position(t)

        result = pos_vmap(times)
        assert result.shape == (4, 3)

        # Spot check one time
        r_direct = body.get_position(times[1])
        assert jnp.allclose(ustrip("AU", result[1]), ustrip("AU", r_direct), atol=1e-8)

    def test_vmap_over_period(self) -> None:
        """Vmap over batched KeplerianBody (different period per element).

        For a circular orbit, orbital speed = 2πa/P, so results must differ
        across period values. Batch by stacking individually-constructed bodies.
        """
        period_values = [0.5, 1.0, 2.0]
        bodies = [_make_circular_body(period=P, a=1.0) for P in period_values]
        bodies_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *bodies)

        t = Quantity(0.0, "yr")  # t = t_peri: position is at (a, 0, 0)

        result_v = jax.vmap(lambda b: b.get_velocity(t))(bodies_batched)
        assert result_v.shape == (3, 3)

        # Strip to plain array so speed comparisons are dimensionless
        result_v_arr = ustrip("AU/yr", result_v)  # (3, 3)
        speeds = jnp.sqrt(jnp.sum(result_v_arr**2, axis=-1))
        for i, (P, body) in enumerate(zip(period_values, bodies, strict=False)):
            expected_speed = 2 * jnp.pi * 1.0 / P
            assert jnp.allclose(speeds[i], expected_speed, rtol=1e-5)
            v_direct_arr = ustrip("AU/yr", body.get_velocity(t))
            assert jnp.allclose(result_v_arr[i], v_direct_arr, atol=1e-7)

    def test_vmap_over_eccentricity(self) -> None:
        """Vmap over batched KeplerianBody (different eccentricity per element).

        At t=t_peri, |r| = a(1-e), so pericenter distance encodes eccentricity.
        """
        ecc_values = [0.0, 0.2, 0.5, 0.7]
        a = 2.0
        bodies = [_make_eccentric_body(eccentricity=e, a=a) for e in ecc_values]
        bodies_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *bodies)

        t = Quantity(0.0, "yr")  # t = t_peri

        result_r = jax.vmap(lambda b: b.get_position(t))(bodies_batched)
        assert result_r.shape == (4, 3)

        result_r_arr = ustrip("AU", result_r)  # (4, 3) plain array
        r_mags = jnp.sqrt(jnp.sum(result_r_arr**2, axis=-1))
        for i, e in enumerate(ecc_values):
            assert jnp.allclose(r_mags[i], a * (1 - e), rtol=1e-5)
