"""Tests for harv.kepler.orbits building blocks."""

import jax
import jax.numpy as jnp
from unxt import Q, ustrip

from harv.kepler import KeplerianOrientation
from harv.kepler.orbits import (
    mean_anomaly,
    rv_shape,
    thiele_innes_ABFG,
    true_anomaly_from_mean,
)


class TestMeanAnomaly:
    def test_known_value(self) -> None:
        # Full orbit: dt = period -> M = 2pi
        M = mean_anomaly(Q(10.0, "day"), Q(10.0, "day"))
        assert jnp.allclose(ustrip("rad", M), 2 * jnp.pi)

    def test_half_orbit(self) -> None:
        M = mean_anomaly(Q(5.0, "day"), Q(10.0, "day"))
        assert jnp.allclose(ustrip("rad", M), jnp.pi)

    def test_zero_dt(self) -> None:
        M = mean_anomaly(Q(0.0, "day"), Q(10.0, "day"))
        assert jnp.allclose(ustrip("rad", M), 0.0)

    def test_jit(self) -> None:
        M = jax.jit(mean_anomaly)(Q(3.0, "day"), Q(10.0, "day"))
        expected = 2 * jnp.pi * 3.0 / 10.0
        assert jnp.allclose(ustrip("rad", M), expected)

    def test_vmap(self) -> None:
        dts = Q(jnp.array([0.0, 5.0, 10.0]), "day")
        Ms = jax.vmap(mean_anomaly, in_axes=(0, None))(dts, Q(10.0, "day"))
        expected = 2 * jnp.pi * jnp.array([0.0, 5.0, 10.0]) / 10.0
        assert jnp.allclose(ustrip("rad", Ms), expected)

    def test_mixed_units(self) -> None:
        # dt in hours, period in days -- should still work
        M = mean_anomaly(Q(24.0, "hr"), Q(1.0, "day"))
        assert jnp.allclose(ustrip("rad", M), 2 * jnp.pi)


class TestTrueAnomalyFromMean:
    def test_circular_orbit(self) -> None:
        # For e=0, true anomaly = mean anomaly
        M = Q(1.0, "rad")
        sin_f, cos_f = true_anomaly_from_mean(M, 0.0)
        assert jnp.allclose(sin_f, jnp.sin(1.0), atol=1e-6)
        assert jnp.allclose(cos_f, jnp.cos(1.0), atol=1e-6)

    def test_pericenter(self) -> None:
        # At M=0, f=0 for any eccentricity
        sin_f, cos_f = true_anomaly_from_mean(Q(0.0, "rad"), 0.3)
        assert jnp.allclose(sin_f, 0.0, atol=1e-10)
        assert jnp.allclose(cos_f, 1.0, atol=1e-10)

    def test_jit(self) -> None:
        sin_f, cos_f = jax.jit(true_anomaly_from_mean)(Q(1.0, "rad"), 0.3)
        assert jnp.isfinite(sin_f)
        assert jnp.isfinite(cos_f)

    def test_vmap(self) -> None:
        Ms = Q(jnp.linspace(0, 2 * jnp.pi, 10), "rad")
        sin_fs, cos_fs = jax.vmap(true_anomaly_from_mean, in_axes=(0, None))(Ms, 0.3)
        assert sin_fs.shape == (10,)
        assert cos_fs.shape == (10,)
        # sin^2f + cos^2f = 1
        assert jnp.allclose(sin_fs**2 + cos_fs**2, 1.0, atol=1e-6)


class TestRvShape:
    def test_circular_at_pericenter(self) -> None:
        # At f=0, cos(omega+0) + e*cos(omega) = cos(omega)(1+e)
        omega = 0.5
        e = 0.3
        result = rv_shape(0.0, 1.0, e, omega)
        expected = jnp.cos(omega) * (1 + e)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_manual_computation(self) -> None:
        sin_f = jnp.array(0.6)
        cos_f = jnp.array(0.8)
        e = 0.2
        omega = 1.2
        result = rv_shape(sin_f, cos_f, e, omega)
        expected = jnp.cos(omega) * cos_f - jnp.sin(omega) * sin_f + e * jnp.cos(omega)
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_jit(self) -> None:
        result = jax.jit(rv_shape)(0.6, 0.8, 0.2, 1.2)
        assert jnp.isfinite(result)

    def test_vmap(self) -> None:
        sin_fs = jnp.array([0.0, 0.5, 1.0])
        cos_fs = jnp.array([1.0, jnp.sqrt(0.75), 0.0])
        results = jax.vmap(rv_shape, in_axes=(0, 0, None, None))(
            sin_fs, cos_fs, 0.3, 0.5
        )
        assert results.shape == (3,)


class TestThieleInnesABFG:
    def test_identity_orientation(self) -> None:
        # omega=0, Omega=0, i=0 -> A=1, B=0, F=0, G=1 (face-on, aligned)
        A, B, F, G = thiele_innes_ABFG(1.0, 0.0, 1.0, 0.0, 1.0)
        assert jnp.allclose(A, 1.0, atol=1e-10)
        assert jnp.allclose(B, 0.0, atol=1e-10)
        assert jnp.allclose(F, 0.0, atol=1e-10)
        assert jnp.allclose(G, 1.0, atol=1e-10)

    def test_matches_keplerian_orientation(self) -> None:
        """Verify building block KeplerianOrientation.thiele_innes_constants()."""
        omega = Q(0.7, "rad")
        Omega = Q(1.3, "rad")
        incl = Q(0.5, "rad")

        orient = KeplerianOrientation.from_angles(
            arg_peri=omega, lon_asc_node=Omega, inclination=incl
        )
        A_ref, B_ref, F_ref, G_ref = orient.thiele_innes_constants()

        A, B, F, G = thiele_innes_ABFG(
            orient.cos_arg_peri,
            orient.sin_arg_peri,
            orient.cos_lon_asc_node,
            orient.sin_lon_asc_node,
            orient.cos_i,
        )
        assert jnp.allclose(A, A_ref, atol=1e-10)
        assert jnp.allclose(B, B_ref, atol=1e-10)
        assert jnp.allclose(F, F_ref, atol=1e-10)
        assert jnp.allclose(G, G_ref, atol=1e-10)

    def test_with_semi_major_axis(self) -> None:
        """Unit T-I x a should match KeplerianOrientation with semi_major_axis."""
        orient = KeplerianOrientation.from_angles(
            arg_peri=Q(0.7, "rad"),
            lon_asc_node=Q(1.3, "rad"),
            inclination=Q(0.5, "rad"),
        )
        a = Q(2.5, "mas")
        A_ref, B_ref, F_ref, G_ref = orient.thiele_innes_constants(semi_major_axis=a)

        A, B, F, G = thiele_innes_ABFG(
            orient.cos_arg_peri,
            orient.sin_arg_peri,
            orient.cos_lon_asc_node,
            orient.sin_lon_asc_node,
            orient.cos_i,
        )
        a_val = ustrip("mas", a)
        assert jnp.allclose(a_val * A, ustrip("mas", A_ref), atol=1e-10)
        assert jnp.allclose(a_val * B, ustrip("mas", B_ref), atol=1e-10)
        assert jnp.allclose(a_val * F, ustrip("mas", F_ref), atol=1e-10)
        assert jnp.allclose(a_val * G, ustrip("mas", G_ref), atol=1e-10)

    def test_jit(self) -> None:
        A, _B, _F, _G = jax.jit(thiele_innes_ABFG)(1.0, 0.0, 1.0, 0.0, 1.0)
        assert jnp.isfinite(A)

    def test_vmap(self) -> None:
        cos_ws = jnp.array([1.0, 0.0, -1.0])
        sin_ws = jnp.array([0.0, 1.0, 0.0])
        vmap_ti = jax.vmap(thiele_innes_ABFG, in_axes=(0, 0, None, None, None))
        As, _Bs, _Fs, _Gs = vmap_ti(cos_ws, sin_ws, 1.0, 0.0, 1.0)
        assert As.shape == (3,)
