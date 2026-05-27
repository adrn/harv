"""Tests for harv.kepler.orbits building blocks."""

import jax
import jax.numpy as jnp
import pytest
from unxt import Q, ustrip

from harv.kepler import KeplerianOrientation
from harv.kepler.orbits import (
    campbell_from_thiele_innes,
    ecc_omega_from_ecosw_esinw,
    ecosw_esinw_from_ecc_omega,
    mean_anomaly,
    rv_shape,
    thiele_innes_ABFG,
    thiele_innes_from_campbell,
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
        result = rv_shape(jnp.array(0.0), jnp.array(1.0), e, omega)
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
        result = jax.jit(rv_shape)(jnp.array(0.6), jnp.array(0.8), 0.2, 1.2)
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


class TestCampbellFromThieleInnes:
    """Round-trip tests"""

    @pytest.mark.parametrize(
        ("arg_peri", "lon_asc_node", "cos_i", "a"),
        [
            (0.5, 1.0, 0.6, 3.0),
            (0.0, 0.0, 1.0, 1.0),  # face-on, aligned
            (1.2, 2.3, 0.3, 5.0),
            (3.1, 0.7, 0.9, 0.5),
            # Retrograde orbits (cos_i < 0). Astrometric fits can converge to
            # either prograde or retrograde solutions; the conversion must
            # preserve the sign so the resulting Campbell elements round-trip
            # back to the same TI constants. Without this, a fitted TI with
            # cos_i < 0 maps to a Campbell orbit that traces a *different*
            # sky-projected ellipse (180° flip).
            (1.2, 2.3, -0.3, 5.0),
            (0.5, 1.0, -0.6, 3.0),
            (3.1, 0.7, -0.9, 0.5),
        ],
    )
    def test_round_trip(
        self,
        arg_peri: float,
        lon_asc_node: float,
        cos_i: float,
        a: float,
    ) -> None:
        # Forward pass: Campbell → TI
        A, B, F, G = thiele_innes_ABFG(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )
        # Inverse pass: TI → Campbell
        result = campbell_from_thiele_innes(
            Q(a * A, "mas"), Q(a * B, "mas"), Q(a * F, "mas"), Q(a * G, "mas")
        )
        # Inversion is 2-fold degenerate: (ω,Ω) and (ω+π, Ω+π) give identical TI
        # constants. Check invariants: a0, cos_i, and the TI constants themselves.
        assert jnp.allclose(ustrip("mas", result["semi_major_axis"]), a, atol=1e-5)
        assert jnp.allclose(ustrip("", result["cos_i"]), cos_i, atol=1e-5)
        # Re-compute TI from the recovered Campbell elements; must recover original
        # A,B,F,G
        A_rt, B_rt, F_rt, G_rt = thiele_innes_ABFG(
            jnp.cos(ustrip("rad", result["arg_peri"])),
            jnp.sin(ustrip("rad", result["arg_peri"])),
            jnp.cos(ustrip("rad", result["lon_asc_node"])),
            jnp.sin(ustrip("rad", result["lon_asc_node"])),
            ustrip("", result["cos_i"]),
        )
        assert jnp.allclose(A_rt, A, atol=1e-5)
        assert jnp.allclose(B_rt, B, atol=1e-5)
        assert jnp.allclose(F_rt, F, atol=1e-5)
        assert jnp.allclose(G_rt, G, atol=1e-5)

    def test_round_trip_quantity(self) -> None:
        arg_peri, lon_asc_node, cos_i = 0.7, 1.3, 0.5
        a = Q(2.5, "mas")
        A, B, F, G = thiele_innes_ABFG(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )
        result = campbell_from_thiele_innes(a * A, a * B, a * F, a * G)
        assert jnp.allclose(
            ustrip("mas", result["semi_major_axis"]), ustrip("mas", a), atol=1e-5
        )
        assert jnp.allclose(ustrip("", result["cos_i"]), cos_i, atol=1e-5)

    def test_jit(self) -> None:
        arg_peri, lon_asc_node, cos_i, a = 0.5, 1.0, 0.6, 3.0
        A, B, F, G = thiele_innes_ABFG(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )
        result = jax.jit(campbell_from_thiele_innes)(
            Q(a * A, "mas"), Q(a * B, "mas"), Q(a * F, "mas"), Q(a * G, "mas")
        )
        assert jnp.isfinite(ustrip("mas", result["semi_major_axis"]))
        assert jnp.isfinite(ustrip("rad", result["arg_peri"]))
        assert jnp.isfinite(ustrip("", result["cos_i"]))


class TestThieleInnesFromCampbell:
    """Tests for thiele_innes_from_campbell (inverse of campbell_from_thiele_innes)."""

    def test_scales_linearly_with_semi_major_axis(self) -> None:
        # Physical TI constants are the unit constants scaled by a_0.
        args = (Q(0.7, "rad"), Q(1.3, "rad"), Q(0.5, ""))
        A1, B1, F1, G1 = thiele_innes_from_campbell(Q(1.0, "mas"), *args)
        A3, B3, F3, G3 = thiele_innes_from_campbell(Q(3.0, "mas"), *args)
        assert jnp.allclose(ustrip("mas", A3), 3.0 * ustrip("mas", A1), atol=1e-6)
        assert jnp.allclose(ustrip("mas", B3), 3.0 * ustrip("mas", B1), atol=1e-6)
        assert jnp.allclose(ustrip("mas", F3), 3.0 * ustrip("mas", F1), atol=1e-6)
        assert jnp.allclose(ustrip("mas", G3), 3.0 * ustrip("mas", G1), atol=1e-6)

    def test_round_trip_with_campbell(self) -> None:
        a0 = Q(3.0, "mas")
        arg_peri, lon_asc_node, cos_i = Q(0.5, "rad"), Q(1.0, "rad"), Q(0.6, "")
        A, B, F, G = thiele_innes_from_campbell(a0, arg_peri, lon_asc_node, cos_i)
        result = campbell_from_thiele_innes(A=A, B=B, F=F, G=G)
        assert jnp.allclose(ustrip("mas", result["semi_major_axis"]), 3.0, atol=1e-5)
        assert jnp.allclose(ustrip("", result["cos_i"]), 0.6, atol=1e-5)
        # Re-derive TI from recovered Campbell elements: must match the originals.
        A_rt, B_rt, F_rt, G_rt = thiele_innes_from_campbell(
            result["semi_major_axis"],
            result["arg_peri"],
            result["lon_asc_node"],
            result["cos_i"],
        )
        assert jnp.allclose(ustrip("mas", A_rt), ustrip("mas", A), atol=1e-5)
        assert jnp.allclose(ustrip("mas", B_rt), ustrip("mas", B), atol=1e-5)
        assert jnp.allclose(ustrip("mas", F_rt), ustrip("mas", F), atol=1e-5)
        assert jnp.allclose(ustrip("mas", G_rt), ustrip("mas", G), atol=1e-5)

    def test_jit(self) -> None:
        out = jax.jit(thiele_innes_from_campbell)(
            Q(2.0, "mas"), Q(0.5, "rad"), Q(1.0, "rad"), Q(0.6, "")
        )
        assert all(bool(jnp.isfinite(ustrip("mas", x))) for x in out)

    def test_vmap(self) -> None:
        A, _B, _F, _G = jax.vmap(thiele_innes_from_campbell)(
            Q(jnp.array([2.0, 3.0]), "mas"),
            Q(jnp.array([0.3, 0.8]), "rad"),
            Q(jnp.array([1.0, 1.5]), "rad"),
            Q(jnp.array([0.6, 0.4]), ""),
        )
        assert A.shape == (2,)


class TestEcoswEsinwConversions:
    """Tests for ecc_omega_from_ecosw_esinw and ecosw_esinw_from_ecc_omega."""

    def test_known_values(self) -> None:
        # omega = 0  ->  ecosw = e, esinw = 0
        ecosw, esinw = ecosw_esinw_from_ecc_omega(Q(0.4, ""), Q(0.0, "rad"))
        assert jnp.allclose(ustrip("", ecosw), 0.4)
        assert jnp.allclose(ustrip("", esinw), 0.0)
        ecc, omega = ecc_omega_from_ecosw_esinw(Q(0.4, ""), Q(0.0, ""))
        assert jnp.allclose(ustrip("", ecc), 0.4)
        assert jnp.allclose(ustrip("rad", omega), 0.0)

    def test_eccentricity_is_norm(self) -> None:
        ecc, _ = ecc_omega_from_ecosw_esinw(Q(0.3, ""), Q(0.4, ""))
        assert jnp.allclose(ustrip("", ecc), 0.5, atol=1e-6)

    def test_round_trip(self) -> None:
        # arg_peri in (-pi, pi] so atan2 recovers it exactly
        ecc_in = Q(jnp.array([0.1, 0.3, 0.5]), "")
        omega_in = Q(jnp.array([-1.0, 0.4, 2.5]), "rad")
        ecosw, esinw = ecosw_esinw_from_ecc_omega(ecc_in, omega_in)
        ecc, omega = ecc_omega_from_ecosw_esinw(ecosw, esinw)
        assert jnp.allclose(ustrip("", ecc), ustrip("", ecc_in), atol=1e-6)
        assert jnp.allclose(ustrip("rad", omega), ustrip("rad", omega_in), atol=1e-6)

    def test_jit(self) -> None:
        ecc, _omega = jax.jit(ecc_omega_from_ecosw_esinw)(Q(0.3, ""), Q(0.4, ""))
        assert jnp.allclose(ustrip("", ecc), 0.5, atol=1e-6)
        ecosw, _esinw = jax.jit(ecosw_esinw_from_ecc_omega)(Q(0.5, ""), Q(0.7, "rad"))
        assert bool(jnp.isfinite(ustrip("", ecosw)))

    def test_vmap(self) -> None:
        ecc, omega = jax.vmap(ecc_omega_from_ecosw_esinw)(
            Q(jnp.array([0.2, 0.0]), ""),
            Q(jnp.array([0.0, 0.3]), ""),
        )
        assert ecc.shape == (2,)
        assert omega.shape == (2,)
