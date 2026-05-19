"""Tests for harv.kepler.masses mass-function helpers."""

import jax
import jax.numpy as jnp
import pytest
import quaxed.numpy as qnp
from unxt import Q, ustrip

from harv.kepler.constants import G
from harv.kepler.masses import (
    astrometric_mass_function,
    binary_mass_function,
    companion_mass_from_mass_function,
    semi_major_axis_physical,
)


def _rv_semiamp(period, m1, m2, eccentricity, sini):
    """Primary RV semi-amplitude K1 for a binary (analytic)."""
    m_total = m1 + m2
    factor = qnp.cbrt(2.0 * jnp.pi * G / period)
    return (
        factor * m2 * sini / (m_total ** (2.0 / 3.0) * qnp.sqrt(1.0 - eccentricity**2))
    )


class TestBinaryMassFunction:
    def test_known_value(self):
        mf = binary_mass_function(Q(100.0, "day"), Q(10.0, "km/s"), 0.1)
        assert mf.unit.is_equivalent(Q(1.0, "Msun").unit)
        assert round(float(ustrip("Msun", mf)), 4) == 0.0102

    def test_zero_eccentricity_vs_finite(self):
        # A more eccentric orbit at fixed (P, K) has a smaller mass function.
        mf0 = binary_mass_function(Q(100.0, "day"), Q(10.0, "km/s"), 0.0)
        mfe = binary_mass_function(Q(100.0, "day"), Q(10.0, "km/s"), 0.5)
        assert float(ustrip("Msun", mfe)) < float(ustrip("Msun", mf0))

    def test_vmap_and_jit(self):
        periods = Q(jnp.array([50.0, 100.0, 200.0]), "day")
        ks = Q(jnp.array([5.0, 10.0, 15.0]), "km/s")
        eccs = jnp.array([0.0, 0.1, 0.3])
        batched = jax.vmap(binary_mass_function)(periods, ks, eccs)
        assert batched.shape == (3,)
        jitted = jax.jit(binary_mass_function)(Q(100.0, "day"), Q(10.0, "km/s"), 0.1)
        assert jnp.allclose(
            ustrip("Msun", jitted),
            ustrip("Msun", binary_mass_function(Q(100.0, "day"), Q(10.0, "km/s"), 0.1)),
        )


class TestAstrometricMassFunction:
    def test_kepler_third_law(self):
        # a = 1 AU, P = 1 yr -> total mass 1 Msun.
        mf = astrometric_mass_function(Q(1.0, "AU"), Q(1.0, "yr"))
        assert round(float(ustrip("Msun", mf)), 3) == 1.0

    def test_scaling(self):
        # f ~ a^3: doubling a multiplies the mass function by 8.
        mf1 = astrometric_mass_function(Q(1.0, "AU"), Q(1.0, "yr"))
        mf2 = astrometric_mass_function(Q(2.0, "AU"), Q(1.0, "yr"))
        assert jnp.allclose(ustrip("Msun", mf2) / ustrip("Msun", mf1), 8.0, atol=1e-3)


class TestCompanionMass:
    @pytest.mark.parametrize(
        ("m1", "m2", "sini"),
        [(1.0, 1.0, 1.0), (1.3, 0.8, 0.7), (0.5, 0.05, 1.0), (2.0, 3.0, 0.9)],
    )
    def test_roundtrip_through_mass_function(self, m1, m2, sini):
        # f = m2^3 sin^3 i / (m1 + m2)^2, solved back for m2.
        mass_function = m2**3 * sini**3 / (m1 + m2) ** 2
        m2_rec = companion_mass_from_mass_function(
            Q(mass_function, "Msun"), Q(m1, "Msun"), sini
        )
        assert jnp.allclose(ustrip("Msun", m2_rec), m2, rtol=1e-5)

    def test_minimum_mass_when_sini_one(self):
        # Smaller sin i implies a larger companion mass for the same f.
        mass_function = Q(0.1, "Msun")
        m1 = Q(1.0, "Msun")
        m2_edge = companion_mass_from_mass_function(mass_function, m1, 1.0)
        m2_incl = companion_mass_from_mass_function(mass_function, m1, 0.5)
        assert float(ustrip("Msun", m2_incl)) > float(ustrip("Msun", m2_edge))

    def test_rv_roundtrip_recovers_companion_mass(self):
        # Build K from known masses, then recover m2 (edge-on).
        period, ecc = Q(300.0, "day"), 0.2
        m1, m2 = Q(1.1, "Msun"), Q(0.6, "Msun")
        k = _rv_semiamp(period, m1, m2, ecc, sini=1.0)
        mf = binary_mass_function(period, k, ecc)
        m2_rec = companion_mass_from_mass_function(mf, m1, 1.0)
        assert jnp.allclose(ustrip("Msun", m2_rec), ustrip("Msun", m2), rtol=1e-4)

    def test_vmap(self):
        mfs = Q(jnp.array([0.05, 0.1, 0.25]), "Msun")
        m1 = Q(jnp.array([1.0, 1.0, 1.0]), "Msun")
        out = jax.vmap(companion_mass_from_mass_function)(mfs, m1)
        assert out.shape == (3,)
        assert jnp.all(ustrip("Msun", out) > 0)


class TestSemiMajorAxisPhysical:
    def test_definition(self):
        # a_AU = a_angular / parallax.
        a = semi_major_axis_physical(Q(10.0, "mas"), Q(5.0, "mas"))
        assert a.unit.is_equivalent(Q(1.0, "AU").unit)
        assert float(ustrip("AU", a)) == 2.0

    def test_unit_agnostic(self):
        # Mixed angle units must still give the correct AU ratio.
        a = semi_major_axis_physical(Q(1.0, "arcsec"), Q(100.0, "mas"))
        assert jnp.allclose(ustrip("AU", a), 10.0)
