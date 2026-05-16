"""Unit tests for RV and astrometry parameterizations."""

import jax
import jax.numpy as jnp
import pytest
from unxt import Q

from harv.data import GaiaAstrometryData
from harv.kepler.orbits import thiele_innes_ABFG
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations.gaia import (
    StandardGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.models.parameterizations.rv import StandardRV


class TestStandardRV:
    def test_params_names(self):
        p = StandardRV()
        names = [pi.name for pi in p.params()]
        assert names == [
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
            "rv_semiamp",
            "v_sys",
        ]

    def test_nonlinear_linear_split(self):
        p = StandardRV()
        nl = p.nonlinear_params()
        lin = p.linear_params()
        assert len(nl) == 4
        assert len(lin) == 2
        assert all(not pi.linear for pi in nl)
        assert all(pi.linear for pi in lin)

    def test_params_returns_paraminfo(self):
        p = StandardRV()
        for pi in p.params():
            assert isinstance(pi, ParamInfo)

    def test_design_matrix_shape(self):
        p = StandardRV()
        n_obs = 10
        key = jax.random.key(42)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))
        nl_values = {"eccentricity": 0.3, "arg_peri": 0.5}
        X = p.design_matrix(sin_f, cos_f, nl_values)
        assert X.shape == (n_obs, 2)

    def test_design_matrix_second_col_ones(self):
        p = StandardRV()
        n_obs = 5
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        nl_values = {"eccentricity": 0.0, "arg_peri": 0.0}
        X = p.design_matrix(sin_f, cos_f, nl_values)
        # Second column (v_sys) should always be ones
        assert jnp.allclose(X[:, 1], 1.0)

    def test_design_matrix_jit(self):
        p = StandardRV()
        sin_f = jnp.array([0.1, 0.2, 0.3])
        cos_f = jnp.array([0.9, 0.8, 0.7])
        nl_values = {"eccentricity": 0.2, "arg_peri": 1.0}

        @jax.jit
        def fn(sf, cf):
            return p.design_matrix(sf, cf, nl_values)

        X = fn(sin_f, cos_f)
        assert X.shape == (3, 2)

    def test_eqx_module_no_leaves(self):
        """StandardRV should be an eqx.Module with no dynamic leaves."""
        p = StandardRV()
        leaves, _ = jax.tree.flatten(p)
        assert leaves == []


class TestStandardGaiaAstrometry:
    def test_params_names(self):
        p = StandardGaiaAstrometry()
        names = [pi.name for pi in p.params()]
        assert names == [
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
            "lon_asc_node",
            "cos_i",
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "semi_major_axis",
        ]

    def test_nonlinear_linear_split(self):
        p = StandardGaiaAstrometry()
        nl = p.nonlinear_params()
        lin = p.linear_params()
        assert len(nl) == 6
        assert len(lin) == 6
        assert all(not pi.linear for pi in nl)
        assert all(pi.linear for pi in lin)

    def test_design_matrix_shape(self):
        p = StandardGaiaAstrometry()
        n_obs = 15
        key = jax.random.key(0)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))
        dt = jax.random.uniform(key, (n_obs,))
        sin_psi = jax.random.normal(key, (n_obs,))
        cos_psi = jax.random.normal(key, (n_obs,))
        parallax_factor = jax.random.uniform(key, (n_obs,))
        nl_values = {
            "eccentricity": 0.3,
            "arg_peri": 0.5,
            "lon_asc_node": 1.0,
            "cos_i": 0.8,
        }
        X = p.design_matrix(
            sin_f, cos_f, dt, sin_psi, cos_psi, parallax_factor, nl_values
        )
        assert X.shape == (n_obs, 6)

    def test_design_matrix_jit(self):
        p = StandardGaiaAstrometry()
        n = 5
        key = jax.random.key(1)
        sin_f = jax.random.normal(key, (n,))
        cos_f = jax.random.normal(key, (n,))
        dt = jax.random.uniform(key, (n,))
        sin_psi = jax.random.normal(key, (n,))
        cos_psi = jax.random.normal(key, (n,))
        pf = jax.random.uniform(key, (n,))
        nl_values = {
            "eccentricity": 0.1,
            "arg_peri": 0.0,
            "lon_asc_node": 0.0,
            "cos_i": 1.0,
        }

        @jax.jit
        def fn(sf, cf, _dt, sp, cp, _pf):
            return p.design_matrix(sf, cf, _dt, sp, cp, _pf, nl_values)

        X = fn(sin_f, cos_f, dt, sin_psi, cos_psi, pf)
        assert X.shape == (n, 6)

    def test_static_registered(self):
        p = StandardGaiaAstrometry()
        leaves, _ = jax.tree.flatten(p)
        assert leaves == []


class TestThieleInnesGaiaAstrometry:
    def test_params_names(self):
        p = ThieleInnesGaiaAstrometry(a_floor=0.01)
        names = [pi.name for pi in p.params()]
        assert names == [
            "period",
            "eccentricity",
            "phase_peri",
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "ti_A",
            "ti_B",
            "ti_F",
            "ti_G",
        ]

    def test_nonlinear_linear_split(self):
        p = ThieleInnesGaiaAstrometry(a_floor=0.01)
        nl = p.nonlinear_params()
        lin = p.linear_params()
        assert len(nl) == 3
        assert len(lin) == 9
        assert all(not pi.linear for pi in nl)
        assert all(pi.linear for pi in lin)

    def test_design_matrix_shape(self):
        p = ThieleInnesGaiaAstrometry(a_floor=0.01)
        n_obs = 15
        key = jax.random.key(0)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))
        dt = jax.random.uniform(key, (n_obs,))
        sin_psi = jax.random.normal(key, (n_obs,))
        cos_psi = jax.random.normal(key, (n_obs,))
        parallax_factor = jax.random.uniform(key, (n_obs,))
        nl_values = {"eccentricity": 0.3}
        X = p.design_matrix(
            sin_f, cos_f, dt, sin_psi, cos_psi, parallax_factor, nl_values
        )
        assert X.shape == (n_obs, 9)

    def test_design_matrix_jit(self):
        p = ThieleInnesGaiaAstrometry(a_floor=0.01)
        n = 5
        key = jax.random.key(1)
        sin_f = jax.random.normal(key, (n,))
        cos_f = jax.random.normal(key, (n,))
        dt = jax.random.uniform(key, (n,))
        sin_psi = jax.random.normal(key, (n,))
        cos_psi = jax.random.normal(key, (n,))
        pf = jax.random.uniform(key, (n,))
        nl_values = {"eccentricity": 0.1}

        @jax.jit
        def fn(sf, cf, _dt, sp, cp, _pf):
            return p.design_matrix(sf, cf, _dt, sp, cp, _pf, nl_values)

        X = fn(sin_f, cos_f, dt, sin_psi, cos_psi, pf)
        assert X.shape == (n, 9)

    def test_design_matrix_matches_standard(self):
        """TI parameterization orbit columns should equal StandardGaiaAstrometry's.

        When ti_A, ti_B, ti_F, ti_G equal the Thiele-Innes constants derived from
        the Campbell angles, the orbit contribution X @ theta should match.
        """
        n = 8
        key = jax.random.key(42)
        sin_f = jax.random.normal(key, (n,))
        cos_f = jax.random.normal(key, (n,))
        dt = jax.random.uniform(key, (n,))
        sin_psi = jax.random.normal(key, (n,))
        cos_psi = jax.random.normal(key, (n,))
        pf = jax.random.uniform(key, (n,))

        a0 = 2.5
        ecc, arg_peri, lon_asc_node, cos_i = 0.3, 0.8, 1.1, 0.6
        A, B, F, G = thiele_innes_ABFG(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )

        std = StandardGaiaAstrometry()
        nl_std = {
            "eccentricity": ecc,
            "arg_peri": arg_peri,
            "lon_asc_node": lon_asc_node,
            "cos_i": cos_i,
        }
        X_std = std.design_matrix(sin_f, cos_f, dt, sin_psi, cos_psi, pf, nl_std)
        orbit_std = X_std[:, 5] * a0  # Standard: 6th col scaled by a0

        ti = ThieleInnesGaiaAstrometry(a_floor=0.01)
        nl_ti = {"eccentricity": ecc}
        X_ti = ti.design_matrix(sin_f, cos_f, dt, sin_psi, cos_psi, pf, nl_ti)
        # Orbit contribution from TI: X_ti columns 5-8 @ [A, B, F, G]
        orbit_ti = (
            X_ti[:, 5] * A * a0
            + X_ti[:, 6] * B * a0
            + X_ti[:, 7] * F * a0
            + X_ti[:, 8] * G * a0
        )

        assert jnp.allclose(orbit_std, orbit_ti, atol=1e-5)

    def test_jacobian_correction_known_value(self):
        """Verify correction at hand-crafted (a0, sin²i) = (2.0, 0.75)."""
        p = ThieleInnesGaiaAstrometry(a_floor=0.0, sin2i_floor=0.0)
        # Choose simple TI constants: a_perp=0, circular face-on-ish orbit
        # A = a0, B=0, F=0, G=a0*cos_i with cos_i = sqrt(1 - sin2i)
        a0 = 2.0
        sin2i = 0.75
        cos_i = jnp.sqrt(1.0 - sin2i)
        # A*G - B*F = v = a0^2 * cos_i
        # A^2+B^2+F^2+G^2 = 2*a0^2 * u=... let's just use the formula directly
        # For a pure A, G solution: A=a0, B=0, F=0, G=a0*cos_i
        A = a0
        B = 0.0
        F = 0.0
        G = a0 * cos_i
        linear_map = {
            "ti_A": jnp.array(A),
            "ti_B": jnp.array(B),
            "ti_F": jnp.array(F),
            "ti_G": jnp.array(G),
        }
        corr = p.linear_log_prior_correction(linear_map)
        expected = -3.0 * jnp.log(a0) - jnp.log(sin2i)
        assert jnp.allclose(corr, expected, atol=1e-5), f"{corr} != {expected}"

    def test_jacobian_correction_floors(self):
        """Floors keep the correction finite for degenerate inputs.

        Two degenerate cases:
        1. All-zero TI → a0 = 0, floor on a0 triggers.  sin²i = 1 (degenerate but
           not face-on), so only the a0 floor fires.
        2. Face-on orbit (cos_i = 1 ↔ sin²i = 0): floor on sin²i triggers.
        """
        # Case 1: all-zero TI
        p = ThieleInnesGaiaAstrometry(a_floor=0.05, sin2i_floor=0.01)
        linear_map_zero = {
            "ti_A": jnp.array(0.0),
            "ti_B": jnp.array(0.0),
            "ti_F": jnp.array(0.0),
            "ti_G": jnp.array(0.0),
        }
        corr_zero = p.linear_log_prior_correction(linear_map_zero)
        assert jnp.isfinite(corr_zero)

        # Case 2: face-on orbit A=a0, B=0, F=0, G=a0 → sin²i = 0
        a0_val = 1.0
        linear_map_faceon = {
            "ti_A": jnp.array(a0_val),
            "ti_B": jnp.array(0.0),
            "ti_F": jnp.array(0.0),
            "ti_G": jnp.array(a0_val),
        }
        corr_faceon = p.linear_log_prior_correction(linear_map_faceon)
        assert jnp.isfinite(corr_faceon)
        expected_faceon = -3.0 * jnp.log(a0_val + 0.05) - jnp.log(0.0 + 0.01)
        assert jnp.allclose(corr_faceon, expected_faceon, atol=1e-4)

    def test_log_uniform_in_a(self):
        """log_uniform_in_a=True should use m=4 instead of m=3."""
        a_floor = 0.0
        sin2i_floor = 0.0
        p3 = ThieleInnesGaiaAstrometry(a_floor=a_floor, sin2i_floor=sin2i_floor)
        p4 = ThieleInnesGaiaAstrometry(
            a_floor=a_floor, sin2i_floor=sin2i_floor, log_uniform_in_a=True
        )
        # Use A=2, B=F=G=0 → a0=2 (edge-on orbit), log(a0)=log(2)>0 so c4 < c3
        linear_map = {
            "ti_A": jnp.array(2.0),
            "ti_B": jnp.array(0.0),
            "ti_F": jnp.array(0.0),
            "ti_G": jnp.array(0.0),
        }
        # Both should be finite; p4 should give a more negative value (larger |m|)
        c3 = p3.linear_log_prior_correction(linear_map)
        c4 = p4.linear_log_prior_correction(linear_map)
        assert jnp.isfinite(c3)
        assert jnp.isfinite(c4)
        assert c4 < c3  # larger m → more negative correction

    def test_from_data(self):
        """from_data sets a_floor = Med(sigma_AL)/sqrt(N) correctly."""
        errs = jnp.array([0.04, 0.05, 0.06, 0.03])
        data = GaiaAstrometryData(
            time=Q(jnp.zeros(4), "day"),
            al_position=Q(jnp.zeros(4), "mas"),
            al_position_err=Q(errs, "mas"),
            scan_angle=Q(jnp.zeros(4), "rad"),
            parallax_factor=jnp.zeros(4),
        )
        p = ThieleInnesGaiaAstrometry.from_data(data)
        expected_floor = float(jnp.median(errs) / jnp.sqrt(4))
        assert abs(p.a_floor - expected_floor) < 1e-10

    def test_linear_log_prior_correction_default_none(self):
        """StandardGaiaAstrometry returns None from the base hook."""
        p = StandardGaiaAstrometry()
        result = p.linear_log_prior_correction({})
        assert result is None

    @pytest.mark.parametrize("log_uniform", [False, True])
    def test_jit_compatible(self, log_uniform):
        p = ThieleInnesGaiaAstrometry(
            a_floor=0.01, sin2i_floor=0.01, log_uniform_in_a=log_uniform
        )
        linear_map = {
            "ti_A": jnp.array(1.5),
            "ti_B": jnp.array(0.3),
            "ti_F": jnp.array(-0.4),
            "ti_G": jnp.array(1.0),
        }

        @jax.jit
        def fn(A, B, F, G):
            lm = {"ti_A": A, "ti_B": B, "ti_F": F, "ti_G": G}
            return p.linear_log_prior_correction(lm)

        result = fn(
            linear_map["ti_A"],
            linear_map["ti_B"],
            linear_map["ti_F"],
            linear_map["ti_G"],
        )
        assert jnp.isfinite(result)

    def test_apply_jacobian_correction_disabled_returns_none(self):
        """With the switch off, linear_log_prior_correction returns None."""
        p = ThieleInnesGaiaAstrometry(apply_jacobian_correction=False)
        assert p.apply_jacobian_correction is False
        assert p.a_floor is None
        # linear_map is ignored entirely when the correction is disabled.
        assert p.linear_log_prior_correction({}) is None

    def test_correction_enabled_by_default(self):
        """apply_jacobian_correction defaults to True (current behavior)."""
        p = ThieleInnesGaiaAstrometry(a_floor=0.01)
        assert p.apply_jacobian_correction is True
        linear_map = {
            "ti_A": jnp.array(1.5),
            "ti_B": jnp.array(0.3),
            "ti_F": jnp.array(-0.4),
            "ti_G": jnp.array(1.0),
        }
        assert p.linear_log_prior_correction(linear_map) is not None

    def test_requires_a_floor_when_correction_enabled(self):
        """a_floor is mandatory when the correction is on (the default)."""
        with pytest.raises(ValueError, match="a_floor is required"):
            ThieleInnesGaiaAstrometry()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"a_floor": 0.01},
            {"sin2i_floor": 0.01},
            {"log_uniform_in_a": True},
            {"a_floor": 0.01, "sin2i_floor": 0.02},
        ],
    )
    def test_rejects_floor_params_when_correction_disabled(self, kwargs):
        """Floor params must be left as None when the correction is disabled."""
        with pytest.raises(ValueError, match="must be left unset"):
            ThieleInnesGaiaAstrometry(apply_jacobian_correction=False, **kwargs)

    def test_sin2i_floor_falls_back_to_default(self):
        """sin2i_floor=None reproduces the documented 0.01 fallback."""
        linear_map = {
            "ti_A": jnp.array(1.5),
            "ti_B": jnp.array(0.3),
            "ti_F": jnp.array(-0.4),
            "ti_G": jnp.array(1.0),
        }
        p_default = ThieleInnesGaiaAstrometry(a_floor=0.05)
        p_explicit = ThieleInnesGaiaAstrometry(a_floor=0.05, sin2i_floor=0.01)
        assert jnp.allclose(
            p_default.linear_log_prior_correction(linear_map),
            p_explicit.linear_log_prior_correction(linear_map),
        )

    def test_from_data_disabled_skips_a_floor(self):
        """from_data(..., apply_jacobian_correction=False) needs no a_floor."""
        data = GaiaAstrometryData(
            time=Q(jnp.zeros(4), "day"),
            al_position=Q(jnp.zeros(4), "mas"),
            al_position_err=Q(jnp.array([0.04, 0.05, 0.06, 0.03]), "mas"),
            scan_angle=Q(jnp.zeros(4), "rad"),
            parallax_factor=jnp.zeros(4),
        )
        p = ThieleInnesGaiaAstrometry.from_data(data, apply_jacobian_correction=False)
        assert p.apply_jacobian_correction is False
        assert p.a_floor is None
        assert p.linear_log_prior_correction({}) is None
