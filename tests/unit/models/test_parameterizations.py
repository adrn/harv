"""Unit tests for RV and astrometry parameterizations."""

import jax
import jax.numpy as jnp

from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations.gaia import StandardGaiaAstrometry
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
