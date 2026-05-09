"""Tests for the EcoswEsinwRV parameterization and its use in RVModel."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV
from harv.models.rv import RVModel


def _make_rv_data(n_obs=20):
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
    )


def _rv_prior():
    return {
        "rv_semiamp": dist.Normal(0.0, 100.0),
        "v_sys": dist.Normal(0.0, 100.0),
    }


# Reference orbital parameters (standard parameterization)
_ECC = 0.3
_ARG_PERI = 1.0  # rad
_ECOSW = _ECC * float(jnp.cos(_ARG_PERI))
_ESINW = _ECC * float(jnp.sin(_ARG_PERI))


class TestEcoswEsinwRVParameterization:
    """Unit tests for the EcoswEsinwRV parameterization object."""

    def test_params_names(self):
        p = EcoswEsinwRV()
        names = [pi.name for pi in p.params()]
        assert names == [
            "period",
            "ecosw",
            "esinw",
            "phase_peri",
            "rv_semiamp",
            "v_sys",
        ]

    def test_nonlinear_linear_split(self):
        p = EcoswEsinwRV()
        nl = p.nonlinear_params()
        lin = p.linear_params()
        assert len(nl) == 4
        assert len(lin) == 2
        assert all(not pi.linear for pi in nl)
        assert all(pi.linear for pi in lin)

    def test_eccentricity(self):
        p = EcoswEsinwRV()
        nl = {"ecosw": _ECOSW, "esinw": _ESINW}
        ecc = p.eccentricity(nl)
        assert jnp.allclose(ecc, _ECC, atol=1e-6)

    def test_eccentricity_zero(self):
        p = EcoswEsinwRV()
        nl = {"ecosw": 0.0, "esinw": 0.0}
        assert float(p.eccentricity(nl)) == 0.0

    def test_strip_nl_for_design(self):
        p = EcoswEsinwRV()
        nl = {"ecosw": 0.3, "esinw": 0.1, "other": "kept"}
        stripped = p.strip_nl_for_design(nl)
        assert "ecosw" in stripped
        assert "esinw" in stripped
        assert "other" in stripped

    def test_design_matrix_shape(self):
        p = EcoswEsinwRV()
        n_obs = 10
        key = jax.random.key(42)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))
        nl_values = {"ecosw": _ECOSW, "esinw": _ESINW}
        X = p.design_matrix(sin_f, cos_f, nl_values)
        assert X.shape == (n_obs, 2)

    def test_design_matrix_second_col_ones(self):
        p = EcoswEsinwRV()
        n_obs = 5
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        nl_values = {"ecosw": 0.0, "esinw": 0.0}
        X = p.design_matrix(sin_f, cos_f, nl_values)
        assert jnp.allclose(X[:, 1], 1.0)

    def test_design_matrix_matches_standard(self):
        """EcoswEsinwRV must produce the same design matrix as StandardRV."""
        std = StandardRV()
        eco = EcoswEsinwRV()
        n_obs = 20
        key = jax.random.key(7)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))

        dm_std = std.design_matrix(
            sin_f, cos_f, {"eccentricity": _ECC, "arg_peri": _ARG_PERI}
        )
        dm_eco = eco.design_matrix(sin_f, cos_f, {"ecosw": _ECOSW, "esinw": _ESINW})
        assert jnp.allclose(dm_std, dm_eco, atol=1e-6)

    def test_static_registered(self):
        p = EcoswEsinwRV()
        leaves, _ = jax.tree.flatten(p)
        assert leaves == []

    def test_design_matrix_jit(self):
        p = EcoswEsinwRV()
        sin_f = jnp.array([0.1, 0.2, 0.3])
        cos_f = jnp.array([0.9, 0.8, 0.7])
        nl_values = {"ecosw": 0.2, "esinw": 0.1}

        @jax.jit
        def fn(sf, cf):
            return p.design_matrix(sf, cf, nl_values)

        X = fn(sin_f, cos_f)
        assert X.shape == (3, 2)


class TestStandardRVHelpers:
    """Verify the new helper methods on StandardRV too."""

    def test_eccentricity(self):
        p = StandardRV()
        nl = {"eccentricity": 0.4}
        assert float(p.eccentricity(nl)) == 0.4

    def test_strip_nl_for_design(self):
        p = StandardRV()
        nl = {
            "eccentricity": Q(0.3, ""),
            "arg_peri": Q(1.0, "rad"),
            "period": Q(100.0, "day"),
        }
        stripped = p.strip_nl_for_design(nl)
        # Should be plain floats after stripping
        assert not hasattr(stripped["eccentricity"], "unit")
        assert not hasattr(stripped["arg_peri"], "unit")


class TestRVModelWithEcoswEsinw:
    """RVModel constructed with EcoswEsinwRV parameterization."""

    def test_construction(self):
        model = RVModel(parameterization=EcoswEsinwRV())
        assert isinstance(model.parameterization, EcoswEsinwRV)

    def test_param_names(self):
        model = RVModel(parameterization=EcoswEsinwRV())
        assert set(model._all_nonlinear_names()) == {
            "period",
            "ecosw",
            "esinw",
            "phase_peri",
        }
        assert set(model._all_linear_names()) == {"rv_semiamp", "v_sys"}

    def test_base_design_matrix_shape(self):
        data = _make_rv_data(n_obs=10)
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "phase_peri": 0.0,
        }
        X = model._base_design_matrix(nl, data)
        assert X.shape == (10, 2)

    def test_explicit_is_finite(self):
        data = _make_rv_data()
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "phase_peri": 0.0,
        }
        linear = {
            "rv_semiamp": jnp.float32(5.0),
            "v_sys": jnp.float32(0.0),
        }
        ll = model.log_prob(nl, data, linear_values=linear)
        assert jnp.isfinite(ll)

    def test_marginalized_is_finite(self):
        data = _make_rv_data()
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "phase_peri": 0.0,
        }
        ll = model.log_prob(nl, data, linear_prior=_rv_prior())
        assert jnp.isfinite(ll)

    def test_jit(self):
        data = _make_rv_data()
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "phase_peri": 0.0,
        }

        @jax.jit
        def fn():
            return model.log_prob(nl, data, linear_prior=_rv_prior())

        ll = fn()
        assert jnp.isfinite(ll)


class TestNumericalEquivalence:
    """EcoswEsinw model must match StandardRV for equivalent parameters."""

    def test_explicit_equivalence(self):
        data = _make_rv_data(n_obs=30)

        std_model = RVModel()
        eco_model = RVModel(parameterization=EcoswEsinwRV())

        nl_std = {
            "period": Q(100.0, "day"),
            "eccentricity": _ECC,
            "phase_peri": 0.0,
            "arg_peri": _ARG_PERI,
        }
        nl_eco = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "phase_peri": 0.0,
        }
        linear = {
            "rv_semiamp": jnp.float32(5.0),
            "v_sys": jnp.float32(0.0),
        }

        ll_std = std_model.log_prob(nl_std, data, linear_values=linear)
        ll_eco = eco_model.log_prob(nl_eco, data, linear_values=linear)

        assert jnp.allclose(ll_std, ll_eco, atol=1e-6), (
            f"std={float(ll_std)}, eco={float(ll_eco)}"
        )

    def test_marginalized_equivalence(self):
        data = _make_rv_data(n_obs=30)
        prior = _rv_prior()

        std_model = RVModel()
        eco_model = RVModel(parameterization=EcoswEsinwRV())

        nl_std = {
            "period": Q(100.0, "day"),
            "eccentricity": _ECC,
            "phase_peri": 0.0,
            "arg_peri": _ARG_PERI,
        }
        nl_eco = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "phase_peri": 0.0,
        }

        ll_std = std_model.log_prob(nl_std, data, linear_prior=prior)
        ll_eco = eco_model.log_prob(nl_eco, data, linear_prior=prior)

        assert jnp.allclose(ll_std, ll_eco, atol=1e-6), (
            f"std={float(ll_std)}, eco={float(ll_eco)}"
        )
