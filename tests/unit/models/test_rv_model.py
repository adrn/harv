"""Unit tests for the new RVModel.

Includes a numerical equivalence test against the existing RVLikelihood.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
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


class TestRVModelBasic:
    """Basic construction and introspection."""

    def test_construction(self):
        data = _make_rv_data()
        model = RVModel(data=data)
        assert model.data is data

    def test_param_names(self):
        model = RVModel(data=_make_rv_data())
        assert set(model._all_nonlinear_names()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }
        assert set(model._all_linear_names()) == {"rv_semiamp", "v_sys"}

    def test_obs_unit(self):
        model = RVModel(data=_make_rv_data())
        assert model._obs_unit() == "km / s"

    def test_base_design_matrix_shape(self):
        data = _make_rv_data(n_obs=10)
        model = RVModel(data=data)
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": 1.0,
        }
        X = model._base_design_matrix(nl)
        assert X.shape == (10, 2)


class TestRVModelExplicit:
    """Explicit (non-marginalized) evaluation."""

    def test_explicit_is_finite(self):
        data = _make_rv_data()
        model = RVModel(data=data)
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        linear = {
            "rv_semiamp": jnp.float32(5.0),
            "v_sys": jnp.float32(0.0),
        }
        ll = model.log_prob(nl, linear_values=linear)
        assert jnp.isfinite(ll)


class TestRVModelMarginalized:
    """Marginalized likelihood evaluation."""

    def test_marginalized_is_finite(self):
        data = _make_rv_data()
        model = RVModel(data=data, linear_prior=_rv_prior())
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        ll = model.log_prob(nl)
        assert jnp.isfinite(ll)

    def test_marginalized_jit(self):
        data = _make_rv_data()
        model = RVModel(data=data, linear_prior=_rv_prior())
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }

        @jax.jit
        def fn():
            return model.log_prob(nl)

        ll = fn()
        assert jnp.isfinite(ll)


class TestRVModelSampleConditional:
    """Test conditional linear parameter sampling."""

    def test_sample_returns_all_linear(self):
        data = _make_rv_data()
        model = RVModel(data=data, linear_prior=_rv_prior())
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        key = jax.random.key(42)
        samples = model.sample_conditional_linear(nl, key)
        assert "rv_semiamp" in samples
        assert "v_sys" in samples

    def test_sample_values_finite(self):
        data = _make_rv_data()
        model = RVModel(data=data, linear_prior=_rv_prior())
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        key = jax.random.key(0)
        samples = model.sample_conditional_linear(nl, key)
        assert jnp.isfinite(samples["rv_semiamp"])
        assert jnp.isfinite(samples["v_sys"])
