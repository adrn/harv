"""Tests for model factory functions."""

import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
from harv.extensions.jitter import Jitter
from harv.models.factories import rv_model
from harv.models.parametrizations.rv import EcoswEsinwRV, StandardRV
from harv.models.rv import RVModel


def _make_rv_data(n_obs=20):
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
    )


class TestRVModelFactory:
    def test_default_returns_rvmodel(self):
        model = rv_model(_make_rv_data())
        assert isinstance(model, RVModel)

    def test_default_parameterization_is_standard(self):
        model = rv_model(_make_rv_data())
        assert isinstance(model.parameterization, StandardRV)

    def test_default_linear_prior_set(self):
        model = rv_model(_make_rv_data())
        assert model.linear_prior is not None
        assert "rv_semiamp" in model.linear_prior
        assert "v_sys" in model.linear_prior

    def test_explicit_no_prior(self):
        model = rv_model(_make_rv_data(), linear_prior=False)
        assert model.linear_prior is None

    def test_custom_prior(self):
        prior = {"rv_semiamp": dist.Normal(0.0, 10.0)}
        model = rv_model(_make_rv_data(), linear_prior=prior)
        assert model.linear_prior is prior

    def test_ecosw_parameterization(self):
        model = rv_model(_make_rv_data(), parameterization=EcoswEsinwRV())
        assert isinstance(model.parameterization, EcoswEsinwRV)

    def test_extensions_passed_through(self):
        jitter = Jitter("km/s")
        model = rv_model(_make_rv_data(), extensions=(jitter,))
        assert len(model.extensions) == 1

    def test_factory_model_is_functional(self):
        model = rv_model(_make_rv_data())
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        ll = model.log_prob(nl)
        assert jnp.isfinite(ll)
