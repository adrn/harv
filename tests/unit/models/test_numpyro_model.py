"""Unit tests for numpyro model generation on AbstractComponentModel."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from harv.distributions import QuantityDistribution as QD
from harv.extensions import Jitter, MonomialTrend
from harv.models import RVModel

# Re-alias shared fixtures to shorter names used throughout this module.
nonlinear_priors = pytest.fixture(name="nonlinear_priors")(
    lambda rv_nonlinear_priors: rv_nonlinear_priors
)
linear_prior = pytest.fixture(name="linear_prior")(
    lambda rv_linear_prior: rv_linear_prior
)


def _get_factor_value(trace, name):
    """Extract the log-probability from a numpyro.factor site."""
    site = trace[name]
    return site["fn"].log_prob(site["value"])


class TestNumpyroModelMarginalized:
    def test_returns_callable(self, rv_data, nonlinear_priors, linear_prior):
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )
        assert callable(model_fn)

    def test_model_traces(self, rv_data, nonlinear_priors, linear_prior):
        """Model can be traced and produces expected sites."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        # Should have sample sites for nonlinear params
        assert "period" in trace
        assert "eccentricity" in trace
        assert "phase_peri" in trace
        assert "arg_peri" in trace

        # Should have a factor site for log_lik
        assert "log_lik" in trace

    def test_log_lik_is_finite(self, rv_data, nonlinear_priors, linear_prior):
        """Log-likelihood in the trace is finite."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        log_lik = _get_factor_value(trace, "log_lik")
        assert jnp.isfinite(log_lik)

    def test_no_linear_sites(self, rv_data, nonlinear_priors, linear_prior):
        """Marginalized model should NOT have linear param sample sites."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        sample_sites = {k for k, v in trace.items() if v["type"] == "sample"}
        assert "rv_semiamp" not in sample_sites
        assert "v_sys" not in sample_sites
        assert "_linear" not in sample_sites


class TestNumpyroModelFull:
    def test_returns_callable(self, rv_data, nonlinear_priors, linear_prior):
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=False
        )
        assert callable(model_fn)

    def test_model_traces(self, rv_data, nonlinear_priors, linear_prior):
        """Full model has both nonlinear and linear sample sites."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=False
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        # Nonlinear params
        assert "period" in trace
        assert "eccentricity" in trace
        # Linear params (sampled jointly as _linear)
        assert "_linear" in trace
        assert trace["_linear"]["type"] == "sample"
        # Deterministic sites for individual linear params
        assert "rv_semiamp" in trace
        assert "v_sys" in trace

    def test_log_lik_is_finite(self, rv_data, nonlinear_priors, linear_prior):
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=False
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        log_lik = _get_factor_value(trace, "log_lik")
        assert jnp.isfinite(log_lik)

    def test_requires_linear_prior(self, rv_data, nonlinear_priors):
        """Full model without linear_prior raises ValueError."""
        model = RVModel()
        with pytest.raises(ValueError, match="linear_prior"):
            model.numpyro_model(nonlinear_priors, rv_data, None, marginalized=False)


class TestNumpyroModelWithExtensions:
    def test_jitter_in_trace(self, rv_data, linear_prior):
        """Jitter extension adds a sample site for jitter."""
        nonlinear_priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            "jitter": QD(dist.HalfNormal(1.0), "km/s"),
        }
        model = RVModel(extensions=(Jitter(param_unit="km/s"),))
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert "jitter" in trace
        assert trace["jitter"]["type"] == "sample"
        assert jnp.isfinite(_get_factor_value(trace, "log_lik"))

    def test_trend_in_trace(self, rv_data):
        """Trend extension adds linear param; marginalized model traces OK."""
        nonlinear_priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        }
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "trend_1": dist.Normal(0.0, 1.0),
        }
        model = RVModel(extensions=(MonomialTrend(order=1, time_unit="day"),))
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert jnp.isfinite(_get_factor_value(trace, "log_lik"))


class TestNumpyroModelUnitConversion:
    def test_period_unit_conversion(self, rv_data, linear_prior):
        """Period prior in different units gets converted correctly."""
        # Period prior in years, data in days
        nonlinear_priors = {
            "period": QD(dist.Uniform(0.1, 2.0), "yr"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        }
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_prior, marginalized=True
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        assert jnp.isfinite(_get_factor_value(trace, "log_lik"))
