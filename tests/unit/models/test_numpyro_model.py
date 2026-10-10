"""Unit tests for numpyro model generation on AbstractComponentModel."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from unxt import Q

from harv.distributions import QuantityDistribution as QD
from harv.models import RVModel
from harv.models.extensions import Jitter, MonomialTrend
from harv.models.priors.callables import PeriodDependentKPrior

# Re-alias shared fixtures to shorter names used throughout this module.
nonlinear_priors = pytest.fixture(name="nonlinear_priors")(
    lambda rv_nonlinear_priors: rv_nonlinear_priors
)
linear_priors = pytest.fixture(name="linear_priors")(
    lambda rv_linear_prior: rv_linear_prior
)


def _get_factor_value(trace, name):
    """Extract the log-probability from a numpyro.factor site."""
    site = trace[name]
    return site["fn"].log_prob(site["value"])


class TestNumpyroModelMarginalized:
    def test_returns_callable(self, rv_data, nonlinear_priors, linear_priors):
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )
        assert callable(model_fn)

    def test_model_traces(self, rv_data, nonlinear_priors, linear_priors):
        """Model can be traced and produces expected sites."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        # Should have sample sites for nonlinear params
        assert "period" in trace
        assert "eccentricity" in trace
        assert "phase_peri" in trace
        assert "arg_peri" in trace

        # Should have a factor site for ln_lik
        assert "ln_lik" in trace

    def test_log_lik_is_finite(self, rv_data, nonlinear_priors, linear_priors):
        """Log-likelihood in the trace is finite."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        ln_lik = _get_factor_value(trace, "ln_lik")
        assert jnp.isfinite(ln_lik)

    def test_no_linear_sites(self, rv_data, nonlinear_priors, linear_priors):
        """Marginalized model should NOT have linear param sample sites."""
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        sample_sites = {k for k, v in trace.items() if v["type"] == "sample"}
        assert "rv_semiamp" not in sample_sites
        assert "v_sys" not in sample_sites
        assert "_linear" not in sample_sites


class TestNumpyroModelFull:
    def test_returns_callable(self, rv_data, nonlinear_priors, linear_priors):
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=False
        )
        assert callable(model_fn)

    def test_model_traces(self, rv_data, nonlinear_priors, linear_priors):
        """Full model has a sample site per parameter, nonlinear and linear.

        Each linear parameter gets its own named site. The priors are
        independent, so the joint site this replaced carried a diagonal
        covariance and was the same distribution.
        """
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=False
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert "period" in trace
        assert "eccentricity" in trace
        assert trace["rv_semiamp"]["type"] == "sample"
        assert trace["v_sys"]["type"] == "sample"
        assert "_linear" not in trace

    def test_log_lik_is_finite(self, rv_data, nonlinear_priors, linear_priors):
        model = RVModel()
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=False
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        ln_lik = _get_factor_value(trace, "ln_lik")
        assert jnp.isfinite(ln_lik)

    def test_requires_linear_prior(self, rv_data, nonlinear_priors):
        """Full model without linear_priors raises ValueError."""
        model = RVModel()
        with pytest.raises(ValueError, match="linear_priors"):
            model.numpyro_model(nonlinear_priors, rv_data, None, marginalized=False)


class TestNumpyroModelWithExtensions:
    def test_jitter_in_trace(self, rv_data, linear_priors):
        """Jitter extension adds a sample site for jitter."""
        nonlinear_priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            "jitter": QD(dist.HalfNormal(1.0), "km/s"),
        }
        model = RVModel(extensions=(Jitter(obs_unit="km/s"),))
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert "jitter" in trace
        assert trace["jitter"]["type"] == "sample"
        assert jnp.isfinite(_get_factor_value(trace, "ln_lik"))

    def test_trend_in_trace(self, rv_data):
        """Trend extension adds linear param; marginalized model traces OK."""
        nonlinear_priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        }
        linear_priors = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "trend_1": dist.Normal(0.0, 1.0),
        }
        model = RVModel(extensions=(MonomialTrend(order=1, time_unit="day"),))
        model_fn = model.numpyro_model(
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert jnp.isfinite(_get_factor_value(trace, "ln_lik"))


class TestNumpyroModelUnitConversion:
    def test_period_unit_conversion(self, rv_data, linear_priors):
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
            nonlinear_priors, rv_data, linear_priors, marginalized=True
        )

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(model_fn).get_trace()

        assert jnp.isfinite(_get_factor_value(trace, "ln_lik"))


class TestNumpyroModelFullExplicitLinearSites:
    """Linear priors the joint ``_linear`` MVN cannot represent.

    That MVN is a single untruncated Gaussian, so a truncated prior, a callable
    that returns one, and a ``Delta`` each need their own treatment (see
    ``docs/spec.md`` -> Scope, and -> Linear prior classification).
    """

    def test_truncated_linear_prior_gets_its_own_site(
        self, rv_data, nonlinear_priors, linear_priors
    ):
        priors = {
            **linear_priors,
            "rv_semiamp": QD(dist.TruncatedNormal(0.0, 30.0, low=0.0), "km/s"),
        }
        model_fn = RVModel().numpyro_model(
            nonlinear_priors, rv_data, priors, marginalized=False
        )
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert trace["rv_semiamp"]["type"] == "sample"
        assert float(trace["rv_semiamp"]["value"]) >= 0.0
        assert trace["v_sys"]["type"] == "sample"

    def test_callable_resolving_to_a_truncated_prior_respects_its_support(
        self, rv_data, nonlinear_priors, linear_priors
    ):
        """The signed-SB2 case: a callable resolving to a truncated Normal."""
        priors = {
            **linear_priors,
            "rv_semiamp": PeriodDependentKPrior(
                Q(30.0, "km/s"), Q(1.0, "yr"), support="positive"
            ),
        }
        model_fn = RVModel().numpyro_model(
            nonlinear_priors, rv_data, priors, marginalized=False
        )
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert trace["rv_semiamp"]["type"] == "sample"
        assert float(trace["rv_semiamp"]["value"]) >= 0.0

    def test_callable_without_a_support_is_unconstrained(
        self, rv_data, nonlinear_priors, linear_priors
    ):
        """No declaration means unconstrained, so the site spans the real line."""
        priors = {
            **linear_priors,
            "rv_semiamp": PeriodDependentKPrior(Q(30.0, "km/s"), Q(1.0, "yr")),
        }
        model_fn = RVModel().numpyro_model(
            nonlinear_priors, rv_data, priors, marginalized=False
        )
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert trace["rv_semiamp"]["type"] == "sample"
        assert isinstance(trace["rv_semiamp"]["fn"], dist.Normal)

    def test_delta_linear_prior_is_deterministic_not_a_sample_site(
        self, rv_data, nonlinear_priors, linear_priors
    ):
        """A fixed value must not become a NUTS site whose log-prob is -inf.

        Sampling ``Delta`` would give stuck chains with no error at all -- the
        classification table calls this prior *Fixed*, not *sampled*.
        """
        priors = {**linear_priors, "v_sys": QD(dist.Delta(3.0), "km/s")}
        model_fn = RVModel().numpyro_model(
            nonlinear_priors, rv_data, priors, marginalized=False
        )
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert trace["v_sys"]["type"] == "deterministic"
        assert float(trace["v_sys"]["value"]) == pytest.approx(3.0)
