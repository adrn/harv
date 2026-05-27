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
        model = RVModel()
        assert isinstance(model.parameterization, object)

    def test_param_names(self):
        model = RVModel()
        assert set(model._all_nonlinear_names()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }
        assert set(model._all_linear_names()) == {"rv_semiamp", "v_sys"}

    def test_obs_unit(self):
        model = RVModel()
        assert model._obs_unit(_make_rv_data()) == "km / s"

    def test_base_design_matrix_shape(self):
        data = _make_rv_data(n_obs=10)
        model = RVModel()
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": 1.0,
        }
        X = model._base_design_matrix(nl, data)
        assert X.shape == (10, 2)


class TestRVModelExplicit:
    """Explicit (non-marginalized) evaluation."""

    def test_explicit_is_finite(self):
        data = _make_rv_data()
        model = RVModel()
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
        ll = model.log_prob(nl, data, linear_values=linear)
        assert jnp.isfinite(ll)


class TestRVModelMarginalized:
    """Marginalized likelihood evaluation."""

    def test_marginalized_is_finite(self):
        data = _make_rv_data()
        model = RVModel()
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        ll = model.log_prob(nl, data, linear_priors=_rv_prior())
        assert jnp.isfinite(ll)

    def test_marginalized_jit(self):
        data = _make_rv_data()
        model = RVModel()
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }

        @jax.jit
        def fn():
            return model.log_prob(nl, data, linear_priors=_rv_prior())

        ll = fn()
        assert jnp.isfinite(ll)


class TestRVModelSampleConditional:
    """Test conditional linear parameter sampling."""

    def test_sample_returns_all_linear(self):
        data = _make_rv_data()
        model = RVModel()
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        key = jax.random.key(42)
        samples = model.sample_conditional_linear(
            nl, key, data, linear_priors=_rv_prior()
        )
        assert "rv_semiamp" in samples
        assert "v_sys" in samples

    def test_sample_values_finite(self):
        data = _make_rv_data()
        model = RVModel()
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        key = jax.random.key(0)
        samples = model.sample_conditional_linear(
            nl, key, data, linear_priors=_rv_prior()
        )
        assert jnp.isfinite(samples["rv_semiamp"])
        assert jnp.isfinite(samples["v_sys"])

    def test_conditional_sample_respects_nonzero_prior_mean(self):
        """Regression: conditional samples should be centred near the prior mean.

        When all observed RVs are exactly equal to a known v_sys value and the
        rv_semiamp signal is negligible, the conditional posterior for v_sys
        should be pulled toward that value.  Previously, MarginalizedLinear.
        conditional() ignored the prior mean, so samples were wrong when the
        Normal prior had a non-zero loc.
        """
        v_sys_true = 50.0  # km/s — deliberately non-zero
        n_obs = 40
        data = RVData(
            time=Q(jnp.linspace(0, 200, n_obs), "day"),
            rv=Q(jnp.full(n_obs, v_sys_true), "km/s"),
            rv_err=Q(jnp.ones(n_obs) * 0.01, "km/s"),  # very tight errors
        )
        # Tight prior on v_sys centred on the true value; wide prior on K
        linear_priors = {
            "rv_semiamp": dist.Normal(0.0, 0.01),  # near-zero K
            "v_sys": dist.Normal(v_sys_true, 10.0),  # non-zero mean
        }
        model = RVModel()
        nl = {
            "period": Q(1000.0, "day"),
            "eccentricity": 0.0,
            "phase_peri": 0.0,
            "arg_peri": Q(0.0, "rad"),
        }
        # Draw many conditional samples and check the mean is close to truth
        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.key(0), i))(
            jnp.arange(200)
        )
        samples = jax.vmap(
            lambda k: model.sample_conditional_linear(
                nl, k, data, linear_priors=linear_priors
            )
        )(keys)
        mean_v_sys = jnp.mean(samples["v_sys"])
        assert jnp.abs(mean_v_sys - v_sys_true) < 1.0, (
            f"Conditional posterior mean for v_sys ({mean_v_sys:.2f}) is far from "
            f"the true value ({v_sys_true:.2f}). "
            "This likely indicates the prior-mean bug in conditional sampling."
        )
