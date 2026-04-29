"""Unit tests for JointModel."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from unxt import Q

from harv.distributions import QuantityDistribution as QD
from harv.extensions import Jitter
from harv.models import JointModel, RVModel
from harv.samplers import RejectionPrior, RejectionSampler

# Re-alias shared fixtures to shorter names used throughout this module.
linear_prior = pytest.fixture(name="linear_prior")(
    lambda rv_linear_prior: rv_linear_prior
)
nl_values = pytest.fixture(name="nl_values")(lambda rv_nl_values: rv_nl_values)


class TestJointModelBasic:
    def test_construction(self, rv_data_primary, rv_data_secondary, linear_prior):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
        )
        assert joint.component_names == ("primary", "secondary")

    def test_shared_params_factory_default(self, rv_data_primary, rv_data_secondary):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary),
                "secondary": RVModel(data=rv_data_secondary),
            },
        )
        assert set(joint.shared_params) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }

    def test_log_prob_is_finite(
        self, rv_data_primary, rv_data_secondary, linear_prior, nl_values
    ):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
        )
        lp = joint.log_prob(nl_values)
        assert jnp.isfinite(lp)

    def test_log_prob_equals_sum(
        self, rv_data_primary, rv_data_secondary, linear_prior, nl_values
    ):
        """Joint log_prob should equal sum of individual component log_probs."""
        model_p = RVModel(data=rv_data_primary, linear_prior=linear_prior)
        model_s = RVModel(data=rv_data_secondary, linear_prior=linear_prior)

        joint = JointModel.for_sb2(
            components={"primary": model_p, "secondary": model_s},
        )

        lp_joint = joint.log_prob(nl_values)
        lp_p = model_p.log_prob(nl_values)
        lp_s = model_s.log_prob(nl_values)

        assert jnp.allclose(lp_joint, lp_p + lp_s, atol=1e-5)


class TestJointModelComponentSpecific:
    def test_per_component_jitter(
        self, rv_data_primary, rv_data_secondary, linear_prior
    ):
        """Each component can have its own jitter via dot-separated key."""
        model_p = RVModel(
            data=rv_data_primary,
            linear_prior=linear_prior,
            extensions=(Jitter(param_unit="km/s"),),
        )
        model_s = RVModel(
            data=rv_data_secondary,
            linear_prior=linear_prior,
            extensions=(Jitter(param_unit="km/s"),),
        )

        joint = JointModel.for_sb2(
            components={"primary": model_p, "secondary": model_s},
        )

        # Per-component jitter using "component.param" convention
        nl_values = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.3),
            "phase_peri": jnp.float32(0.1),
            "arg_peri": Q(1.0, "rad"),
            "primary.jitter": 0.5,
            "secondary.jitter": 0.3,
        }
        lp = joint.log_prob(nl_values)
        assert jnp.isfinite(lp)

    def test_jitter_affects_result(
        self, rv_data_primary, rv_data_secondary, linear_prior
    ):
        """Different jitter values produce different log-likelihoods."""
        model_p = RVModel(
            data=rv_data_primary,
            linear_prior=linear_prior,
            extensions=(Jitter(param_unit="km/s"),),
        )
        model_s = RVModel(
            data=rv_data_secondary,
            linear_prior=linear_prior,
            extensions=(Jitter(param_unit="km/s"),),
        )
        joint = JointModel.for_sb2(
            components={"primary": model_p, "secondary": model_s},
        )

        base_nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.3),
            "phase_peri": jnp.float32(0.1),
            "arg_peri": Q(1.0, "rad"),
        }

        lp1 = joint.log_prob(
            {**base_nl, "primary.jitter": 0.5, "secondary.jitter": 0.3}
        )
        lp2 = joint.log_prob(
            {**base_nl, "primary.jitter": 2.0, "secondary.jitter": 2.0}
        )
        assert not jnp.allclose(lp1, lp2)


class TestJointModelSampleConditional:
    def test_sample_returns_per_component(
        self, rv_data_primary, rv_data_secondary, linear_prior, nl_values
    ):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
        )
        key = jax.random.PRNGKey(42)
        samples = joint.sample_conditional_linear(nl_values, key)

        assert "primary" in samples
        assert "secondary" in samples
        assert "rv_semiamp" in samples["primary"]
        assert "v_sys" in samples["primary"]
        assert "rv_semiamp" in samples["secondary"]
        assert "v_sys" in samples["secondary"]
        assert all(jnp.isfinite(v) for s in samples.values() for v in s.values())


class TestJointModelNumpyro:
    def test_marginalized_traces(
        self, rv_data_primary, rv_data_secondary, linear_prior
    ):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
        )
        priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        }
        model_fn = joint.numpyro_model(priors, marginalized=True)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert "period" in trace
        assert "eccentricity" in trace
        assert "log_lik" in trace

        # Verify finite log-likelihood
        site = trace["log_lik"]
        log_lik = site["fn"].log_prob(site["value"])
        assert jnp.isfinite(log_lik)

    def test_with_per_component_jitter(
        self, rv_data_primary, rv_data_secondary, linear_prior
    ):
        model_p = RVModel(
            data=rv_data_primary,
            linear_prior=linear_prior,
            extensions=(Jitter(param_unit="km/s"),),
        )
        model_s = RVModel(
            data=rv_data_secondary,
            linear_prior=linear_prior,
            extensions=(Jitter(param_unit="km/s"),),
        )
        joint = JointModel.for_sb2(
            components={"primary": model_p, "secondary": model_s},
        )

        priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            "primary.jitter": QD(dist.HalfNormal(1.0), "km/s"),
            "secondary.jitter": QD(dist.HalfNormal(0.5), "km/s"),
        }
        model_fn = joint.numpyro_model(priors, marginalized=True)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert "primary.jitter" in trace
        assert "secondary.jitter" in trace

    def test_full_model_returns_callable(
        self, rv_data_primary, rv_data_secondary, linear_prior
    ):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
        )
        priors = {"period": dist.Uniform(10.0, 500.0)}
        model_fn = joint.numpyro_model(priors, marginalized=False)
        assert callable(model_fn)


class TestJointModelJit:
    def test_log_prob_jit(self, rv_data_primary, rv_data_secondary, linear_prior):
        joint = JointModel.for_sb2(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
        )

        @jax.jit
        def _lp(period, ecc, phase, arg_peri):
            return joint.log_prob(
                {
                    "period": Q(period, "day"),
                    "eccentricity": ecc,
                    "phase_peri": phase,
                    "arg_peri": Q(arg_peri, "rad"),
                }
            )

        result = _lp(100.0, 0.3, 0.1, 1.0)
        assert jnp.isfinite(result)


class TestSB2RejectionSamplerLinearKeys:
    """SB2 rejection sampler: colliding param names must be namespaced."""

    def test_sb2_linear_keys_are_namespaced(self, rv_data_primary, rv_data_secondary):
        """primary.rv_semiamp and secondary.rv_semiamp must both appear in
        Samples.linear; the old flat merge silently dropped one of them.
        """
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
        }
        model_p = RVModel(data=rv_data_primary, linear_prior=linear_prior)
        model_s = RVModel(data=rv_data_secondary, linear_prior=linear_prior)
        joint = JointModel.for_sb2(
            components={"primary": model_p, "secondary": model_s}
        )
        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.Uniform(10.0, 500.0), "day"),
                "eccentricity": dist.Uniform(0.0, 0.9),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior=linear_prior,
        )
        sampler = RejectionSampler.from_model(model=joint, prior=prior)
        samples = sampler.run(seed=0, n_prior_samples=20)
        assert "primary.rv_semiamp" in samples.linear
        assert "secondary.rv_semiamp" in samples.linear
        assert "primary.v_sys" in samples.linear
        assert "secondary.v_sys" in samples.linear
        # The bare (un-namespaced) keys must NOT appear when there is a collision.
        assert "rv_semiamp" not in samples.linear
        assert "v_sys" not in samples.linear
