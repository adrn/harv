"""Unit tests for JointModel."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from unxt import Q

from harv.distributions import QuantityDistribution as QD
from harv.models import HarvPrior, JointModel, RVModel
from harv.models.extensions import Jitter
from harv.samplers import RejectionSampler

# Re-alias shared fixtures to shorter names used throughout this module.
linear_priors = pytest.fixture(name="linear_priors")(
    lambda rv_linear_prior: rv_linear_prior
)
nl_values = pytest.fixture(name="nl_values")(lambda rv_nl_values: rv_nl_values)

# Joint model linear priors use qualified keys ("comp.param").
_JOINT_LP_SHARED_VSYS = {
    "primary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
    "secondary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
    "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),  # shared
}

_JOINT_LP_UNSHARED = {
    "primary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
    "secondary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
    "primary.v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
    "secondary.v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
}


class TestJointModelBasic:
    def test_construction(self, rv_data_primary, rv_data_secondary, linear_priors):
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        assert joint.component_names == ("primary", "secondary")

    def test_shared_params_factory_default(self, rv_data_primary, rv_data_secondary):
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=(),
        )
        assert set(joint.shared_params) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }

    def test_log_prob_is_finite(
        self, rv_data_primary, rv_data_secondary, linear_priors, nl_values
    ):
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        lp = joint.log_prob(nl_values, data, linear_priors=linear_priors)
        assert jnp.isfinite(lp)

    def test_log_prob_equals_sum(
        self, rv_data_primary, rv_data_secondary, linear_priors, nl_values
    ):
        """With shared_linear_params=(), joint log_prob equals sum of per-component."""
        model_p = RVModel()
        model_s = RVModel()

        joint = JointModel(
            components={"primary": model_p, "secondary": model_s},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=(),
        )

        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        lp_joint = joint.log_prob(nl_values, data, linear_priors=_JOINT_LP_UNSHARED)
        lp_p = model_p.log_prob(nl_values, rv_data_primary, linear_priors=linear_priors)
        lp_s = model_s.log_prob(
            nl_values, rv_data_secondary, linear_priors=linear_priors
        )

        assert jnp.allclose(lp_joint, lp_p + lp_s, atol=1e-5)


class TestJointModelComponentSpecific:
    def test_per_component_jitter(
        self, rv_data_primary, rv_data_secondary, linear_priors
    ):
        """Each component can have its own jitter via dot-separated key."""
        model_p = RVModel(extensions=(Jitter(param_unit="km/s"),))
        model_s = RVModel(extensions=(Jitter(param_unit="km/s"),))

        joint = JointModel(
            components={"primary": model_p, "secondary": model_s},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
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
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        lp = joint.log_prob(nl_values, data, linear_priors=linear_priors)
        assert jnp.isfinite(lp)

    def test_jitter_affects_result(
        self, rv_data_primary, rv_data_secondary, linear_priors
    ):
        """Different jitter values produce different log-likelihoods."""
        model_p = RVModel(extensions=(Jitter(param_unit="km/s"),))
        model_s = RVModel(extensions=(Jitter(param_unit="km/s"),))
        joint = JointModel(
            components={"primary": model_p, "secondary": model_s},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )

        base_nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.3),
            "phase_peri": jnp.float32(0.1),
            "arg_peri": Q(1.0, "rad"),
        }
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}

        lp1 = joint.log_prob(
            {**base_nl, "primary.jitter": 0.5, "secondary.jitter": 0.3},
            data,
            linear_priors=linear_priors,
        )
        lp2 = joint.log_prob(
            {**base_nl, "primary.jitter": 2.0, "secondary.jitter": 2.0},
            data,
            linear_priors=linear_priors,
        )
        assert not jnp.allclose(lp1, lp2)


class TestJointModelSampleConditional:
    def test_sample_returns_per_component(
        self, rv_data_primary, rv_data_secondary, linear_priors, nl_values
    ):
        """With shared_linear_params=(v_sys,), v_sys at top; rv_semiamp per-comp."""
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        key = jax.random.PRNGKey(42)
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        samples = joint.sample_conditional_linear(
            nl_values, key, data, linear_priors=_JOINT_LP_SHARED_VSYS
        )

        # v_sys is shared — appears at the top level
        assert "v_sys" in samples
        assert not isinstance(samples["v_sys"], dict)
        # rv_semiamp is per-component
        assert "primary" in samples
        assert "secondary" in samples
        assert "rv_semiamp" in samples["primary"]
        assert "v_sys" not in samples["primary"]
        assert "rv_semiamp" in samples["secondary"]
        assert "v_sys" not in samples["secondary"]
        assert jnp.isfinite(samples["v_sys"]).all()
        assert all(
            jnp.isfinite(v).all()
            for s in samples.values()
            if isinstance(s, dict)
            for v in s.values()
        )


class TestJointModelNumpyro:
    def test_marginalized_traces(
        self, rv_data_primary, rv_data_secondary, linear_priors
    ):
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        }
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        model_fn = joint.numpyro_model(priors, data, linear_priors, marginalized=True)

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
        self, rv_data_primary, rv_data_secondary, linear_priors
    ):
        model_p = RVModel(extensions=(Jitter(param_unit="km/s"),))
        model_s = RVModel(extensions=(Jitter(param_unit="km/s"),))
        joint = JointModel(
            components={"primary": model_p, "secondary": model_s},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )

        priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            "primary.jitter": QD(dist.HalfNormal(1.0), "km/s"),
            "secondary.jitter": QD(dist.HalfNormal(0.5), "km/s"),
        }
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        model_fn = joint.numpyro_model(priors, data, linear_priors, marginalized=True)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        assert "primary.jitter" in trace
        assert "secondary.jitter" in trace

    def test_full_model_returns_callable(
        self, rv_data_primary, rv_data_secondary, linear_priors
    ):
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        priors = {"period": dist.Uniform(10.0, 500.0)}
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        model_fn = joint.numpyro_model(priors, data, linear_priors, marginalized=False)
        assert callable(model_fn)

    def test_full_model_traces_with_qualified_per_component_sites(
        self, rv_data_primary, rv_data_secondary, linear_priors
    ):
        """Non-shared linear names appear once per component as qualified sites.

        Regression: when both components have a Gaussian linear prior under
        the same bare name (e.g. ``rv_semiamp``) and that name is *not* in
        ``shared_linear_params``, the full numpyro model used to call
        ``numpyro.deterministic("rv_semiamp", ...)`` twice and crash with a
        duplicate-site assertion.  Qualified site names are required.
        """
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 6.28), "rad"),
        }
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        model_fn = joint.numpyro_model(
            priors, data, _JOINT_LP_SHARED_VSYS, marginalized=False
        )
        seeded = handlers.seed(model_fn, jax.random.PRNGKey(0))
        trace = handlers.trace(seeded).get_trace()

        # Per-component non-shared linear params are qualified.
        assert "primary.rv_semiamp" in trace
        assert "secondary.rv_semiamp" in trace
        # The bare name must NOT appear (would imply silent collision).
        assert "rv_semiamp" not in trace
        # Shared linear params keep bare names.
        assert "v_sys" in trace
        assert "primary.v_sys" not in trace
        assert "secondary.v_sys" not in trace


class TestJointModelJit:
    def test_log_prob_jit(self, rv_data_primary, rv_data_secondary, linear_priors):
        joint = JointModel(
            components={
                "primary": RVModel(),
                "secondary": RVModel(),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        data = {"primary": rv_data_primary, "secondary": rv_data_secondary}

        @jax.jit
        def _lp(period, ecc, phase, arg_peri):
            return joint.log_prob(
                {
                    "period": Q(period, "day"),
                    "eccentricity": ecc,
                    "phase_peri": phase,
                    "arg_peri": Q(arg_peri, "rad"),
                },
                data,
                linear_priors=linear_priors,
            )

        result = _lp(100.0, 0.3, 0.1, 1.0)
        assert jnp.isfinite(result)


class TestSB2RejectionSamplerLinearKeys:
    """SB2 rejection sampler: colliding param names must be namespaced."""

    def test_sb2_linear_keys_are_namespaced(self, rv_data_primary, rv_data_secondary):
        """primary.rv_semiamp and secondary.rv_semiamp must both appear in
        Samples.linear; the old flat merge silently dropped one of them.
        """
        linear_priors = _JOINT_LP_SHARED_VSYS
        model_p = RVModel()
        model_s = RVModel()
        joint = JointModel(
            components={"primary": model_p, "secondary": model_s},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.Uniform(10.0, 500.0), "day"),
                "eccentricity": dist.Uniform(0.0, 0.9),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors=linear_priors,
        )
        sampler = RejectionSampler(prior, joint)
        joint_data = {"primary": rv_data_primary, "secondary": rv_data_secondary}
        samples = sampler.run(joint_data, seed=0, n_prior_samples=20)
        assert "primary.rv_semiamp" in samples.linear
        assert "secondary.rv_semiamp" in samples.linear
        # v_sys is shared: appears bare (not namespaced per component)
        assert "v_sys" in samples.linear
        assert "primary.v_sys" not in samples.linear
        assert "secondary.v_sys" not in samples.linear
        # rv_semiamp appears namespaced (collision)
        assert "rv_semiamp" not in samples.linear
