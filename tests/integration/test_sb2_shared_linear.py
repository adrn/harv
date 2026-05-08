"""Integration tests for shared linear parameters in JointModel (SB2 use-case).

Tests cover:
- Construction validation (Step 1)
- Numerical correctness of joint marginalization (Step 3/4)
- sample_conditional_linear return shape (Step 5)
- Rejection-sampler flattening (Step 6)
- Numpyro shared explicit linear (Step 7)
- for_sb2 factory (Step 2)
"""

import re

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from unxt import Q

import harv
from harv.data import RVData, SystemData
from harv.distributions import QuantityDistribution as QD
from harv.models import JointModel, RVModel
from harv.samplers import RejectionPrior, RejectionSampler
from harv.samplers.custom_priors import PeriodDependentKPrior

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_LINEAR_PRIOR = {
    "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
    "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
}

_NL_VALUES = {
    "period": Q(100.0, "day"),
    "eccentricity": jnp.float64(0.3),
    "phase_peri": jnp.float64(0.1),
    "arg_peri": Q(1.0, "rad"),
}


def _make_rv_data(seed: int, n: int = 6) -> RVData:
    key = jax.random.PRNGKey(seed)
    t = jnp.linspace(0.0, 200.0, n)
    rv = jax.random.normal(key, (n,)) * 0.5 + 2.0
    err = jnp.full(n, 0.5)
    return RVData(
        time=Q(t, "day"),
        rv=Q(rv, "km/s"),
        rv_err=Q(err, "km/s"),
    )


@pytest.fixture
def rv_data_primary() -> RVData:
    return _make_rv_data(0)


@pytest.fixture
def rv_data_secondary() -> RVData:
    return _make_rv_data(1)


@pytest.fixture
def system_data(rv_data_primary, rv_data_secondary) -> SystemData:
    return SystemData(primary=rv_data_primary, secondary=rv_data_secondary)


@pytest.fixture
def sb2_prior() -> RejectionPrior:
    return RejectionPrior.default_sb2(
        period_min=Q(10.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


# ---------------------------------------------------------------------------
# Test 1: Construction validation
# ---------------------------------------------------------------------------


class TestSharedLinearValidation:
    def test_shared_linear_must_be_in_every_component(
        self, rv_data_primary, rv_data_secondary
    ):
        """A shared name missing from a component's linear_prior raises."""
        primary = RVModel(
            data=rv_data_primary,
            linear_prior={
                "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            },
        )
        secondary = RVModel(
            data=rv_data_secondary,
            linear_prior={"rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s")},
            # v_sys intentionally absent
        )
        with pytest.raises(ValueError, match="v_sys"):
            JointModel(
                components={"primary": primary, "secondary": secondary},
                shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
                shared_linear_params=("v_sys",),
            )

    def test_shared_linear_priors_must_match(self, rv_data_primary, rv_data_secondary):
        """Differing priors for a shared linear param raises."""
        primary = RVModel(
            data=rv_data_primary,
            linear_prior={
                "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            },
        )
        secondary = RVModel(
            data=rv_data_secondary,
            linear_prior={
                "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 5.0), "km/s"),  # different sigma
            },
        )
        with pytest.raises(ValueError, match="v_sys"):
            JointModel(
                components={"primary": primary, "secondary": secondary},
                shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
                shared_linear_params=("v_sys",),
            )

    def test_shared_callable_priors_must_match(
        self, rv_data_primary, rv_data_secondary
    ):
        """Differing eqx.Module callable priors must be detected.

        Regression test: a previous ``__dict__``-based fallback in
        ``_priors_equal`` silently returned True for any two
        ``__slots__``-using ``eqx.Module`` priors and mishandled array-shaped
        Quantity fields.  Validation must now reject mismatched callable
        priors and accept matched ones.
        """
        # Mismatched sigma_K0 -> must raise.
        primary = RVModel(
            data=rv_data_primary,
            linear_prior={
                "rv_semiamp": PeriodDependentKPrior(
                    sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr")
                ),
            },
        )
        secondary = RVModel(
            data=rv_data_secondary,
            linear_prior={
                "rv_semiamp": PeriodDependentKPrior(
                    sigma_K0=Q(99.0, "km/s"), P0=Q(1.0, "yr")
                ),
            },
        )
        with pytest.raises(ValueError, match="rv_semiamp"):
            JointModel(
                components={"primary": primary, "secondary": secondary},
                shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
                shared_linear_params=("rv_semiamp",),
            )

        # Matched eqx.Module priors -> must construct without error.
        secondary_match = RVModel(
            data=rv_data_secondary,
            linear_prior={
                "rv_semiamp": PeriodDependentKPrior(
                    sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr")
                ),
            },
        )
        joint = JointModel(
            components={"primary": primary, "secondary": secondary_match},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("rv_semiamp",),
        )
        assert joint.shared_linear_params == ("rv_semiamp",)

    def test_shared_linear_collision_with_shared_params_raises(
        self, rv_data_primary, rv_data_secondary
    ):
        """A name in both shared_params and shared_linear_params raises."""
        primary = RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR)
        secondary = RVModel(data=rv_data_secondary, linear_prior=_LINEAR_PRIOR)
        with pytest.raises(ValueError, match="cannot be both"):
            JointModel(
                components={"primary": primary, "secondary": secondary},
                shared_params=(
                    "period",
                    "eccentricity",
                    "phase_peri",
                    "arg_peri",
                    "v_sys",
                ),
                shared_linear_params=("v_sys",),
            )


# ---------------------------------------------------------------------------
# Test 2: Numerical correctness
# ---------------------------------------------------------------------------


class TestJointMargLogProb:
    def test_joint_marg_logprob_is_finite(self, rv_data_primary, rv_data_secondary):
        """Joint log_prob with shared v_sys is finite."""
        joint = JointModel(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR),
                "secondary": RVModel(
                    data=rv_data_secondary, linear_prior=_LINEAR_PRIOR
                ),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        lp = joint.log_prob(_NL_VALUES)
        assert jnp.isfinite(lp)

    def test_joint_marg_logprob_differs_from_summed(
        self, rv_data_primary, rv_data_secondary
    ):
        """Shared-v_sys joint log_prob differs from naive per-component sum."""
        components = {
            "primary": RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR),
            "secondary": RVModel(data=rv_data_secondary, linear_prior=_LINEAR_PRIOR),
        }
        joint_shared = JointModel(
            components=components,
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        joint_unshared = JointModel(
            components=components,
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=(),
        )
        lp_shared = joint_shared.log_prob(_NL_VALUES)
        lp_unshared = joint_unshared.log_prob(_NL_VALUES)
        assert jnp.isfinite(lp_shared)
        assert jnp.isfinite(lp_unshared)
        assert not jnp.isclose(lp_shared, lp_unshared, atol=1e-3)

    def test_joint_marg_matches_manual_block_integral(
        self, rv_data_primary, rv_data_secondary
    ):
        """Shared-v_sys log_prob matches a hand-coded joint Gaussian integral."""

        v_sys_prior = QD(dist.Normal(0.0, 10.0), "km/s")
        k_prior = QD(dist.Normal(5.0, 5.0), "km/s")
        linear_prior = {"rv_semiamp": k_prior, "v_sys": v_sys_prior}

        primary = RVModel(data=rv_data_primary, linear_prior=linear_prior)
        secondary = RVModel(data=rv_data_secondary, linear_prior=linear_prior)
        joint = JointModel(
            components={"primary": primary, "secondary": secondary},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        lp_joint = joint.log_prob(_NL_VALUES)

        # Hand-build the joint MarginalizedLinear directly.
        # Get per-component building blocks.
        shared_nl = joint._shared_param_names()
        per_comp_nl = joint._per_component_nonlinear_names()
        comp_nl = harv.models.joint._split_nl_values(
            _NL_VALUES, shared_nl, joint.component_names, per_comp_nl
        )
        per_comp_marg = {
            comp_name: comp._auto_marginalized_names()
            for comp_name, comp in joint.components.items()
        }
        joint._route_explicit_linear(_NL_VALUES, comp_nl, per_comp_marg)

        marg_dist_manual, y_joint_manual, _, _ = joint._build_joint_marginalized_linear(
            comp_nl, per_comp_marg
        )
        lp_manual = marg_dist_manual.log_prob(y_joint_manual)

        assert jnp.allclose(lp_joint, lp_manual, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: sample_conditional_linear shape
# ---------------------------------------------------------------------------


class TestSampleConditionalLinearShape:
    def test_shared_at_top_level(self, rv_data_primary, rv_data_secondary):
        """v_sys appears at top of dict, rv_semiamp in component sub-dicts."""
        joint = JointModel(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR),
                "secondary": RVModel(
                    data=rv_data_secondary, linear_prior=_LINEAR_PRIOR
                ),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        key = jax.random.PRNGKey(0)
        samples = joint.sample_conditional_linear(_NL_VALUES, key)

        # Shared v_sys at top level
        assert "v_sys" in samples
        assert not isinstance(samples["v_sys"], dict)

        # Per-component rv_semiamp in sub-dicts
        assert "primary" in samples
        assert "rv_semiamp" in samples["primary"]
        assert "v_sys" not in samples["primary"]

        assert "secondary" in samples
        assert "rv_semiamp" in samples["secondary"]
        assert "v_sys" not in samples["secondary"]

        # All values finite
        assert jnp.isfinite(samples["v_sys"]).all()
        assert jnp.isfinite(samples["primary"]["rv_semiamp"]).all()

    def test_per_component_path_unchanged(self, rv_data_primary, rv_data_secondary):
        """With shared_linear_params=(), result is per-component sub-dicts only."""
        joint = JointModel(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR),
                "secondary": RVModel(
                    data=rv_data_secondary, linear_prior=_LINEAR_PRIOR
                ),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=(),
        )
        key = jax.random.PRNGKey(0)
        samples = joint.sample_conditional_linear(_NL_VALUES, key)

        assert "primary" in samples
        assert "secondary" in samples
        assert "v_sys" in samples["primary"]
        assert "v_sys" in samples["secondary"]


# ---------------------------------------------------------------------------
# Test 5: Rejection-sampler flattening
# ---------------------------------------------------------------------------


class TestRejectionSamplerFlattening:
    def test_samples_v_sys_unnamespaced(self, rv_data_primary, rv_data_secondary):
        """After RejectionSampler on SB2 joint model, samples['v_sys'] is bare."""
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
        }
        joint = JointModel(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
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
        sampler = RejectionSampler(prior, joint)
        samples = sampler.run(seed=0, n_prior_samples=1_000, max_posterior_samples=4)

        # v_sys must appear as a bare key (shared)
        assert "v_sys" in samples.linear
        # rv_semiamp collides across components — must be namespaced
        assert "primary.rv_semiamp" in samples.linear
        assert "secondary.rv_semiamp" in samples.linear
        # Namespaced v_sys keys must NOT appear
        assert "primary.v_sys" not in samples.linear
        assert "secondary.v_sys" not in samples.linear


# ---------------------------------------------------------------------------
# Test 6: Numpyro shared explicit linear (sampled once)
# ---------------------------------------------------------------------------


class TestNumpyroSharedExplicitLinear:
    def test_shared_explicit_sampled_once(self, rv_data_primary, rv_data_secondary):
        """Shared explicit linear param appears exactly once in numpyro trace."""
        # Use a non-Gaussian prior so v_sys is classified as explicit.
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.HalfNormal(10.0), "km/s"),
        }
        joint = JointModel(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=linear_prior),
                "secondary": RVModel(data=rv_data_secondary, linear_prior=linear_prior),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        nonlinear_priors = {
            "period": QD(dist.Uniform(10.0, 500.0), "day"),
            "eccentricity": dist.Uniform(0.0, 0.9),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        }
        model_fn = joint.numpyro_model(nonlinear_priors, marginalized=True)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        # "v_sys" must appear exactly once (shared explicit site).
        v_sys_sites = [k for k in trace if k == "v_sys"]
        assert len(v_sys_sites) == 1, f"Expected 1 site for v_sys, got: {v_sys_sites}"


# ---------------------------------------------------------------------------
# Test 7: for_sb2 factory
# ---------------------------------------------------------------------------


class TestForSb2Factory:
    def test_builds_correct_models(self, system_data, sb2_prior):
        """Factory routes shared and per-component linear priors correctly."""
        joint = JointModel.for_sb2(data=system_data, prior=sb2_prior)

        assert joint.shared_linear_params == ("v_sys",)
        assert (
            joint.components["primary"].linear_prior["rv_semiamp"]
            is sb2_prior.linear_prior["primary.rv_semiamp"]
        )
        assert (
            joint.components["secondary"].linear_prior["rv_semiamp"]
            is sb2_prior.linear_prior["secondary.rv_semiamp"]
        )
        assert (
            joint.components["primary"].linear_prior["v_sys"]
            is sb2_prior.linear_prior["v_sys"]
        )

    def test_extensions_tuple_applied_to_all(self, system_data, sb2_prior):
        """extensions as tuple is applied to all components."""
        jitter = harv.Jitter("km/s")
        joint = JointModel.for_sb2(
            data=system_data, prior=sb2_prior, extensions=(jitter,)
        )
        for comp in joint.components.values():
            assert any(isinstance(ext, harv.Jitter) for ext in comp.extensions)

    def test_extensions_dict_per_component(self, system_data, sb2_prior):
        """extensions as dict applies per-component extensions."""
        jitter = harv.Jitter("km/s")
        joint = JointModel.for_sb2(
            data=system_data,
            prior=sb2_prior,
            extensions={"primary": (jitter,)},
        )
        assert any(
            isinstance(e, harv.Jitter) for e in joint.components["primary"].extensions
        )
        assert len(joint.components["secondary"].extensions) == 0

    def test_wrong_component_count_raises(self, rv_data_primary, sb2_prior):
        """Factory raises when data has != 2 components."""
        data_single = SystemData(only=rv_data_primary)
        with pytest.raises(ValueError, match="2 components"):
            JointModel.for_sb2(data=data_single, prior=sb2_prior)

    def test_missing_sb2_keys_raises(self, system_data):
        """Factory raises when prior lacks component-qualified semi-amplitudes."""
        bad_prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.Uniform(10.0, 500.0), "day"),
                "eccentricity": dist.Uniform(0.0, 0.9),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            },
        )
        with pytest.raises(
            ValueError, match=re.escape("prior.linear_prior is missing SB2 keys")
        ):
            JointModel.for_sb2(data=system_data, prior=bad_prior)

    def test_shared_linear_name_must_be_bare(self, system_data):
        """Shared linear params must not use component-qualified keys."""
        bad_prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.Uniform(10.0, 500.0), "day"),
                "eccentricity": dist.Uniform(0.0, 0.9),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "primary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "secondary.rv_semiamp": QD(dist.Normal(4.0, 5.0), "km/s"),
                "primary.v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            },
        )

        with pytest.raises(ValueError, match="is shared and must not be prefixed"):
            JointModel.for_sb2(
                data=system_data,
                prior=bad_prior,
                shared_linear_params=("v_sys",),
            )

    def test_non_shared_linear_name_must_be_qualified(self, system_data):
        """Non-shared linear params must use component-qualified keys."""
        bad_prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.Uniform(10.0, 500.0), "day"),
                "eccentricity": dist.Uniform(0.0, 0.9),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "primary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "secondary.rv_semiamp": QD(dist.Normal(4.0, 5.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            },
        )

        with pytest.raises(ValueError, match="is not shared and must be qualified"):
            JointModel.for_sb2(
                data=system_data,
                prior=bad_prior,
                shared_linear_params=(),
            )

    def test_shared_nonlinear_name_must_be_bare(self, system_data):
        """Shared nonlinear params must not use component-qualified keys."""
        bad_prior = RejectionPrior(
            nonlinear_priors={
                "primary.period": QD(dist.Uniform(10.0, 500.0), "day"),
                "eccentricity": dist.Uniform(0.0, 0.9),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "primary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "secondary.rv_semiamp": QD(dist.Normal(4.0, 5.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            },
        )

        with pytest.raises(ValueError, match="is shared and must not be prefixed"):
            JointModel.for_sb2(data=system_data, prior=bad_prior)

    def test_non_shared_nonlinear_name_must_be_qualified(self, system_data):
        """Non-shared nonlinear params must use component-qualified keys."""
        bad_prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.Uniform(10.0, 500.0), "day"),
                "primary.eccentricity": dist.Uniform(0.0, 0.9),
                "primary.phase_peri": dist.Uniform(0.0, 1.0),
                "primary.arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
                "secondary.eccentricity": dist.Uniform(0.0, 0.9),
                "secondary.phase_peri": dist.Uniform(0.0, 1.0),
                "secondary.arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "primary.rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
                "secondary.rv_semiamp": QD(dist.Normal(4.0, 5.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            },
        )

        with pytest.raises(ValueError, match="is not shared and must be qualified"):
            JointModel.for_sb2(
                data=system_data,
                prior=bad_prior,
                shared_params=(
                    "eccentricity",
                    "phase_peri",
                    "arg_peri",
                ),
            )

    def test_log_prob_is_finite(self, system_data, sb2_prior):
        """Factory-built joint model produces finite log_prob."""
        joint = JointModel.for_sb2(data=system_data, prior=sb2_prior)
        lp = joint.log_prob(_NL_VALUES)
        assert jnp.isfinite(lp)

    def test_jit_compatible(self, system_data, sb2_prior):
        """for_sb2 model is JIT-compatible."""
        joint = JointModel.for_sb2(data=system_data, prior=sb2_prior)

        @jax.jit
        def _lp(period):
            return joint.log_prob({**_NL_VALUES, "period": Q(period, "day")})

        result = _lp(100.0)
        assert jnp.isfinite(result)


# ---------------------------------------------------------------------------
# Test 9: Regression — existing per-component path
# ---------------------------------------------------------------------------


class TestPerComponentPathRegression:
    """Existing behavior is preserved when shared_linear_params=()."""

    def test_log_prob_equals_sum_when_no_shared_lin(
        self, rv_data_primary, rv_data_secondary
    ):
        """With shared_linear_params=(), joint log_prob equals sum of per-component."""
        model_p = RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR)
        model_s = RVModel(data=rv_data_secondary, linear_prior=_LINEAR_PRIOR)

        joint = JointModel(
            components={"primary": model_p, "secondary": model_s},
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=(),
        )

        lp_joint = joint.log_prob(_NL_VALUES)
        lp_p = model_p.log_prob(_NL_VALUES)
        lp_s = model_s.log_prob(_NL_VALUES)

        assert jnp.allclose(lp_joint, lp_p + lp_s, atol=1e-5)

    def test_joint_model_stores_shared_linear_params(
        self, rv_data_primary, rv_data_secondary
    ):
        """JointModel stores shared_linear_params correctly."""
        joint = JointModel(
            components={
                "primary": RVModel(data=rv_data_primary, linear_prior=_LINEAR_PRIOR),
                "secondary": RVModel(
                    data=rv_data_secondary, linear_prior=_LINEAR_PRIOR
                ),
            },
            shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
            shared_linear_params=("v_sys",),
        )
        assert joint.shared_linear_params == ("v_sys",)
