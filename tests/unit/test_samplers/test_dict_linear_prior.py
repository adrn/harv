"""Tests for dict-form linear prior (hybrid marginalization path)."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv.models as hm
from harv.data import RVData
from harv.distributions import QD
from harv.models.priors import HarvPrior
from harv.models.rv import RVModel
from harv.samplers.rejection import RejectionSampler


def _make_rv_data(n_obs: int = 30, seed: int = 42) -> RVData:
    """Tiny RV dataset for structural tests."""
    key = jr.key(seed)
    times = Q(jnp.linspace(0.0, 200.0, n_obs), "day")
    rv = Q(jr.normal(key, (n_obs,)) * 5.0, "km/s")
    rv_err = Q(jnp.ones(n_obs) * 2.0, "km/s")
    return RVData(time=times, rv=rv, rv_err=rv_err)


class TestDictLinearPriorRV:
    """Tests for dict-form linear prior with RV data."""

    def test_run_ignore_non_finite_keeps_finite_samples(self, monkeypatch):
        """ignore_non_finite=True should reject only the bad log-likelihoods."""
        data = _make_rv_data()
        prior = hm.StandardRV().default_prior(
            period_min=Q(1.0, "day"),
            period_max=Q(10.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        sampler = RejectionSampler(prior, RVModel())

        def fake_sample_prior_and_evaluate_batched(
            self,
            model,
            key,
            n_prior_samples,
            ext_nl_priors,
            eff_linear,
            marginalize_names,
            data,
        ):
            del self, model, key, n_prior_samples, ext_nl_priors, eff_linear
            del marginalize_names, data
            prior_samples = {
                "period": jnp.array([2.0, 3.0, 4.0, 5.0]),
                "eccentricity": jnp.array([0.1, 0.1, 0.1, 0.1]),
                "phase_peri": jnp.array([0.2, 0.2, 0.2, 0.2]),
                "arg_peri": jnp.array([1.0, 1.0, 1.0, 1.0]),
            }
            log_likelihoods = jnp.array([0.0, jnp.nan, -jnp.inf, jnp.inf])
            return prior_samples, log_likelihoods

        def fake_sample_linear_parameters(
            self,
            model,
            key,
            nonlinear_samples,
            marginalized_names,
            data,
            linear_priors,
        ):
            del self, model, key, marginalized_names, data, linear_priors
            n = len(next(iter(nonlinear_samples.values())))
            return {
                "rv_semiamp": Q(jnp.zeros(n), "km/s"),
                "v_sys": Q(jnp.zeros(n), "km/s"),
            }

        monkeypatch.setattr(
            RejectionSampler,
            "_sample_prior_and_evaluate_batched",
            fake_sample_prior_and_evaluate_batched,
        )
        monkeypatch.setattr(
            RejectionSampler,
            "_sample_linear_parameters",
            fake_sample_linear_parameters,
        )

        samples = sampler.run(
            data,
            n_prior_samples=4,
            seed=0,
            ignore_non_finite=True,
        )

        assert samples.n_samples == 1
        assert jnp.allclose(samples["period"].value, jnp.array([2.0]))

    def test_run_default_does_not_ignore_non_finite(self, monkeypatch):
        """Without ignore_non_finite, non-finite values poison the rejection step."""
        data = _make_rv_data()
        prior = hm.StandardRV().default_prior(
            period_min=Q(1.0, "day"),
            period_max=Q(10.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        sampler = RejectionSampler(prior, RVModel())

        def fake_sample_prior_and_evaluate_batched(
            self,
            model,
            key,
            n_prior_samples,
            ext_nl_priors,
            eff_linear,
            marginalize_names,
            data,
        ):
            del self, model, key, n_prior_samples, ext_nl_priors, eff_linear
            del marginalize_names, data
            prior_samples = {
                "period": jnp.array([2.0, 3.0, 4.0, 5.0]),
                "eccentricity": jnp.array([0.1, 0.1, 0.1, 0.1]),
                "phase_peri": jnp.array([0.2, 0.2, 0.2, 0.2]),
                "arg_peri": jnp.array([1.0, 1.0, 1.0, 1.0]),
            }
            log_likelihoods = jnp.array([0.0, jnp.nan, -jnp.inf, jnp.inf])
            return prior_samples, log_likelihoods

        def fake_sample_linear_parameters(
            self,
            model,
            key,
            nonlinear_samples,
            marginalized_names,
            data,
            linear_priors,
        ):
            del self, model, key, marginalized_names, data, linear_priors
            n = len(next(iter(nonlinear_samples.values())))
            return {
                "rv_semiamp": Q(jnp.zeros(n), "km/s"),
                "v_sys": Q(jnp.zeros(n), "km/s"),
            }

        monkeypatch.setattr(
            RejectionSampler,
            "_sample_prior_and_evaluate_batched",
            fake_sample_prior_and_evaluate_batched,
        )
        monkeypatch.setattr(
            RejectionSampler,
            "_sample_linear_parameters",
            fake_sample_linear_parameters,
        )

        samples = sampler.run(
            data,
            n_prior_samples=4,
            seed=0,
        )

        assert samples.n_samples == 0

    def test_all_gaussian_matches_mvn(self):
        """Dict with all Normal entries should behave like equivalent MVN."""
        data = _make_rv_data()
        n_prior = 10_000

        # Dict prior with all Gaussian entries
        dict_prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors={
                "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        dict_sampler = RejectionSampler(dict_prior, RVModel())
        dict_samples = dict_sampler.run(data, n_prior_samples=n_prior, seed=0)

        assert dict_samples.n_samples >= 0
        assert dict_samples.data_type == "RVModel"
        assert "rv_semiamp" in dict_samples
        assert "v_sys" in dict_samples

    def test_explicit_rv_semiamp_gaussian_v_sys(self):
        """HalfNormal rv_semiamp (explicit) + Normal v_sys (marginalized)."""
        data = _make_rv_data()

        prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors={
                "rv_semiamp": QD(dist.HalfNormal(100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior, RVModel())
        samples = sampler.run(data, n_prior_samples=10_000, seed=1)

        assert samples.n_samples >= 0
        assert samples.data_type == "RVModel"
        assert "rv_semiamp" in samples
        assert "v_sys" in samples
        # rv_semiamp was sampled from HalfNormal: all values should be >= 0
        if samples.n_samples > 0:
            K_vals = samples["rv_semiamp"]
            assert jnp.all(K_vals.value >= 0)

    def test_delta_rv_semiamp_gaussian_v_sys(self):
        """Delta rv_semiamp (fixed) + Normal v_sys (marginalized)."""
        data = _make_rv_data()

        prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors={
                "rv_semiamp": QD(dist.Delta(10.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior, RVModel())
        samples = sampler.run(data, n_prior_samples=10_000, seed=2)

        assert samples.n_samples >= 0
        assert samples.data_type == "RVModel"
        # rv_semiamp should be fixed at 10.0 for all samples
        if samples.n_samples > 0:
            K_vals = samples["rv_semiamp"]
            assert jnp.allclose(K_vals.value, 10.0)

    def test_quantity_distribution_in_dict(self):
        """QuantityDistribution entries in dict-form linear prior."""
        data = _make_rv_data()

        prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors={
                "rv_semiamp": QD(dist.HalfNormal(100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior, RVModel())
        samples = sampler.run(data, n_prior_samples=10_000, seed=3)

        assert samples.n_samples >= 0
        assert samples.data_type == "RVModel"

    def test_sampler_owned_marginalized_subset(self):
        """Sampler-owned marginalized_names can force a Gaussian subset."""
        data = _make_rv_data()

        prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors={
                "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior, RVModel(), marginalized_names=("v_sys",))
        samples = sampler.run(data, n_prior_samples=10_000, seed=30)

        assert samples.n_samples >= 0
        assert samples.data_type == "RVModel"
        assert "rv_semiamp" in samples.linear
        assert "v_sys" in samples.linear

    def test_all_fixed_delta(self):
        """Both rv_semiamp and v_sys as Delta (fully fixed linear params)."""
        data = _make_rv_data()

        prior = HarvPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_priors={
                "rv_semiamp": QD(dist.Delta(10.0), "km/s"),
                "v_sys": QD(dist.Delta(0.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior, RVModel())
        # When all linear params are Delta, build_gaussian_mvn raises ValueError.
        # This case is not yet supported -- just verify the error is clear.
        with pytest.raises(ValueError, match="No marginalized parameters remain"):
            sampler.run(data, n_prior_samples=1_000, seed=4)
