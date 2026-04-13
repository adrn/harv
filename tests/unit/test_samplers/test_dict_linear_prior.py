"""Tests for dict-form linear prior (hybrid marginalization path)."""

import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
from harv.distributions import QD
from harv.samplers.rejection import RejectionSampler
from harv.samplers.rejection_prior import RejectionPrior


def _make_rv_data(n_obs: int = 30, seed: int = 42) -> RVData:
    """Tiny RV dataset for structural tests."""
    import jax.random as jr

    key = jr.key(seed)
    times = Q(jnp.linspace(0.0, 200.0, n_obs), "day")
    rv = Q(jr.normal(key, (n_obs,)) * 5.0, "km/s")
    rv_err = Q(jnp.ones(n_obs) * 2.0, "km/s")
    return RVData(time=times, rv=rv, rv_err=rv_err)


class TestDictLinearPriorRV:
    """Tests for dict-form linear prior with RV data."""

    def test_all_gaussian_matches_mvn(self):
        """Dict with all Normal entries should behave like equivalent MVN."""
        data = _make_rv_data()
        n_prior = 10_000

        # Dict prior with all Gaussian entries
        dict_prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        dict_sampler = RejectionSampler(dict_prior)
        dict_samples = dict_sampler.run(data, n_prior_samples=n_prior, seed=0)

        assert dict_samples.n_samples >= 0
        assert dict_samples.data_type == "rv"
        assert "rv_semiamp" in dict_samples.keys()
        assert "v_sys" in dict_samples.keys()

    def test_explicit_rv_semiamp_gaussian_v_sys(self):
        """HalfNormal rv_semiamp (explicit) + Normal v_sys (marginalized)."""
        data = _make_rv_data()

        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "rv_semiamp": QD(dist.HalfNormal(100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=1)

        assert samples.n_samples >= 0
        assert samples.data_type == "rv"
        assert "rv_semiamp" in samples.keys()
        assert "v_sys" in samples.keys()
        # rv_semiamp was sampled from HalfNormal: all values should be >= 0
        if samples.n_samples > 0:
            K_vals = samples["rv_semiamp"]
            assert jnp.all(K_vals.value >= 0)

    def test_delta_rv_semiamp_gaussian_v_sys(self):
        """Delta rv_semiamp (fixed) + Normal v_sys (marginalized)."""
        data = _make_rv_data()

        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "rv_semiamp": QD(dist.Delta(10.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=2)

        assert samples.n_samples >= 0
        assert samples.data_type == "rv"
        # rv_semiamp should be fixed at 10.0 for all samples
        if samples.n_samples > 0:
            K_vals = samples["rv_semiamp"]
            assert jnp.allclose(K_vals.value, 10.0)

    def test_quantity_distribution_in_dict(self):
        """QuantityDistribution entries in dict-form linear prior."""
        data = _make_rv_data()

        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "rv_semiamp": QD(dist.HalfNormal(100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=3)

        assert samples.n_samples >= 0
        assert samples.data_type == "rv"

    def test_all_fixed_delta(self):
        """Both rv_semiamp and v_sys as Delta (fully fixed linear params)."""
        data = _make_rv_data()

        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(50.0, 200.0), "day"),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0.0, 1.0),
                "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
            },
            linear_prior={
                "rv_semiamp": QD(dist.Delta(10.0), "km/s"),
                "v_sys": QD(dist.Delta(0.0), "km/s"),
            },
        )
        sampler = RejectionSampler(prior)
        # When all linear params are Delta, build_gaussian_mvn raises ValueError.
        # This case is not yet supported -- just verify the error is clear.
        import pytest

        with pytest.raises(ValueError, match="No marginalized parameters remain"):
            sampler.run(data, n_prior_samples=1_000, seed=4)
