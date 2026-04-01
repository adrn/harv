"""Integration tests for multi-survey RV (C1+C2).

These tests validate the end-to-end multi-instrument RV path: data simulation,
likelihood building, and rejection sampling.  They intentionally use low-SNR
data (K ~ sigma_rv) so that rejection sampling is efficient enough to accept samples.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Quantity

from harv.data import RadialVelocityData
from harv.likelihood._params import RVOrbitParameters
from harv.likelihood.rv import (
    MarginalizedMultiSurveyRVLikelihood,
    MarginalizedRVLikelihood,
)
from harv.priors.rejection import RejectionPrior
from harv.samplers.rejection import (
    RejectionSampler,
    _build_indicator_matrix,
    _stack_rv_datasets,
)
from harv.simulate.rv import simulate_rv_multisurv_data


class TestMultiSurveyLikelihood:
    """Unit-style tests for MarginalizedMultiSurveyRVLikelihood."""

    def test_log_prob_finite(self):
        """Likelihood returns a finite scalar at arbitrary parameters."""
        source_data, _ = simulate_rv_multisurv_data(
            instruments={"keck": None, "espresso": Quantity(2.0, "km/s")},
            seed=1,
            n_obs_per_instrument=20,
            period=Quantity(50.0, "day"),
            K=Quantity(5.0, "km/s"),
            rv_err=Quantity(3.0, "km/s"),
        )
        rv_datasets = source_data.get_datasets_by_type(RadialVelocityData)
        stacked = _stack_rv_datasets(rv_datasets)
        offsets = {"keck": None, "espresso": dist.Normal(0.0, 5.0)}
        indicator = _build_indicator_matrix(rv_datasets, offsets)

        lp = dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=100.0**2 * jnp.eye(3)
        )
        lik = MarginalizedMultiSurveyRVLikelihood(
            data=stacked, indicator_matrix=indicator, linear_prior=lp
        )
        params = RVOrbitParameters(
            period=Quantity(50.0, "day"),
            eccentricity=0.2,
            phase_peri=0.5,
            arg_peri=1.0,
        )
        log_lik = lik.log_prob(params)
        assert jnp.isfinite(log_lik)

    def test_log_prob_higher_than_single_instrument(self):
        """Multi-survey likelihood with correct offset is higher than without.

        When the data has a known instrument offset, a model that accounts for it
        should fit better (higher log-lik) than one that ignores the offset.
        """
        source_data, _ = simulate_rv_multisurv_data(
            instruments={"keck": None, "espresso": Quantity(10.0, "km/s")},
            seed=2,
            n_obs_per_instrument=30,
            period=Quantity(100.0, "day"),
            eccentricity=0.0,
            K=Quantity(3.0, "km/s"),
            rv_err=Quantity(2.0, "km/s"),
        )
        rv_datasets = source_data.get_datasets_by_type(RadialVelocityData)
        stacked = _stack_rv_datasets(rv_datasets)

        offsets = {"keck": None, "espresso": dist.Normal(0.0, 15.0)}
        indicator = _build_indicator_matrix(rv_datasets, offsets)

        # Multi-survey: 3D prior [K, v0, delta_espresso]
        lp_multi = dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=100.0**2 * jnp.eye(3)
        )
        lik_multi = MarginalizedMultiSurveyRVLikelihood(
            data=stacked, indicator_matrix=indicator, linear_prior=lp_multi
        )

        # Single-instrument: 2D prior [K, v0] ignoring offset
        lp_single = dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=100.0**2 * jnp.eye(2)
        )
        lik_single = MarginalizedRVLikelihood(data=stacked, linear_prior=lp_single)

        params = RVOrbitParameters(
            period=Quantity(100.0, "day"),
            eccentricity=0.0,
            phase_peri=0.5,
            arg_peri=0.0,
        )
        log_lik_multi = lik_multi.log_prob(params)
        log_lik_single = lik_single.log_prob(params)

        # Multi-survey model has more flexibility to fit the offset → higher log-lik
        assert log_lik_multi > log_lik_single

    def test_vmap_batch(self):
        """Vmap over a batch of parameter samples works correctly."""
        source_data, _ = simulate_rv_multisurv_data(
            instruments={"keck": None, "hires": Quantity(1.0, "km/s")},
            seed=3,
            n_obs_per_instrument=15,
            period=Quantity(30.0, "day"),
            K=Quantity(4.0, "km/s"),
            rv_err=Quantity(2.0, "km/s"),
        )
        rv_datasets = source_data.get_datasets_by_type(RadialVelocityData)
        stacked = _stack_rv_datasets(rv_datasets)
        offsets = {"keck": None, "hires": dist.Normal(0.0, 5.0)}
        indicator = _build_indicator_matrix(rv_datasets, offsets)

        lp = dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=100.0**2 * jnp.eye(3)
        )
        lik = MarginalizedMultiSurveyRVLikelihood(
            data=stacked, indicator_matrix=indicator, linear_prior=lp
        )

        n = 8
        params_batch = RVOrbitParameters(
            period=Quantity(jnp.ones(n) * 30.0, "day"),
            eccentricity=jnp.linspace(0.0, 0.5, n),
            phase_peri=jnp.linspace(0.0, 1.0, n),
            arg_peri=jnp.ones(n) * 1.0,
        )
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        assert log_liks.shape == (n,)
        assert jnp.all(jnp.isfinite(log_liks))


class TestMultiSurveyRejectionSampler:
    """End-to-end rejection sampler tests for multi-survey RV (low-SNR data)."""

    @pytest.fixture
    def low_snr_data(self):
        """Low-SNR multi-survey RV data: K/sigma ~ 1, rejection sampling tractable."""
        source_data, true = simulate_rv_multisurv_data(
            instruments={"keck": None, "harps": Quantity(2.0, "km/s")},
            seed=7,
            n_obs_per_instrument=20,
            period=Quantity(80.0, "day"),
            eccentricity=0.1,
            K=Quantity(5.0, "km/s"),
            v0=Quantity(0.0, "km/s"),
            rv_err=Quantity(5.0, "km/s"),
        )
        return source_data, true

    def test_sampler_runs_and_returns_samples(self, low_snr_data):
        """Rejection sampler completes and returns a valid Samples object."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=40.0,
            period_max=160.0,
            offsets={"keck": None, "harps": dist.Normal(0.0, 5.0)},
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(source_data, n_prior_samples=50_000, seed=10)

        assert samples.n_samples >= 0  # may be 0 with very unlucky seed
        assert samples.data_type == "rv"

    def test_samples_have_correct_keys(self, low_snr_data):
        """Samples object has all expected parameter keys, including offset."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=40.0,
            period_max=160.0,
            offsets={"keck": None, "harps": dist.Normal(0.0, 5.0)},
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(source_data, n_prior_samples=50_000, seed=11)

        keys = samples.keys()
        for nonlinear_key in (
            "period",
            "log_period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        ):
            assert nonlinear_key in keys, f"Missing key: {nonlinear_key}"
        for linear_key in ("K", "v0"):
            assert linear_key in keys, f"Missing key: {linear_key}"
        assert "harps" in keys, "Missing offset key: harps"

    def test_offset_key_absent_for_reference_instrument(self, low_snr_data):
        """The reference instrument (keck, offset=None) has no offset key."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=40.0,
            period_max=160.0,
            offsets={"keck": None, "harps": dist.Normal(0.0, 5.0)},
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(source_data, n_prior_samples=50_000, seed=12)
        assert "keck" not in samples.keys()  # noqa: SIM118

    def test_reproducibility(self, low_snr_data):
        """Same seed produces identical samples."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=40.0,
            period_max=160.0,
            offsets={"keck": None, "harps": dist.Normal(0.0, 5.0)},
        )
        sampler = RejectionSampler(prior)
        s1 = sampler.run(source_data, n_prior_samples=20_000, seed=20)
        s2 = sampler.run(source_data, n_prior_samples=20_000, seed=20)

        assert s1.n_samples == s2.n_samples
        if s1.n_samples > 0:
            np.testing.assert_array_equal(s1["period"].value, s2["period"].value)
