"""Integration tests for multi-survey RV (C1+C2).

These tests validate the end-to-end multi-instrument RV path: data simulation,
likelihood building, and rejection sampling.  They intentionally use low-SNR
data (K ~ sigma_rv) so that rejection sampling is efficient enough to accept samples.
"""

import jax
import numpy as np
import numpyro.distributions as dist
import pytest
import quaxed.numpy as jnp
from unxt import Q, uconvert

from harv.data import RVData, build_indicator_matrix
from harv.distributions import QD
from harv.likelihood.params import RVParameters
from harv.likelihood.rv import RVLikelihood
from harv.model import Model
from harv.samplers.rejection import RejectionSampler
from harv.samplers.rejection_prior import RejectionPrior
from harv.simulate.rv import simulate_rv_multisurv_data


class TestMultiSurveyLikelihood:
    """Unit-style tests for RVLikelihood."""

    def test_log_prob_finite(self):
        """Likelihood returns a finite scalar at arbitrary parameters."""
        source_data, _ = simulate_rv_multisurv_data(
            instruments={"keck": None, "espresso": Q(2.0, "km/s")},
            seed=1,
            n_obs_per_instrument=20,
            period=Q(50.0, "day"),
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(3.0, "km/s"),
        )
        rv_datasets = source_data.get_datasets_by_type(RVData)
        stacked, indicator, instrument_names = build_indicator_matrix(
            rv_datasets, reference="keck"
        )

        linear_prior = {
            "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 100.0), "km/s"),
        }
        offsets_prior = {
            "espresso": QD(dist.Normal(0.0, 5.0), "km/s"),
        }
        lik = RVLikelihood(
            data=stacked,
            linear_marginalized_prior=linear_prior,
            offsets_marginalized_prior=offsets_prior,
            indicator_matrix=indicator,
            instrument_names=instrument_names,
        )
        params = RVParameters.marginalized(
            period=Q(50.0, "day"),
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
            instruments={"keck": None, "espresso": Q(10.0, "km/s")},
            seed=2,
            n_obs_per_instrument=30,
            period=Q(100.0, "day"),
            eccentricity=0.0,
            rv_semiamp=Q(3.0, "km/s"),
            rv_err=Q(2.0, "km/s"),
        )
        rv_datasets = source_data.get_datasets_by_type(RVData)
        stacked, indicator, instrument_names = build_indicator_matrix(
            rv_datasets, reference="keck"
        )

        # Multi-survey: prior on rv_semiamp, v_sys, + espresso offset
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 100.0), "km/s"),
        }
        offsets_prior = {
            "espresso": QD(dist.Normal(0.0, 15.0), "km/s"),
        }
        lik_multi = RVLikelihood(
            data=stacked,
            linear_marginalized_prior=linear_prior,
            offsets_marginalized_prior=offsets_prior,
            indicator_matrix=indicator,
            instrument_names=instrument_names,
        )

        # Single-instrument: prior on rv_semiamp, v_sys only, ignoring offset
        lik_single = RVLikelihood(
            data=stacked,
            linear_marginalized_prior=linear_prior,
        )

        params = RVParameters.marginalized(
            period=Q(100.0, "day"),
            eccentricity=0.0,
            phase_peri=0.5,
            arg_peri=0.0,
        )
        log_lik_multi = lik_multi.log_prob(params)
        log_lik_single = lik_single.log_prob(params)

        # Multi-survey model has more flexibility to fit the offset -> higher log-lik
        assert log_lik_multi > log_lik_single

    def test_vmap_batch(self):
        """Vmap over a batch of parameter samples works correctly."""
        source_data, _ = simulate_rv_multisurv_data(
            instruments={"keck": None, "hires": Q(1.0, "km/s")},
            seed=3,
            n_obs_per_instrument=15,
            period=Q(30.0, "day"),
            rv_semiamp=Q(4.0, "km/s"),
            rv_err=Q(2.0, "km/s"),
        )
        rv_datasets = source_data.get_datasets_by_type(RVData)
        stacked, indicator, instrument_names = build_indicator_matrix(
            rv_datasets, reference="keck"
        )

        linear_prior = {
            "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 100.0), "km/s"),
        }
        offsets_prior = {
            "hires": QD(dist.Normal(0.0, 5.0), "km/s"),
        }
        lik = RVLikelihood(
            data=stacked,
            linear_marginalized_prior=linear_prior,
            offsets_marginalized_prior=offsets_prior,
            indicator_matrix=indicator,
            instrument_names=instrument_names,
        )

        n = 8
        params_batch = RVParameters.marginalized(
            period=Q(jnp.ones(n) * 30.0, "day"),
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
            instruments={"keck": None, "harps": Q(2.0, "km/s")},
            seed=7,
            n_obs_per_instrument=20,
            period=Q(80.0, "day"),
            eccentricity=0.1,
            rv_semiamp=Q(5.0, "km/s"),
            v_sys=Q(0.0, "km/s"),
            rv_err=Q(1.0, "km/s"),
        )
        return source_data, true

    def test_sampler_runs_and_returns_samples(self, low_snr_data):
        """Rejection sampler completes and returns a valid Samples object."""
        source_data, truth = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=Q(40.0, "day"),
            period_max=Q(160.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
            offsets={
                "keck": None,
                "harps": QD(dist.Normal(0.0, 5.0), "km/s"),
            },
        )
        sampler = RejectionSampler(Model(prior, source_data))
        samples = sampler.run(n_prior_samples=500_000, seed=10)

        period_samples = uconvert("day", samples["period"])
        period_true = uconvert("day", truth["period"])
        assert jnp.all(jnp.abs(period_samples - period_true).value < 2.0)

        offset_samples = uconvert("km/s", samples["harps"])
        offset_true = uconvert("km/s", truth["offset_harps"])
        assert jnp.all(jnp.abs(offset_samples - offset_true).value < 1.0)

        assert jnp.all(jnp.abs(samples["eccentricity"] - truth["eccentricity"]) < 0.1)

        K_samples = jnp.abs(uconvert("km/s", samples["rv_semiamp"]))
        K_true = uconvert("km/s", truth["rv_semiamp"])
        assert jnp.all(jnp.abs(K_samples - K_true).value < 1.0)

        assert samples.n_samples > 0
        assert samples.data_type == "rv"

    def test_samples_have_correct_keys(self, low_snr_data):
        """Samples object has all expected parameter keys, including offset."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=Q(40.0, "day"),
            period_max=Q(160.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
            offsets={
                "keck": None,
                "harps": QD(dist.Normal(0.0, 5.0), "km/s"),
            },
        )
        sampler = RejectionSampler(Model(prior, source_data))
        samples = sampler.run(n_prior_samples=50_000, seed=11)

        keys = samples.keys()
        for nonlinear_key in (
            "period",
            "log_period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        ):
            assert nonlinear_key in keys, f"Missing key: {nonlinear_key}"
        for linear_key in ("rv_semiamp", "v_sys"):
            assert linear_key in keys, f"Missing key: {linear_key}"
        assert "harps" in keys, "Missing offset key: harps"

    def test_offset_key_absent_for_reference_instrument(self, low_snr_data):
        """The reference instrument (keck, offset=None) has no offset key."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=Q(40.0, "day"),
            period_max=Q(160.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
            offsets={
                "keck": None,
                "harps": QD(dist.Normal(0.0, 5.0), "km/s"),
            },
        )
        sampler = RejectionSampler(Model(prior, source_data))
        samples = sampler.run(n_prior_samples=50_000, seed=12)
        assert "keck" not in samples.keys()  # noqa: SIM118

    def test_reproducibility(self, low_snr_data):
        """Same seed produces identical samples."""
        source_data, _ = low_snr_data
        prior = RejectionPrior.default_rv(
            period_min=Q(40.0, "day"),
            period_max=Q(160.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
            offsets={
                "keck": None,
                "harps": QD(dist.Normal(0.0, 5.0), "km/s"),
            },
        )
        sampler = RejectionSampler(Model(prior, source_data))
        s1 = sampler.run(n_prior_samples=20_000, seed=20)
        s2 = sampler.run(n_prior_samples=20_000, seed=20)

        assert s1.n_samples == s2.n_samples
        if s1.n_samples > 0:
            np.testing.assert_array_equal(s1["period"].value, s2["period"].value)
