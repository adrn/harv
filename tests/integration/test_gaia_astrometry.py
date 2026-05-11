"""Integration tests for Gaia astrometry rejection sampling.

These tests validate the end-to-end astrometry path: data simulation,
likelihood building, and rejection sampling with parameter recovery.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q
from unxt import ustrip as _ustrip

from harv.distributions import QD
from harv.models.astrometry import GaiaAstrometryModel
from harv.samplers.rejection import RejectionSampler
from harv.samplers.rejection_prior import RejectionPrior
from harv.simulate.astrometry import simulate_gaia_epoch_astrometry


def _linear_prior():
    return {
        "ra0": QD(dist.Normal(0.0, 1e3), "mas"),
        "dec0": QD(dist.Normal(0.0, 1e3), "mas"),
        "pmra": QD(dist.Normal(0.0, 1e3), "mas/yr"),
        "pmdec": QD(dist.Normal(0.0, 1e3), "mas/yr"),
        "parallax": QD(dist.Normal(0.0, 1e3), "mas"),
        "semi_major_axis": QD(dist.Normal(0.0, 1e3), "mas"),
    }


class TestGaiaAstrometryModel:
    """Unit-style tests for GaiaAstrometryModel."""

    @pytest.fixture
    def astro_data(self):
        """Simulated Gaia astrometry data with known parameters."""
        data, true = simulate_gaia_epoch_astrometry(
            seed=42,
            n_obs=80,
            period=Q(1.5, "yr"),
            eccentricity=0.3,
            semi_major_axis=Q(5.0, "mas"),
            al_error=Q(0.05, "mas"),
        )
        return data, true

    def test_log_prob_finite(self, astro_data):
        """Model returns a finite scalar at arbitrary parameters."""
        data, _ = astro_data
        model = GaiaAstrometryModel()
        nl = {
            "period": Q(1.5, "yr"),
            "eccentricity": 0.3,
            "phase_peri": 0.5,
            "arg_peri": Q(1.0, "rad"),
            "cos_i": 0.5,
            "lon_asc_node": Q(1.0, "rad"),
        }
        log_lik = model.log_prob(nl, data, linear_prior=_linear_prior())
        assert jnp.isfinite(log_lik)

    def test_vmap_batch(self, astro_data):
        """Vmap over a batch of parameter samples works correctly."""
        data, _ = astro_data
        model = GaiaAstrometryModel()
        lp = _linear_prior()
        n = 8
        nl_batch = {
            "period": Q(jnp.ones(n) * 1.5, "yr"),
            "eccentricity": jnp.linspace(0.0, 0.5, n),
            "phase_peri": jnp.linspace(0.0, 1.0, n),
            "arg_peri": Q(jnp.ones(n) * 1.0, "rad"),
            "cos_i": jnp.linspace(-0.5, 0.5, n),
            "lon_asc_node": Q(jnp.ones(n) * 1.0, "rad"),
        }
        log_liks = jax.jit(
            jax.vmap(lambda nl: model.log_prob(nl, data, linear_prior=lp))
        )(nl_batch)

        assert log_liks.shape == (n,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_true_params_have_higher_loglik(self, astro_data):
        """True parameters should give a higher log-likelihood than random ones."""
        data, true = astro_data
        model = GaiaAstrometryModel()
        lp = _linear_prior()

        cos_i_true = float(np.cos(float(true["inclination"].value)))

        t_peri_yr = float(_ustrip("yr", true["t_peri"]))
        period_yr = float(_ustrip("yr", true["period"]))
        phase_peri_true = (t_peri_yr / period_yr) % 1.0
        nl_true = {
            "period": true["period"],
            "eccentricity": float(true["eccentricity"]),
            "phase_peri": phase_peri_true,
            "arg_peri": Q(float(true["arg_peri"].value), "rad"),
            "cos_i": cos_i_true,
            "lon_asc_node": Q(float(true["lon_asc_node"].value), "rad"),
        }
        nl_rng = {
            "period": Q(0.5, "yr"),
            "eccentricity": 0.0,
            "phase_peri": 0.1,
            "arg_peri": Q(0.0, "rad"),
            "cos_i": 0.0,
            "lon_asc_node": Q(0.0, "rad"),
        }
        ll_true = model.log_prob(nl_true, data, linear_prior=lp)
        ll_rng = model.log_prob(nl_rng, data, linear_prior=lp)
        assert ll_true > ll_rng


class TestGaiaAstrometryRejectionSampler:
    """End-to-end rejection sampler tests for Gaia astrometry."""

    @pytest.fixture
    def sim_data(self):
        """Simulated Gaia astrometry with moderate SNR for tractable rejection."""
        data, true = simulate_gaia_epoch_astrometry(
            seed=100,
            n_obs=80,
            period=Q(1.0, "yr"),
            eccentricity=0.2,
            semi_major_axis=Q(3.0, "mas"),
            al_error=Q(0.1, "mas"),
        )
        return data, true

    def _make_sampler(self, data):
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Q(0.3, "yr"),
            period_max=Q(3.0, "yr"),
            sigma_a0=Q(1e3, "AU"),
            sigma_parallax=Q(100.0, "mas"),
            sigma_pos=Q(1e3, "mas"),
            sigma_vtan=Q(200.0, "km/s"),
        )
        return RejectionSampler(prior, GaiaAstrometryModel(), batch_size=10_000), data

    def test_sampler_runs_and_returns_samples(self, sim_data):
        """Rejection sampler completes and returns a valid Samples object."""
        data, _ = sim_data
        sampler, data = self._make_sampler(data)
        samples = sampler.run(data, n_prior_samples=50_000, seed=42)

        assert samples.n_samples > 0
        assert samples.data_type == "GaiaAstrometryModel"

    def test_samples_have_correct_keys(self, sim_data):
        """Samples object has all expected parameter keys."""
        data, _ = sim_data
        sampler, data = self._make_sampler(data)
        samples = sampler.run(data, n_prior_samples=50_000, seed=43)

        keys = samples.keys()
        for nl_key in (
            "period",
            "log_period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
            "cos_i",
            "lon_asc_node",
        ):
            assert nl_key in keys, f"Missing key: {nl_key}"
        for lin_key in (
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "semi_major_axis",
        ):
            assert lin_key in keys, f"Missing key: {lin_key}"

    def test_reproducibility(self, sim_data):
        """Same seed produces identical samples."""
        data, _ = sim_data
        sampler, data = self._make_sampler(data)
        s1 = sampler.run(data, n_prior_samples=20_000, seed=44)
        s2 = sampler.run(data, n_prior_samples=20_000, seed=44)

        assert s1.n_samples == s2.n_samples
        np.testing.assert_array_equal(s1["period"].value, s2["period"].value)
