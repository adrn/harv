"""Integration tests for Gaia astrometry rejection sampling.

These tests validate the end-to-end astrometry path: data simulation,
likelihood building, and rejection sampling with parameter recovery.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Quantity

from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.params import GaiaAstrometryParameters
from harv.priors.rejection import RejectionPrior
from harv.quantity_distribution import QuantityDistribution
from harv.samplers.rejection import RejectionSampler
from harv.simulate.astrometry import simulate_gaia_epoch_astrometry


class TestGaiaAstrometryLikelihood:
    """Unit-style tests for GaiaAstrometryLikelihood."""

    @pytest.fixture
    def astro_data(self):
        """Simulated Gaia astrometry data with known parameters."""
        data, true = simulate_gaia_epoch_astrometry(
            seed=42,
            n_obs=80,
            period=Quantity(1.5, "yr"),
            eccentricity=0.3,
            semi_major_axis=Quantity(5.0, "mas"),
            al_error=Quantity(0.05, "mas"),
        )
        return data, true

    def test_log_prob_finite(self, astro_data):
        """Likelihood returns a finite scalar at arbitrary parameters."""
        data, _ = astro_data
        linear_prior = {
            "ra0": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "dec0": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "pmra": QuantityDistribution(dist.Normal(0.0, 1e3), "mas/yr"),
            "pmdec": QuantityDistribution(dist.Normal(0.0, 1e3), "mas/yr"),
            "parallax": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "semi_major_axis": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
        }
        lik = GaiaAstrometryLikelihood(
            data=data,
            linear_marginalized_prior=linear_prior,
        )
        params = GaiaAstrometryParameters.marginalized(
            period=Quantity(1.5, "yr"),
            eccentricity=0.3,
            phase_peri=0.5,
            arg_peri=1.0,
            cos_i=0.5,
            lon_asc_node=1.0,
        )
        log_lik = lik.log_prob(params)
        assert jnp.isfinite(log_lik)

    def test_vmap_batch(self, astro_data):
        """Vmap over a batch of parameter samples works correctly."""
        data, _ = astro_data
        linear_prior = {
            "ra0": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "dec0": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "pmra": QuantityDistribution(dist.Normal(0.0, 1e3), "mas/yr"),
            "pmdec": QuantityDistribution(dist.Normal(0.0, 1e3), "mas/yr"),
            "parallax": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "semi_major_axis": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
        }
        lik = GaiaAstrometryLikelihood(
            data=data,
            linear_marginalized_prior=linear_prior,
        )
        n = 8
        params_batch = GaiaAstrometryParameters.marginalized(
            period=Quantity(jnp.ones(n) * 1.5, "yr"),
            eccentricity=jnp.linspace(0.0, 0.5, n),
            phase_peri=jnp.linspace(0.0, 1.0, n),
            arg_peri=jnp.ones(n) * 1.0,
            cos_i=jnp.linspace(-0.5, 0.5, n),
            lon_asc_node=jnp.ones(n) * 1.0,
        )
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        assert log_liks.shape == (n,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_true_params_have_higher_loglik(self, astro_data):
        """True parameters should give a higher log-likelihood than random ones."""
        data, true = astro_data
        linear_prior = {
            "ra0": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "dec0": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "pmra": QuantityDistribution(dist.Normal(0.0, 1e3), "mas/yr"),
            "pmdec": QuantityDistribution(dist.Normal(0.0, 1e3), "mas/yr"),
            "parallax": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
            "semi_major_axis": QuantityDistribution(dist.Normal(0.0, 1e3), "mas"),
        }
        lik = GaiaAstrometryLikelihood(
            data=data,
            linear_marginalized_prior=linear_prior,
        )

        # Construct params at / near the truth
        import numpy as np

        cos_i_true = float(np.cos(float(true["inclination"].value)))
        # phase_peri = t_peri / period (see _solve_kepler in helpers.py)
        from unxt import ustrip as _ustrip

        t_peri_yr = float(_ustrip("yr", true["t_peri"]))
        period_yr = float(_ustrip("yr", true["period"]))
        phase_peri_true = (t_peri_yr / period_yr) % 1.0
        params_true = GaiaAstrometryParameters.marginalized(
            period=true["period"],
            eccentricity=float(true["eccentricity"]),
            phase_peri=phase_peri_true,
            arg_peri=float(true["arg_peri"].value),
            cos_i=cos_i_true,
            lon_asc_node=float(true["lon_asc_node"].value),
        )
        # Random params
        params_rng = GaiaAstrometryParameters.marginalized(
            period=Quantity(0.5, "yr"),
            eccentricity=0.0,
            phase_peri=0.1,
            arg_peri=0.0,
            cos_i=0.0,
            lon_asc_node=0.0,
        )
        ll_true = lik.log_prob(params_true)
        ll_rng = lik.log_prob(params_rng)
        assert ll_true > ll_rng


class TestGaiaAstrometryRejectionSampler:
    """End-to-end rejection sampler tests for Gaia astrometry."""

    @pytest.fixture
    def sim_data(self):
        """Simulated Gaia astrometry with moderate SNR for tractable rejection."""
        data, true = simulate_gaia_epoch_astrometry(
            seed=100,
            n_obs=80,
            period=Quantity(1.0, "yr"),
            eccentricity=0.2,
            semi_major_axis=Quantity(3.0, "mas"),
            al_error=Quantity(0.1, "mas"),
        )
        return data, true

    def test_sampler_runs_and_returns_samples(self, sim_data):
        """Rejection sampler completes and returns a valid Samples object."""
        data, _ = sim_data
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Quantity(0.3, "yr"),
            period_max=Quantity(3.0, "yr"),
            sigma_a0=Quantity(1e3, "AU"),
            sigma_parallax=Quantity(100.0, "mas"),
            sigma_pos=Quantity(1e3, "mas"),
            sigma_vtan=Quantity(200.0, "km/s"),
        )
        sampler = RejectionSampler(prior, batch_size=10_000)
        samples = sampler.run(data, n_prior_samples=50_000, seed=42)

        assert samples.n_samples > 0
        assert samples.data_type == "astrometry"

    def test_samples_have_correct_keys(self, sim_data):
        """Samples object has all expected parameter keys."""
        data, _ = sim_data
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Quantity(0.3, "yr"),
            period_max=Quantity(3.0, "yr"),
            sigma_a0=Quantity(1e3, "AU"),
            sigma_parallax=Quantity(100.0, "mas"),
            sigma_pos=Quantity(1e3, "mas"),
            sigma_vtan=Quantity(200.0, "km/s"),
        )
        sampler = RejectionSampler(prior, batch_size=10_000)
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
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Quantity(0.3, "yr"),
            period_max=Quantity(3.0, "yr"),
            sigma_a0=Quantity(1e3, "AU"),
            sigma_parallax=Quantity(100.0, "mas"),
            sigma_pos=Quantity(1e3, "mas"),
            sigma_vtan=Quantity(200.0, "km/s"),
        )
        sampler = RejectionSampler(prior, batch_size=10_000)
        s1 = sampler.run(data, n_prior_samples=20_000, seed=44)
        s2 = sampler.run(data, n_prior_samples=20_000, seed=44)

        assert s1.n_samples == s2.n_samples
        np.testing.assert_array_equal(s1["period"].value, s2["period"].value)
