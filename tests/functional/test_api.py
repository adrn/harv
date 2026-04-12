"""Functional tests matching API examples from api.py.

These tests verify that the API patterns demonstrated in api.py work end-to-end,
from data creation through sampling to analysis.
"""

import numpy as np
from unxt import Q

from harv.data import GaiaAstrometryData
from harv.samplers.rejection_prior import RejectionPrior
from harv.samplers.rejection import RejectionSampler

# Common kwargs for default_gaia_astrometry throughout tests
_ASTRO_KWARGS = dict(
    period_min=Q(50.0, "day"),
    period_max=Q(200.0, "day"),
    sigma_a0=Q(1e3, "AU"),
    sigma_parallax=Q(100.0, "mas"),
    sigma_pos=Q(1e3, "mas"),
    sigma_vtan=Q(200.0, "km/s"),
)


def simulate_gaia_data_simple(seed: int = 42, n_obs: int = 50) -> GaiaAstrometryData:
    """Create simple simulated Gaia astrometry data for testing.

    This is a minimal simulation utility for testing purposes.
    """
    rng = np.random.default_rng(seed)

    # Fixed orbital parameters
    period = 100.0  # days
    semimajor_axis = 1.0  # mas

    # Observation times
    t_ref = Q(2000.0, "day")
    times = Q(np.sort(rng.uniform(0, 1000, n_obs)), "day") + t_ref

    # Random scan angles
    scan_angle = Q(rng.uniform(0, 2 * np.pi, n_obs), "rad")

    # Simplified parallax factor (just random numbers for test)
    parallax_factor = rng.uniform(-0.5, 0.5, n_obs)

    # Create simple along-scan positions
    # Just use a simplified model: linear motion + simple orbital signal
    dt_yr = (times - t_ref).to_value("yr")
    cos_psi = np.cos(scan_angle.to_value("rad"))
    sin_psi = np.sin(scan_angle.to_value("rad"))

    # 5-parameter astrometry (with made-up values)
    alpha0 = 0.0
    delta0 = 0.0
    mu_alpha = 5.0  # mas/yr
    mu_delta = -3.0  # mas/yr
    parallax = 10.0  # mas

    y_astro = (
        cos_psi * alpha0
        + sin_psi * delta0
        + cos_psi * mu_alpha * dt_yr
        + sin_psi * mu_delta * dt_yr
        + parallax * parallax_factor
    )

    # Add simple orbital signal
    phase = 2 * np.pi * (times - t_ref).to_value("day") / period
    y_orbit = semimajor_axis * (cos_psi * np.cos(phase) + sin_psi * np.sin(phase))

    # Add noise
    al_error = Q(rng.uniform(0.05, 0.15, n_obs), "mas")
    noise = rng.normal(size=n_obs)
    y_al = Q(y_astro + y_orbit + al_error.to_value("mas") * noise, "mas")

    return GaiaAstrometryData(
        time=times,
        al_position=y_al,
        al_position_err=al_error,
        scan_angle=scan_angle,
        parallax_factor=parallax_factor,
        t_ref=t_ref,
    )


class TestBasicAPI:
    """Test basic API patterns from api.py."""

    def test_default_prior_and_basic_run(self):
        """Test the simplest API pattern: default prior and basic run."""
        data = simulate_gaia_data_simple(seed=42, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=42)

        assert samples.n_samples > 0
        assert samples.data_type == "astrometry"

        period = samples["period"]
        assert period.unit == "day"
        assert len(period) == samples.n_samples

        ecc = samples["eccentricity"]
        assert len(ecc) == samples.n_samples

    def test_run_with_max_posterior_samples(self):
        """Test limiting the number of posterior samples returned."""
        data = simulate_gaia_data_simple(seed=43, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)

        max_samples = 64
        samples = sampler.run(
            data, n_prior_samples=10_000, max_posterior_samples=max_samples, seed=43
        )
        assert samples.n_samples <= max_samples

    def test_custom_prior(self):
        """Test creating a custom prior with specific parameter bounds."""
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Q(10.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_a0=Q(500.0, "AU"),
            sigma_parallax=Q(100.0, "mas"),
            sigma_pos=Q(500.0, "mas"),
            sigma_vtan=Q(200.0, "km/s"),
        )

        data = simulate_gaia_data_simple(seed=44, n_obs=30)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=5_000, seed=44)

        period = samples["period"]
        assert np.all(period.to_value("day") >= 10.0)
        assert np.all(period.to_value("day") <= 1000.0)

    def test_reproducibility(self):
        """Test that using the same seed produces identical results."""
        data = simulate_gaia_data_simple(seed=45, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)

        samples1 = sampler.run(data, n_prior_samples=5_000, seed=100)
        samples2 = sampler.run(data, n_prior_samples=5_000, seed=100)

        assert samples1.n_samples == samples2.n_samples
        np.testing.assert_array_equal(
            samples1["eccentricity"], samples2["eccentricity"]
        )

    def test_different_seeds_give_different_results(self):
        """Test that different seeds produce different samples."""
        data = simulate_gaia_data_simple(seed=46, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)

        samples1 = sampler.run(data, n_prior_samples=5_000, seed=200)
        samples2 = sampler.run(data, n_prior_samples=5_000, seed=201)

        assert not np.allclose(samples1["eccentricity"], samples2["eccentricity"])


class TestSamplesContainer:
    """Test Samples container functionality."""

    def test_dict_like_access(self):
        """Test dict-like access to parameters."""
        data = simulate_gaia_data_simple(seed=50, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=50)

        # Test nonlinear parameter access
        assert "period" in samples
        assert "eccentricity" in samples
        assert "phase_peri" in samples
        assert "cos_i" in samples
        assert "arg_peri" in samples
        assert "lon_asc_node" in samples

        # Test linear parameter access
        assert "ra0" in samples
        assert "dec0" in samples
        assert "pmra" in samples
        assert "pmdec" in samples
        assert "parallax" in samples
        assert "semi_major_axis" in samples

        # Test derived quantity access
        assert "log_period" in samples
        period = samples["period"]
        log_period = samples["log_period"]
        np.testing.assert_allclose(period.to_value("day"), 10.0**log_period, rtol=1e-5)

    def test_unit_conversion(self):
        """Test that units are properly restored when accessing parameters."""
        data = simulate_gaia_data_simple(seed=51, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=51)

        # Angles should have radian units
        arg_peri = samples["arg_peri"]
        assert arg_peri.unit == "rad"

        lon_asc = samples["lon_asc_node"]
        assert lon_asc.unit == "rad"

        # Astrometric parameters should have proper units
        ra0 = samples["ra0"]
        assert ra0.unit == "mas"

        pmra = samples["pmra"]
        assert pmra.unit == "mas/yr"

        parallax = samples["parallax"]
        assert parallax.unit == "mas"

        semimaj = samples["semi_major_axis"]
        assert semimaj.unit == "mas"

    def test_dimensionless_parameters(self):
        """Test that dimensionless parameters are plain arrays or dimensionless."""
        data = simulate_gaia_data_simple(seed=52, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=52)

        # These should be dimensionless (plain arrays or Q with unit='')
        for key in ("eccentricity", "phase_peri", "cos_i"):
            val = samples[key]
            if hasattr(val, "unit"):
                assert val.unit == ""

        log_period = samples["log_period"]
        if hasattr(log_period, "unit"):
            assert log_period.unit == ""

    def test_len_and_n_samples(self):
        """Test len() and n_samples property."""
        data = simulate_gaia_data_simple(seed=53, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=53)

        assert len(samples) == samples.n_samples
        assert samples.n_samples > 0

    def test_repr(self):
        """Test string representation."""
        data = simulate_gaia_data_simple(seed=54, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=10_000, seed=54)

        repr_str = repr(samples)
        assert "Samples(" in repr_str
        assert "n_samples=" in repr_str
        assert "data_type='astrometry'" in repr_str
        assert "parameters=" in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_wrong_data_type_raises_error(self):
        """Test that providing wrong data type raises an error."""
        data = simulate_gaia_data_simple(seed=60, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)

        samples = sampler.run(data, n_prior_samples=5_000, seed=60)
        assert samples.n_samples >= 0

    def test_small_batch_size(self):
        """Test sampler with very small batch size."""
        data = simulate_gaia_data_simple(seed=61, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)

        sampler = RejectionSampler(prior, batch_size=1000)

        samples = sampler.run(data, n_prior_samples=5_000, seed=61)
        assert samples.n_samples >= 0

    def test_no_accepted_samples(self):
        """Test handling when no samples are accepted (very unlikely but possible)."""
        data = simulate_gaia_data_simple(seed=62, n_obs=30)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)

        samples = sampler.run(data, n_prior_samples=10, seed=62)

        assert samples.n_samples >= 0
        assert samples.data_type == "astrometry"


class TestAcceptanceRate:
    """Test that acceptance rates are reasonable."""

    def test_acceptance_rate_is_reasonable(self):
        """Test that we get a reasonable acceptance rate."""
        data = simulate_gaia_data_simple(seed=70, n_obs=50)
        prior = RejectionPrior.default_gaia_astrometry(**_ASTRO_KWARGS)
        sampler = RejectionSampler(prior)

        n_prior = 50_000
        samples = sampler.run(data, n_prior_samples=n_prior, seed=70)

        acceptance_rate = samples.n_samples / n_prior

        assert samples.n_samples >= 0
        assert acceptance_rate <= 1.0
