"""Unit tests for rejection sampling priors."""

import jax.random as jr
import numpyro.distributions as dist
import pytest

from epochalypse.priors.rejection import RejectionPrior


class TestRejectionPriorAstrometry:
    """Tests for astrometry-only priors."""

    def test_default_astrometry_creation(self):
        """Test creating default astrometry prior."""
        prior = RejectionPrior.default_astrometry()

        assert (
            prior.n_nonlinear == 6
        )  # log_period, ecc, phase_peri, cos_i, arg_peri, lon_asc_node
        assert prior.cos_i is not None
        assert prior.lon_asc_node is not None
        assert prior.arg_peri is not None  # Required for astrometry

    def test_astrometry_prior_sampling(self):
        """Test sampling from astrometry prior."""
        prior = RejectionPrior.default_astrometry()
        key = jr.PRNGKey(42)

        samples = prior.sample_nonlinear(key, n_samples=100)

        assert set(samples.keys()) == {
            "log_period",
            "eccentricity",
            "phase_peri",
            "cos_i",
            "arg_peri",
            "lon_asc_node",
        }
        assert samples["log_period"].shape == (100,)
        assert samples["eccentricity"].shape == (100,)

        # Check ranges
        assert (samples["eccentricity"] >= 0).all()
        assert (samples["eccentricity"] < 1).all()
        assert (samples["phase_peri"] >= 0).all()
        assert (samples["phase_peri"] <= 1).all()
        assert (samples["cos_i"] >= -1).all()
        assert (samples["cos_i"] <= 1).all()

    def test_custom_period_bounds(self):
        """Test custom period bounds."""
        prior = RejectionPrior.default_astrometry(
            log_period_min=0.0, log_period_max=3.0
        )
        key = jr.PRNGKey(123)

        samples = prior.sample_nonlinear(key, n_samples=1000)

        assert (samples["log_period"] >= 0.0).all()
        assert (samples["log_period"] <= 3.0).all()

    def test_linear_prior_distribution(self):
        """Test getting linear prior distribution."""
        prior = RejectionPrior.default_astrometry(linear_prior_scale=500.0)

        linear_dist = prior.get_linear_prior_distribution()

        assert isinstance(linear_dist, dist.Normal)
        # Sample from it to verify it works
        key = jr.PRNGKey(42)
        linear_samples = linear_dist.sample(key, (10,))
        assert linear_samples.shape == (10,)


class TestRejectionPriorRV:
    """Tests for RV-only priors."""

    def test_default_rv_creation(self):
        """Test creating default RV prior."""
        prior = RejectionPrior.default_rv()

        assert prior.n_nonlinear == 4  # log_period, ecc, arg_peri, phase_peri
        assert prior.arg_peri is not None
        assert prior.cos_i is None  # Not used for RV-only
        assert prior.lon_asc_node is None

    def test_rv_with_offsets(self):
        """Test RV prior with multi-instrument offsets."""
        offsets = {
            "keck": None,  # Reference instrument
            "espresso": dist.Normal(0, 5.0),
            "harps": dist.Normal(0, 10.0),
        }
        prior = RejectionPrior.default_rv(offsets=offsets)

        assert prior.offsets is not None
        assert len(prior.offsets) == 3
        # Count non-None offsets (reference instrument has None)
        n_offsets = sum(1 for v in prior.offsets.values() if v is not None)
        assert n_offsets == 2  # espresso and harps

    def test_rv_prior_sampling(self):
        """Test sampling from RV prior."""
        prior = RejectionPrior.default_rv()
        key = jr.PRNGKey(42)

        samples = prior.sample_nonlinear(key, n_samples=100)

        assert set(samples.keys()) == {
            "log_period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }
        assert samples["arg_peri"].shape == (100,)


class TestRejectionPriorCombined:
    """Tests for combined astrometry + RV priors."""

    def test_default_combined_creation(self):
        """Test creating default combined prior."""
        prior = RejectionPrior.default_combined()

        assert prior.n_nonlinear == 6  # All 6 parameters
        assert prior.cos_i is not None
        assert prior.arg_peri is not None
        assert prior.lon_asc_node is not None

    def test_combined_prior_sampling(self):
        """Test sampling from combined prior."""
        prior = RejectionPrior.default_combined()
        key = jr.PRNGKey(42)

        samples = prior.sample_nonlinear(key, n_samples=50)

        assert set(samples.keys()) == {
            "log_period",
            "eccentricity",
            "phase_peri",
            "cos_i",
            "arg_peri",
            "lon_asc_node",
        }


class TestRejectionPriorSB2:
    """Tests for SB2 (double-lined) priors."""

    def test_default_sb2_creation(self):
        """Test creating default SB2 prior."""
        prior = RejectionPrior.default_sb2()

        # SB2 has same nonlinear params as RV (no orientation)
        assert prior.n_nonlinear == 4  # log_period, ecc, phase_peri, arg_peri
        assert prior.arg_peri is not None
        assert prior.cos_i is None
        assert prior.lon_asc_node is None


class TestPriorValidation:
    """Tests for prior validation."""

    def test_negative_linear_prior_scale(self):
        """Test that negative linear prior scale raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            RejectionPrior(
                log_period=dist.Uniform(-1, 4),
                eccentricity=dist.Beta(0.867, 3.03),
                phase_peri=dist.Uniform(0, 1),
                linear_prior_scale=-10.0,  # Invalid!
                arg_peri=dist.Uniform(0, 6.28),
            )

    def test_prior_can_be_created_with_any_params(self):
        """Test that priors can be created with any combination of params.

        Validation of required params happens in the sampler, not the prior.
        """
        # This is valid - missing orientation params are fine at prior level
        prior = RejectionPrior(
            log_period=dist.Uniform(-1, 4),
            eccentricity=dist.Beta(0.867, 3.03),
            phase_peri=dist.Uniform(0, 1),
            linear_prior_scale=1000.0,
            # Missing cos_i, arg_peri, lon_asc_node - OK at prior level
        )
        assert prior.n_nonlinear == 3  # Only the 3 required params

    def test_prior_with_all_params(self):
        """Test creating a prior with all optional params."""
        prior = RejectionPrior(
            log_period=dist.Uniform(-1, 4),
            eccentricity=dist.Beta(0.867, 3.03),
            phase_peri=dist.Uniform(0, 1),
            cos_i=dist.Uniform(-1, 1),
            arg_peri=dist.Uniform(0, 6.28),
            lon_asc_node=dist.Uniform(0, 6.28),
            linear_prior_scale=1000.0,
        )
        assert prior.n_nonlinear == 6


class TestPriorProperties:
    """Tests for prior property methods."""

    def test_parameter_counting(self):
        """Test that parameter counting is correct."""
        # Astrometry - 6 nonlinear params
        astro_prior = RejectionPrior.default_astrometry()
        assert astro_prior.n_nonlinear == 6

        # RV - 4 nonlinear params
        rv_prior = RejectionPrior.default_rv()
        assert rv_prior.n_nonlinear == 4

        # RV with offsets
        rv_prior_offsets = RejectionPrior.default_rv(
            offsets={"inst1": None, "inst2": dist.Normal(0, 5)}
        )
        assert rv_prior_offsets.n_nonlinear == 4
        assert rv_prior_offsets.offsets is not None
        n_offsets = sum(1 for v in rv_prior_offsets.offsets.values() if v is not None)
        assert n_offsets == 1

    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with same seed."""
        prior = RejectionPrior.default_astrometry()

        samples1 = prior.sample_nonlinear(jr.PRNGKey(42), n_samples=100)
        samples2 = prior.sample_nonlinear(jr.PRNGKey(42), n_samples=100)

        # Should be identical
        assert (samples1["log_period"] == samples2["log_period"]).all()
        assert (samples1["eccentricity"] == samples2["eccentricity"]).all()
