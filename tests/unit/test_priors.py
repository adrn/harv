"""Unit tests for rejection sampling priors."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist

from harv.priors.rejection import RejectionPrior


class TestRejectionPriorAstrometry:
    """Tests for astrometry-only priors."""

    def test_default_astrometry_creation(self):
        """Test creating default astrometry prior."""
        prior = RejectionPrior.default_astrometry()

        assert (
            prior.n_nonlinear == 6
        )  # period, ecc, phase_peri, cos_i, arg_peri, lon_asc_node
        assert "cos_i" in prior.nonlinear_priors
        assert "lon_asc_node" in prior.nonlinear_priors
        assert "arg_peri" in prior.nonlinear_priors

    def test_astrometry_prior_sampling(self):
        """Test sampling from astrometry prior."""
        prior = RejectionPrior.default_astrometry()
        key = jr.PRNGKey(42)

        samples = prior.sample_nonlinear(key, n_samples=100)

        assert set(samples.keys()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "cos_i",
            "arg_peri",
            "lon_asc_node",
        }
        assert samples["period"].shape == (100,)
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
        prior = RejectionPrior.default_astrometry(period_min=1.0, period_max=1000.0)
        key = jr.PRNGKey(123)

        samples = prior.sample_nonlinear(key, n_samples=1000)

        assert (samples["period"] >= 1.0).all()
        assert (samples["period"] <= 1000.0).all()

    def test_linear_prior_distribution(self):
        """Test linear prior is a MultivariateNormal."""
        prior = RejectionPrior.default_astrometry(linear_prior_scale=500.0)

        assert isinstance(prior.linear_prior, dist.MultivariateNormal)
        # Astrometry has 6 linear parameters
        assert prior.linear_prior.loc.shape == (6,)
        # Sample from it to verify it works
        key = jr.PRNGKey(42)
        linear_samples = prior.linear_prior.sample(key, (10,))
        assert linear_samples.shape == (10, 6)


class TestRejectionPriorRV:
    """Tests for RV-only priors."""

    def test_default_rv_creation(self):
        """Test creating default RV prior."""
        prior = RejectionPrior.default_rv()

        assert prior.n_nonlinear == 4  # period, ecc, arg_peri, phase_peri
        assert "arg_peri" in prior.nonlinear_priors
        assert "cos_i" not in prior.nonlinear_priors
        assert "lon_asc_node" not in prior.nonlinear_priors

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
            "period",
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
        assert "cos_i" in prior.nonlinear_priors
        assert "arg_peri" in prior.nonlinear_priors
        assert "lon_asc_node" in prior.nonlinear_priors

    def test_combined_prior_sampling(self):
        """Test sampling from combined prior."""
        prior = RejectionPrior.default_combined()
        key = jr.PRNGKey(42)

        samples = prior.sample_nonlinear(key, n_samples=50)

        assert set(samples.keys()) == {
            "period",
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
        assert prior.n_nonlinear == 4  # period, ecc, phase_peri, arg_peri
        assert "arg_peri" in prior.nonlinear_priors
        assert "cos_i" not in prior.nonlinear_priors
        assert "lon_asc_node" not in prior.nonlinear_priors


class TestPriorValidation:
    """Tests for prior validation."""

    def test_prior_can_be_created_with_any_params(self):
        """Test that priors can be created with any combination of params.

        Validation of required params happens in the sampler, not the prior.
        """
        # Missing orientation params are fine at prior level
        prior = RejectionPrior(
            nonlinear_priors={
                "period": dist.LogUniform(1.0, 1000.0),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0, 1),
            },
            linear_prior=dist.MultivariateNormal(
                loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * 1000.0**2
            ),
        )
        assert prior.n_nonlinear == 3  # Only the 3 provided params

    def test_prior_with_all_params(self):
        """Test creating a prior with all optional params."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": dist.LogUniform(1.0, 1000.0),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0, 1),
                "cos_i": dist.Uniform(-1, 1),
                "arg_peri": dist.Uniform(0, 6.28),
                "lon_asc_node": dist.Uniform(0, 6.28),
            },
            linear_prior=dist.MultivariateNormal(
                loc=jnp.zeros(6), covariance_matrix=jnp.eye(6) * 1000.0**2
            ),
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
        assert (samples1["period"] == samples2["period"]).all()
        assert (samples1["eccentricity"] == samples2["eccentricity"]).all()
