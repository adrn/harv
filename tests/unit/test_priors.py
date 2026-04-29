"""Unit tests for rejection sampling priors."""

import jax.random as jr
import numpyro.distributions as dist
from unxt import Q

from harv.distributions import QD
from harv.samplers.custom_priors import (
    ParallaxDependentProperMotionPrior,
    PeriodDependentSemiMajorAxisPrior,
)
from harv.samplers.rejection_prior import RejectionPrior

# Common default_rv kwargs used throughout tests
_DEFAULT_RV_KWARGS = {
    "period_min": Q(50.0, "day"),
    "period_max": Q(200.0, "day"),
    "sigma_K0": Q(30.0, "km/s"),
    "sigma_v0": Q(30.0, "km/s"),
}

# Common default_gaia_astrometry kwargs used throughout tests
_DEFAULT_ASTRO_KWARGS = {
    "period_min": Q(50.0, "day"),
    "period_max": Q(200.0, "day"),
    "sigma_a0": Q(1e3, "AU"),
    "sigma_parallax": Q(100.0, "mas"),
    "sigma_pos": Q(1e3, "mas"),
    "sigma_vtan": Q(200.0, "km/s"),
}


class TestRejectionPriorAstrometry:
    """Tests for astrometry-only priors."""

    def test_default_gaia_astrometry_creation(self):
        """Test creating default Gaia astrometry prior."""
        prior = RejectionPrior.default_gaia_astrometry(**_DEFAULT_ASTRO_KWARGS)

        assert (
            prior.n_nonlinear == 6
        )  # period, ecc, phase_peri, cos_i, arg_peri, lon_asc_node
        assert "cos_i" in prior.nonlinear_priors
        assert "lon_asc_node" in prior.nonlinear_priors
        assert "arg_peri" in prior.nonlinear_priors

    def test_astrometry_prior_sampling(self):
        """Test sampling from astrometry prior."""
        prior = RejectionPrior.default_gaia_astrometry(**_DEFAULT_ASTRO_KWARGS)
        key = jr.key(42)

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
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Q(1.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_a0=Q(1e3, "AU"),
            sigma_parallax=Q(100.0, "mas"),
            sigma_pos=Q(1e3, "mas"),
            sigma_vtan=Q(200.0, "km/s"),
        )
        key = jr.key(123)

        samples = prior.sample_nonlinear(key, n_samples=1000)

        assert (samples["period"] >= 1.0).all()
        assert (samples["period"] <= 1000.0).all()

    def test_linear_prior_structure(self):
        """Test linear prior is a dict with correct keys and types."""
        prior = RejectionPrior.default_gaia_astrometry(**_DEFAULT_ASTRO_KWARGS)

        assert isinstance(prior.linear_prior, dict)
        assert set(prior.linear_prior.keys()) == {
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "semi_major_axis",
        }
        # parallax should be HalfNormal (explicit)
        assert isinstance(prior.linear_prior["parallax"], QD)
        # semi_major_axis should be PeriodDependentSemiMajorAxisPrior (callable)
        assert isinstance(
            prior.linear_prior["semi_major_axis"],
            PeriodDependentSemiMajorAxisPrior,
        )
        # ra0/dec0 should be QuantityDistribution wrapping Normal
        for key in ("ra0", "dec0"):
            assert isinstance(prior.linear_prior[key], QD)
            assert isinstance(prior.linear_prior[key].distribution, dist.Normal)
        # pmra/pmdec should be ParallaxDependentProperMotionPrior (callable)
        for key in ("pmra", "pmdec"):
            assert isinstance(
                prior.linear_prior[key],
                ParallaxDependentProperMotionPrior,
            )


class TestRejectionPriorRV:
    """Tests for RV-only priors."""

    def test_default_rv_creation(self):
        """Test creating default RV prior."""
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS)

        assert prior.n_nonlinear == 4  # period, ecc, arg_peri, phase_peri
        assert "arg_peri" in prior.nonlinear_priors
        assert "cos_i" not in prior.nonlinear_priors
        assert "lon_asc_node" not in prior.nonlinear_priors

    def test_rv_with_extension_priors(self):
        """Test RV prior with multi-instrument offset extension priors."""
        prior = RejectionPrior.default_rv(
            **_DEFAULT_RV_KWARGS,
            espresso=QD(dist.Normal(0, 5.0), "km/s"),
            harps=QD(dist.Normal(0, 10.0), "km/s"),
        )

        # Extension priors are stored in extension_priors, not linear_prior
        assert "espresso" in prior.extension_priors
        assert "harps" in prior.extension_priors
        # They are not in linear_prior (routing happens at run-time)
        assert "espresso" not in prior.linear_prior
        assert "harps" not in prior.linear_prior

    def test_rv_prior_sampling(self):
        """Test sampling from RV prior."""
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS)
        key = jr.key(42)

        samples = prior.sample_nonlinear(key, n_samples=100)

        assert set(samples.keys()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }
        assert samples["arg_peri"].shape == (100,)


class TestParameterOverrides:
    """Tests for overriding nonlinear and linear priors via **kwargs."""

    def test_rv_override_nonlinear(self):
        """Nonlinear prior can be overridden via kwargs."""
        custom_ecc = dist.Uniform(0.0, 0.5)
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS, eccentricity=custom_ecc)
        assert prior.nonlinear_priors["eccentricity"] is custom_ecc

    def test_rv_override_linear(self):
        """Linear prior can be overridden via kwargs."""
        custom_K = QD(dist.Normal(0.0, 50.0), "km/s")
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS, rv_semiamp=custom_K)
        assert prior.linear_prior["rv_semiamp"] is custom_K

    def test_rv_override_both(self):
        """Nonlinear and linear overrides can be combined."""
        custom_ecc = dist.Uniform(0.0, 0.3)
        custom_v0 = QD(dist.Normal(0.0, 5.0), "km/s")
        prior = RejectionPrior.default_rv(
            **_DEFAULT_RV_KWARGS, eccentricity=custom_ecc, v_sys=custom_v0
        )
        assert prior.nonlinear_priors["eccentricity"] is custom_ecc
        assert prior.linear_prior["v_sys"] is custom_v0

    def test_rv_unknown_kwarg_becomes_extension_prior(self):
        """Unknown kwarg is accepted and stored in extension_priors."""
        bogus_dist = dist.Normal(0, 1)
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS, bogus=bogus_dist)
        assert "bogus" in prior.extension_priors
        assert prior.extension_priors["bogus"] is bogus_dist

    def test_astro_override_nonlinear(self):
        """Nonlinear prior can be overridden in astrometry constructor."""
        custom_cos_i = dist.Uniform(0.0, 1.0)
        prior = RejectionPrior.default_gaia_astrometry(
            **_DEFAULT_ASTRO_KWARGS, cos_i=custom_cos_i
        )
        assert prior.nonlinear_priors["cos_i"] is custom_cos_i

    def test_astro_override_linear(self):
        """Linear prior can be overridden in astrometry constructor."""
        custom_parallax = QD(dist.Normal(5.0, 0.5), "mas")
        prior = RejectionPrior.default_gaia_astrometry(
            **_DEFAULT_ASTRO_KWARGS, parallax=custom_parallax
        )
        assert prior.linear_prior["parallax"] is custom_parallax

    def test_astro_unknown_kwarg_becomes_extension_prior(self):
        """Unknown kwarg is accepted and stored in extension_priors."""
        fake_dist = dist.Normal(0, 1)
        prior = RejectionPrior.default_gaia_astrometry(
            **_DEFAULT_ASTRO_KWARGS, fake=fake_dist
        )
        assert "fake" in prior.extension_priors
        assert prior.extension_priors["fake"] is fake_dist


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
            linear_prior={
                "rv_semiamp": dist.Normal(0.0, 1000.0),
                "v_sys": dist.Normal(0.0, 1000.0),
            },
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
            linear_prior={
                "ra0": dist.Normal(0.0, 1000.0),
                "dec0": dist.Normal(0.0, 1000.0),
                "pmra": dist.Normal(0.0, 1000.0),
                "pmdec": dist.Normal(0.0, 1000.0),
                "parallax": dist.Normal(0.0, 1000.0),
                "semi_major_axis": dist.Normal(0.0, 1000.0),
            },
        )
        assert prior.n_nonlinear == 6


class TestDictLinearPrior:
    """Tests for dict-form linear prior."""

    def test_dict_linear_prior_is_dict(self):
        """Dict with all Normal entries is stored as dict."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": dist.LogUniform(1.0, 1000.0),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0, 1),
                "arg_peri": dist.Uniform(0, 6.28),
            },
            linear_prior={
                "rv_semiamp": dist.Normal(0.0, 100.0),
                "v_sys": dist.Normal(0.0, 50.0),
            },
        )
        assert isinstance(prior.linear_prior, dict)
        assert set(prior.linear_prior.keys()) == {"rv_semiamp", "v_sys"}

    def test_dict_with_delta(self):
        """Delta entries are accepted in dict-form linear prior."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": dist.LogUniform(1.0, 1000.0),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0, 1),
                "arg_peri": dist.Uniform(0, 6.28),
            },
            linear_prior={
                "rv_semiamp": dist.Delta(10.0),
                "v_sys": dist.Normal(0.0, 50.0),
            },
        )
        assert isinstance(prior.linear_prior, dict)

    def test_dict_with_mixed_types(self):
        """All three categories (Gaussian, Delta, explicit) in one dict."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": dist.LogUniform(1.0, 1000.0),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0, 1),
                "cos_i": dist.Uniform(-1, 1),
                "arg_peri": dist.Uniform(0, 6.28),
                "lon_asc_node": dist.Uniform(0, 6.28),
            },
            linear_prior={
                "ra0": dist.Normal(0.0, 1000.0),
                "dec0": dist.Normal(0.0, 1000.0),
                "pmra": dist.Normal(0.0, 100.0),
                "pmdec": dist.Normal(0.0, 100.0),
                "parallax": dist.Delta(5.0),
                "semi_major_axis": dist.HalfNormal(10.0),
            },
        )
        assert isinstance(prior.linear_prior, dict)
        assert len(prior.linear_prior) == 6

    def test_default_rv_has_dict_linear_prior(self):
        """default_rv returns a prior with dict-form linear_prior."""
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS)
        assert isinstance(prior.linear_prior, dict)

    def test_dict_with_quantity_distribution(self):
        """QuantityDistribution entries in dict are accepted."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": dist.LogUniform(1.0, 1000.0),
                "eccentricity": dist.Beta(0.867, 3.03),
                "phase_peri": dist.Uniform(0, 1),
                "arg_peri": dist.Uniform(0, 6.28),
            },
            linear_prior={
                "rv_semiamp": QD(dist.HalfNormal(100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        assert isinstance(prior.linear_prior, dict)
        assert set(prior.linear_prior.keys()) == {"rv_semiamp", "v_sys"}


class TestPriorProperties:
    """Tests for prior property methods."""

    def test_parameter_counting_rv(self):
        """Test that RV prior has 4 nonlinear parameters."""
        rv_prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS)
        assert rv_prior.n_nonlinear == 4

    def test_parameter_counting_rv_with_extension_priors(self):
        """Test that RV with extension priors still has 4 nonlinear parameters."""
        rv_prior_offsets = RejectionPrior.default_rv(
            **_DEFAULT_RV_KWARGS,
            inst2=QD(dist.Normal(0, 5), "km/s"),
        )
        assert rv_prior_offsets.n_nonlinear == 4
        assert "inst2" in rv_prior_offsets.extension_priors
        # inst2 is NOT in linear_prior until routing happens at run-time
        assert "inst2" not in rv_prior_offsets.linear_prior

    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with same seed."""
        prior = RejectionPrior.default_rv(**_DEFAULT_RV_KWARGS)

        samples1 = prior.sample_nonlinear(jr.key(42), n_samples=100)
        samples2 = prior.sample_nonlinear(jr.key(42), n_samples=100)

        # Should be identical
        assert (samples1["period"] == samples2["period"]).all()
        assert (samples1["eccentricity"] == samples2["eccentricity"]).all()
