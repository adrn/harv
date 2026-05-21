"""Unit tests for rejection sampling priors."""

import jax.random as jr
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv.models as hm
from harv.distributions import QD
from harv.models.parameterizations._base import AbstractParameterization
from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
from harv.models.priors import (
    HarvPrior,
    ParallaxDependentProperMotionPrior,
    PeriodDependentSemiMajorAxisPrior,
    default_sb2_prior,
)

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


class TestHarvPriorAstrometry:
    """Tests for astrometry-only priors."""

    def test_default_gaia_astrometry_creation(self):
        """Test creating default Gaia astrometry prior."""
        prior = hm.StandardGaiaAstrometry().default_prior(**_DEFAULT_ASTRO_KWARGS)
        assert "cos_i" in prior.nonlinear_priors
        assert "lon_asc_node" in prior.nonlinear_priors
        assert "arg_peri" in prior.nonlinear_priors

    def test_astrometry_prior_sampling(self):
        """Test sampling from astrometry prior."""
        prior = hm.StandardGaiaAstrometry().default_prior(**_DEFAULT_ASTRO_KWARGS)
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
        prior = hm.StandardGaiaAstrometry().default_prior(
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
        prior = hm.StandardGaiaAstrometry().default_prior(**_DEFAULT_ASTRO_KWARGS)

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


class TestHarvPriorRV:
    """Tests for RV-only priors."""

    def test_default_rv_creation(self):
        """Test creating default RV prior."""
        prior = hm.StandardRV().default_prior(**_DEFAULT_RV_KWARGS)

        assert "arg_peri" in prior.nonlinear_priors
        assert "cos_i" not in prior.nonlinear_priors
        assert "lon_asc_node" not in prior.nonlinear_priors

    def test_rv_with_extension_priors(self):
        """Test RV prior with multi-instrument offset extension priors."""
        prior = hm.StandardRV().default_prior(
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
        prior = hm.StandardRV().default_prior(**_DEFAULT_RV_KWARGS)
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
        prior = hm.StandardRV().default_prior(
            **_DEFAULT_RV_KWARGS, eccentricity=custom_ecc
        )
        assert prior.nonlinear_priors["eccentricity"] is custom_ecc

    def test_rv_override_linear(self):
        """Linear prior can be overridden via kwargs (omit conflicting sigma_K0)."""
        custom_K = QD(dist.Normal(0.0, 50.0), "km/s")
        kwargs = {k: v for k, v in _DEFAULT_RV_KWARGS.items() if k != "sigma_K0"}
        prior = hm.StandardRV().default_prior(**kwargs, rv_semiamp=custom_K)
        assert prior.linear_prior["rv_semiamp"] is custom_K

    def test_rv_override_both(self):
        """Nonlinear and linear overrides can be combined."""
        custom_ecc = dist.Uniform(0.0, 0.3)
        custom_v0 = QD(dist.Normal(0.0, 5.0), "km/s")
        # Omit sigma_v0 since v_sys is being overridden directly.
        kwargs = {k: v for k, v in _DEFAULT_RV_KWARGS.items() if k != "sigma_v0"}
        prior = hm.StandardRV().default_prior(
            **kwargs, eccentricity=custom_ecc, v_sys=custom_v0
        )
        assert prior.nonlinear_priors["eccentricity"] is custom_ecc
        assert prior.linear_prior["v_sys"] is custom_v0

    def test_rv_unknown_kwarg_becomes_extension_prior(self):
        """Unknown kwarg is accepted and stored in extension_priors."""
        bogus_dist = dist.Normal(0, 1)
        prior = hm.StandardRV().default_prior(**_DEFAULT_RV_KWARGS, bogus=bogus_dist)
        assert "bogus" in prior.extension_priors
        assert prior.extension_priors["bogus"] is bogus_dist

    def test_astro_override_nonlinear(self):
        """Nonlinear prior can be overridden in astrometry constructor."""
        custom_cos_i = dist.Uniform(0.0, 1.0)
        prior = hm.StandardGaiaAstrometry().default_prior(
            **_DEFAULT_ASTRO_KWARGS, cos_i=custom_cos_i
        )
        assert prior.nonlinear_priors["cos_i"] is custom_cos_i

    def test_astro_override_linear(self):
        """Linear prior can be overridden in astrometry constructor."""
        # HalfNormal is non-Gaussian → explicitly sampled → dependent callables work
        custom_parallax = QD(dist.HalfNormal(0.5), "mas")
        kwargs = {
            k: v for k, v in _DEFAULT_ASTRO_KWARGS.items() if k != "sigma_parallax"
        }
        prior = hm.StandardGaiaAstrometry().default_prior(
            **kwargs, parallax=custom_parallax
        )
        assert prior.linear_prior["parallax"] is custom_parallax

    def test_astro_unknown_kwarg_becomes_extension_prior(self):
        """Unknown kwarg is accepted and stored in extension_priors."""
        fake_dist = dist.Normal(0, 1)
        prior = hm.StandardGaiaAstrometry().default_prior(
            **_DEFAULT_ASTRO_KWARGS, fake=fake_dist
        )
        assert "fake" in prior.extension_priors
        assert prior.extension_priors["fake"] is fake_dist

    def test_astro_override_parallax_omits_sigma_parallax(self):
        """Passing parallax= directly lets sigma_parallax be omitted."""
        custom_parallax = QD(dist.Normal(5.0, 0.5), "mas")
        kwargs = {
            k: v for k, v in _DEFAULT_ASTRO_KWARGS.items() if k != "sigma_parallax"
        }
        prior = hm.StandardGaiaAstrometry().default_prior(
            **kwargs, parallax=custom_parallax
        )
        assert prior.linear_prior["parallax"] is custom_parallax

    def test_astro_override_pos_omits_sigma_pos(self):
        """Passing ra0= and dec0= directly lets sigma_pos be omitted."""
        custom_ra0 = QD(dist.Normal(0.0, 10.0), "mas")
        custom_dec0 = QD(dist.Normal(0.0, 10.0), "mas")
        kwargs = {k: v for k, v in _DEFAULT_ASTRO_KWARGS.items() if k != "sigma_pos"}
        prior = hm.StandardGaiaAstrometry().default_prior(
            **kwargs, ra0=custom_ra0, dec0=custom_dec0
        )
        assert prior.linear_prior["ra0"] is custom_ra0
        assert prior.linear_prior["dec0"] is custom_dec0

    def test_astro_override_semi_major_axis_omits_sigma_a0(self):
        """Passing semi_major_axis= directly lets sigma_a0 be omitted."""
        custom_sma = QD(dist.HalfNormal(5.0), "mas")
        kwargs = {k: v for k, v in _DEFAULT_ASTRO_KWARGS.items() if k != "sigma_a0"}
        prior = hm.StandardGaiaAstrometry().default_prior(
            **kwargs, semi_major_axis=custom_sma
        )
        assert prior.linear_prior["semi_major_axis"] is custom_sma

    def test_astro_override_pmra_pmdec_omits_sigma_vtan(self):
        """Passing pmra= and pmdec= directly lets sigma_vtan be omitted."""
        custom_pmra = QD(dist.Normal(0.0, 10.0), "mas/yr")
        custom_pmdec = QD(dist.Normal(0.0, 10.0), "mas/yr")
        kwargs = {k: v for k, v in _DEFAULT_ASTRO_KWARGS.items() if k != "sigma_vtan"}
        prior = hm.StandardGaiaAstrometry().default_prior(
            **kwargs, pmra=custom_pmra, pmdec=custom_pmdec
        )
        assert prior.linear_prior["pmra"] is custom_pmra
        assert prior.linear_prior["pmdec"] is custom_pmdec

    def test_astro_conflict_raises_parallax(self):
        """Providing both parallax= and sigma_parallax= raises TypeError."""

        custom_parallax = QD(dist.Normal(5.0, 0.5), "mas")
        with pytest.raises(TypeError, match="Cannot specify both"):
            hm.StandardGaiaAstrometry().default_prior(
                **_DEFAULT_ASTRO_KWARGS, parallax=custom_parallax
            )

    def test_astro_conflict_raises_sigma_a0(self):
        """Providing both semi_major_axis= and sigma_a0= raises TypeError."""

        custom_sma = QD(dist.HalfNormal(5.0), "mas")
        with pytest.raises(TypeError, match="Cannot specify both"):
            hm.StandardGaiaAstrometry().default_prior(
                **_DEFAULT_ASTRO_KWARGS, semi_major_axis=custom_sma
            )

    def test_astro_missing_all_raises(self):
        """Omitting sigma_parallax and parallax= raises TypeError."""

        kwargs = {
            k: v for k, v in _DEFAULT_ASTRO_KWARGS.items() if k != "sigma_parallax"
        }
        with pytest.raises(TypeError, match="Must specify either"):
            hm.StandardGaiaAstrometry().default_prior(**kwargs)


class TestPriorValidation:
    """Tests for prior validation."""

    def test_prior_can_be_created_with_any_params(self):
        """Test that priors can be created with any combination of params.

        Validation of required params happens in the sampler, not the prior.
        """
        # Missing orientation params are fine at prior level
        HarvPrior(
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

    def test_prior_with_all_params(self):
        """Test creating a prior with all optional params."""
        HarvPrior(
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


class TestDictLinearPrior:
    """Tests for dict-form linear prior."""

    def test_dict_linear_prior_is_dict(self):
        """Dict with all Normal entries is stored as dict."""
        prior = HarvPrior(
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
        prior = HarvPrior(
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
        prior = HarvPrior(
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
        prior = hm.StandardRV().default_prior(**_DEFAULT_RV_KWARGS)
        assert isinstance(prior.linear_prior, dict)

    def test_dict_with_quantity_distribution(self):
        """QuantityDistribution entries in dict are accepted."""
        prior = HarvPrior(
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

    def test_parameter_counting_rv_with_extension_priors(self):
        """Test that RV with extension priors still has 4 nonlinear parameters."""
        rv_prior_offsets = hm.StandardRV().default_prior(
            **_DEFAULT_RV_KWARGS,
            inst2=QD(dist.Normal(0, 5), "km/s"),
        )
        assert "inst2" in rv_prior_offsets.extension_priors
        # inst2 is NOT in linear_prior until routing happens at run-time
        assert "inst2" not in rv_prior_offsets.linear_prior

    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with same seed."""
        prior = hm.StandardRV().default_prior(**_DEFAULT_RV_KWARGS)

        samples1 = prior.sample_nonlinear(jr.key(42), n_samples=100)
        samples2 = prior.sample_nonlinear(jr.key(42), n_samples=100)

        # Should be identical
        assert (samples1["period"] == samples2["period"]).all()
        assert (samples1["eccentricity"] == samples2["eccentricity"]).all()


class TestParameterizationDefaultPriors:
    """Tests for ``parameterization.default_prior(...)`` on each parameterization.

    Each concrete parameterization owns its default-prior construction; these
    tests cover the parameter sets and override behavior of all four supported
    parameterizations.
    """

    def test_standard_rv_keys(self):
        prior = hm.StandardRV().default_prior(**_DEFAULT_RV_KWARGS)
        assert sorted(prior.nonlinear_priors.keys()) == [
            "arg_peri",
            "eccentricity",
            "period",
            "phase_peri",
        ]
        assert sorted(prior.linear_prior.keys()) == ["rv_semiamp", "v_sys"]

    def test_ecosw_esinw_rv_keys(self):
        prior = hm.EcoswEsinwRV().default_prior(**_DEFAULT_RV_KWARGS)
        assert sorted(prior.nonlinear_priors.keys()) == [
            "ecosw",
            "esinw",
            "period",
            "phase_peri",
        ]
        assert sorted(prior.linear_prior.keys()) == ["rv_semiamp", "v_sys"]

    def test_ecosw_esinw_sampling_ranges(self):
        prior = hm.EcoswEsinwRV().default_prior(**_DEFAULT_RV_KWARGS)
        samples = prior.sample_nonlinear(jr.key(0), n_samples=200)
        for key in ("ecosw", "esinw"):
            assert samples[key].shape == (200,)
            assert (samples[key] >= -1.0).all()
            assert (samples[key] <= 1.0).all()

    def test_ecosw_override(self):
        custom = dist.Uniform(-0.5, 0.5)
        prior = hm.EcoswEsinwRV().default_prior(**_DEFAULT_RV_KWARGS, ecosw=custom)
        assert prior.nonlinear_priors["ecosw"] is custom

    def test_standard_gaia_keys(self):
        prior = hm.StandardGaiaAstrometry().default_prior(**_DEFAULT_ASTRO_KWARGS)
        assert sorted(prior.nonlinear_priors.keys()) == [
            "arg_peri",
            "cos_i",
            "eccentricity",
            "lon_asc_node",
            "period",
            "phase_peri",
        ]
        assert sorted(prior.linear_prior.keys()) == [
            "dec0",
            "parallax",
            "pmdec",
            "pmra",
            "ra0",
            "semi_major_axis",
        ]

    def test_thiele_innes_keys(self):
        prior = ThieleInnesGaiaAstrometry().default_prior(**_DEFAULT_ASTRO_KWARGS)
        assert sorted(prior.nonlinear_priors.keys()) == [
            "eccentricity",
            "period",
            "phase_peri",
        ]
        assert sorted(prior.linear_prior.keys()) == [
            "dec0",
            "parallax",
            "pmdec",
            "pmra",
            "ra0",
            "ti_A",
            "ti_B",
            "ti_F",
            "ti_G",
        ]

    def test_thiele_innes_ti_priors_use_period_dependent_scale(self):
        """Each TI constant should default to PeriodDependentSemiMajorAxisPrior."""
        prior = ThieleInnesGaiaAstrometry().default_prior(**_DEFAULT_ASTRO_KWARGS)
        for name in ("ti_A", "ti_B", "ti_F", "ti_G"):
            assert isinstance(
                prior.linear_prior[name], PeriodDependentSemiMajorAxisPrior
            )

    def test_thiele_innes_ti_override(self):
        custom = QD(dist.Normal(0.0, 1.0), "mas")
        prior = ThieleInnesGaiaAstrometry().default_prior(
            **_DEFAULT_ASTRO_KWARGS, ti_A=custom
        )
        assert prior.linear_prior["ti_A"] is custom

    def test_abstract_default_prior_raises(self):
        """The base-class stub raises NotImplementedError."""

        class Bogus(AbstractParameterization):
            def params(self):
                return ()

        with pytest.raises(NotImplementedError, match="default_prior"):
            Bogus().default_prior()


class TestDefaultSB2Prior:
    """Tests for the module-level ``default_sb2_prior`` factory."""

    def test_creation_and_keys(self):

        prior = default_sb2_prior(**_DEFAULT_RV_KWARGS)
        assert sorted(prior.nonlinear_priors.keys()) == [
            "arg_peri",
            "eccentricity",
            "period",
            "phase_peri",
        ]
        assert sorted(prior.linear_prior.keys()) == [
            "primary.rv_semiamp",
            "secondary.rv_semiamp",
            "v_sys",
        ]

    def test_custom_component_names(self):

        prior = default_sb2_prior(**_DEFAULT_RV_KWARGS, component_names=("A", "B"))
        assert sorted(prior.linear_prior.keys()) == [
            "A.rv_semiamp",
            "B.rv_semiamp",
            "v_sys",
        ]
