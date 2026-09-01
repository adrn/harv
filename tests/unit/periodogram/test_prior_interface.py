"""Tests for the periodogram's explicit-prior interface.

The periodogram takes a standard ``HarvPrior`` built from a Fourier
parameterization — there are no data-driven defaults and no hidden scale
assumptions. These tests cover prior validation, period-dependent amplitude
priors, and linear-column extensions.
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv.models as hm
import harv.periodogram as hp
from harv.data import SourceData
from harv.distributions import QuantityDistribution as QD
from harv.models.priors.custom_priors import PeriodDependentKPrior
from harv.simulate import simulate_gaia_epoch_astrometry, simulate_rv_sb1_data

RV_KW = {
    "period_min": Q(1.0, "day"),
    "period_max": Q(5000.0, "day"),
    "sigma_amp": Q(30.0, "km/s"),
    "sigma_v0": Q(50.0, "km/s"),
}
GAIA_KW = {
    "period_min": Q(1.0, "day"),
    "period_max": Q(5000.0, "day"),
    "sigma_amp": Q(20.0, "mas"),
    "sigma_pos": Q(500.0, "mas"),
    "sigma_pm": Q(500.0, "mas/yr"),
    "sigma_parallax": Q(500.0, "mas"),
}


def _rv():
    data, _ = simulate_rv_sb1_data(
        seed=7,
        n_obs=40,
        period=Q(37.0, "day"),
        eccentricity=0.2,
        rv_semiamp=Q(6.0, "km/s"),
        rv_err=Q(0.2, "km/s"),
    )
    return data


def _gaia():
    data, _ = simulate_gaia_epoch_astrometry(
        seed=3,
        n_obs=80,
        period=Q(100.0, "day"),
        eccentricity=0.2,
        semi_major_axis=Q(2.0, "mas"),
        parallax=Q(20.0, "mas"),
        mu_alpha=Q(15.0, "mas/yr"),
        mu_delta=Q(-8.0, "mas/yr"),
        al_error=Q(0.05, "mas"),
    )
    return data


class TestPriorValidation:
    def test_keplerian_prior_rejected(self):
        # A StandardRV prior has extra nonlinear params the Fourier trial
        # model cannot scan.
        data = _rv()
        bad = hm.StandardRV().default_prior(
            period_min=Q(1.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(10.0, "km/s"),
        )
        with pytest.raises(TypeError, match="no nonlinear parameters besides"):
            hp.periodogram(data, prior=bad, period_min=Q(5.0, "day"))

    def test_unknown_linear_name_rejected(self):
        data = _rv()
        prior = hm.FourierRV(n_terms=2).default_prior(**RV_KW)
        prior.linear_priors["bogus"] = QD(dist.Normal(0.0, 1.0), "km/s")
        with pytest.raises(TypeError, match="not parameters of"):
            hp.periodogram(data, prior=prior, period_min=Q(5.0, "day"))

    def test_missing_linear_entry_rejected(self):
        data = _rv()
        prior = hm.FourierRV(n_terms=2).default_prior(**RV_KW)
        del prior.linear_priors["sin_amp_2"]
        with pytest.raises(TypeError, match="missing entries"):
            hp.periodogram(data, prior=prior, period_min=Q(5.0, "day"))

    def test_prior_for_wrong_n_terms_rejected(self):
        # A 1-term prior cannot drive a 2-term model.
        data = _rv()
        prior = hm.FourierRV(n_terms=1).default_prior(**RV_KW)
        with pytest.raises(TypeError, match="missing entries"):
            hp.periodogram(data, prior=prior, period_min=Q(5.0, "day"), n_terms=2)


class TestPeriodDependentAmplitudePrior:
    def test_callable_amplitude_prior_resolves_per_period(self):
        """A LinearPriorCallable amplitude prior is resolved at each trial period.

        It flows through the standard model machinery with no special
        plumbing, and tilts Delta relative to a constant-scale prior.
        """
        data = _rv()
        k_prior = PeriodDependentKPrior(sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr"))
        tilted = hm.FourierRV(n_terms=1).default_prior(
            period_min=Q(1.0, "day"),
            period_max=Q(5000.0, "day"),
            sigma_v0=Q(50.0, "km/s"),
            cos_amp_1=k_prior,
            sin_amp_1=k_prior,
        )
        flat = hm.FourierRV(n_terms=1).default_prior(**RV_KW)
        f = hp.frequency_grid(data, period_min=Q(5.0, "day"))
        r_tilt = hp.periodogram(data, f, prior=tilted, n_terms=1)
        r_flat = hp.periodogram(data, f, prior=flat, n_terms=1)
        assert jnp.all(jnp.isfinite(r_tilt.delta_ln_likelihood))
        # The period-dependent prior genuinely changes the statistic:
        assert not jnp.allclose(r_tilt.delta_ln_likelihood, r_flat.delta_ln_likelihood)


class TestExtensions:
    def test_multi_survey_offset_columns(self):
        """Survey offsets ride the standard extension machinery."""
        data = _rv()
        n = data.time.shape[0]
        indicator = np.zeros((n, 1))
        indicator[n // 2 :, 0] = 1.0
        ext = hm.MultiSurveyOffset(
            indicator_matrix=jnp.asarray(indicator), instrument_names=("b",)
        )
        prior = hm.FourierRV(n_terms=1).default_prior(
            **RV_KW, b=QD(dist.Normal(0.0, 5.0), "km/s")
        )
        result = hp.periodogram(
            data,
            prior=prior,
            period_min=Q(5.0, "day"),
            n_terms=1,
            extensions=(ext,),
        )
        assert jnp.all(jnp.isfinite(result.delta_ln_likelihood))

    def test_missing_extension_prior_raises(self):
        data = _rv()
        n = data.time.shape[0]
        indicator = np.zeros((n, 1))
        indicator[n // 2 :, 0] = 1.0
        ext = hm.MultiSurveyOffset(
            indicator_matrix=jnp.asarray(indicator), instrument_names=("b",)
        )
        prior = hm.FourierRV(n_terms=1).default_prior(**RV_KW)
        with pytest.raises(ValueError, match="Missing required prior"):
            hp.periodogram(
                data,
                prior=prior,
                period_min=Q(5.0, "day"),
                n_terms=1,
                extensions=(ext,),
            )

    def test_nonlinear_extension_rejected(self):
        # Jitter adds a nonlinear parameter the periodogram cannot scan.
        data = _rv()
        prior = hm.FourierRV(n_terms=1).default_prior(
            **RV_KW, jitter=QD(dist.HalfNormal(0.5), "km/s")
        )
        with pytest.raises(TypeError, match="nonlinear parameter"):
            hp.periodogram(
                data,
                prior=prior,
                period_min=Q(5.0, "day"),
                n_terms=1,
                extensions=(hm.Jitter("km/s"),),
            )


class TestContainerPriors:
    def test_mapping_missing_entry_raises(self):
        gaia = _gaia()
        rv, _ = simulate_rv_sb1_data(
            seed=11,
            n_obs=30,
            period=Q(100.0, "day"),
            eccentricity=0.2,
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(0.3, "km/s"),
        )
        source = SourceData(gaia=gaia, rv=rv)
        priors = {"gaia": hm.FourierGaiaAstrometry(n_terms=2).default_prior(**GAIA_KW)}
        with pytest.raises(TypeError, match="no entry for dataset"):
            hp.periodogram(source, prior=priors, period_min=Q(20.0, "day"))
