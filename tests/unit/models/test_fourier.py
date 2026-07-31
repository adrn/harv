"""Tests for the Kepler-free Fourier parameterizations."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q, ustrip

import harv
import harv.models as hm
from harv.distributions import QuantityDistribution as QD
from harv.samplers._prior_resolution import effective_linear_prior_from_prior
from harv.simulate import simulate_gaia_epoch_astrometry, simulate_rv_sb1_data
from harv.stats import MarginalizedLinear

RV_SCALES = {
    "period_min": Q(5.0, "day"),
    "period_max": Q(1000.0, "day"),
    "sigma_amp": Q(30.0, "km/s"),
    "sigma_v0": Q(10.0, "km/s"),
}
GAIA_SCALES = {
    "period_min": Q(20.0, "day"),
    "period_max": Q(2000.0, "day"),
    "sigma_amp": Q(5.0, "mas"),
    "sigma_pos": Q(100.0, "mas"),
    "sigma_pm": Q(50.0, "mas/yr"),
    "sigma_parallax": Q(50.0, "mas"),
}


def _rv_data():
    data, _ = simulate_rv_sb1_data(
        seed=7,
        n_obs=40,
        period=Q(37.0, "day"),
        eccentricity=0.2,
        rv_semiamp=Q(6.0, "km/s"),
        rv_err=Q(0.2, "km/s"),
    )
    return data


def _gaia_data():
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


class TestParams:
    def test_rv_params_order_and_count(self):
        p = hm.FourierRV(n_terms=3)
        names = [pi.name for pi in p.params()]
        assert names[0] == "period"
        assert names[1:] == [
            "cos_amp_1", "sin_amp_1", "cos_amp_2", "sin_amp_2",
            "cos_amp_3", "sin_amp_3", "v_sys",
        ]
        assert all(pi.linear for pi in p.linear_params())
        assert len(p.nonlinear_params()) == 1

    def test_gaia_params_order_and_count(self):
        p = hm.FourierGaiaAstrometry(n_terms=2)
        names = [pi.name for pi in p.params()]
        assert names[:6] == ["period", "ra0", "dec0", "pmra", "pmdec", "parallax"]
        assert names[6:] == [
            "ti_A_1", "ti_B_1", "ti_F_1", "ti_G_1",
            "ti_A_2", "ti_B_2", "ti_F_2", "ti_G_2",
        ]

    def test_zero_terms_null_model(self):
        assert [pi.name for pi in hm.FourierRV(n_terms=0).linear_params()] == ["v_sys"]
        gaia_null = hm.FourierGaiaAstrometry(n_terms=0)
        assert [pi.name for pi in gaia_null.linear_params()] == [
            "ra0", "dec0", "pmra", "pmdec", "parallax"
        ]

    def test_negative_terms_raises(self):
        with pytest.raises(ValueError, match="n_terms"):
            hm.FourierRV(n_terms=-1)


class TestDesignMatrix:
    def test_rv_columns_match_explicit_trig(self):
        # The recurrence-built harmonics must equal explicit cos(kM)/sin(kM).
        data = _rv_data()
        model = hm.RVModel(parameterization=hm.FourierRV(n_terms=3))
        P = Q(37.0, "day")
        X = model._base_design_matrix({"period": P}, data)
        t = ustrip("day", data.time - data.t_ref)
        M = 2.0 * np.pi * t / 37.0
        expected = np.stack(
            [np.cos(1*M), np.sin(1*M), np.cos(2*M), np.sin(2*M),
             np.cos(3*M), np.sin(3*M), np.ones_like(M)], axis=-1,
        )
        # float32 suite: trig at phases of hundreds of radians carries
        # ~|M|*eps argument error, so compare loosely (structural errors
        # would be O(1)).
        assert np.allclose(np.asarray(X), expected, atol=1e-3)

    def test_gaia_columns_match_explicit_trig(self):
        data = _gaia_data()
        model = hm.GaiaAstrometryModel(
            parameterization=hm.FourierGaiaAstrometry(n_terms=2)
        )
        P = Q(100.0, "day")
        X = np.asarray(model._base_design_matrix({"period": P}, data))
        t = ustrip("day", data.time - data.t_ref)
        dt_yr = ustrip("yr", data.time - data.t_ref)
        psi = ustrip("rad", data.scan_angle)
        sp, cp = np.sin(psi), np.cos(psi)
        M = 2.0 * np.pi * t / 100.0
        base = [sp, cp, sp * dt_yr, cp * dt_yr, np.asarray(data.parallax_factor)]
        harm = []
        for k in (1, 2):
            ck, sk = np.cos(k * M), np.sin(k * M)
            harm += [ck * cp, ck * sp, sk * cp, sk * sp]
        expected = np.stack(base + harm, axis=-1)
        assert X.shape == (len(t), 5 + 8)
        assert np.allclose(X, expected, atol=1e-3)  # float32 trig at large phase

    def test_log_prob_equals_direct_marginalized_linear(self):
        # RVModel.log_prob with FourierRV == a direct MarginalizedLinear call
        # with the same design/priors/noise — the machinery adds nothing else.
        data = _rv_data()
        p = hm.FourierRV(n_terms=2)
        prior = p.default_prior(**RV_SCALES)
        model = hm.RVModel(parameterization=p)
        P = Q(41.0, "day")
        lp = model.log_prob({"period": P}, data, linear_priors=prior.linear_priors)

        X = model._base_design_matrix({"period": P}, data)
        y = jnp.asarray(ustrip("km/s", data.rv))
        err = jnp.asarray(ustrip("km/s", data.rv_err))
        scales = jnp.array([30.0, 30.0, 30.0, 30.0, 10.0])  # amp x4, v_sys
        direct = MarginalizedLinear(
            X, dist.Normal(0.0, scales), dist.Normal(0.0, err)
        ).log_prob(y)
        assert jnp.allclose(lp, direct, atol=1e-8)


class TestDefaultPrior:
    def test_rv_prior_structure(self):
        prior = hm.FourierRV(n_terms=2).default_prior(**RV_SCALES)
        assert set(prior.nonlinear_priors) == {"period"}
        assert set(prior.linear_priors) == {
            "cos_amp_1", "sin_amp_1", "cos_amp_2", "sin_amp_2", "v_sys"
        }

    def test_sigma_amp_required(self):
        kwargs = {k: v for k, v in RV_SCALES.items() if k != "sigma_amp"}
        with pytest.raises(TypeError, match="sigma_amp"):
            hm.FourierRV(n_terms=1).default_prior(**kwargs)
        # ...but not for the null model:
        prior = hm.FourierRV(n_terms=0).default_prior(**kwargs)
        assert set(prior.linear_priors) == {"v_sys"}

    def test_per_amplitude_override(self):
        prior = hm.FourierRV(n_terms=1).default_prior(
            **RV_SCALES, cos_amp_1=QD(dist.Normal(0.0, 1.0), "km/s")
        )
        assert float(prior.linear_priors["cos_amp_1"].distribution.scale) == 1.0
        assert float(prior.linear_priors["sin_amp_1"].distribution.scale) == 30.0

    def test_gaia_prior_structure_and_required_scales(self):
        prior = hm.FourierGaiaAstrometry(n_terms=1).default_prior(**GAIA_SCALES)
        assert set(prior.linear_priors) == {
            "ra0", "dec0", "pmra", "pmdec", "parallax",
            "ti_A_1", "ti_B_1", "ti_F_1", "ti_G_1",
        }
        kwargs = {k: v for k, v in GAIA_SCALES.items() if k != "sigma_pm"}
        with pytest.raises(TypeError, match="sigma_pm"):
            hm.FourierGaiaAstrometry(n_terms=1).default_prior(**kwargs)


class TestSamplerIntegration:
    def test_rejection_sampler_smoke(self):
        # Fourier parameterizations are first-class: the rejection sampler
        # runs and returns samples with the Fourier parameter names.
        data = _rv_data()
        p = hm.FourierRV(n_terms=1)
        prior = p.default_prior(**RV_SCALES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = harv.RejectionSampler(prior, hm.RVModel(parameterization=p)).run(
                data, n_prior_samples=50_000, seed=0
            )
        assert samples.n_samples > 0
        assert "period" in samples.nonlinear
        assert "cos_amp_1" in samples.linear
        # Kepler-free samples: t_peri is not advertised (no phase_peri).
        assert "t_peri" not in samples
        assert "log_period" in samples

    def test_vmap_and_jit_log_prob(self):
        data = _rv_data()
        p = hm.FourierRV(n_terms=2)
        prior = p.default_prior(**RV_SCALES)
        model = hm.RVModel(parameterization=p)
        pv = jnp.linspace(10.0, 100.0, 32)
        fn = jax.jit(
            jax.vmap(
                lambda x: model.log_prob(
                    {"period": Q(x, "day")}, data, linear_priors=prior.linear_priors
                )
            )
        )
        out = fn(pv)
        assert out.shape == (32,)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_multi_survey_offset_extension(self):
        # Linear-column extensions work with Fourier parameterizations.
        data = _rv_data()
        n = data.time.shape[0]
        indicator = np.zeros((n, 1))
        indicator[n // 2 :, 0] = 1.0
        ext = hm.MultiSurveyOffset(
            indicator_matrix=jnp.asarray(indicator), instrument_names=("b",)
        )
        p = hm.FourierRV(n_terms=1)
        model = hm.RVModel(parameterization=p, extensions=(ext,))
        offset_names = [pi.name for pi in ext.extra_params()]
        prior = p.default_prior(
            **RV_SCALES,
            **{nm: QD(dist.Normal(0.0, 5.0), "km/s") for nm in offset_names},
        )
        eff = effective_linear_prior_from_prior(prior, model)
        lp = model.log_prob({"period": Q(37.0, "day")}, data, linear_priors=eff)
        assert bool(jnp.isfinite(lp))
