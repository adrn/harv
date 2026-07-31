"""End-to-end tests: periodogram -> interim period prior -> sampler -> reweighting.

These exercise the full pipeline promised by the periodogram feature:
rejection-sampling acceptance improves dramatically with a tailored interim
period prior, and per-source interim priors remain valid for downstream
hierarchical (Hogg/Myers/Bovy-style) importance reweighting.
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest
from jax.scipy.special import logsumexp
from unxt import Q, ustrip

import harv.models as hm
import harv.periodogram as hp
from harv.data import SourceData
from harv.distributions import QD
from harv.models import (
    GaiaAstrometryModel,
    HarvPrior,
    JointModel,
    RVModel,
)
from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
from harv.samplers import RejectionSampler
from harv.simulate import simulate_gaia_epoch_astrometry, simulate_rv_sb1_data

P_MIN = Q(5.0, "day")
P_MAX = Q(2000.0, "day")
RV_SCALES = {"sigma_K0": Q(30.0, "km/s"), "sigma_v0": Q(30.0, "km/s")}


def _fourier_rv_prior(n_terms: int = 2, p_min: Q = P_MIN, p_max: Q = P_MAX):
    """Explicit Fourier prior driving the RV periodogram."""
    return hm.FourierRV(n_terms=n_terms).default_prior(
        period_min=p_min,
        period_max=p_max,
        sigma_amp=Q(30.0, "km/s"),
        sigma_v0=Q(30.0, "km/s"),
    )


def _fourier_gaia_prior(n_terms: int = 2, p_min: Q = Q(20.0, "day"), p_max: Q = P_MAX):
    """Explicit Fourier prior driving the Gaia periodogram."""
    return hm.FourierGaiaAstrometry(n_terms=n_terms).default_prior(
        period_min=p_min,
        period_max=p_max,
        sigma_amp=Q(20.0, "mas"),
        sigma_pos=Q(500.0, "mas"),
        sigma_pm=Q(500.0, "mas/yr"),
        sigma_parallax=Q(500.0, "mas"),
    )


class TestRVAcceptance:
    @pytest.mark.parametrize(("builder", "min_ratio"), [("tempered", 20), ("peaks", 3)])
    def test_acceptance_improvement(self, builder: str, min_ratio: int):
        # Moderate-SNR regime (the Joker's): period is the acceptance
        # bottleneck. At extreme SNR the other nonlinear parameters dominate
        # rejection and a period prior alone cannot raise the acceptance rate.
        data, _ = simulate_rv_sb1_data(
            seed=42,
            n_obs=16,
            period=Q(123.0, "day"),
            eccentricity=0.3,
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(1.5, "km/s"),
        )
        result = hp.periodogram(
            data, prior=_fourier_rv_prior(), period_min=P_MIN, period_max=P_MAX
        )
        if builder == "tempered":
            period_prior = hp.tempered_period_prior(result, beta=1.0, floor=0.1)
        else:
            period_prior = hp.peak_period_prior(result, floor=0.1)

        base = hm.StandardRV().default_prior(
            period_min=P_MIN, period_max=P_MAX, **RV_SCALES
        )
        tailored = hm.StandardRV().default_prior(period=period_prior, **RV_SCALES)

        n_prior = 100_000
        s_base = RejectionSampler(base, RVModel()).run(
            data, n_prior_samples=n_prior, seed=0
        )
        s_tail = RejectionSampler(tailored, RVModel()).run(
            data, n_prior_samples=n_prior, seed=0
        )

        # Same prior-sample budget: the tailored prior accepts far more.
        assert s_tail.n_samples >= min_ratio * max(s_base.n_samples, 1)
        # And its posterior contains the truth:
        p = ustrip("day", s_tail["period"])
        assert abs(float(jnp.median(p)) - 123.0) < 5.0


class TestGaiaAcceptance:
    def test_acceptance_improvement(self):
        # Moderate astrometric SNR (~2 per epoch), so that period is the
        # acceptance bottleneck rather than the other nonlinear parameters:
        data, _ = simulate_gaia_epoch_astrometry(
            seed=3,
            n_obs=60,
            period=Q(100.0, "day"),
            eccentricity=0.2,
            semi_major_axis=Q(1.0, "mas"),
            parallax=Q(20.0, "mas"),
            mu_alpha=Q(15.0, "mas/yr"),
            mu_delta=Q(-8.0, "mas/yr"),
            al_error=Q(0.5, "mas"),
        )
        param = ThieleInnesGaiaAstrometry.from_data(data)
        scales = {
            "sigma_a0": Q(5.0, "AU"),
            "sigma_parallax": Q(50.0, "mas"),
            "sigma_pos": Q(100.0, "mas"),
            "sigma_vtan": Q(100.0, "km/s"),
        }
        base = param.default_prior(
            period_min=Q(20.0, "day"), period_max=P_MAX, **scales
        )
        result = hp.periodogram(
            data,
            prior=_fourier_gaia_prior(),
            period_min=Q(20.0, "day"),
            period_max=P_MAX,
        )
        tailored = param.default_prior(
            period=hp.tempered_period_prior(result, beta=1.0, floor=0.1), **scales
        )

        model = GaiaAstrometryModel(parameterization=param)
        n_prior = 100_000
        s_base = RejectionSampler(base, model).run(
            data, n_prior_samples=n_prior, seed=1
        )
        s_tail = RejectionSampler(tailored, model).run(
            data, n_prior_samples=n_prior, seed=1
        )

        assert s_tail.n_samples >= 5 * max(s_base.n_samples, 1)
        p = ustrip("day", s_tail["period"])
        assert abs(float(jnp.median(p)) - 100.0) < 5.0


class TestJointEndToEnd:
    def test_joint_periodogram_prior_run(self):
        gaia_data, _ = simulate_gaia_epoch_astrometry(
            seed=3,
            n_obs=80,
            period=Q(100.0, "day"),
            eccentricity=0.2,
            semi_major_axis=Q(2.0, "mas"),
            parallax=Q(20.0, "mas"),
            al_error=Q(0.05, "mas"),
        )
        rv_data, _ = simulate_rv_sb1_data(
            seed=11,
            n_obs=30,
            period=Q(100.0, "day"),
            eccentricity=0.2,
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(0.3, "km/s"),
        )
        source = SourceData(astro=gaia_data, rv=rv_data)

        result = hp.periodogram(
            source,
            prior={"astro": _fourier_gaia_prior(), "rv": _fourier_rv_prior()},
            period_min=Q(20.0, "day"),
            period_max=P_MAX,
        )
        period_prior = hp.tempered_period_prior(result, beta=1.0, floor=0.1)

        two_pi = 2.0 * float(jnp.pi)
        prior = HarvPrior(
            nonlinear_priors={
                "period": period_prior,
                "eccentricity": ndist.Beta(0.867, 3.03),
                "phase_peri": ndist.Uniform(0.0, 1.0),
                "arg_peri": QD(ndist.Uniform(0.0, two_pi), "rad"),
                "cos_i": ndist.Uniform(-1.0, 1.0),
                "lon_asc_node": QD(ndist.Uniform(0.0, two_pi), "rad"),
            },
            linear_priors={
                "astro.ra0": QD(ndist.Normal(0.0, 100.0), "mas"),
                "astro.dec0": QD(ndist.Normal(0.0, 100.0), "mas"),
                "astro.pmra": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
                "astro.pmdec": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
                "astro.parallax": QD(ndist.Normal(20.0, 5.0), "mas"),
                "astro.semi_major_axis": QD(ndist.Normal(0.0, 20.0), "mas"),
                "rv.rv_semiamp": QD(ndist.Normal(0.0, 30.0), "km/s"),
                "rv.v_sys": QD(ndist.Normal(0.0, 30.0), "km/s"),
            },
        )
        joint = JointModel.for_rv_and_gaia(
            components={"astro": GaiaAstrometryModel(), "rv": RVModel()}
        )
        samples = RejectionSampler(prior, joint).run(
            source, n_prior_samples=100_000, seed=2, ignore_non_finite=True
        )
        assert samples.n_samples > 0
        p = ustrip("day", samples["period"])
        assert abs(float(jnp.median(p)) - 100.0) < 5.0


class TestReweightingConsistency:
    """Hyperparameter estimates agree between log-uniform and per-source priors.

    Simulates a small population with ln-period ~ Normal(mu, sigma), fits each
    source with (a) a shared log-uniform interim period prior and (b) a
    per-source tempered periodogram prior, then estimates mu with the
    Hogg/Myers/Bovy importance-reweighting estimator using the per-sample
    ``ln_pint_period`` column. The two estimates must agree within Monte Carlo
    error — per-source interim priors do not bias the population inference.
    """

    MU_TRUE = float(np.log(100.0))
    SIGMA_POP = 0.8
    N_SOURCES = 8

    def _simulate_population(self):
        rng = np.random.default_rng(2026)
        sources = []
        for i in range(self.N_SOURCES):
            period = float(
                np.clip(np.exp(rng.normal(self.MU_TRUE, self.SIGMA_POP)), 10.0, 800.0)
            )
            data, _ = simulate_rv_sb1_data(
                seed=1000 + i,
                n_obs=16,
                baseline=Q(6.0, "yr"),
                period=Q(period, "day"),
                eccentricity=0.1,
                rv_semiamp=Q(float(rng.uniform(2.5, 5.0)), "km/s"),
                rv_err=Q(1.5, "km/s"),
            )
            sources.append(data)
        return sources

    @staticmethod
    def _mu_hat(samples_list, mu_grid: np.ndarray, sigma: float) -> float:
        """Argmax over mu of the reweighting population log-likelihood."""
        total = np.zeros_like(mu_grid)
        for s in samples_list:
            ln_p = jnp.log(ustrip("day", s["period"]))
            ln_pint = ustrip("", s[hp.LN_PINT_PERIOD_KEY])
            for k, mu in enumerate(mu_grid):
                ln_pop = ndist.Normal(mu, sigma).log_prob(ln_p)
                total[k] += float(logsumexp(ln_pop - ln_pint) - jnp.log(len(ln_p)))
        return float(mu_grid[int(np.argmax(total))])

    def test_population_mu_agrees(self):
        sources = self._simulate_population()
        base_prior = hm.StandardRV().default_prior(
            period_min=P_MIN, period_max=P_MAX, **RV_SCALES
        )
        base_period_prior = QD(
            ndist.LogUniform(float(ustrip("day", P_MIN)), float(ustrip("day", P_MAX))),
            "day",
        )
        # One shared grid config for the whole population (same knot count):
        frequency = hp.frequency_grid(
            t_span=Q(6.0, "yr"), period_min=P_MIN, period_max=P_MAX
        )

        samples_base, samples_tail = [], []
        for i, data in enumerate(sources):
            s_a = RejectionSampler(base_prior, RVModel()).run(
                data, n_prior_samples=600_000, max_posterior_samples=128, seed=i
            )
            assert s_a.n_samples > 5, "population setup must yield accepted samples"
            samples_base.append(hp.attach_ln_pint(s_a, base_period_prior))

            result = hp.periodogram(data, frequency, prior=_fourier_rv_prior())
            period_prior = hp.tempered_period_prior(result, beta=1.0, floor=0.1)
            tailored = hm.StandardRV().default_prior(period=period_prior, **RV_SCALES)
            s_b = RejectionSampler(tailored, RVModel()).run(
                data, n_prior_samples=100_000, max_posterior_samples=128, seed=i
            )
            samples_tail.append(hp.attach_ln_pint(s_b, period_prior))

        mu_grid = np.linspace(np.log(30.0), np.log(300.0), 231)
        mu_a = self._mu_hat(samples_base, mu_grid, self.SIGMA_POP)
        mu_b = self._mu_hat(samples_tail, mu_grid, self.SIGMA_POP)

        # The two interim-prior choices give consistent population estimates,
        # and both recover the truth within ~2 standard errors
        # (SE ~ sigma/sqrt(N) ~ 0.28):
        assert abs(mu_a - mu_b) < 0.15
        assert abs(mu_a - self.MU_TRUE) < 0.6
        assert abs(mu_b - self.MU_TRUE) < 0.6
