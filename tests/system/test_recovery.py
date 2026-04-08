"""System tests: posterior recovery with simulated data.

These tests verify that the rejection sampler returns posterior samples that are
statistically consistent with the true parameters used to generate the data.
They are organized into three regimes:

High-SNR recovery (tight prior)
    K/sigma ≈ 2-3, period prior within 2x of truth.  The posterior should
    concentrate near the true parameter values; we verify by checking that the
    true period is covered by the posterior's central 90% credible interval.

Multi-survey offset recovery
    Two-instrument RV with a known zero-point offset.  The offset posterior
    should contain the injected value within its 90% credible interval.

Low-SNR / low-epoch (broad / multi-modal posterior)
    K/sigma < 1 with few observations and a broad period prior.  The posterior
    is naturally broad and may be multi-modal (period aliases).  We verify two
    things:
      - The posterior is genuinely broad (many period scales are plausible).
      - The true period is covered somewhere inside the posterior's full range.

Astrometry likelihood sanity
    With the full Thiele-Innes astrometry model and proper simulation, high-SNR
    rejection sampling is computationally intractable (very narrow posterior in
    6 nonlinear dimensions).  Instead we verify that the likelihood is maximized
    near the true parameters by showing that the true-parameter log-prob
    significantly exceeds the median log-prob under the prior.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from unxt import Quantity, ustrip

from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.params import GaiaAstrometryParameters
from harv.priors.rejection import RejectionPrior
from harv.samplers.rejection import RejectionSampler
from harv.simulate.astrometry import simulate_gaia_epoch_astrometry
from harv.simulate.rv import simulate_rv_multisurv_data, simulate_rv_sb1_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _period_quantile(samples, q: float) -> float:
    """Return the q-th percentile (0-100) of the period samples in days."""
    return float(jnp.percentile(samples["period"].value, q))


# ---------------------------------------------------------------------------
# High-SNR RV: posterior should cover the true period
# ---------------------------------------------------------------------------


class TestHighSNRRVRecovery:
    """High-SNR single-instrument RV with a tight period prior.

    True parameters: P=100 d, e=0.3, K=5 km/s, s=2 km/s (K/s=2.5), 30 obs.
    Period prior: log-uniform [50, 200] days.

    Expected: ~800 accepted samples from 500 k draws; the posterior 90%
    credible interval for period should contain the true value.
    """

    @pytest.fixture(scope="class")
    def rv_samples_high_snr(self):
        data, true = simulate_rv_sb1_data(
            seed=42,
            n_obs=30,
            period=Quantity(100.0, "day"),
            eccentricity=0.3,
            K=Quantity(5.0, "km/s"),
            v0=Quantity(0.0, "km/s"),
            rv_err=Quantity(2.0, "km/s"),
        )
        prior = RejectionPrior.default_rv(period_min=50.0, period_max=200.0)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=500_000, seed=42)
        return samples, true

    def test_enough_accepted_samples(self, rv_samples_high_snr):
        samples, _ = rv_samples_high_snr
        assert samples.n_samples >= 20, (
            f"Expected ≥20 accepted samples; got {samples.n_samples}. "
            "Acceptance rate may be too low — check simulation or prior."
        )

    def test_true_period_in_90pct_credible_interval(self, rv_samples_high_snr):
        """True period should lie within the posterior 90% CI."""
        samples, true = rv_samples_high_snr
        true_period = float(ustrip("day", true["period"]))
        p5 = _period_quantile(samples, 5)
        p95 = _period_quantile(samples, 95)
        assert (
            p5 <= true_period <= p95
        ), f"True period {true_period:.1f} d not in 90% CI [{p5:.1f}, {p95:.1f}] d."

    def test_k_v0_recovered(self, rv_samples_high_snr):
        """K and v0 (marginalized linear params) should be consistent with truth."""
        samples, true = rv_samples_high_snr
        true_K = float(ustrip("km/s", true["K"]))
        true_v0 = float(ustrip("km/s", true["v0"]))

        K_samples = samples["K"].value
        v0_samples = samples["v0"].value

        K_lo, K_hi = (
            float(jnp.percentile(K_samples, 5)),
            float(jnp.percentile(K_samples, 95)),
        )
        v0_lo, v0_hi = (
            float(jnp.percentile(v0_samples, 5)),
            float(jnp.percentile(v0_samples, 95)),
        )

        assert (
            K_lo <= true_K <= K_hi
        ), f"True K={true_K:.2f} km/s not in 90% CI [{K_lo:.2f}, {K_hi:.2f}]."
        assert (
            v0_lo <= true_v0 <= v0_hi
        ), f"True v0={true_v0:.2f} km/s not in 90% CI [{v0_lo:.2f}, {v0_hi:.2f}]."

    def test_eccentricity_positive(self, rv_samples_high_snr):
        """Sampled eccentricities must lie in [0, 1)."""
        samples, _ = rv_samples_high_snr
        ecc = samples["eccentricity"]
        assert jnp.all(ecc >= 0.0)
        assert jnp.all(ecc < 1.0)


# ---------------------------------------------------------------------------
# Multi-survey RV: offset should be recovered
# ---------------------------------------------------------------------------


class TestMultiSurveyRVRecovery:
    """Two-instrument RV with a known 2 km/s zero-point offset.

    Period prior: tight [50, 200] days.  The injected offset for the second
    instrument should lie within the posterior 90% CI.
    """

    @pytest.fixture(scope="class")
    def multisurv_samples(self):
        instruments = {"keck": None, "harps": Quantity(2.0, "km/s")}
        source_data, true = simulate_rv_multisurv_data(
            instruments=instruments,
            seed=10,
            n_obs_per_instrument=20,
            period=Quantity(100.0, "day"),
            eccentricity=0.3,
            K=Quantity(5.0, "km/s"),
            v0=Quantity(0.0, "km/s"),
            rv_err=Quantity(2.0, "km/s"),
        )
        prior = RejectionPrior.default_rv(
            period_min=50.0,
            period_max=200.0,
            offsets={"keck": None, "harps": dist.Normal(0.0, 5.0)},
        )
        sampler = RejectionSampler(prior)
        samples = sampler.run(source_data, n_prior_samples=500_000, seed=10)
        return samples, true

    def test_enough_accepted_samples(self, multisurv_samples):
        samples, _ = multisurv_samples
        assert samples.n_samples >= 10

    def test_true_period_in_90pct_credible_interval(self, multisurv_samples):
        samples, true = multisurv_samples
        true_period = float(ustrip("day", true["period"]))
        p5 = _period_quantile(samples, 5)
        p95 = _period_quantile(samples, 95)
        assert (
            p5 <= true_period <= p95
        ), f"True period {true_period:.1f} d not in 90% CI [{p5:.1f}, {p95:.1f}] d."

    def test_injected_offset_in_90pct_credible_interval(self, multisurv_samples):
        """Injected harps offset (2 km/s) should be covered by the posterior."""
        samples, true = multisurv_samples
        true_offset = float(ustrip("km/s", true["offset_harps"]))

        offset_samples = samples["harps"].value
        off_lo = float(jnp.percentile(offset_samples, 5))
        off_hi = float(jnp.percentile(offset_samples, 95))
        assert off_lo <= true_offset <= off_hi, (
            f"True offset {true_offset:.2f} km/s not in 90% CI [{off_lo:.2f}, "
            f"{off_hi:.2f}]."
        )

    def test_offset_key_present(self, multisurv_samples):
        samples, _ = multisurv_samples
        assert "harps" in samples.keys()  # noqa: SIM118
        assert "keck" not in samples.keys()  # noqa: SIM118


# ---------------------------------------------------------------------------
# Low-SNR / low-epoch: broad (multi-modal) posterior
# ---------------------------------------------------------------------------


class TestLowSNRBroadPosterior:
    """Low-SNR RV with very few epochs: posterior is broad and likely multi-modal.

    True parameters: P=100 d, K=2 km/s, s=5 km/s (K/s=0.4), 15 obs.
    Prior: log-uniform [20, 500] days.

    We verify:
    - Posterior is broad (std of log₁₀(period) > 0.3, i.e. spanning more than
      a factor of 2 in period).
    - True period is contained within the posterior's full range
      [min, max] of accepted samples.  This is satisfied if the true period is
      within the prior and samples span the prior reasonably.
    """

    @pytest.fixture(scope="class")
    def low_snr_samples(self):
        data, true = simulate_rv_sb1_data(
            seed=7,
            n_obs=15,
            period=Quantity(100.0, "day"),
            eccentricity=0.2,
            K=Quantity(2.0, "km/s"),
            v0=Quantity(0.0, "km/s"),
            rv_err=Quantity(5.0, "km/s"),
        )
        prior = RejectionPrior.default_rv(period_min=20.0, period_max=500.0)
        sampler = RejectionSampler(prior)
        samples = sampler.run(data, n_prior_samples=200_000, seed=7)
        return samples, true

    def test_enough_accepted_samples(self, low_snr_samples):
        """Low-SNR data should still yield accepted samples from the broad prior."""
        samples, _ = low_snr_samples
        assert (
            samples.n_samples >= 10
        ), f"Expected ≥10 accepted samples; got {samples.n_samples}."

    def test_posterior_is_broad(self, low_snr_samples):
        """With low SNR, the period posterior should span > factor of 2 in period."""
        samples, _ = low_snr_samples
        log_periods = jnp.log10(samples["period"].value)
        std_log_p = float(jnp.std(log_periods))
        assert std_log_p > 0.3, (
            f"std(log₁₀ period) = {std_log_p:.3f} — posterior not broad enough. "
            "Low-SNR data should produce a wide period distribution."
        )

    def test_true_period_within_posterior_range(self, low_snr_samples):
        """True period (100 d) should be contained in the posterior's range."""
        samples, true = low_snr_samples
        true_period = float(ustrip("day", true["period"]))
        p_min = float(jnp.min(samples["period"].value))
        p_max = float(jnp.max(samples["period"].value))
        assert p_min <= true_period <= p_max, (
            f"True period {true_period:.1f} d not within posterior range "
            f"[{p_min:.1f}, {p_max:.1f}] d."
        )

    def test_period_spans_multiple_aliases(self, low_snr_samples):
        """With few epochs, period aliases should create a multi-modal distribution.

        We check that the posterior covers at least a factor of 3 in period
        (the range between P/2 and 2P aliases of a 100-day orbit).
        """
        samples, _ = low_snr_samples
        periods = samples["period"].value
        period_range_factor = float(jnp.max(periods) / jnp.min(periods))
        assert period_range_factor >= 3.0, (
            f"Period range factor {period_range_factor:.1f} < 3 — "
            "expected multi-modal coverage across period aliases."
        )


# ---------------------------------------------------------------------------
# Astrometry: likelihood correctness at true parameters
# ---------------------------------------------------------------------------


class TestAstrometryLikelihoodSanity:
    """Verify that the astrometry likelihood is near its maximum at true params.

    Rejection sampling is computationally intractable for high-SNR astrometry
    (the posterior is very narrow in 6 nonlinear dimensions).  Instead we
    confirm that the likelihood computed at the true parameter values is
    substantially higher than the median likelihood under the prior — i.e., the
    likelihood function is correctly implemented and truly peaks near the truth.
    """

    @pytest.fixture(scope="class")
    def astro_lik_and_truth(self):
        data, true = simulate_gaia_epoch_astrometry(
            seed=42,
            n_obs=50,
            period=Quantity(300.0, "day"),
            eccentricity=0.3,
            semimajor_axis=Quantity(5.0, "mas"),
            al_error=Quantity(0.2, "mas"),
        )
        lp = dist.MultivariateNormal(
            loc=jnp.zeros(6), covariance_matrix=1000.0**2 * jnp.eye(6)
        )
        lik = GaiaAstrometryLikelihood(data=data, linear_prior=lp)
        return lik, data, true

    def test_true_params_log_prob_finite(self, astro_lik_and_truth):
        lik, _, true = astro_lik_and_truth
        period_day = float(ustrip("day", true["period"]))
        phase_peri = float(ustrip("day", true["t_peri"])) / period_day
        params = GaiaAstrometryParameters.marginalized(
            period=true["period"],
            eccentricity=true["eccentricity"],
            phase_peri=phase_peri,
            cos_i=float(jnp.cos(ustrip("rad", true["inclination"]))),
            arg_peri=float(ustrip("rad", true["arg_peri"])),
            lon_asc_node=float(ustrip("rad", true["lon_asc_node"])),
        )
        log_lik = lik.log_prob(params)
        assert jnp.isfinite(
            log_lik
        ), f"log_prob at true params is not finite: {log_lik}"

    def test_true_params_better_than_prior_median(self, astro_lik_and_truth):
        """log_prob at true params >> median log_prob under the prior.

        For informative data (SNR=25, 50 obs), the likelihood at the true
        parameters should be several hundred nats above the median prior sample.
        We use a conservative threshold of 50 nats.
        """
        lik, _, true = astro_lik_and_truth
        period_day = float(ustrip("day", true["period"]))
        phase_peri = float(ustrip("day", true["t_peri"])) / period_day

        params_true = GaiaAstrometryParameters.marginalized(
            period=true["period"],
            eccentricity=true["eccentricity"],
            phase_peri=phase_peri,
            cos_i=float(jnp.cos(ustrip("rad", true["inclination"]))),
            arg_peri=float(ustrip("rad", true["arg_peri"])),
            lon_asc_node=float(ustrip("rad", true["lon_asc_node"])),
        )
        log_lik_true = float(lik.log_prob(params_true))

        # Sample 1000 random nonlinear parameter sets from the prior
        prior = RejectionPrior.default_astrometry(period_min=100.0, period_max=1000.0)
        prior_nl = prior.sample_nonlinear(jr.PRNGKey(0), 1_000)
        prior_batch = GaiaAstrometryParameters.marginalized(
            period=Quantity(prior_nl["period"], "day"),
            eccentricity=prior_nl["eccentricity"],
            phase_peri=prior_nl["phase_peri"],
            cos_i=prior_nl["cos_i"],
            arg_peri=prior_nl["arg_peri"],
            lon_asc_node=prior_nl["lon_asc_node"],
        )
        log_liks_prior = jax.jit(jax.vmap(lik.log_prob))(prior_batch)
        median_prior_log_lik = float(jnp.median(log_liks_prior))

        improvement = log_lik_true - median_prior_log_lik
        assert improvement > 50.0, (
            f"log_prob improvement at true params over prior median = {improvement:.1f}"
            " nats. Expected > 50 nats for informative data."
        )

    def test_grid_maximum_near_true_period(self, astro_lik_and_truth):
        """A grid search over period (all other params fixed) peaks near truth."""
        lik, _, true = astro_lik_and_truth
        period_day = float(ustrip("day", true["period"]))
        phase_peri = float(ustrip("day", true["t_peri"])) / period_day
        ecc = true["eccentricity"]
        cos_i = float(jnp.cos(ustrip("rad", true["inclination"])))
        arg_peri = float(ustrip("rad", true["arg_peri"]))
        lon_asc = float(ustrip("rad", true["lon_asc_node"]))

        n_grid = 100
        test_periods = jnp.linspace(200.0, 400.0, n_grid)
        params_batch = GaiaAstrometryParameters.marginalized(
            period=Quantity(test_periods, "day"),
            eccentricity=jnp.ones(n_grid) * ecc,
            phase_peri=jnp.ones(n_grid) * phase_peri,
            cos_i=jnp.ones(n_grid) * cos_i,
            arg_peri=jnp.ones(n_grid) * arg_peri,
            lon_asc_node=jnp.ones(n_grid) * lon_asc,
        )
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
        best_period = float(test_periods[jnp.argmax(log_liks)])

        # The best grid period should be within 5% of the true value
        assert abs(best_period - period_day) / period_day < 0.05, (
            f"Grid peak period {best_period:.1f} d is more than 5% from "
            f"true period {period_day:.1f} d."
        )
