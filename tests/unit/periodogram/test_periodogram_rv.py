"""Unit tests for the RV path of harv.periodogram.periodogram."""

import warnings

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxt import Q, ustrip

import harv
import harv.models as hm
import harv.periodogram as hp
from harv.data import RVData, SourceData
from harv.distributions import QD
from harv.models.priors.custom_priors import PeriodDependentKPrior
from harv.simulate import simulate_rv_sb1_data

P_TRUE = Q(37.0, "day")


def _sim(seed: int = 7, **kwargs: object):
    defaults: dict[str, object] = {
        "n_obs": 40,
        "period": P_TRUE,
        "eccentricity": 0.2,
        "rv_semiamp": Q(6.0, "km/s"),
        "rv_err": Q(0.2, "km/s"),
    }
    defaults.update(kwargs)
    return simulate_rv_sb1_data(seed=seed, **defaults)


def _prior(n_terms: int = 2, **kwargs: object):
    """Explicit Fourier prior for the RV periodogram (no data-driven defaults)."""
    scales: dict[str, object] = {
        "period_min": Q(1.0, "day"),
        "period_max": Q(5000.0, "day"),
        "sigma_amp": Q(30.0, "km/s"),
        "sigma_v0": Q(50.0, "km/s"),
    }
    if "v_sys" in kwargs:  # explicit v_sys prior replaces the sigma_v0 scale
        scales.pop("sigma_v0")
    scales.update(kwargs)
    return hm.FourierRV(n_terms=n_terms).default_prior(**scales)


def _peak_within_grid_steps(result, p_true, n_steps: int = 2) -> bool:
    f = ustrip("1/day", result.frequency)
    df = float(f[1] - f[0])
    f_true = 1.0 / float(ustrip("day", p_true))
    f_peak = float(f[jnp.argmax(result.delta_ln_likelihood)])
    return abs(f_peak - f_true) < n_steps * df


class TestRecovery:
    def test_circular_recovery(self):
        data, _ = _sim(eccentricity=0.0)
        result = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        assert _peak_within_grid_steps(result, P_TRUE)

    def test_eccentric_recovery(self):
        data, _ = _sim(eccentricity=0.5)
        result = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        assert _peak_within_grid_steps(result, P_TRUE)

    def test_multi_term_beats_single_term_when_eccentric(self):
        data, _ = _sim(eccentricity=0.5)
        f = hp.frequency_grid(data, period_min=Q(5.0, "day"))
        r1 = hp.periodogram(data, f, prior=_prior(1), n_terms=1)
        r2 = hp.periodogram(data, f, prior=_prior(2), n_terms=2)
        i_true = jnp.argmin(
            jnp.abs(ustrip("1/day", f) - 1.0 / float(ustrip("day", P_TRUE)))
        )
        assert float(r2.delta_ln_likelihood[i_true]) > float(
            r1.delta_ln_likelihood[i_true]
        )

    def test_pure_noise_has_low_power(self):
        data, _ = _sim(rv_semiamp=Q(0.0, "km/s"))
        result = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        assert float(jnp.max(result.delta_ln_likelihood)) < 5.0


class TestInvariance:
    def test_constant_offset_invariance(self):
        """Delta is invariant to a constant shift when v_sys can absorb it.

        There is no data centering: the v_sys column carries the offset, and
        it appears in both the trial and base models, so as the v_sys prior
        widens (the shift becoming unpenalized) Delta is unchanged. Checked in
        float64 — lnL here is O(1e4), so float32 cancellation noise (~1 nat)
        swamps the effect.
        """
        with jax.enable_x64(new_val=True):
            data, _ = _sim()
            shifted = RVData(
                time=data.time,
                rv=data.rv + Q(100.0, "km/s"),
                rv_err=data.rv_err,
                t_ref=data.t_ref,
            )
            wide = _prior(2, v_sys=harv.QD(dist.Normal(0.0, 1e4), "km/s"))
            f = hp.frequency_grid(data, period_min=Q(5.0, "day"))
            r0 = hp.periodogram(data, f, prior=wide)
            r1 = hp.periodogram(shifted, f, prior=wide)
            assert jnp.allclose(
                r0.delta_ln_likelihood, r1.delta_ln_likelihood, atol=1e-4
            )

    def test_deterministic_across_calls(self):
        data, _ = _sim()
        r0 = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        r1 = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        assert jnp.array_equal(r0.delta_ln_likelihood, r1.delta_ln_likelihood)


class TestApi:
    def test_explicit_grid_conflicts_with_grid_kwargs(self):
        data, _ = _sim()
        f = hp.frequency_grid(data, period_min=Q(5.0, "day"))
        with pytest.raises(TypeError, match="Cannot specify both"):
            hp.periodogram(data, f, prior=_prior(), period_min=Q(5.0, "day"))

    def test_period_min_required_without_grid(self):
        data, _ = _sim()
        with pytest.raises(TypeError, match="period_min"):
            hp.periodogram(data, prior=_prior())

    def test_prior_is_required(self):
        data, _ = _sim()
        with pytest.raises(TypeError, match="prior"):
            hp.periodogram(data, period_min=Q(5.0, "day"))  # type: ignore[call-arg]

    def test_result_fields(self):
        data, _ = _sim()
        result = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        assert result.per_dataset is None
        assert result.frequency.shape == result.delta_ln_likelihood.shape
        assert result.ln_likelihood_base.shape == ()
        assert result.n_terms == 2
        # period property is descending (frequency ascending):
        p = ustrip("day", result.period)
        assert bool(jnp.all(jnp.diff(p) < 0))
        assert jnp.isclose(
            ustrip("day", result.max_period()),
            float(p[jnp.argmax(result.delta_ln_likelihood)]),
        )

    def test_unsupported_data_type_raises(self):
        class FakeData:
            pass

        with pytest.raises((NotImplementedError, TypeError, AttributeError)):
            hp.periodogram(FakeData(), prior=_prior(), period_min=Q(5.0, "day"))


class TestHarmonicCap:
    """The term count is capped to avoid overfitting sparse data.

    With too few observations per linear column the trial model fits almost
    any trial period, so spurious alias peaks dominate the periodogram and the
    tailored prior can hurt acceptance. See harv.periodogram.core.
    """

    def test_sparse_data_caps_and_warns(self):
        data, _ = _sim(n_obs=8, eccentricity=0.0)
        with pytest.warns(UserWarning, match="overfits data"):
            result = hp.periodogram(
                data,
                prior=_prior(),
                period_min=Q(5.0, "day"),
                period_max=Q(2000.0, "day"),
            )
        # 8 obs -> at most 4 columns -> 1 + 2H <= 4 -> H = 1:
        assert result.n_terms == 1
        # The (overdetermined) H=1 fit is not driven to spurious extremes the
        # way an overfit H=2 fit is: a well-sampled short-baseline circular
        # signal is recovered cleanly.
        dense, _ = simulate_rv_sb1_data(
            seed=7,
            n_obs=8,
            baseline=Q(120.0, "day"),
            period=P_TRUE,
            eccentricity=0.0,
            rv_semiamp=Q(10.0, "km/s"),
            rv_err=Q(0.3, "km/s"),
        )
        with pytest.warns(UserWarning, match="overfits data"):
            r_dense = hp.periodogram(dense, prior=_prior(), period_min=Q(5.0, "day"))
        assert _peak_within_grid_steps(r_dense, P_TRUE, n_steps=3)

    def test_adequate_data_keeps_requested_terms(self):
        # Enough observations to support H=2: no cap, no warning.
        data, _ = _sim(n_obs=40, eccentricity=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = hp.periodogram(data, prior=_prior(), period_min=Q(5.0, "day"))
        assert result.n_terms == 2

    def test_eccentric_adequate_data_not_harmed(self):
        # Where multi-term actually helps (eccentric orbit, enough data),
        # the cap does not engage, so eccentric systems are unaffected.
        data, _ = _sim(n_obs=40, eccentricity=0.5)
        r2 = hp.periodogram(data, prior=_prior(2), period_min=Q(5.0, "day"), n_terms=2)
        assert r2.n_terms == 2
        assert _peak_within_grid_steps(r2, P_TRUE)


class TestNTermsValidation:
    """A periodogram needs at least one harmonic (see docs/spec.md)."""

    @pytest.mark.parametrize("n_terms", [0, -1, -5])
    def test_rejects_fewer_than_one_term(self, n_terms):
        data, _ = _sim()
        with pytest.raises(ValueError, match="n_terms must be at least 1"):
            hp.periodogram(
                data, prior=_prior(), period_min=Q(5.0, "day"), n_terms=n_terms
            )

    def test_rejects_before_touching_the_prior(self):
        """The error names n_terms, not linear-prior entries never asked for."""
        data, _ = _sim()
        with pytest.raises(ValueError, match="n_terms"):
            hp.periodogram(
                data,
                prior=hm.FourierRV(n_terms=0).default_prior(
                    period_min=Q(1.0, "day"),
                    period_max=Q(5000.0, "day"),
                    sigma_v0=Q(50.0, "km/s"),
                ),
                period_min=Q(5.0, "day"),
                n_terms=0,
            )


class TestPeriodDependentBasePrior:
    """The base model is only period-independent when its own priors are."""

    @staticmethod
    def _run(v_sys_prior) -> hp.PeriodogramResult:
        data, _ = _sim()
        return hp.periodogram(
            data,
            prior=_prior(v_sys=v_sys_prior),
            period_min=Q(20.0, "day"),
            period_max=Q(200.0, "day"),
        )

    def test_base_likelihood_is_per_frequency(self):
        result = self._run(
            PeriodDependentKPrior(sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr"))
        )
        # A callable v_sys prior resolves per trial period, so the baseline
        # varies across the grid and must be evaluated there.
        assert result.ln_likelihood_base.shape == result.frequency.shape
        assert (
            float(
                jnp.max(result.ln_likelihood_base) - jnp.min(result.ln_likelihood_base)
            )
            > 0.0
        )

    def test_plain_prior_keeps_the_scalar_fast_path(self):
        result = self._run(QD(dist.Normal(0.0, 50.0), "km/s"))
        assert result.ln_likelihood_base.shape == ()

    def test_container_mixes_scalar_and_per_frequency_baselines(self):
        """One dataset on the slow path, one on the fast path, must still add up."""
        d1, _ = _sim(seed=3)
        d2, _ = _sim(seed=4)
        source = SourceData(a=d1, b=d2)
        result = hp.periodogram(
            source,
            prior={
                "a": _prior(
                    v_sys=PeriodDependentKPrior(
                        sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr")
                    )
                ),
                "b": _prior(),
            },
            period_min=Q(20.0, "day"),
            period_max=Q(200.0, "day"),
        )
        # Broadcast, not stack: a scalar and an (n,) baseline sum to (n,), and
        # the total must not collapse over the grid axis.
        assert result.ln_likelihood_base.shape == result.frequency.shape
        assert result.delta_ln_likelihood.shape == result.frequency.shape

    def test_delta_uses_the_matching_baseline(self):
        """Delta must not be tilted by a baseline taken at one period."""
        result = self._run(
            PeriodDependentKPrior(sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr"))
        )
        # Reconstructing lnL and subtracting a single-period baseline (the old
        # behavior) gives a visibly different, tilted statistic.
        lnl = result.delta_ln_likelihood + result.ln_likelihood_base
        tilted = lnl - result.ln_likelihood_base[0]
        assert not bool(jnp.allclose(tilted, result.delta_ln_likelihood, atol=1e-4))
