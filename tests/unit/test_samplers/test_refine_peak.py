"""Tests for likelihood-peak refinement and the refined acceptance threshold."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q, ustrip

import harv.models as hm
from harv.distributions import QD
from harv.samplers import RejectionSampler
from harv.samplers._peak import maximize_log_likelihood
from harv.simulate import simulate_rv_sb1_data

RV_SCALES = {"sigma_K0": Q(30.0, "km/s"), "sigma_v0": Q(30.0, "km/s")}


def _loguniform_prior():
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"), period_max=Q(2000.0, "day"), **RV_SCALES
    )


def _peaked_data():
    """High SNR, densely sampled -> the library lands far below the peak."""
    data, _ = simulate_rv_sb1_data(
        seed=42,
        n_obs=16,
        baseline=Q(100.0, "day"),
        period=Q(35.0, "day"),
        eccentricity=0.3,
        rv_semiamp=Q(10.0, "km/s"),
    )
    return data


def _broad_data():
    """Low SNR -> a broad likelihood, so many samples are accepted."""
    data, _ = simulate_rv_sb1_data(
        seed=1,
        n_obs=40,
        baseline=Q(400.0, "day"),
        period=Q(120.0, "day"),
        eccentricity=0.1,
        rv_semiamp=Q(2.0, "km/s"),
        rv_err=Q(1.5, "km/s"),
    )
    return data


class TestMaximizeLogLikelihood:
    def test_recovers_a_known_peak(self):
        lnl, params = maximize_log_likelihood(
            lambda v: -((v["x"] - 7.0) ** 2) - 3.0,
            {"x": jnp.array([1.0, 50.0])},
            {"x": dist.LogUniform(0.01, 1000.0)},
        )
        assert float(params["x"]) == pytest.approx(7.0, abs=1e-3)
        assert float(lnl) == pytest.approx(-3.0, abs=1e-5)

    def test_respects_the_prior_support(self):
        """The optimum is outside [0, 1); the transform must keep us inside."""
        _, params = maximize_log_likelihood(
            lambda v: -((v["e"] - 5.0) ** 2),
            {"e": jnp.array([0.3, 0.6])},
            {"e": dist.Uniform(0.0, 1.0)},
        )
        assert 0.0 <= float(params["e"]) < 1.0

    def test_multi_start_returns_the_best(self):
        """A start in the shallow well must not win over one in the deep well."""

        def two_wells(v):
            x = v["x"]
            return jnp.maximum(-10.0 * (x - 1.0) ** 2 - 5.0, -10.0 * (x - 9.0) ** 2)

        lnl, params = maximize_log_likelihood(
            two_wells,
            {"x": jnp.array([1.05, 8.9])},
            {"x": dist.Uniform(0.0, 10.0)},
        )
        assert float(params["x"]) == pytest.approx(9.0, abs=1e-2)
        assert float(lnl) == pytest.approx(0.0, abs=1e-4)

    def test_deterministic(self):
        args = (
            lambda v: -((v["x"] - 2.0) ** 2),
            {"x": jnp.array([0.5, 5.0])},
            {"x": dist.LogUniform(0.01, 100.0)},
        )
        a = maximize_log_likelihood(*args)
        b = maximize_log_likelihood(*args)
        assert float(a[0]) == float(b[0])
        assert float(a[1]["x"]) == float(b[1]["x"])

    def test_non_optimized_keys_are_passed_through_fixed(self):
        """Keys without a prior are held at their starting value, not optimized."""
        lnl, params = maximize_log_likelihood(
            lambda v: -((v["x"] - 2.0) ** 2) - (v["fixed"] ** 2),
            {"x": jnp.array([0.0]), "fixed": jnp.array([3.0])},
            {"x": dist.Uniform(-10.0, 10.0)},
        )
        assert float(params["fixed"]) == 3.0
        assert float(lnl) == pytest.approx(-9.0, abs=1e-4)

    def test_requires_something_to_optimize(self):
        with pytest.raises(ValueError, match="No parameter to optimize"):
            maximize_log_likelihood(lambda v: -v["x"], {"x": jnp.array([1.0])}, {})


class TestRejectionStep:
    """The threshold argument, exercised without a model or data."""

    LL = jnp.array([0.0, -1.0, -2.0, -10.0])

    def test_default_matches_explicit_none(self):
        key = jax.random.key(0)
        a = RejectionSampler._rejection_step(key, self.LL)
        b = RejectionSampler._rejection_step(key, self.LL, None)
        assert bool(jnp.all(a == b))

    def test_below_library_max_is_a_no_op(self):
        """A threshold under the library max would bias; it must be clamped up."""
        key = jax.random.key(0)
        base = RejectionSampler._rejection_step(key, self.LL)
        lower = RejectionSampler._rejection_step(key, self.LL, jnp.asarray(-5.0))
        assert bool(jnp.all(base == lower))

    def test_above_library_max_shrinks_the_accepted_set(self):
        key = jax.random.key(0)
        base = RejectionSampler._rejection_step(key, self.LL)
        higher = RejectionSampler._rejection_step(key, self.LL, jnp.asarray(5.0))
        assert int(higher.sum()) < int(base.sum())
        # Acceptance is a fixed uniform draw against a smaller weight, so the
        # survivors are a subset -- never a different set.
        assert bool(jnp.all(~higher | base))

    def test_non_finite_library_still_accepts_nothing(self):
        """A finite threshold must not rescue a poisoned library."""
        poisoned = jnp.array([0.0, jnp.nan, -jnp.inf, jnp.inf])
        mask = RejectionSampler._rejection_step(
            jax.random.key(0), poisoned, jnp.asarray(0.0)
        )
        assert int(mask.sum()) == 0

    def test_library_argmax_is_always_accepted_by_default(self):
        """The property that makes the default count self-referential."""
        mask = RejectionSampler._rejection_step(jax.random.key(3), self.LL)
        assert bool(mask[int(jnp.argmax(self.LL))])


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a_s, b_s = np.sort(a), np.sort(b)
    grid = np.concatenate([a_s, b_s])
    cdf_a = np.searchsorted(a_s, grid, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, grid, side="right") / b_s.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


class TestThresholdChangesCountNotDistribution:
    def test_period_distribution_is_unchanged(self):
        """Raising the threshold rescales every acceptance probability equally."""
        data = _broad_data()
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lib = sampler.run(
                data,
                n_prior_samples=200_000,
                top_k=200_000,
                seed=0,
                return_logprobs=True,
            )
        ln_l = np.asarray(lib.ln_likelihood)
        period = np.asarray(ustrip("day", lib["period"]))
        rng = np.random.default_rng(0)
        draws = rng.uniform(size=ln_l.size)

        base = period[draws < np.exp(ln_l - ln_l.max())]
        raised = period[draws < np.exp(ln_l - (ln_l.max() + 3.0))]

        assert 0 < raised.size < base.size
        # Two-sample KS against the 1% critical value. Written out rather than
        # pulled from scipy, which harv does not depend on.
        d = _ks_statistic(np.log(base), np.log(raised))
        critical = 1.63 * np.sqrt((base.size + raised.size) / (base.size * raised.size))
        assert d < critical, (
            f"KS D={d:.3f} exceeds the 1% critical value {critical:.3f}"
        )


class TestRefinePeak:
    def test_gap_is_non_negative_under_an_informative_prior(self):
        """The regression case for optimizing the likelihood, not the posterior.

        A sharp period prior offset from the likelihood peak pulls the posterior
        MAP away from it, so a MAP-based refinement reports a *negative* gap and
        the threshold silently collapses to the library maximum.
        """
        data = _peaked_data()
        offset_prior = hm.StandardRV().default_prior(
            period=QD(dist.Normal(34.9, 0.05), "day"), **RV_SCALES
        )
        sampler = RejectionSampler(offset_prior, hm.RVModel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = sampler.run(
                data, n_prior_samples=200_000, seed=0, return_evidence_stats=True
            )
            diag = s.acceptance_diagnostics(sampler=sampler, data=data)
        assert diag["peak_gap_nats"] >= 0.0
        assert diag["max_log_likelihood_refined"] >= diag["max_log_likelihood"]

    def test_peak_is_prior_independent(self):
        """The likelihood peak belongs to the data, not to the prior."""
        data = _peaked_data()
        peaks = []
        for prior in (
            _loguniform_prior(),
            hm.StandardRV().default_prior(
                period=QD(dist.Normal(34.9, 0.05), "day"), **RV_SCALES
            ),
        ):
            sampler = RejectionSampler(prior, hm.RVModel())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = sampler.run(data, n_prior_samples=200_000, seed=0)
                peak, params = sampler.refine_peak(s, data)
            peaks.append((float(peak), float(params["period"])))
        assert peaks[0][0] == pytest.approx(peaks[1][0], abs=0.05)
        assert peaks[0][1] == pytest.approx(peaks[1][1], rel=1e-3)

    def test_run_with_refine_peak_shrinks_and_records(self):
        data = _broad_data()
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = sampler.run(
                data, n_prior_samples=200_000, seed=0, return_evidence_stats=True
            )
            refined = sampler.run(
                data,
                n_prior_samples=200_000,
                seed=0,
                return_evidence_stats=True,
                refine_peak=4,
            )
        assert "refined_max_log_likelihood" not in plain.metadata
        assert refined.n_samples < plain.n_samples
        assert (
            refined.metadata["refined_max_log_likelihood"]
            >= refined.metadata["max_log_likelihood"]
        )
        # The library maximum keeps its meaning.
        assert (
            refined.metadata["max_log_likelihood"]
            == plain.metadata["max_log_likelihood"]
        )

    def test_non_finite_peak_falls_back_with_a_warning(self, monkeypatch):
        data = _broad_data()
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())

        def _nan_peak(*_args: object, **_kwargs: object) -> tuple[jax.Array, dict]:
            return jnp.asarray(jnp.nan), {}

        monkeypatch.setattr(RejectionSampler, "_refine_peak", _nan_peak)
        with pytest.warns(UserWarning, match="non-finite peak"):
            s = sampler.run(data, n_prior_samples=20_000, seed=0, refine_peak=2)
        assert s.n_samples > 0

    def test_top_k_is_rejected(self):
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())
        with pytest.raises(ValueError, match="mutually exclusive"):
            sampler.run(
                _broad_data(), n_prior_samples=1000, seed=0, top_k=4, refine_peak=2
            )

    def test_negative_refine_peak_is_rejected(self):
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())
        with pytest.raises(ValueError, match="non-negative"):
            sampler.run(_broad_data(), n_prior_samples=1000, seed=0, refine_peak=-1)


class TestAcceptanceDiagnosticsRefinement:
    def test_keys_appear_only_with_sampler_and_data(self):
        data = _peaked_data()
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = sampler.run(
                data, n_prior_samples=200_000, seed=0, return_evidence_stats=True
            )
        plain = s.acceptance_diagnostics()
        assert "peak_gap_nats" not in plain

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = s.acceptance_diagnostics(sampler=sampler, data=data)
        assert full["peak_gap_nats"] > 0.0
        # The honest count: exp(ln M + logZ_int - peak).
        expected = np.exp(
            np.log(full["n_prior_samples"])
            + full["logZ_int"]
            - full["max_log_likelihood_refined"]
        )
        assert full["n_accepted_at_refined_peak"] == pytest.approx(expected, rel=1e-6)

    def test_requires_both_or_neither(self):
        sampler = RejectionSampler(_loguniform_prior(), hm.RVModel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = sampler.run(
                _broad_data(),
                n_prior_samples=20_000,
                seed=0,
                return_evidence_stats=True,
            )
        with pytest.raises(ValueError, match="both sampler and data"):
            s.acceptance_diagnostics(sampler=sampler)
        with pytest.raises(ValueError, match="both sampler and data"):
            s.acceptance_diagnostics(data=_broad_data())
