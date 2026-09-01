"""Unit tests for the periodogram -> interim prior builders."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv.models as hm
import harv.periodogram as hp
from harv.periodogram.core import PeriodogramResult

P_LO, P_HI = 10.0, 1000.0


def _fake_result(
    n: int = 2000,
    peaks: tuple[tuple[float, float], ...] = ((100.0, 30.0), (300.0, 20.0)),
    width_u: float = 0.05,
) -> PeriodogramResult:
    """Synthetic periodogram with Gaussian bumps in ln-period."""
    f = jnp.linspace(1.0 / P_HI, 1.0 / P_LO, n)
    u = jnp.log(1.0 / f)
    delta = jnp.zeros(n)
    for period, amp in peaks:
        delta = delta + amp * jnp.exp(-0.5 * ((u - jnp.log(period)) / width_u) ** 2)
    return PeriodogramResult(
        frequency=Q(f, "1/day"),
        delta_ln_likelihood=delta,
        ln_likelihood_base=jnp.asarray(0.0),
        t_span=Q(2000.0, "day"),
        t_ref=Q(0.0, "day"),
    )


def _mass_between(prior, p_lo: float, p_hi: float) -> float:
    d = prior.distribution
    return float(d.cdf(p_hi) - d.cdf(p_lo))


def _peak_mass(prior, p: float, floor: float, half_width: float = 0.2) -> float:
    """Mass of the top-hat at ``p``, with the log-uniform floor subtracted.

    The window must be wide enough to contain the whole top-hat; the floor is
    flat in ln-period, so its share of the window is exactly computable.
    """
    total = _mass_between(prior, p * np.exp(-half_width), p * np.exp(half_width))
    return total - floor * (2.0 * half_width) / np.log(P_HI / P_LO)


class TestTempered:
    def test_beta_zero_is_loguniform(self):
        prior = hp.tempered_period_prior(_fake_result(), beta=0.0, floor=0.1)
        lu = dist.LogUniform(P_LO, P_HI)
        x = jnp.geomspace(P_LO * 1.01, P_HI * 0.99, 301)
        assert jnp.allclose(prior.distribution.log_prob(x), lu.log_prob(x), atol=1e-4)

    def test_concentrates_mass_at_peaks(self):
        result = _fake_result()
        prior = hp.tempered_period_prior(result, beta=1.0, floor=0.1)
        mass_peak = _mass_between(prior, 90.0, 110.0)
        lu_mass = np.log(110.0 / 90.0) / np.log(P_HI / P_LO)
        assert mass_peak > 20 * lu_mass

    def test_more_tempering_concentrates_more(self):
        result = _fake_result()
        m_lo = _mass_between(
            hp.tempered_period_prior(result, beta=0.3, floor=0.1), 95.0, 105.0
        )
        m_hi = _mass_between(
            hp.tempered_period_prior(result, beta=1.0, floor=0.1), 95.0, 105.0
        )
        assert m_hi > m_lo

    def test_floor_lower_bound(self):
        floor = 0.15
        prior = hp.tempered_period_prior(_fake_result(), beta=1.0, floor=floor)
        x = jnp.geomspace(P_LO * 1.01, P_HI * 0.99, 501)
        density_ln = jnp.exp(prior.distribution.log_prob_ln(x))
        bound = floor / np.log(P_HI / P_LO)
        assert bool(jnp.all(density_ln >= bound * (1.0 - 1e-3)))

    @pytest.mark.parametrize(
        ("period_min", "period_max"),
        [
            (None, Q(6000.0, "day")),  # above the grid
            (Q(1.0, "day"), None),  # below the grid
            (Q(1.0, "day"), Q(6000.0, "day")),  # both sides
            (Q(2000.0, "day"), Q(5000.0, "day")),  # entirely above
            (Q(0.1, "day"), Q(5.0, "day")),  # entirely below
        ],
    )
    def test_domain_outside_the_grid_is_refused(self, period_min, period_max):
        """The periodogram is evidence only where it was evaluated."""
        with pytest.raises(ValueError, match="reaches outside the periodogram grid"):
            hp.tempered_period_prior(
                _fake_result(), period_min=period_min, period_max=period_max
            )

    def test_domain_equal_to_the_grid_bounds_is_accepted(self):
        """The round trip through 1/f and log() must not trip the check."""
        prior = hp.tempered_period_prior(
            _fake_result(), period_min=Q(P_LO, "day"), period_max=Q(P_HI, "day")
        )
        assert np.isclose(float(prior.distribution.low), P_LO, rtol=1e-6)
        assert np.isclose(float(prior.distribution.high), P_HI, rtol=1e-6)
        # ... and matches the default (domain omitted) prior.
        default = hp.tempered_period_prior(_fake_result())
        assert np.isclose(
            float(prior.distribution.log_prob(100.0)),
            float(default.distribution.log_prob(100.0)),
            rtol=1e-5,
        )

    def test_domain_subset_of_the_grid_is_supported(self):
        """Narrowing the domain is still allowed, and renormalizes."""
        prior = hp.tempered_period_prior(
            _fake_result(), period_min=Q(50.0, "day"), period_max=Q(200.0, "day")
        )
        assert np.isclose(float(prior.distribution.low), 50.0, rtol=1e-6)
        assert np.isclose(float(prior.distribution.high), 200.0, rtol=1e-6)
        assert _mass_between(prior, 50.0, 200.0) == pytest.approx(1.0, abs=1e-6)
        # The 100 d peak survives; the 300 d peak is outside the domain.
        assert np.isneginf(float(prior.distribution.log_prob(300.0)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_delta_is_refused(self, bad):
        """A NaN/inf Delta must name itself, not surface from LogGridDensity."""
        base = _fake_result()
        delta = base.delta_ln_likelihood.at[17].set(bad)
        result = PeriodogramResult(
            frequency=base.frequency,
            delta_ln_likelihood=delta,
            ln_likelihood_base=base.ln_likelihood_base,
            t_span=base.t_span,
            t_ref=base.t_ref,
        )
        with pytest.raises(ValueError, match="non-finite at 1 of"):
            hp.tempered_period_prior(result)
        with pytest.raises(ValueError, match="non-finite at 1 of"):
            hp.peak_period_prior(result)

    def test_invalid_args(self):
        result = _fake_result()
        with pytest.raises(ValueError, match="beta"):
            hp.tempered_period_prior(result, beta=-1.0)
        with pytest.raises(ValueError, match="floor"):
            hp.tempered_period_prior(result, floor=1.5)
        with pytest.warns(UserWarning, match="floor=0"):
            hp.tempered_period_prior(result, floor=0.0)


class TestPeaks:
    def test_equal_mass_regardless_of_amplitude(self):
        """Peaks with 30 vs 20 delta-ln-L get exactly equal mass."""
        floor = 0.1
        # height_drop=15 admits both peaks (global max 30, so keep delta >= 15):
        prior = hp.peak_period_prior(_fake_result(), height_drop=15.0, floor=floor)
        target = (1.0 - floor) / 2.0
        m1 = _peak_mass(prior, 100.0, floor)
        m2 = _peak_mass(prior, 300.0, floor)
        # Each top-hat is normalized by its mass as the knots sample it, so the
        # documented (1 - floor) / n_peaks share is exact, not approximate.
        assert m1 == pytest.approx(target, rel=1e-4)
        assert m2 == pytest.approx(target, rel=1e-4)
        assert _mass_between(prior, P_LO, P_HI) == pytest.approx(1.0, abs=1e-6)

    def test_height_drop_excludes_weak_peaks(self):
        # global max 30, drop 5 -> keep delta >= 25 -> only the 30 peak:
        prior = hp.peak_period_prior(_fake_result(), height_drop=5.0, floor=0.1)
        m1 = _mass_between(prior, 100.0 * np.exp(-0.2), 100.0 * np.exp(0.2))
        m2 = _mass_between(prior, 300.0 * np.exp(-0.2), 300.0 * np.exp(0.2))
        assert m1 > 0.5
        assert m2 < 0.1

    def test_relative_criterion_is_scale_invariant(self):
        """The same peaks are selected after an overall shift of delta."""
        base = _fake_result()
        shifted = PeriodogramResult(
            frequency=base.frequency,
            delta_ln_likelihood=base.delta_ln_likelihood - 500.0,
            ln_likelihood_base=base.ln_likelihood_base,
            t_span=base.t_span,
            t_ref=base.t_ref,
        )
        p0 = hp.peak_period_prior(base, height_drop=15.0, floor=0.1)
        p1 = hp.peak_period_prior(shifted, height_drop=15.0, floor=0.1)
        x = jnp.geomspace(P_LO * 1.01, P_HI * 0.99, 201)
        assert jnp.allclose(
            p0.distribution.log_prob(x), p1.distribution.log_prob(x), atol=1e-5
        )

    def test_max_peaks_caps_peak_count(self):
        result = _fake_result(
            peaks=((50.0, 30.0), (100.0, 25.0), (300.0, 20.0)), width_u=0.03
        )
        floor = 0.1
        prior = hp.peak_period_prior(result, height_drop=15.0, max_peaks=2, floor=floor)
        # The two strongest peaks (50 d, 100 d) share the mass; the 300 d
        # peak is dropped:
        m3 = _mass_between(prior, 300.0 * np.exp(-0.2), 300.0 * np.exp(0.2))
        assert m3 < 0.1
        # Each kept peak carries exactly (1 - floor) / max_peaks -- the bound
        # the docs state, which the max_peaks cap exists to guarantee.
        m1 = _peak_mass(prior, 50.0, floor)
        m2 = _peak_mass(prior, 100.0, floor)
        assert m1 == pytest.approx((1.0 - floor) / 2.0, rel=1e-4)
        assert m2 == pytest.approx((1.0 - floor) / 2.0, rel=1e-4)

    def test_flat_periodogram_falls_back_to_loguniform(self):
        result = _fake_result(peaks=())
        with pytest.warns(UserWarning, match="flat or monotonic"):
            prior = hp.peak_period_prior(result, floor=0.1)
        lu = dist.LogUniform(P_LO, P_HI)
        x = jnp.geomspace(P_LO * 1.01, P_HI * 0.99, 101)
        assert jnp.allclose(prior.distribution.log_prob(x), lu.log_prob(x), atol=1e-4)

    def test_same_tree_structure_as_tempered(self):
        """Both builders on one grid config yield one pytree structure."""
        result = _fake_result()
        p_t = hp.tempered_period_prior(result, beta=1.0, floor=0.1)
        p_p = hp.peak_period_prior(result, floor=0.1)
        s_t = jax.tree_util.tree_structure(p_t.distribution)
        s_p = jax.tree_util.tree_structure(p_p.distribution)
        assert s_t == s_p


class TestSamplerDropIn:
    def test_prior_samples_follow_peaks(self):
        prior_dist = hp.tempered_period_prior(_fake_result(), beta=1.0, floor=0.1)
        prior = hm.StandardRV().default_prior(
            period=prior_dist,
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
        )
        samples = prior.sample_nonlinear(jax.random.key(0), n_samples=4096)
        p = samples["period"]
        frac_near_peak = float(jnp.mean((p > 90.0) & (p < 110.0)))
        lu_frac = np.log(110.0 / 90.0) / np.log(P_HI / P_LO)
        assert frac_near_peak > 10 * lu_frac
