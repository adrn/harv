"""Unit tests for harv.periodogram.grid.frequency_grid."""

import warnings

import jax.numpy as jnp
import pytest
from unxt import Q, ustrip

from harv.data import SourceData
from harv.models import FourierGaiaAstrometry, FourierRV
from harv.periodogram import frequency_grid
from harv.periodogram.core import _effective_n_terms
from harv.simulate import simulate_rv_sb1_data


def _capped(fourier_cls, n_requested, n_obs, n_ext_linear):
    """Profile mode: assert the warning fires and return the reduced count."""
    with pytest.warns(UserWarning, match="reducing to n_terms"):
        return _effective_n_terms(
            fourier_cls, n_requested, n_obs, n_ext_linear, profile=True
        )


def _warned(fourier_cls, n_requested, n_obs, n_ext_linear):
    """Marginal mode: assert the warning fires and return the (uncapped) count."""
    with pytest.warns(UserWarning, match="is not reduced"):
        return _effective_n_terms(
            fourier_cls, n_requested, n_obs, n_ext_linear, profile=False
        )


class TestEffectiveNTermsProfile:
    """Profile mode caps: at least 2 observations per fitted linear column.

    The unregularized least-squares solve goes identically flat once the
    columns outnumber the observations, so the reduction is a correctness
    safeguard. Column counts are derived from the parameterization itself, so
    the cap automatically accounts for extension-added linear columns.
    """

    def test_rv_cap(self):
        # RV trial model has 1 + 2H columns.
        assert _capped(FourierRV, 5, 8, 0) == 1  # 8/2=4 cols -> H=1
        assert _capped(FourierRV, 5, 10, 0) == 2  # 5 cols -> H=2
        assert _capped(FourierRV, 20, 40, 0) == 9

    def test_gaia_cap(self):
        # Gaia trial model has 5 + 4H columns.
        assert _capped(FourierGaiaAstrometry, 5, 20, 0) == 1  # 10 cols
        assert _capped(FourierGaiaAstrometry, 20, 60, 0) == 6

    def test_floored_at_one(self):
        assert _capped(FourierRV, 2, 2, 0) == 1
        assert _capped(FourierGaiaAstrometry, 2, 4, 0) == 1

    def test_extension_columns_count_against_the_budget(self):
        # 3 linear extension columns eat into the same budget.
        assert _capped(FourierRV, 5, 10, 0) == 2
        assert _capped(FourierRV, 5, 10, 3) == 1


class TestEffectiveNTermsMarginal:
    """Marginal mode warns at the same threshold but never reduces n_terms.

    The amplitude prior regularizes, so ``M = I + B^T B`` stays invertible at
    any column count and there is no breakdown point to cap against.
    """

    def test_request_is_kept(self):
        # Every case the profile path would have reduced.
        assert _warned(FourierRV, 5, 8, 0) == 5
        assert _warned(FourierRV, 5, 10, 0) == 5
        assert _warned(FourierGaiaAstrometry, 5, 20, 0) == 5

    def test_kept_even_with_more_columns_than_observations(self):
        # 4 observations, 1 + 2*10 = 21 linear columns: still well-posed.
        assert _warned(FourierRV, 10, 4, 0) == 10

    def test_extension_columns_still_trigger_the_warning(self):
        assert _warned(FourierRV, 5, 10, 3) == 5

    def test_threshold_matches_the_profile_path(self):
        # Same h_max in both modes; only the action past it differs.
        for n_obs in (8, 10, 20, 40):
            for n_req in (1, 2, 3, 5):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    prof = _effective_n_terms(FourierRV, n_req, n_obs, 0, profile=True)
                    n_prof = len(caught)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    marg = _effective_n_terms(FourierRV, n_req, n_obs, 0, profile=False)
                    n_marg = len(caught)
                assert n_prof == n_marg  # same trigger condition
                assert marg == n_req  # marginal never reduces
                if n_marg == 0:
                    assert prof == n_req  # below the bar, both pass through


class TestWarningMessage:
    """The message reports the observations-per-column ratio it measured.

    The bar sits above the rank limit, so a flagged model is usually still
    overdetermined (7 columns against 10 observations). Calling that
    "overfitting" would be wrong and would not survive a reader checking the
    arithmetic, so the message states the criterion instead.
    """

    @staticmethod
    def _message(*, profile: bool) -> str:
        with pytest.warns(UserWarning, match="observations per linear column") as rec:
            _effective_n_terms(FourierRV, 3, 10, 0, profile=profile)
        return str(rec[0].message)

    @pytest.mark.parametrize("profile", [True, False])
    def test_reports_the_ratio_and_its_inputs(self, profile):
        msg = self._message(profile=profile)
        # 1 + 2*3 = 7 columns against 10 observations -> 1.4 per column.
        assert "1.4 observations per linear column" in msg
        assert "7 columns, 10 observations" in msg
        assert "2 per column" in msg  # states the bar being applied

    @pytest.mark.parametrize("profile", [True, False])
    def test_does_not_claim_overfitting(self, profile):
        # The trial model here has fewer columns than data points.
        assert "overfit" not in self._message(profile=profile).lower()

    def test_marginal_says_n_terms_is_kept(self):
        assert "is not reduced" in self._message(profile=False)

    def test_profile_says_what_it_reduced_to(self):
        # h_max = int((10/2 - 1) // 2) = 2.
        assert "reducing to n_terms=2 (5 columns)" in self._message(profile=True)


class TestEffectiveNTermsShared:
    def test_no_warning_when_comfortably_overdetermined(self):
        for profile in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert _effective_n_terms(FourierRV, 2, 1000, 0, profile=profile) == 2


class TestBounds:
    def test_endpoints_and_unit(self):
        f = frequency_grid(
            t_span=Q(1000.0, "day"),
            period_min=Q(10.0, "day"),
            period_max=Q(500.0, "day"),
        )
        vals = ustrip("1/day", f)
        assert jnp.isclose(vals[0], 1.0 / 500.0)
        assert jnp.isclose(vals[-1], 1.0 / 10.0)

    def test_spacing(self):
        span = 1000.0
        spp = 7
        f = frequency_grid(
            t_span=Q(span, "day"),
            period_min=Q(10.0, "day"),
            samples_per_peak=spp,
        )
        df = jnp.diff(ustrip("1/day", f))
        assert bool(jnp.all(df <= (1.0 / (spp * span)) * (1.0 + 1e-4)))

    def test_default_period_max_from_t_span(self):
        f = frequency_grid(t_span=Q(1000.0, "day"), period_min=Q(10.0, "day"))
        assert jnp.isclose(ustrip("1/day", f)[0], 1.0 / 1000.0)
        f2 = frequency_grid(
            t_span=Q(1000.0, "day"),
            period_min=Q(10.0, "day"),
            max_period_factor=2.0,
        )
        assert jnp.isclose(ustrip("1/day", f2)[0], 1.0 / 2000.0)

    def test_n_grid_override(self):
        f = frequency_grid(
            t_span=Q(1000.0, "day"), period_min=Q(10.0, "day"), n_grid=37
        )
        assert f.shape == (37,)

    def test_unit_follows_period_min(self):
        f = frequency_grid(t_span=Q(3.0, "yr"), period_min=Q(0.1, "yr"))
        assert jnp.isclose(ustrip("1/yr", f)[-1], 10.0)


class TestDataPath:
    def test_data_matches_t_span(self):
        data, _ = simulate_rv_sb1_data(seed=0, n_obs=20)
        span = data.time.max() - data.time.min()
        f_data = frequency_grid(data, period_min=Q(10.0, "day"))
        f_span = frequency_grid(t_span=span, period_min=Q(10.0, "day"))
        assert f_data.shape == f_span.shape
        assert jnp.allclose(ustrip("1/day", f_data), ustrip("1/day", f_span))

    def test_container_spans_all_datasets(self):
        d1, _ = simulate_rv_sb1_data(seed=1, n_obs=20, baseline=Q(2.0, "yr"))
        d2, _ = simulate_rv_sb1_data(seed=2, n_obs=20, baseline=Q(6.0, "yr"))
        both = SourceData(a=d1, b=d2)
        f_both = frequency_grid(both, period_min=Q(10.0, "day"))
        f_short = frequency_grid(d1, period_min=Q(10.0, "day"))
        # The container baseline is at least as long -> finer or equal spacing.
        assert f_both.shape[0] >= f_short.shape[0]


class TestErrors:
    def test_requires_exactly_one_of_data_t_span(self):
        with pytest.raises(TypeError, match="Exactly one"):
            frequency_grid(period_min=Q(10.0, "day"))
        data, _ = simulate_rv_sb1_data(seed=0, n_obs=10)
        with pytest.raises(TypeError, match="Exactly one"):
            frequency_grid(data, t_span=Q(1.0, "yr"), period_min=Q(10.0, "day"))

    def test_bad_period_bounds(self):
        with pytest.raises(ValueError, match="positive"):
            frequency_grid(t_span=Q(1.0, "yr"), period_min=Q(-1.0, "day"))
        with pytest.raises(ValueError, match="greater than"):
            frequency_grid(
                t_span=Q(1.0, "yr"),
                period_min=Q(100.0, "day"),
                period_max=Q(50.0, "day"),
            )
