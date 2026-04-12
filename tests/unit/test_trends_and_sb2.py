"""Unit tests for polynomial trend support and SB2 likelihood."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData, SystemData
from harv.likelihood.params import RVParameters, SB2RVParameters
from harv.likelihood.rv import (
    RVLikelihood,
    SB2RVLikelihood,
    _build_trend_columns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rv_data(n_obs=20, t_ref=None):
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
        t_ref=t_ref,
    )


def _make_rv_params(**kw):
    defaults = dict(
        period=Q(100.0, "day"),
        eccentricity=0.3,
        phase_peri=0.0,
        arg_peri=Q(1.0, "rad"),
    )
    defaults.update(kw)
    return RVParameters.marginalized(**defaults)


def _rv_prior():
    return {"rv_semiamp": dist.Normal(0.0, 100.0), "v_sys": dist.Normal(0.0, 100.0)}


def _sb2_prior():
    return {
        "rv_semiamp_1": dist.Normal(0.0, 100.0),
        "rv_semiamp_2": dist.Normal(0.0, 100.0),
        "v_sys": dist.Normal(0.0, 100.0),
    }


def _trend_prior(order):
    return {f"trend_{k}": dist.Normal(0.0, 10.0) for k in range(1, order + 1)}


# ===========================================================================
# FEAT 1: Polynomial trends
# ===========================================================================


class TestBuildTrendColumns:
    """Tests for _build_trend_columns helper."""

    def test_shape_order1(self):
        times = jnp.linspace(0, 1, 10)
        cols = _build_trend_columns(times, t_ref=0.5, order=1)
        assert cols.shape == (10, 1)

    def test_shape_order3(self):
        times = jnp.linspace(0, 1, 10)
        cols = _build_trend_columns(times, t_ref=0.5, order=3)
        assert cols.shape == (10, 3)

    def test_columns_are_powers(self):
        times = jnp.array([1.0, 2.0, 3.0])
        cols = _build_trend_columns(times, t_ref=0.0, order=3)
        dt = times
        assert jnp.allclose(cols[:, 0], dt**1)
        assert jnp.allclose(cols[:, 1], dt**2)
        assert jnp.allclose(cols[:, 2], dt**3)

    def test_t_ref_offset(self):
        times = jnp.array([10.0, 20.0, 30.0])
        cols = _build_trend_columns(times, t_ref=20.0, order=1)
        expected = jnp.array([-10.0, 0.0, 10.0]).reshape(-1, 1)
        assert jnp.allclose(cols, expected)


class TestRVLikelihoodTrend:
    """Tests for RVLikelihood with polynomial trends."""

    def test_trend_order_zero_default(self):
        data = _make_rv_data()
        lik = RVLikelihood(data=data, linear_marginalized_prior=_rv_prior())
        assert lik.trend_order == 0
        assert lik.trend_column_names == ()

    def test_trend_column_names(self):
        data = _make_rv_data()
        lik = RVLikelihood(
            data=data,
            linear_marginalized_prior=_rv_prior(),
            trend_marginalized_prior=_trend_prior(2),
            trend_order=2,
        )
        assert lik.trend_column_names == ("trend_1", "trend_2")

    def test_design_matrix_no_trend(self):
        data = _make_rv_data(n_obs=10)
        lik = RVLikelihood(data=data, linear_marginalized_prior=_rv_prior())
        params = _make_rv_params()
        X = lik.design_matrix(params)
        assert X.shape == (10, 2)  # [rv_shape, 1]

    def test_design_matrix_with_trend(self):
        data = _make_rv_data(n_obs=10)
        lik = RVLikelihood(
            data=data,
            linear_marginalized_prior=_rv_prior(),
            trend_marginalized_prior=_trend_prior(2),
            trend_order=2,
        )
        params = _make_rv_params()
        X = lik.design_matrix(params)
        # 2 base + 2 trend = 4
        assert X.shape == (10, 4)

    def test_design_matrix_trend_plus_offsets(self):
        """Trend columns sit between base and indicator columns."""
        n_obs = 10
        data = _make_rv_data(n_obs=n_obs)
        indicator = jnp.ones((n_obs, 1))
        lik = RVLikelihood(
            data=data,
            linear_marginalized_prior=_rv_prior(),
            trend_marginalized_prior=_trend_prior(1),
            trend_order=1,
            offsets_marginalized_prior={"survey2": dist.Normal(0.0, 10.0)},
            indicator_matrix=indicator,
            instrument_names=("survey2",),
        )
        params = _make_rv_params()
        X = lik.design_matrix(params)
        # 2 base + 1 trend + 1 offset = 4
        assert X.shape == (n_obs, 4)

    def test_log_prob_with_trend_is_finite(self):
        data = _make_rv_data()
        lik = RVLikelihood(
            data=data,
            linear_marginalized_prior=_rv_prior(),
            trend_marginalized_prior=_trend_prior(2),
            trend_order=2,
        )
        params = _make_rv_params()
        ll = lik.log_prob(params)
        assert jnp.isfinite(ll)

    def test_log_prob_jit_compatible(self):
        data = _make_rv_data()
        lik = RVLikelihood(
            data=data,
            linear_marginalized_prior=_rv_prior(),
            trend_marginalized_prior=_trend_prior(1),
            trend_order=1,
        )
        params = _make_rv_params()
        ll_jit = eqx.filter_jit(lik.log_prob)(params)
        assert jnp.isfinite(ll_jit)

    def test_log_prob_vmap_compatible(self):
        data = _make_rv_data()
        lik = RVLikelihood(
            data=data,
            linear_marginalized_prior=_rv_prior(),
            trend_marginalized_prior=_trend_prior(1),
            trend_order=1,
        )
        n_samples = 5
        params_batch = RVParameters.marginalized(
            period=Q(jnp.full(n_samples, 100.0), "day"),
            eccentricity=jnp.linspace(0.0, 0.5, n_samples),
            phase_peri=jnp.zeros(n_samples),
            arg_peri=Q(jnp.ones(n_samples), "rad"),
        )
        lls = jax.vmap(lik.log_prob)(params_batch)
        assert lls.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(lls))


# ===========================================================================
# FEAT 2: SB2 support
# ===========================================================================


class TestSystemData:
    """Tests for SystemData container."""

    def test_construction(self):
        primary = _make_rv_data(n_obs=10)
        secondary = _make_rv_data(n_obs=8)
        sd = SystemData(primary=primary, secondary=secondary)
        assert sd["primary"] is primary
        assert sd["secondary"] is secondary

    def test_dict_interface(self):
        primary = _make_rv_data(n_obs=10)
        secondary = _make_rv_data(n_obs=8)
        sd = SystemData(primary=primary, secondary=secondary)
        assert len(sd) == 2
        assert "primary" in sd
        assert list(sd.keys()) == ["primary", "secondary"]
        assert list(sd.values()) == [primary, secondary]

    def test_get_datasets_by_type(self):
        primary = _make_rv_data(n_obs=10)
        secondary = _make_rv_data(n_obs=8)
        sd = SystemData(primary=primary, secondary=secondary)
        rv_datasets = sd.get_datasets_by_type(RVData)
        assert len(rv_datasets) == 2

    def test_empty_raises(self):
        import pytest

        with pytest.raises(ValueError, match="At least one"):
            SystemData()

    def test_t_ref_from_primary(self):
        primary = _make_rv_data(n_obs=10, t_ref=Q(50.0, "day"))
        secondary = _make_rv_data(n_obs=8)
        sd = SystemData(primary=primary, secondary=secondary)
        assert float(sd.t_ref.value) == 50.0

    def test_stacked_obs(self):
        primary = RVData(
            time=Q(jnp.array([1.0, 2.0]), "day"),
            rv=Q(jnp.array([10.0, 20.0]), "km/s"),
            rv_err=Q(jnp.array([0.1, 0.1]), "km/s"),
        )
        secondary = RVData(
            time=Q(jnp.array([1.5, 2.5, 3.5]), "day"),
            rv=Q(jnp.array([30.0, 40.0, 50.0]), "km/s"),
            rv_err=Q(jnp.array([0.2, 0.2, 0.2]), "km/s"),
        )
        sd = SystemData(primary=primary, secondary=secondary)
        obs = sd._get_obs()
        err = sd._get_obs_err()
        assert len(obs) == 5
        assert len(err) == 5
        assert np.allclose(obs.value[:2], [10.0, 20.0])
        assert np.allclose(obs.value[2:], [30.0, 40.0, 50.0])


class TestSB2RVParameters:
    """Tests for SB2RVParameters struct."""

    def test_linear_param_names(self):
        assert SB2RVParameters.linear_param_names == (
            "rv_semiamp_1",
            "rv_semiamp_2",
            "v_sys",
        )

    def test_nonlinear_param_names(self):
        assert "period" in SB2RVParameters.nonlinear_param_names
        assert "eccentricity" in SB2RVParameters.nonlinear_param_names

    def test_marginalized_construction(self):
        mp = SB2RVParameters.marginalized(
            period=Q(100.0, "day"),
            eccentricity=0.3,
            phase_peri=0.0,
            arg_peri=Q(1.0, "rad"),
        )
        assert mp.period == Q(100.0, "day")
        assert set(mp.marginalized_names) == {
            "rv_semiamp_1",
            "rv_semiamp_2",
            "v_sys",
        }


class TestSB2RVLikelihood:
    """Tests for SB2RVLikelihood."""

    def _make_sb2_data(self):
        primary = RVData(
            time=Q(jnp.linspace(0, 100, 15), "day"),
            rv=Q(jnp.zeros(15), "km/s"),
            rv_err=Q(jnp.ones(15) * 0.1, "km/s"),
        )
        secondary = RVData(
            time=Q(jnp.linspace(0, 100, 12), "day"),
            rv=Q(jnp.zeros(12), "km/s"),
            rv_err=Q(jnp.ones(12) * 0.2, "km/s"),
        )
        return SystemData(primary=primary, secondary=secondary)

    def _make_sb2_params(self):
        return SB2RVParameters.marginalized(
            period=Q(100.0, "day"),
            eccentricity=0.3,
            phase_peri=0.0,
            arg_peri=Q(1.0, "rad"),
        )

    def test_design_matrix_shape(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        X = lik.design_matrix(params)
        assert X.shape == (15 + 12, 3)  # [K1, K2, v_sys]

    def test_design_matrix_primary_k2_zero(self):
        """K2 column should be zero for primary rows."""
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        X = lik.design_matrix(params)
        # Primary rows: first 15
        assert jnp.allclose(X[:15, 1], 0.0)

    def test_design_matrix_secondary_k1_zero(self):
        """K1 column should be zero for secondary rows."""
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        X = lik.design_matrix(params)
        # Secondary rows: last 12
        assert jnp.allclose(X[15:, 0], 0.0)

    def test_design_matrix_vsys_column_ones(self):
        """The v_sys column should be all ones."""
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        X = lik.design_matrix(params)
        assert jnp.allclose(X[:, 2], 1.0)

    def test_design_matrix_antiphase(self):
        """Secondary K2 column has opposite sign to primary K1 column."""
        # Use same times for both components
        times = Q(jnp.linspace(0, 100, 10), "day")
        primary = RVData(
            time=times,
            rv=Q(jnp.zeros(10), "km/s"),
            rv_err=Q(jnp.ones(10) * 0.1, "km/s"),
        )
        secondary = RVData(
            time=times,
            rv=Q(jnp.zeros(10), "km/s"),
            rv_err=Q(jnp.ones(10) * 0.1, "km/s"),
        )
        data = SystemData(primary=primary, secondary=secondary)

        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        X = lik.design_matrix(params)

        # Primary K1[:10] and Secondary K2[10:] differ by sign
        assert jnp.allclose(X[:10, 0], -X[10:, 1])

    def test_log_prob_is_finite(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        ll = lik.log_prob(params)
        assert jnp.isfinite(ll)

    def test_log_prob_jit_compatible(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        params = self._make_sb2_params()
        ll = eqx.filter_jit(lik.log_prob)(params)
        assert jnp.isfinite(ll)

    def test_log_prob_vmap_compatible(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        n = 5
        params_batch = SB2RVParameters.marginalized(
            period=Q(jnp.full(n, 100.0), "day"),
            eccentricity=jnp.linspace(0.0, 0.5, n),
            phase_peri=jnp.zeros(n),
            arg_peri=Q(jnp.ones(n), "rad"),
        )
        lls = jax.vmap(lik.log_prob)(params_batch)
        assert lls.shape == (n,)
        assert jnp.all(jnp.isfinite(lls))

    def test_linear_param_units(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(data=data, linear_marginalized_prior=_sb2_prior())
        units = lik.linear_param_units
        assert set(units.keys()) == {"rv_semiamp_1", "rv_semiamp_2", "v_sys"}

    def test_design_matrix_with_trend(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(
            data=data,
            linear_marginalized_prior=_sb2_prior(),
            trend_marginalized_prior=_trend_prior(1),
            trend_order=1,
        )
        params = self._make_sb2_params()
        X = lik.design_matrix(params)
        # 3 base + 1 trend = 4 columns, 15+12 = 27 rows
        assert X.shape == (27, 4)

    def test_log_prob_with_trend_is_finite(self):
        data = self._make_sb2_data()
        lik = SB2RVLikelihood(
            data=data,
            linear_marginalized_prior=_sb2_prior(),
            trend_marginalized_prior=_trend_prior(2),
            trend_order=2,
        )
        params = self._make_sb2_params()
        ll = lik.log_prob(params)
        assert jnp.isfinite(ll)


class TestRejectionPriorTrend:
    """Tests for RejectionPrior default factories with trend support."""

    def test_default_rv_trend_order(self):
        from harv.samplers.rejection_prior import RejectionPrior

        prior = RejectionPrior.default_rv(
            period_min=Q(1.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
            trend_order=2,
            trend_priors=_trend_prior(2),
        )
        assert prior.trend_order == 2
        assert prior.trend_priors is not None
        assert len(prior.trend_priors) == 2

    def test_default_sb2(self):
        from harv.samplers.rejection_prior import RejectionPrior

        prior = RejectionPrior.default_sb2(
            period_min=Q(1.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
        )
        assert "rv_semiamp_1" in prior.linear_prior
        assert "rv_semiamp_2" in prior.linear_prior
        assert "v_sys" in prior.linear_prior
