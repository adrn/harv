"""Unit tests for built-in extensions: Jitter, MonomialTrend, MultiSurveyOffset."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxt import Q

from harv.data import GaiaAstrometryData, RVData
from harv.distributions import QuantityDistribution as QD
from harv.extensions import (
    AbstractExtension,
    Jitter,
    MonomialTrend,
    MultiSurveyOffset,
)
from harv.models import RVModel
from harv.samplers import RejectionPrior, RejectionSampler

# ======================================================================
# Jitter
# ======================================================================


class TestJitter:
    def test_is_extension(self):
        assert isinstance(Jitter(), AbstractExtension)

    def test_extra_params(self):
        j = Jitter(param_unit="km/s")
        params = j.extra_params()
        assert len(params) == 1
        assert params[0].name == "jitter"
        assert not params[0].linear
        assert params[0].unit == "km/s"

    def test_modify_design_matrix_passthrough(self):
        j = Jitter()
        X = jnp.ones((5, 2))
        result = j.modify_design_matrix(X, None, {})
        assert jnp.array_equal(result, X)

    def test_modify_covariance_diagonal(self):
        j = Jitter()
        cov = jnp.array([1.0, 4.0, 9.0])
        result = j.modify_covariance(cov, None, {"jitter": 2.0})
        expected = jnp.array([5.0, 8.0, 13.0])
        assert jnp.allclose(result, expected)

    def test_modify_covariance_full(self):
        j = Jitter()
        cov = jnp.eye(3) * 4.0
        result = j.modify_covariance(cov, None, {"jitter": 3.0})
        expected = jnp.eye(3) * 4.0 + jnp.eye(3) * 9.0
        assert jnp.allclose(result, expected)

    def test_with_rv_model(self):
        """Jitter extension integrates correctly with RVModel."""
        data = RVData(
            time=Q([0.0, 50.0, 100.0], "day"),
            rv=Q([1.0, -2.0, 0.5], "km/s"),
            rv_err=Q([0.5, 0.5, 0.5], "km/s"),
        )
        jitter_ext = Jitter(param_unit="km/s")
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
        }
        model = RVModel(extensions=(jitter_ext,))
        # Check that jitter appears in nonlinear params
        assert "jitter" in model._all_nonlinear_names()

        # Evaluate log_prob with jitter value
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.3),
            "phase_peri": jnp.float32(0.1),
            "arg_peri": Q(1.0, "rad"),
            "jitter": 0.5,  # in km/s, unit-stripped
        }
        lp = model.log_prob(nl, data, linear_prior=linear_prior)
        assert jnp.isfinite(lp)

        # Compare: jitter=0 should give same result as no jitter extension
        model_no_jitter = RVModel()
        nl_no_jitter = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.3),
            "phase_peri": jnp.float32(0.1),
            "arg_peri": Q(1.0, "rad"),
        }
        nl_zero = {**nl_no_jitter, "jitter": 0.0}
        lp_zero_jitter = model.log_prob(nl_zero, data, linear_prior=linear_prior)
        lp_no_ext = model_no_jitter.log_prob(
            nl_no_jitter, data, linear_prior=linear_prior
        )
        assert jnp.allclose(lp_zero_jitter, lp_no_ext, atol=1e-5)

    def test_jitter_jit(self):
        """Jitter extension works under jit."""
        data = RVData(
            time=Q([0.0, 50.0, 100.0], "day"),
            rv=Q([1.0, -2.0, 0.5], "km/s"),
            rv_err=Q([0.5, 0.5, 0.5], "km/s"),
        )
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
        }
        model = RVModel(extensions=(Jitter(param_unit="km/s"),))
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.3),
            "phase_peri": jnp.float32(0.1),
            "arg_peri": Q(1.0, "rad"),
            "jitter": 0.5,
        }

        @jax.jit
        def _lp(jitter_val):
            vals = {**nl, "jitter": jitter_val}
            return model.log_prob(vals, data, linear_prior=linear_prior)

        result = _lp(0.5)
        assert jnp.isfinite(result)


# ======================================================================
# MonomialTrend
# ======================================================================


class TestMonomialTrend:
    def test_is_extension(self):
        assert isinstance(MonomialTrend(order=1), AbstractExtension)

    def test_bad_order(self):
        with pytest.raises(ValueError, match="order must be"):
            MonomialTrend(order=0)

    def test_extra_params_rv(self):
        t = MonomialTrend(order=2, obs_unit="km/s")
        params = t.extra_params()
        assert len(params) == 2
        assert params[0].name == "trend_1"
        assert params[1].name == "trend_2"
        assert all(p.linear for p in params)

    def test_extra_params_astrometry(self):
        t = MonomialTrend(order=2, obs_unit="mas", astrometry=True)
        params = t.extra_params()
        assert len(params) == 4
        names = [p.name for p in params]
        assert names == ["trend_ra_1", "trend_dec_1", "trend_ra_2", "trend_dec_2"]

    def test_modify_design_matrix_rv(self):
        """Trend appends correct monomial columns for RV."""
        data = RVData(
            time=Q([10.0, 20.0, 30.0], "day"),
            rv=Q([1.0, -2.0, 0.5], "km/s"),
            rv_err=Q([0.5, 0.5, 0.5], "km/s"),
            t_ref=Q(20.0, "day"),
        )
        t = MonomialTrend(order=2, time_unit="day")
        X = jnp.ones((3, 2))  # base design matrix
        result = t.modify_design_matrix(X, data, {})

        assert result.shape == (3, 4)
        # First two columns unchanged
        assert jnp.array_equal(result[:, :2], X)
        # dt = [-10, 0, 10]
        dt = jnp.array([-10.0, 0.0, 10.0])
        assert jnp.allclose(result[:, 2], dt)
        assert jnp.allclose(result[:, 3], dt**2)

    def test_modify_design_matrix_astrometry(self):
        """Astrometry trend uses dt^(k+1) with sin/cos scan angle."""
        data = GaiaAstrometryData(
            time=Q([0.0, 365.25, 730.5], "day"),
            al_position=Q([0.1, -0.2, 0.05], "mas"),
            al_position_err=Q([0.01, 0.01, 0.01], "mas"),
            scan_angle=Q([0.0, jnp.pi / 2, jnp.pi], "rad"),
            parallax_factor=jnp.array([0.3, -0.1, 0.4]),
            t_ref=Q(365.25, "day"),
        )
        t = MonomialTrend(order=1, time_unit="yr", astrometry=True)
        X = jnp.ones((3, 6))
        result = t.modify_design_matrix(X, data, {})
        # 6 base + 2 trend columns
        assert result.shape == (3, 8)

    def test_modify_covariance_passthrough(self):
        t = MonomialTrend(order=1)
        cov = jnp.ones(5)
        assert jnp.array_equal(t.modify_covariance(cov, None, {}), cov)

    def test_with_rv_model(self):
        """MonomialTrend integrates with RVModel for marginalized likelihood."""
        data = RVData(
            time=Q([0.0, 50.0, 100.0, 150.0, 200.0], "day"),
            rv=Q([1.0, -2.0, 0.5, 3.0, -1.0], "km/s"),
            rv_err=Q([0.5, 0.5, 0.5, 0.5, 0.5], "km/s"),
        )
        linear_prior = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "trend_1": dist.Normal(0.0, 1.0),  # dimensionless trend
        }
        trend = MonomialTrend(order=1, time_unit="day", obs_unit="km/s")
        model = RVModel(extensions=(trend,))

        # trend_1 should be a linear param
        assert "trend_1" in model._all_linear_names()

        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.1),
            "phase_peri": jnp.float32(0.25),
            "arg_peri": Q(0.5, "rad"),
        }
        lp = model.log_prob(nl, data, linear_prior=linear_prior)
        assert jnp.isfinite(lp)

    def test_rejection_sampler_requires_all_trend_priors(self):
        """Sampler must reject missing priors for declared trend coefficients."""
        data = RVData(
            time=Q([0.0, 50.0, 100.0, 150.0, 200.0], "day"),
            rv=Q([1.0, -2.0, 0.5, 3.0, -1.0], "km/s"),
            rv_err=Q([0.5, 0.5, 0.5, 0.5, 0.5], "km/s"),
        )
        prior = RejectionPrior.default_rv(
            period_min=Q(1.0, "day"),
            period_max=Q(1_000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
            trend_1=QD(dist.Normal(0.0, 1.0), "km/s"),
        )
        sampler = RejectionSampler(
            prior,
            RVModel(extensions=(MonomialTrend(order=2, time_unit="day"),)),
        )

        with pytest.raises(ValueError, match="trend_2"):
            sampler.run(data, n_prior_samples=8, max_posterior_samples=2, seed=0)


# ======================================================================
# MultiSurveyOffset
# ======================================================================


class TestMultiSurveyOffset:
    def test_is_extension(self):
        indicator = jnp.array([[0.0], [1.0], [0.0]])
        assert isinstance(
            MultiSurveyOffset(indicator, ("espresso",), "km/s"), AbstractExtension
        )

    def test_bad_indicator_ndim(self):
        with pytest.raises(ValueError, match="2-d"):
            MultiSurveyOffset(jnp.array([1.0, 0.0]), ("a",))

    def test_bad_indicator_columns(self):
        with pytest.raises(ValueError, match="columns"):
            MultiSurveyOffset(jnp.ones((3, 2)), ("a",))

    def test_extra_params(self):
        indicator = jnp.ones((3, 2))
        ext = MultiSurveyOffset(indicator, ("espresso", "keck"), "km/s")
        params = ext.extra_params()
        assert len(params) == 2
        assert params[0].name == "espresso"
        assert params[1].name == "keck"
        assert all(p.linear for p in params)

    def test_modify_design_matrix(self):
        indicator = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        ext = MultiSurveyOffset(indicator, ("a", "b"), "km/s")
        X = jnp.ones((3, 2))
        result = ext.modify_design_matrix(X, None, {})
        assert result.shape == (3, 4)
        assert jnp.array_equal(result[:, :2], X)
        assert jnp.array_equal(result[:, 2:], indicator)

    def test_modify_covariance_passthrough(self):
        indicator = jnp.ones((3, 1))
        ext = MultiSurveyOffset(indicator, ("a",))
        cov = jnp.ones(3)
        assert jnp.array_equal(ext.modify_covariance(cov, None, {}), cov)

    def test_with_rv_model(self):
        """MultiSurveyOffset integrates with RVModel."""
        # Simulate two surveys: ref (3 obs) + espresso (2 obs)
        times = Q([0.0, 50.0, 100.0, 25.0, 75.0], "day")
        rvs = Q([1.0, -2.0, 0.5, 1.5, -1.5], "km/s")
        errs = Q([0.5, 0.5, 0.5, 0.3, 0.3], "km/s")
        data = RVData(time=times, rv=rvs, rv_err=errs)

        # ref: obs 0,1,2; espresso: obs 3,4
        indicator = jnp.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
            ]
        )
        linear_prior_mso = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "espresso": QD(dist.Normal(0.0, 5.0), "km/s"),
        }
        offset_ext = MultiSurveyOffset(indicator, ("espresso",), "km/s")
        model = RVModel(extensions=(offset_ext,))

        assert "espresso" in model._all_linear_names()

        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.1),
            "phase_peri": jnp.float32(0.25),
            "arg_peri": Q(0.5, "rad"),
        }
        lp = model.log_prob(nl, data, linear_prior=linear_prior_mso)
        assert jnp.isfinite(lp)

    def test_sample_conditional_with_offset(self):
        """Can sample conditional linear params including the offset."""
        times = Q([0.0, 50.0, 100.0, 25.0, 75.0], "day")
        rvs = Q([1.0, -2.0, 0.5, 1.5, -1.5], "km/s")
        errs = Q([0.5, 0.5, 0.5, 0.3, 0.3], "km/s")
        data = RVData(time=times, rv=rvs, rv_err=errs)

        indicator = jnp.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
            ]
        )
        linear_prior_off = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "espresso": QD(dist.Normal(0.0, 5.0), "km/s"),
        }
        offset_ext = MultiSurveyOffset(indicator, ("espresso",), "km/s")
        model = RVModel(extensions=(offset_ext,))

        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.1),
            "phase_peri": jnp.float32(0.25),
            "arg_peri": Q(0.5, "rad"),
        }
        key = jax.random.PRNGKey(42)
        samples = model.sample_conditional_linear(
            nl, key, data, linear_prior=linear_prior_off
        )
        assert "espresso" in samples
        assert "rv_semiamp" in samples
        assert "v_sys" in samples
        assert all(jnp.isfinite(v) for v in samples.values())


# ======================================================================
# Combined extensions
# ======================================================================


class TestCombinedExtensions:
    """Test combining multiple extensions on one model."""

    def test_jitter_plus_trend(self):
        data = RVData(
            time=Q([0.0, 50.0, 100.0, 150.0], "day"),
            rv=Q([1.0, -2.0, 0.5, 3.0], "km/s"),
            rv_err=Q([0.5, 0.5, 0.5, 0.5], "km/s"),
        )
        linear_prior_jt = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "trend_1": dist.Normal(0.0, 1.0),
        }
        model = RVModel(
            extensions=(
                Jitter(param_unit="km/s"),
                MonomialTrend(order=1, time_unit="day"),
            ),
        )

        assert "jitter" in model._all_nonlinear_names()
        assert "trend_1" in model._all_linear_names()

        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.1),
            "phase_peri": jnp.float32(0.25),
            "arg_peri": Q(0.5, "rad"),
            "jitter": 0.3,
        }
        lp = model.log_prob(nl, data, linear_prior=linear_prior_jt)
        assert jnp.isfinite(lp)

    def test_all_three_extensions(self):
        """Jitter + Trend + MultiSurvey offset, all at once."""
        times = Q([0.0, 50.0, 100.0, 150.0, 25.0, 75.0], "day")
        rvs = Q([1.0, -2.0, 0.5, 3.0, 1.5, -1.5], "km/s")
        errs = Q([0.5, 0.5, 0.5, 0.5, 0.3, 0.3], "km/s")
        data = RVData(time=times, rv=rvs, rv_err=errs)

        indicator = jnp.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
            ]
        )
        linear_prior_all = {
            "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
            "trend_1": dist.Normal(0.0, 1.0),
            "other_surv": QD(dist.Normal(0.0, 5.0), "km/s"),
        }
        model = RVModel(
            extensions=(
                Jitter(param_unit="km/s"),
                MonomialTrend(order=1, time_unit="day"),
                MultiSurveyOffset(indicator, ("other_surv",), "km/s"),
            ),
        )

        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": jnp.float32(0.1),
            "phase_peri": jnp.float32(0.25),
            "arg_peri": Q(0.5, "rad"),
            "jitter": 0.2,
        }
        lp = model.log_prob(nl, data, linear_prior=linear_prior_all)
        assert jnp.isfinite(lp)

        # Sample conditional
        key = jax.random.PRNGKey(0)
        samples = model.sample_conditional_linear(
            nl, key, data, linear_prior=linear_prior_all
        )
        assert set(samples) == {"rv_semiamp", "v_sys", "trend_1", "other_surv"}
        assert all(jnp.isfinite(v) for v in samples.values())
