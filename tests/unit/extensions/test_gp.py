"""Tests for the GP covariance extension."""

from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
from harv.extensions.base import AbstractExtension, ParamInfo
from harv.extensions.gp import GP
from harv.extensions.jitter import Jitter
from harv.models.rv import RVModel

# ---------------------------------------------------------------------------
# Mock kernel (avoids tinygp dependency in tests)
# ---------------------------------------------------------------------------


class _MockKernel:
    """Minimal kernel-like object with an ``evaluate(X, Xp)`` method."""

    def __init__(self, amp: float) -> None:
        self.amp = amp

    def evaluate(self, X: jax.Array, Xp: jax.Array) -> jax.Array:
        # Simple squared-exponential-like kernel: amp^2 * exp(-0.5 * |X-Xp|^2)
        diff = X[:, None] - Xp[None, :]
        return self.amp**2 * jnp.exp(-0.5 * diff**2)


def _mock_kernel_builder(nl_values: dict[str, Any]) -> _MockKernel:
    return _MockKernel(amp=nl_values["gp_amp"])


def _make_gp(time_unit: str = "day") -> GP:
    return GP(
        kernel_builder=_mock_kernel_builder,
        hyperparams=(ParamInfo("gp_amp", "km/s"),),
        time_unit=time_unit,
    )


def _make_rv_data(n_obs: int = 10) -> RVData:
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.5, "km/s"),
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestGPProtocol:
    def test_is_extension(self):
        gp = _make_gp()
        assert isinstance(gp, AbstractExtension)

    def test_extra_params(self):
        gp = _make_gp()
        params = gp.extra_params()
        assert len(params) == 1
        assert params[0].name == "gp_amp"
        assert not params[0].linear

    def test_multiple_hyperparams(self):
        gp = GP(
            kernel_builder=_mock_kernel_builder,
            hyperparams=(
                ParamInfo("gp_amp", "km/s"),
                ParamInfo("gp_length_scale", "day"),
            ),
        )
        assert len(gp.extra_params()) == 2


# ---------------------------------------------------------------------------
# Covariance modification
# ---------------------------------------------------------------------------


class TestGPModifyCovariance:
    def test_promotes_diagonal_to_full(self):
        gp = _make_gp()
        data = _make_rv_data(n_obs=5)
        cov_diag = jnp.ones(5) * 0.25  # variance = 0.5^2

        nl = {"gp_amp": 1.0}
        cov_full = gp.modify_covariance(cov_diag, data, nl)
        assert cov_full.shape == (5, 5)

    def test_diagonal_preserved_after_promotion(self):
        gp = _make_gp()
        data = _make_rv_data(n_obs=5)
        cov_diag = jnp.ones(5) * 0.25

        # With amp=0, kernel matrix is zeros -> only diagonal remains
        nl = {"gp_amp": 0.0}
        cov_full = gp.modify_covariance(cov_diag, data, nl)
        assert jnp.allclose(jnp.diag(cov_full), 0.25)
        # Off-diagonal should be 0
        off_diag = cov_full - jnp.diag(jnp.diag(cov_full))
        assert jnp.allclose(off_diag, 0.0, atol=1e-10)

    def test_adds_kernel_to_full_cov(self):
        gp = _make_gp()
        data = _make_rv_data(n_obs=5)
        cov_full = jnp.diag(jnp.ones(5) * 0.25)

        nl = {"gp_amp": 1.0}
        result = gp.modify_covariance(cov_full, data, nl)
        # Diagonal should be 0.25 + kernel(0,0) = 0.25 + 1.0
        assert jnp.all(jnp.diag(result) > 0.25)
        # Off-diagonal should have kernel contributions
        assert result.shape == (5, 5)

    def test_kernel_diagonal_is_amp_squared(self):
        gp = _make_gp()
        data = _make_rv_data(n_obs=5)
        cov_diag = jnp.zeros(5)

        amp = 2.5
        nl = {"gp_amp": amp}
        cov_full = gp.modify_covariance(cov_diag, data, nl)
        # Kernel diagonal: amp^2 * exp(0) = amp^2
        assert jnp.allclose(jnp.diag(cov_full), amp**2)

    def test_does_not_modify_design_matrix(self):
        gp = _make_gp()
        data = _make_rv_data(n_obs=5)
        X = jnp.ones((5, 2))
        result = gp.modify_design_matrix(X, data, {})
        assert jnp.array_equal(X, result)


# ---------------------------------------------------------------------------
# Integration with RVModel
# ---------------------------------------------------------------------------


class TestGPWithRVModel:
    def test_rv_model_with_gp_construction(self):
        gp = _make_gp()
        model = RVModel(extensions=(gp,))
        assert "gp_amp" in model._all_nonlinear_names()

    def test_rv_model_log_prob_is_finite(self):
        data = _make_rv_data()
        gp = _make_gp()
        linear_prior = {
            "rv_semiamp": dist.Normal(0.0, 100.0),
            "v_sys": dist.Normal(0.0, 100.0),
        }
        model = RVModel(extensions=(gp,))
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
            "gp_amp": 1.0,
        }
        ll = model.log_prob(nl, data, linear_prior=linear_prior)
        assert jnp.isfinite(ll)

    def test_rv_model_with_gp_jit(self):
        data = _make_rv_data()
        gp = _make_gp()
        linear_prior = {
            "rv_semiamp": dist.Normal(0.0, 100.0),
            "v_sys": dist.Normal(0.0, 100.0),
        }
        model = RVModel(extensions=(gp,))
        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
            "gp_amp": 1.0,
        }

        @jax.jit
        def fn():
            return model.log_prob(nl, data, linear_prior=linear_prior)

        ll = fn()
        assert jnp.isfinite(ll)

    def test_gp_changes_log_prob(self):
        """Adding a GP should change the log-likelihood value."""
        data = _make_rv_data()
        prior = {
            "rv_semiamp": dist.Normal(0.0, 100.0),
            "v_sys": dist.Normal(0.0, 100.0),
        }

        model_no_gp = RVModel()
        model_gp = RVModel(extensions=(_make_gp(),))

        nl_base = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
        }
        nl_gp = {**nl_base, "gp_amp": 2.0}

        ll_no_gp = model_no_gp.log_prob(nl_base, data, linear_prior=prior)
        ll_gp = model_gp.log_prob(nl_gp, data, linear_prior=prior)

        assert not jnp.allclose(ll_no_gp, ll_gp)
        assert jnp.isfinite(ll_gp)

    def test_gp_plus_jitter(self):
        """GP and Jitter should compose correctly."""
        data = _make_rv_data()
        gp = _make_gp()
        jitter = Jitter("km/s")
        linear_prior = {
            "rv_semiamp": dist.Normal(0.0, 100.0),
            "v_sys": dist.Normal(0.0, 100.0),
        }
        model = RVModel(extensions=(jitter, gp))
        # Both jitter and gp_amp should be nonlinear params
        nl_names = model._all_nonlinear_names()
        assert "jitter" in nl_names
        assert "gp_amp" in nl_names

        nl = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(1.0, "rad"),
            "jitter": 0.5,
            "gp_amp": 1.0,
        }
        ll = model.log_prob(nl, data, linear_prior=linear_prior)
        assert jnp.isfinite(ll)
