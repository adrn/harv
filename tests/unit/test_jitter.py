"""Unit tests for the jitter (excess variance) feature."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
from harv.distributions import QuantityDistribution as QD
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    RVParameters,
    SB2RVParameters,
)
from harv.likelihood.rv import RVLikelihood
from harv.samplers.rejection_prior import RejectionPrior

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rv_data(n_obs=20):
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
    )


def _make_rv_params(**overrides):
    defaults = dict(
        period=Q(100.0, "day"),
        eccentricity=0.3,
        phase_peri=0.0,
        arg_peri=Q(1.0, "rad"),
        rv_semiamp=Q(10.0, "km/s"),
        v_sys=Q(0.0, "km/s"),
    )
    defaults.update(overrides)
    return RVParameters(**defaults)


def _make_rv_marg_params(**overrides):
    defaults = dict(
        period=Q(100.0, "day"),
        eccentricity=0.3,
        phase_peri=0.0,
        arg_peri=Q(1.0, "rad"),
    )
    defaults.update(overrides)
    return RVParameters.marginalized(**defaults)


# ---------------------------------------------------------------------------
# Parameter struct tests
# ---------------------------------------------------------------------------


class TestJitterFieldTypes:
    """Jitter field should have proper Quantity type on each param class."""

    def test_rv_jitter_default_none(self):
        params = _make_rv_params()
        assert params.jitter is None

    def test_rv_jitter_is_quantity(self):
        params = _make_rv_params(jitter=Q(1.0, "km/s"))
        assert hasattr(params.jitter, "unit")

    def test_gaia_jitter_default_none(self):
        params = GaiaAstrometryParameters(
            period=Q(100.0, "day"),
            eccentricity=0.3,
            phase_peri=0.0,
            arg_peri=Q(1.0, "rad"),
            cos_i=0.5,
            lon_asc_node=Q(1.0, "rad"),
            ra0=Q(0.0, "mas"),
            dec0=Q(0.0, "mas"),
            pmra=Q(0.0, "mas/yr"),
            pmdec=Q(0.0, "mas/yr"),
            parallax=Q(1.0, "mas"),
            semi_major_axis=Q(1.0, "AU"),
        )
        assert params.jitter is None

    def test_sb2_jitter_default_none(self):
        params = SB2RVParameters(
            period=Q(100.0, "day"),
            eccentricity=0.3,
            phase_peri=0.0,
            arg_peri=Q(1.0, "rad"),
            rv_semiamp_1=Q(10.0, "km/s"),
            rv_semiamp_2=Q(8.0, "km/s"),
            v_sys=Q(0.0, "km/s"),
        )
        assert params.jitter is None

    def test_jitter_not_in_nonlinear_param_names(self):
        """Jitter is optional and should NOT be in nonlinear_param_names."""
        assert "jitter" not in RVParameters.nonlinear_param_names
        assert "jitter" not in GaiaAstrometryParameters.nonlinear_param_names
        assert "jitter" not in SB2RVParameters.nonlinear_param_names

    def test_jitter_in_optional_nonlinear_param_names(self):
        assert "jitter" in RVParameters._optional_nonlinear_param_names
        assert "jitter" in GaiaAstrometryParameters._optional_nonlinear_param_names
        assert "jitter" in SB2RVParameters._optional_nonlinear_param_names


# ---------------------------------------------------------------------------
# Likelihood tests
# ---------------------------------------------------------------------------


class TestJitterLikelihood:
    """Jitter should inflate errors, reducing log-likelihood magnitude."""

    def test_jitter_none_identity(self):
        """jitter=None should give same result as before (no inflation)."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data)
        params_no_jitter = _make_rv_marg_params()
        lp = lik.log_prob(params_no_jitter)
        assert jnp.isfinite(lp)

    def test_jitter_inflates_errors(self):
        """Non-zero jitter should change the marginalized log-prob."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data)

        params_no_jitter = _make_rv_marg_params()
        params_with_jitter = _make_rv_marg_params(jitter=Q(5.0, "km/s"))

        lp_no = lik.log_prob(params_no_jitter)
        lp_yes = lik.log_prob(params_with_jitter)

        # Jitter should change the log-prob (not necessarily in one direction
        # for the marginalized case, since it changes the evidence integral).
        assert lp_no != lp_yes

    def test_explicit_jitter_inflates_errors(self):
        """Jitter should also work in explicit (non-marginalized) mode."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data)

        params_no = _make_rv_params()
        params_yes = _make_rv_params(jitter=Q(5.0, "km/s"))

        lp_no = lik.log_prob(params_no)
        lp_yes = lik.log_prob(params_yes)

        assert lp_yes > lp_no

    def test_vmap_with_jitter_none(self):
        """Vmap should work when jitter is None (static leaf)."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data)

        n_samples = 5
        params_batch = RVParameters(
            period=Q(jnp.ones(n_samples) * 100.0, "day"),
            eccentricity=jnp.ones(n_samples) * 0.3,
            phase_peri=jnp.zeros(n_samples),
            arg_peri=Q(jnp.ones(n_samples) * 1.0, "rad"),
            rv_semiamp=Q(jnp.ones(n_samples) * 10.0, "km/s"),
            v_sys=Q(jnp.zeros(n_samples), "km/s"),
        )
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
        assert log_liks.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_vmap_with_jitter_batched(self):
        """Vmap should work when jitter is batched."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data)

        n_samples = 5
        params_batch = RVParameters(
            period=Q(jnp.ones(n_samples) * 100.0, "day"),
            eccentricity=jnp.ones(n_samples) * 0.3,
            phase_peri=jnp.zeros(n_samples),
            arg_peri=Q(jnp.ones(n_samples) * 1.0, "rad"),
            rv_semiamp=Q(jnp.ones(n_samples) * 10.0, "km/s"),
            v_sys=Q(jnp.zeros(n_samples), "km/s"),
            jitter=Q(jnp.ones(n_samples) * 1.0, "km/s"),
        )
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
        assert log_liks.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(log_liks))


# ---------------------------------------------------------------------------
# Prior construction tests
# ---------------------------------------------------------------------------


class TestJitterPrior:
    """default_* methods should accept jitter and set jitter_priors."""

    def test_default_rv_no_jitter(self):
        prior = RejectionPrior.default_rv(
            period_min=Q(50.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(10.0, "km/s"),
        )
        assert prior.jitter_priors is None

    def test_default_rv_with_jitter(self):
        prior = RejectionPrior.default_rv(
            period_min=Q(50.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(10.0, "km/s"),
            jitter_scale=Q(1.0, "km/s"),
        )
        assert prior.jitter_priors is not None
        assert "rv" in prior.jitter_priors
        assert isinstance(prior.jitter_priors["rv"], QD)

    def test_default_gaia_astrometry_with_jitter(self):
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Q(50.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_a0=Q(1.0, "AU"),
            sigma_parallax=Q(1.0, "mas"),
            sigma_pos=Q(1.0, "mas"),
            sigma_vtan=Q(10.0, "km/s"),
            jitter_scale=Q(0.1, "mas"),
        )
        assert prior.jitter_priors is not None
        assert "astrometry" in prior.jitter_priors

    def test_default_sb2_with_jitter(self):
        prior = RejectionPrior.default_sb2(
            period_min=Q(50.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(10.0, "km/s"),
            jitter_scale=Q(1.0, "km/s"),
        )
        assert prior.jitter_priors is not None
        assert "rv" in prior.jitter_priors

    def test_manual_jitter_priors(self):
        """User can pass jitter_priors directly to __init__."""
        prior = RejectionPrior.default_rv(
            period_min=Q(50.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(10.0, "km/s"),
        )
        # Construct manually with jitter_priors
        prior2 = RejectionPrior(
            nonlinear_priors=prior.nonlinear_priors,
            linear_prior=prior.linear_prior,
            jitter_priors={"rv": QD(dist.HalfNormal(2.0), "km/s")},
        )
        assert prior2.jitter_priors is not None
        assert "rv" in prior2.jitter_priors
