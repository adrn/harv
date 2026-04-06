"""Unit tests for RV likelihood functions."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Quantity

from harv.data import RadialVelocityData
from harv.likelihood._params import RVMarginalizedParameters
from harv.likelihood.rv import (
    MarginalizedRVLikelihood,
    _get_design_matrix,
    _get_design_matrix_sb2,
)


def _make_rv_params(period_day=100.0, eccentricity=0.3, phase_peri=0.0, arg_peri=1.0):
    return RVMarginalizedParameters(
        period=Quantity(period_day, "day"),
        eccentricity=eccentricity,
        phase_peri=phase_peri,
        arg_peri=arg_peri,
    )


def _make_rv_data(n_obs=20):
    return RadialVelocityData(
        time=Quantity(jnp.linspace(0, 100, n_obs), "day"),
        rv=Quantity(jnp.zeros(n_obs), "km/s"),
        rv_err=Quantity(jnp.ones(n_obs) * 0.1, "km/s"),
    )


def _rv_prior(n=2):
    return dist.MultivariateNormal(
        loc=jnp.zeros(n), covariance_matrix=jnp.eye(n) * 100.0**2
    )


class TestRVDesignMatrix:
    """Tests for RV design matrix construction."""

    def test_design_matrix_shape(self):
        """Test that design matrix has correct shape."""
        n_obs = 20
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        params = _make_rv_params(eccentricity=0.3, arg_peri=1.0)

        dm = _get_design_matrix(params, sin_f, cos_f)

        assert dm.shape == (n_obs, 2)
        # Second column should be all ones (v₀ coefficient)
        assert jnp.allclose(dm[:, 1], 1.0)

    def test_circular_orbit(self):
        """Test design matrix for circular orbit (e=0)."""
        sin_f = jnp.array([0.0, 1.0, 0.0, -1.0])
        cos_f = jnp.array([1.0, 0.0, -1.0, 0.0])
        params = _make_rv_params(eccentricity=0.0, arg_peri=0.0)

        dm = _get_design_matrix(params, sin_f, cos_f)

        # For e=0, ω=0: RV amplitude = cos(f)
        assert jnp.allclose(dm[:, 0], cos_f)

    def test_eccentric_orbit(self):
        """Test design matrix for eccentric orbit."""
        sin_f = jnp.array([0.0])
        cos_f = jnp.array([1.0])
        params = _make_rv_params(eccentricity=0.3, arg_peri=jnp.pi / 2)

        dm = _get_design_matrix(params, sin_f, cos_f)

        # At f=0, ω=π/2: cos(ω+f) = cos(π/2) = 0, e·cos(π/2) = 0
        assert jnp.allclose(dm[0, 0], 0.0, atol=1e-6)


class TestRVDesignMatrixSB2:
    """Tests for SB2 design matrix construction."""

    def test_sb2_primary_shape(self):
        """Test SB2 primary design matrix shape."""
        n_obs = 10
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        params = _make_rv_params(eccentricity=0.2, arg_peri=0.5)

        dm = _get_design_matrix_sb2(params, sin_f, cos_f, primary=True)

        assert dm.shape == (n_obs, 3)
        # Column 1 (K₂) should be zero for primary
        assert jnp.allclose(dm[:, 1], 0.0)
        # Column 2 (v₀) should be all ones
        assert jnp.allclose(dm[:, 2], 1.0)

    def test_sb2_secondary_shape(self):
        """Test SB2 secondary design matrix shape."""
        n_obs = 10
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        params = _make_rv_params(eccentricity=0.2, arg_peri=0.5)

        dm = _get_design_matrix_sb2(params, sin_f, cos_f, primary=False)

        assert dm.shape == (n_obs, 3)
        # Column 0 (K₁) should be zero for secondary
        assert jnp.allclose(dm[:, 0], 0.0)
        # Column 2 (v₀) should be all ones
        assert jnp.allclose(dm[:, 2], 1.0)

    def test_sb2_opposite_phases(self):
        """Test that primary and secondary have opposite RV amplitudes."""
        sin_f = jnp.array([0.0, 1.0])
        cos_f = jnp.array([1.0, 0.0])
        params = _make_rv_params(eccentricity=0.3, arg_peri=0.0)

        primary = _get_design_matrix_sb2(params, sin_f, cos_f, primary=True)
        secondary = _get_design_matrix_sb2(params, sin_f, cos_f, primary=False)

        # Primary K₁ coefficient = -Secondary K₂ coefficient
        assert jnp.allclose(primary[:, 0], -secondary[:, 1])


class TestMarginalizedLikelihoodRV:
    """Tests for marginalized RV likelihood computation."""

    def test_likelihood_is_finite(self):
        """Test that likelihood returns finite value."""
        data = _make_rv_data()
        lik = MarginalizedRVLikelihood(data=data, linear_prior=_rv_prior())
        params = _make_rv_params()

        log_lik = lik.log_prob(params)

        assert jnp.isfinite(log_lik)

    def test_likelihood_decreases_with_noise(self):
        """Test that likelihood is higher with smaller errors."""
        n_obs = 30
        times = Quantity(jnp.linspace(0, 100, n_obs), "day")
        rv = Quantity(jnp.zeros(n_obs), "km/s")

        data_small = RadialVelocityData(
            time=times,
            rv=rv,
            rv_err=Quantity(jnp.ones(n_obs) * 0.01, "km/s"),
        )
        data_large = RadialVelocityData(
            time=times,
            rv=rv,
            rv_err=Quantity(jnp.ones(n_obs) * 1.0, "km/s"),
        )

        params = _make_rv_params(eccentricity=0.2)
        prior = _rv_prior()

        log_lik_small = MarginalizedRVLikelihood(data_small, prior).log_prob(params)
        log_lik_large = MarginalizedRVLikelihood(data_large, prior).log_prob(params)

        assert log_lik_small > log_lik_large

    def test_circular_vs_eccentric(self):
        """Test likelihood for circular vs eccentric orbits."""
        data = RadialVelocityData(
            time=Quantity(jnp.linspace(0, 365, 50), "day"),
            rv=Quantity(jnp.zeros(50), "km/s"),
            rv_err=Quantity(jnp.ones(50) * 0.1, "km/s"),
        )
        prior = _rv_prior()

        log_lik_circ = MarginalizedRVLikelihood(data, prior).log_prob(
            _make_rv_params(eccentricity=0.0, arg_peri=0.0)
        )
        log_lik_ecc = MarginalizedRVLikelihood(data, prior).log_prob(
            _make_rv_params(eccentricity=0.5, arg_peri=0.0)
        )

        assert jnp.isfinite(log_lik_circ)
        assert jnp.isfinite(log_lik_ecc)


class TestBatchLikelihoodRV:
    """Tests for batched RV likelihood via vmap."""

    def test_batch_shape(self):
        """Test that vmap over log_prob returns correct shape."""
        n_samples = 10
        data = _make_rv_data()
        lik = MarginalizedRVLikelihood(data=data, linear_prior=_rv_prior())

        eccentricities = jnp.linspace(0.0, 0.5, n_samples)
        params_batch = RVMarginalizedParameters(
            period=Quantity(jnp.ones(n_samples) * 100.0, "day"),
            eccentricity=eccentricities,
            phase_peri=jnp.zeros(n_samples),
            arg_peri=jnp.ones(n_samples) * 1.0,
        )

        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        assert log_liks.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_batch_vs_single(self):
        """Test that vmap gives same result as serial evaluation."""
        n_obs = 15
        data = RadialVelocityData(
            time=Quantity(jnp.linspace(0, 100, n_obs), "day"),
            rv=Quantity(jnp.zeros(n_obs), "km/s"),
            rv_err=Quantity(jnp.ones(n_obs) * 0.1, "km/s"),
        )
        prior = _rv_prior()
        lik = MarginalizedRVLikelihood(data=data, linear_prior=prior)

        eccs = jnp.linspace(0.0, 0.5, 5)

        params_batch = RVMarginalizedParameters(
            period=Quantity(jnp.ones(5) * 100.0, "day"),
            eccentricity=eccs,
            phase_peri=jnp.zeros(5),
            arg_peri=jnp.ones(5),
        )
        log_liks_batch = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        log_liks_serial = jnp.array(
            [
                lik.log_prob(
                    RVMarginalizedParameters(
                        period=Quantity(100.0, "day"),
                        eccentricity=float(eccs[i]),
                        phase_peri=0.0,
                        arg_peri=1.0,
                    )
                )
                for i in range(5)
            ]
        )

        assert jnp.allclose(log_liks_batch, log_liks_serial, rtol=1e-5)
