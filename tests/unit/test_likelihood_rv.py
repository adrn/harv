"""Unit tests for RV likelihood functions."""

import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Quantity

from harv.likelihood.rv import (
    compute_marginal_log_likelihood_rv,
    compute_marginal_log_likelihood_rv_batch,
    get_rv_design_matrix,
    get_rv_design_matrix_sb2,
)


class TestRVDesignMatrix:
    """Tests for RV design matrix construction."""

    def test_design_matrix_shape(self):
        """Test that design matrix has correct shape."""
        n_obs = 20
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        ecc = 0.3
        arg_peri = 1.0

        design_matrix = get_rv_design_matrix(sin_f, cos_f, ecc, arg_peri)

        assert design_matrix.shape == (n_obs, 2)
        # Second column should be all ones (v₀ coefficient)
        assert jnp.allclose(design_matrix[:, 1], 1.0)

    def test_circular_orbit(self):
        """Test design matrix for circular orbit (e=0)."""
        sin_f = jnp.array([0.0, 1.0, 0.0, -1.0])
        cos_f = jnp.array([1.0, 0.0, -1.0, 0.0])
        ecc = 0.0
        arg_peri = 0.0

        design_matrix = get_rv_design_matrix(sin_f, cos_f, ecc, arg_peri)

        # For e=0, ω=0: RV amplitude = cos(f)
        expected_amplitude = cos_f
        assert jnp.allclose(design_matrix[:, 0], expected_amplitude)

    def test_eccentric_orbit(self):
        """Test design matrix for eccentric orbit."""
        sin_f = jnp.array([0.0])
        cos_f = jnp.array([1.0])
        ecc = 0.3
        arg_peri = jnp.pi / 2  # 90 degrees

        design_matrix = get_rv_design_matrix(sin_f, cos_f, ecc, arg_peri)

        # At f=0, ω=π/2: cos(ω+f) = cos(π/2) = 0
        # RV amplitude = 0 + e·cos(π/2) = 0
        assert jnp.allclose(design_matrix[0, 0], 0.0, atol=1e-6)


class TestRVDesignMatrixSB2:
    """Tests for SB2 design matrix construction."""

    def test_sb2_primary_shape(self):
        """Test SB2 primary design matrix shape."""
        n_obs = 10
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        ecc = 0.2
        arg_peri = 0.5

        design_matrix = get_rv_design_matrix_sb2(
            sin_f, cos_f, ecc, arg_peri, primary=True
        )

        assert design_matrix.shape == (n_obs, 3)
        # Column 1 (K₂) should be zero for primary
        assert jnp.allclose(design_matrix[:, 1], 0.0)
        # Column 2 (v₀) should be all ones
        assert jnp.allclose(design_matrix[:, 2], 1.0)

    def test_sb2_secondary_shape(self):
        """Test SB2 secondary design matrix shape."""
        n_obs = 10
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        ecc = 0.2
        arg_peri = 0.5

        design_matrix = get_rv_design_matrix_sb2(
            sin_f, cos_f, ecc, arg_peri, primary=False
        )

        assert design_matrix.shape == (n_obs, 3)
        # Column 0 (K₁) should be zero for secondary
        assert jnp.allclose(design_matrix[:, 0], 0.0)
        # Column 2 (v₀) should be all ones
        assert jnp.allclose(design_matrix[:, 2], 1.0)

    def test_sb2_opposite_phases(self):
        """Test that primary and secondary have opposite RV amplitudes."""
        sin_f = jnp.array([0.0, 1.0])
        cos_f = jnp.array([1.0, 0.0])
        ecc = 0.3
        arg_peri = 0.0

        primary = get_rv_design_matrix_sb2(sin_f, cos_f, ecc, arg_peri, primary=True)
        secondary = get_rv_design_matrix_sb2(sin_f, cos_f, ecc, arg_peri, primary=False)

        # Primary K₁ coefficient = -Secondary K₂ coefficient
        assert jnp.allclose(primary[:, 0], -secondary[:, 1])


class TestMarginalizedLikelihoodRV:
    """Tests for marginalized RV likelihood computation."""

    def test_likelihood_is_finite(self):
        """Test that likelihood returns finite value."""
        times = Quantity(jnp.linspace(0, 100, 20), "day")
        rv = Quantity(jnp.zeros(20), "km/s")
        rv_err = Quantity(jnp.ones(20) * 0.1, "km/s")
        t_ref = Quantity(0.0, "day")

        log_lik = compute_marginal_log_likelihood_rv(
            log_period=2.0,
            eccentricity=0.2,
            phase_peri=0.0,
            arg_peri=1.0,
            times=times,
            rv=rv,
            rv_err=rv_err,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 100.0),
        )

        assert jnp.isfinite(log_lik)

    def test_likelihood_decreases_with_noise(self):
        """Test that likelihood decreases as we add more noise."""
        n_obs = 30
        times = Quantity(jnp.linspace(0, 100, n_obs), "day")
        rv = Quantity(jnp.zeros(n_obs), "km/s")
        t_ref = Quantity(0.0, "day")

        # Small errors
        rv_err_small = Quantity(jnp.ones(n_obs) * 0.01, "km/s")
        log_lik_small = compute_marginal_log_likelihood_rv(
            log_period=2.0,
            eccentricity=0.2,
            phase_peri=0.0,
            arg_peri=1.0,
            times=times,
            rv=rv,
            rv_err=rv_err_small,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 100.0),
        )

        # Large errors
        rv_err_large = Quantity(jnp.ones(n_obs) * 1.0, "km/s")
        log_lik_large = compute_marginal_log_likelihood_rv(
            log_period=2.0,
            eccentricity=0.2,
            phase_peri=0.0,
            arg_peri=1.0,
            times=times,
            rv=rv,
            rv_err=rv_err_large,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 100.0),
        )

        # Likelihood should be higher (less negative) with smaller errors
        assert log_lik_small > log_lik_large

    def test_circular_vs_eccentric(self):
        """Test likelihood for circular vs eccentric orbits with same data."""
        n_obs = 50
        times = Quantity(jnp.linspace(0, 365, n_obs), "day")
        rv = Quantity(jnp.zeros(n_obs), "km/s")
        rv_err = Quantity(jnp.ones(n_obs) * 0.1, "km/s")
        t_ref = Quantity(0.0, "day")

        # Circular orbit
        log_lik_circular = compute_marginal_log_likelihood_rv(
            log_period=2.0,
            eccentricity=0.0,
            phase_peri=0.0,
            arg_peri=0.0,
            times=times,
            rv=rv,
            rv_err=rv_err,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 100.0),
        )

        # Eccentric orbit
        log_lik_eccentric = compute_marginal_log_likelihood_rv(
            log_period=2.0,
            eccentricity=0.5,
            phase_peri=0.0,
            arg_peri=0.0,
            times=times,
            rv=rv,
            rv_err=rv_err,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 100.0),
        )

        # Both should be finite
        assert jnp.isfinite(log_lik_circular)
        assert jnp.isfinite(log_lik_eccentric)


class TestBatchLikelihoodRV:
    """Tests for batched RV likelihood computation."""

    def test_batch_shape(self):
        """Test that batch likelihood returns correct shape."""
        n_samples = 100
        n_obs = 20

        log_period = jnp.ones(n_samples) * 2.0
        ecc = jnp.linspace(0.0, 0.5, n_samples)
        phase_peri = jnp.zeros(n_samples)
        arg_peri = jnp.ones(n_samples) * 1.0

        times = Quantity(jnp.linspace(0, 100, n_obs), "day")
        rv = Quantity(jnp.zeros(n_obs), "km/s")
        rv_err = Quantity(jnp.ones(n_obs) * 0.1, "km/s")
        t_ref = Quantity(0.0, "day")

        log_liks = compute_marginal_log_likelihood_rv_batch(
            log_period,
            ecc,
            phase_peri,
            arg_peri,
            times,
            rv,
            rv_err,
            t_ref,
            dist.Normal(0.0, 100.0),
        )

        assert log_liks.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_batch_vs_single(self):
        """Test that batch computation matches single computation."""
        n_samples = 10
        n_obs = 15

        log_period = jnp.ones(n_samples) * 2.0
        ecc = jnp.linspace(0.0, 0.5, n_samples)
        phase_peri = jnp.zeros(n_samples)
        arg_peri = jnp.ones(n_samples) * 1.0

        times = Quantity(jnp.linspace(0, 100, n_obs), "day")
        rv = Quantity(jnp.zeros(n_obs), "km/s")
        rv_err = Quantity(jnp.ones(n_obs) * 0.1, "km/s")
        t_ref = Quantity(0.0, "day")
        linear_prior = dist.Normal(0.0, 100.0)

        # Batch computation
        log_liks_batch = compute_marginal_log_likelihood_rv_batch(
            log_period,
            ecc,
            phase_peri,
            arg_peri,
            times,
            rv,
            rv_err,
            t_ref,
            linear_prior,
        )

        # Single computation
        log_liks_single = jnp.array(
            [
                compute_marginal_log_likelihood_rv(
                    log_period[i],
                    ecc[i],
                    phase_peri[i],
                    arg_peri[i],
                    times,
                    rv,
                    rv_err,
                    t_ref,
                    linear_prior,
                )
                for i in range(n_samples)
            ]
        )

        assert jnp.allclose(log_liks_batch, log_liks_single, rtol=1e-5)
