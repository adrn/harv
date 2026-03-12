"""Unit tests for likelihood functions."""

import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Quantity

from epochalypse.likelihood.astrometry import (
    compute_marginal_log_likelihood_astrometry,
    compute_marginal_log_likelihood_astrometry_batch,
    get_astrometry_design_matrix,
)


class TestAstrometryDesignMatrix:
    """Tests for astrometry design matrix construction."""

    def test_design_matrix_shape(self):
        """Test that design matrix has correct shape."""
        n_obs = 50
        times = Quantity(jnp.linspace(0, 1000, n_obs), "day")
        scan_angle = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        parallax_factor = jnp.ones(n_obs) * 0.5
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        t_ref = Quantity(0.0, "day")
        cos_i = 0.5
        arg_peri = 1.0
        lon_asc_node = 2.0

        design_matrix = get_astrometry_design_matrix(
            times,
            scan_angle,
            parallax_factor,
            sin_f,
            cos_f,
            t_ref,
            cos_i,
            arg_peri,
            lon_asc_node,
        )

        assert design_matrix.shape == (n_obs, 6)

    def test_design_matrix_columns(self):
        """Test that design matrix columns are computed correctly."""
        n_obs = 10
        times = Quantity(jnp.array([0.0, 1.0, 2.0] * 3 + [3.0]), "day")
        scan_angle = Quantity(jnp.zeros(n_obs), "rad")  # All zero for simplicity
        parallax_factor = jnp.ones(n_obs)
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        t_ref = Quantity(0.0, "day")
        cos_i = 1.0  # Face-on
        arg_peri = 0.0
        lon_asc_node = 0.0

        design_matrix = get_astrometry_design_matrix(
            times,
            scan_angle,
            parallax_factor,
            sin_f,
            cos_f,
            t_ref,
            cos_i,
            arg_peri,
            lon_asc_node,
        )

        # With scan_angle=0, cos(scan)=1, sin(scan)=0
        # Column 0 (α₀): cos(scan) = 1
        assert jnp.allclose(design_matrix[:, 0], 1.0)
        # Column 1 (δ₀): sin(scan) = 0
        assert jnp.allclose(design_matrix[:, 1], 0.0)

    def test_thiele_innes_computation(self):
        """Test that Thiele-Innes constants are computed correctly."""
        n_obs = 5
        times = Quantity(jnp.zeros(n_obs), "day")
        scan_angle = Quantity(jnp.zeros(n_obs), "rad")
        parallax_factor = jnp.zeros(n_obs)
        sin_f = jnp.ones(n_obs)  # sin(f) = 1
        cos_f = jnp.zeros(n_obs)  # cos(f) = 0
        t_ref = Quantity(0.0, "day")

        # Face-on orbit (i=0, cos(i)=1)
        cos_i = 1.0
        arg_peri = 0.0  # ω = 0
        lon_asc_node = 0.0  # Ω = 0

        design_matrix = get_astrometry_design_matrix(
            times,
            scan_angle,
            parallax_factor,
            sin_f,
            cos_f,
            t_ref,
            cos_i,
            arg_peri,
            lon_asc_node,
        )

        # For face-on, ω=0, Ω=0:
        # A = cos(0)*cos(0) - sin(0)*sin(0)*1 = 1
        # B = cos(0)*sin(0) + sin(0)*cos(0)*1 = 0
        # F = -sin(0)*cos(0) - cos(0)*sin(0)*1 = 0
        # G = -sin(0)*sin(0) + cos(0)*cos(0)*1 = 1
        # semimaj_term = (A*0 + B*1)*0 + (F*0 + G*1)*1 = G = 1
        # Column 5 should be semimaj_term = 1 (with scan_angle=0, sin=0, cos=1)
        # Actually: semimaj_term = (A*sin + B*cos)*cos_f + (F*sin + G*cos)*sin_f
        #         = (1*0 + 0*1)*0 + (0*0 + 1*1)*1 = 1
        assert jnp.allclose(design_matrix[:, 5], 1.0)


class TestMarginalizedLikelihood:
    """Tests for marginalized log-likelihood computation."""

    def test_likelihood_finite(self):
        """Test that likelihood returns finite value."""
        n_obs = 50
        times = Quantity(jnp.linspace(0, 1000, n_obs), "day")
        scan_angle = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        parallax_factor = jnp.ones(n_obs) * 0.5
        y_al = Quantity(jnp.zeros(n_obs), "mas")
        y_al_error = Quantity(jnp.ones(n_obs) * 0.1, "mas")
        t_ref = Quantity(0.0, "day")

        log_lik = compute_marginal_log_likelihood_astrometry(
            log_period=jnp.log10(100.0),
            eccentricity=0.3,
            phase_peri=0.0,
            cos_i=jnp.cos(1.0),
            arg_peri=0.5,
            lon_asc_node=1.0,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            y_al=y_al,
            y_al_error=y_al_error,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 1000.0),
        )

        assert jnp.isfinite(log_lik)
        assert isinstance(log_lik, (float, jnp.ndarray))

    def test_likelihood_decreases_with_noise(self):
        """Test that likelihood decreases as we add more noise."""
        n_obs = 30
        times = Quantity(jnp.linspace(0, 500, n_obs), "day")
        scan_angle = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        parallax_factor = jnp.ones(n_obs) * 0.5
        y_al = Quantity(jnp.zeros(n_obs), "mas")
        t_ref = Quantity(0.0, "day")

        # Small errors
        y_al_error_small = Quantity(jnp.ones(n_obs) * 0.01, "mas")
        log_lik_small = compute_marginal_log_likelihood_astrometry(
            log_period=2.0,
            eccentricity=0.2,
            phase_peri=0.0,
            cos_i=0.5,
            arg_peri=1.0,
            lon_asc_node=2.0,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            y_al=y_al,
            y_al_error=y_al_error_small,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 1000.0),
        )

        # Large errors
        y_al_error_large = Quantity(jnp.ones(n_obs) * 1.0, "mas")
        log_lik_large = compute_marginal_log_likelihood_astrometry(
            log_period=2.0,
            eccentricity=0.2,
            phase_peri=0.0,
            cos_i=0.5,
            arg_peri=1.0,
            lon_asc_node=2.0,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            y_al=y_al,
            y_al_error=y_al_error_large,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 1000.0),
        )

        # Likelihood should be higher (less negative) with smaller errors
        # since the model can fit the data more precisely
        assert log_lik_small > log_lik_large

    def test_circular_vs_eccentric(self):
        """Test likelihood for circular vs eccentric orbits."""
        n_obs = 50
        times = Quantity(jnp.linspace(0, 365, n_obs), "day")
        scan_angle = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        parallax_factor = jnp.ones(n_obs) * 0.5
        y_al = Quantity(jnp.zeros(n_obs), "mas")
        y_al_error = Quantity(jnp.ones(n_obs) * 0.1, "mas")
        t_ref = Quantity(0.0, "day")

        # Circular orbit
        log_lik_circular = compute_marginal_log_likelihood_astrometry(
            log_period=2.0,
            eccentricity=0.0,
            phase_peri=0.0,
            cos_i=0.5,
            arg_peri=0.0,
            lon_asc_node=0.0,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            y_al=y_al,
            y_al_error=y_al_error,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 1000.0),
        )

        # Eccentric orbit
        log_lik_eccentric = compute_marginal_log_likelihood_astrometry(
            log_period=2.0,
            eccentricity=0.7,
            phase_peri=0.0,
            cos_i=0.5,
            arg_peri=1.0,
            lon_asc_node=1.0,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            y_al=y_al,
            y_al_error=y_al_error,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 1000.0),
        )

        # Both should be finite
        assert jnp.isfinite(log_lik_circular)
        assert jnp.isfinite(log_lik_eccentric)


class TestBatchLikelihood:
    """Tests for batch likelihood computation."""

    def test_batch_shape(self):
        """Test that batch likelihood returns correct shape."""
        n_samples = 100
        n_obs = 30

        # Batch parameters
        log_periods = jnp.ones(n_samples) * 2.0
        eccentricities = jnp.linspace(0, 0.5, n_samples)
        phase_peris = jnp.zeros(n_samples)
        cos_is = jnp.ones(n_samples) * 0.5
        arg_peris = jnp.ones(n_samples) * 1.0
        lon_asc_nodes = jnp.ones(n_samples) * 2.0

        # Data (same for all)
        times = Quantity(jnp.linspace(0, 1000, n_obs), "day")
        scan_angle = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        parallax_factor = jnp.ones(n_obs) * 0.5
        y_al = Quantity(jnp.zeros(n_obs), "mas")
        y_al_error = Quantity(jnp.ones(n_obs) * 0.1, "mas")
        t_ref = Quantity(0.0, "day")

        log_liks = compute_marginal_log_likelihood_astrometry_batch(
            log_periods,
            eccentricities,
            phase_peris,
            cos_is,
            arg_peris,
            lon_asc_nodes,
            times,
            scan_angle,
            parallax_factor,
            y_al,
            y_al_error,
            t_ref,
            dist.Normal(0.0, 1000.0),
        )

        assert log_liks.shape == (n_samples,)
        assert jnp.isfinite(log_liks).all()

    def test_batch_vs_single(self):
        """Test that batch computation gives same results as single."""
        n_obs = 20
        times = Quantity(jnp.linspace(0, 500, n_obs), "day")
        scan_angle = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        parallax_factor = jnp.ones(n_obs) * 0.5
        y_al = Quantity(jnp.zeros(n_obs), "mas")
        y_al_error = Quantity(jnp.ones(n_obs) * 0.1, "mas")
        t_ref = Quantity(0.0, "day")

        # Single computation
        log_lik_single = compute_marginal_log_likelihood_astrometry(
            log_period=2.0,
            eccentricity=0.3,
            phase_peri=0.0,
            cos_i=0.5,
            arg_peri=1.0,
            lon_asc_node=2.0,
            times=times,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            y_al=y_al,
            y_al_error=y_al_error,
            t_ref=t_ref,
            linear_prior=dist.Normal(0.0, 1000.0),
        )

        # Batch computation with same parameters
        log_liks_batch = compute_marginal_log_likelihood_astrometry_batch(
            jnp.array([2.0, 2.0]),
            jnp.array([0.3, 0.3]),
            jnp.array([0.0, 0.0]),
            jnp.array([0.5, 0.5]),
            jnp.array([1.0, 1.0]),
            jnp.array([2.0, 2.0]),
            times,
            scan_angle,
            parallax_factor,
            y_al,
            y_al_error,
            t_ref,
            dist.Normal(0.0, 1000.0),
        )

        # Should be identical
        assert jnp.allclose(log_liks_batch[0], log_lik_single)
        assert jnp.allclose(log_liks_batch[1], log_lik_single)
