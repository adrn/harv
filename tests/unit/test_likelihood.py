"""Unit tests for Gaia astrometry likelihood functions."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Quantity

from harv.data import GaiaAstrometryData
from harv.likelihood._params import GaiaAstrometryParameters
from harv.likelihood.gaia_astrometry import (
    MarginalizedGaiaAstrometryLikelihood,
    _get_design_matrix,
)


def _make_astro_data(n_obs=50):
    """Helper: synthetic GaiaAstrometryData."""
    return GaiaAstrometryData(
        time=Quantity(jnp.linspace(0, 1000, n_obs), "day"),
        al_position=Quantity(jnp.zeros(n_obs), "mas"),
        al_position_err=Quantity(jnp.ones(n_obs) * 0.1, "mas"),
        scan_angle=Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad"),
        parallax_factor=jnp.ones(n_obs) * 0.5,
        t_ref=Quantity(0.0, "day"),
    )


def _make_astro_params(
    period_day=100.0,
    eccentricity=0.3,
    phase_peri=0.0,
    arg_peri=1.0,
    cos_i=0.5,
    lon_asc_node=2.0,
):
    return GaiaAstrometryParameters.marginalized(
        period=Quantity(period_day, "day"),
        eccentricity=eccentricity,
        phase_peri=phase_peri,
        arg_peri=arg_peri,
        cos_i=cos_i,
        lon_asc_node=lon_asc_node,
    )


def _astro_prior(n=6):
    return dist.MultivariateNormal(
        loc=jnp.zeros(n), covariance_matrix=jnp.eye(n) * 1000.0**2
    )


class TestAstrometryDesignMatrix:
    """Tests for astrometry design matrix construction."""

    def test_design_matrix_shape(self):
        """Test that design matrix has correct shape (n_obs, 6)."""
        data = _make_astro_data(n_obs=50)
        params = _make_astro_params()
        sin_f = jnp.zeros(50)
        cos_f = jnp.ones(50)

        dm = _get_design_matrix(data, params, sin_f, cos_f)

        assert dm.shape == (50, 6)

    def test_design_matrix_pm_columns(self):
        """With scan_angle=0, RA offset column = 1 and Dec offset column = 0."""
        n_obs = 10
        data = GaiaAstrometryData(
            time=Quantity(jnp.zeros(n_obs), "day"),
            al_position=Quantity(jnp.zeros(n_obs), "mas"),
            al_position_err=Quantity(jnp.ones(n_obs) * 0.1, "mas"),
            scan_angle=Quantity(jnp.zeros(n_obs), "rad"),  # all zero
            parallax_factor=jnp.ones(n_obs),
            t_ref=Quantity(0.0, "day"),
        )
        params = _make_astro_params()
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)

        dm = _get_design_matrix(data, params, sin_f, cos_f)

        # scan_angle=0 → cos(ψ)=1, sin(ψ)=0
        # Column 0 (ra0): cos(scan) = 1
        assert jnp.allclose(dm[:, 0], 1.0)
        # Column 1 (dec0): sin(scan) = 0
        assert jnp.allclose(dm[:, 1], 0.0)


class TestMarginalizedLikelihood:
    """Tests for marginalized astrometry log-likelihood."""

    def test_likelihood_finite(self):
        """Test that likelihood returns a finite scalar."""
        data = _make_astro_data()
        lik = MarginalizedGaiaAstrometryLikelihood(
            data=data, linear_prior=_astro_prior()
        )
        params = _make_astro_params()

        log_lik = lik.log_prob(params)

        assert jnp.isfinite(log_lik)

    def test_likelihood_decreases_with_noise(self):
        """Test that log_prob is higher with smaller errors."""
        n_obs = 30
        times = Quantity(jnp.linspace(0, 500, n_obs), "day")
        scan = Quantity(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad")
        pf = jnp.ones(n_obs) * 0.5

        data_small = GaiaAstrometryData(
            time=times,
            al_position=Quantity(jnp.zeros(n_obs), "mas"),
            al_position_err=Quantity(jnp.ones(n_obs) * 0.01, "mas"),
            scan_angle=scan,
            parallax_factor=pf,
            t_ref=Quantity(0.0, "day"),
        )
        data_large = GaiaAstrometryData(
            time=times,
            al_position=Quantity(jnp.zeros(n_obs), "mas"),
            al_position_err=Quantity(jnp.ones(n_obs) * 1.0, "mas"),
            scan_angle=scan,
            parallax_factor=pf,
            t_ref=Quantity(0.0, "day"),
        )

        params = _make_astro_params()
        prior = _astro_prior()

        log_lik_small = MarginalizedGaiaAstrometryLikelihood(
            data_small, prior
        ).log_prob(params)
        log_lik_large = MarginalizedGaiaAstrometryLikelihood(
            data_large, prior
        ).log_prob(params)

        assert log_lik_small > log_lik_large

    def test_circular_vs_eccentric(self):
        """Both circular and eccentric log_probs should be finite."""
        data = _make_astro_data()
        prior = _astro_prior()

        log_lik_circ = MarginalizedGaiaAstrometryLikelihood(data, prior).log_prob(
            _make_astro_params(eccentricity=0.0, arg_peri=0.0)
        )
        log_lik_ecc = MarginalizedGaiaAstrometryLikelihood(data, prior).log_prob(
            _make_astro_params(eccentricity=0.7, arg_peri=1.0)
        )

        assert jnp.isfinite(log_lik_circ)
        assert jnp.isfinite(log_lik_ecc)


class TestBatchLikelihood:
    """Tests for batched astrometry likelihood via vmap."""

    def test_batch_shape(self):
        """Test that vmap over log_prob returns correct shape."""
        n_samples = 10
        data = _make_astro_data()
        lik = MarginalizedGaiaAstrometryLikelihood(
            data=data, linear_prior=_astro_prior()
        )

        params_batch = GaiaAstrometryParameters.marginalized(
            period=Quantity(jnp.ones(n_samples) * 100.0, "day"),
            eccentricity=jnp.linspace(0.0, 0.5, n_samples),
            phase_peri=jnp.zeros(n_samples),
            arg_peri=jnp.ones(n_samples) * 1.0,
            cos_i=jnp.ones(n_samples) * 0.5,
            lon_asc_node=jnp.ones(n_samples) * 2.0,
        )

        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        assert log_liks.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_batch_vs_single(self):
        """Test that vmap gives same result as serial evaluation."""
        n_obs = 20
        data = _make_astro_data(n_obs=n_obs)
        prior = _astro_prior()
        lik = MarginalizedGaiaAstrometryLikelihood(data=data, linear_prior=prior)

        eccs = jnp.array([0.1, 0.3])
        params_batch = GaiaAstrometryParameters.marginalized(
            period=Quantity(jnp.ones(2) * 100.0, "day"),
            eccentricity=eccs,
            phase_peri=jnp.zeros(2),
            arg_peri=jnp.ones(2),
            cos_i=jnp.ones(2) * 0.5,
            lon_asc_node=jnp.ones(2) * 2.0,
        )
        log_liks_batch = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        log_liks_serial = jnp.array(
            [
                lik.log_prob(_make_astro_params(eccentricity=float(eccs[i])))
                for i in range(2)
            ]
        )

        assert jnp.allclose(log_liks_batch, log_liks_serial, rtol=1e-5)
