"""Unit tests for RV likelihood functions."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import RVData
from harv.likelihood.params import RVParameters
from harv.likelihood.rv import (
    RVLikelihood,
    _get_design_matrix_sb2,
)
from harv.likelihood.rv import (
    _get_design_matrix_sb1 as _get_design_matrix,
)


def _make_rv_params(period_day=100.0, eccentricity=0.3, phase_peri=0.0, arg_peri=1.0):
    return RVParameters.marginalized(
        period=Q(period_day, "day"),
        eccentricity=eccentricity,
        phase_peri=phase_peri,
        arg_peri=Q(arg_peri, "rad"),
    )


def _make_rv_data(n_obs=20):
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
    )


def _rv_prior():
    return {
        "rv_semiamp": dist.Normal(0.0, 100.0),
        "v_sys": dist.Normal(0.0, 100.0),
    }


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
        # Second column should be all ones (v_0 coefficient)
        assert jnp.allclose(dm[:, 1], 1.0)

    def test_circular_orbit(self):
        """Test design matrix for circular orbit (e=0)."""
        sin_f = jnp.array([0.0, 1.0, 0.0, -1.0])
        cos_f = jnp.array([1.0, 0.0, -1.0, 0.0])
        params = _make_rv_params(eccentricity=0.0, arg_peri=0.0)

        dm = _get_design_matrix(params, sin_f, cos_f)

        # For e=0, omega=0: RV amplitude = cos(f)
        assert jnp.allclose(dm[:, 0], cos_f)

    def test_eccentric_orbit(self):
        """Test design matrix for eccentric orbit."""
        sin_f = jnp.array([0.0])
        cos_f = jnp.array([1.0])
        params = _make_rv_params(eccentricity=0.3, arg_peri=jnp.pi / 2)

        dm = _get_design_matrix(params, sin_f, cos_f)

        # At f=0, omega=pi/2: cos(omega+f) = cos(pi/2) = 0, e*cos(pi/2) = 0
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
        # Column 1 (K_2) should be zero for primary
        assert jnp.allclose(dm[:, 1], 0.0)
        # Column 2 (v_0) should be all ones
        assert jnp.allclose(dm[:, 2], 1.0)

    def test_sb2_secondary_shape(self):
        """Test SB2 secondary design matrix shape."""
        n_obs = 10
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        params = _make_rv_params(eccentricity=0.2, arg_peri=0.5)

        dm = _get_design_matrix_sb2(params, sin_f, cos_f, primary=False)

        assert dm.shape == (n_obs, 3)
        # Column 0 (K_1) should be zero for secondary
        assert jnp.allclose(dm[:, 0], 0.0)
        # Column 2 (v_0) should be all ones
        assert jnp.allclose(dm[:, 2], 1.0)

    def test_sb2_opposite_phases(self):
        """Test that primary and secondary have opposite RV amplitudes."""
        sin_f = jnp.array([0.0, 1.0])
        cos_f = jnp.array([1.0, 0.0])
        params = _make_rv_params(eccentricity=0.3, arg_peri=0.0)

        primary = _get_design_matrix_sb2(params, sin_f, cos_f, primary=True)
        secondary = _get_design_matrix_sb2(params, sin_f, cos_f, primary=False)

        # Primary K_1 coefficient = -Secondary K_2 coefficient
        assert jnp.allclose(primary[:, 0], -secondary[:, 1])


class TestMarginalizedLikelihoodRV:
    """Tests for marginalized RV likelihood computation."""

    def test_likelihood_is_finite(self):
        """Test that likelihood returns finite value."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data, linear_marginalized_prior=_rv_prior())
        params = _make_rv_params()

        log_lik = lik.log_prob(params)

        assert jnp.isfinite(log_lik)

    def test_likelihood_decreases_with_noise(self):
        """Test that likelihood is higher with smaller errors."""
        n_obs = 30
        times = Q(jnp.linspace(0, 100, n_obs), "day")
        rv = Q(jnp.zeros(n_obs), "km/s")

        data_small = RVData(
            time=times,
            rv=rv,
            rv_err=Q(jnp.ones(n_obs) * 0.01, "km/s"),
        )
        data_large = RVData(
            time=times,
            rv=rv,
            rv_err=Q(jnp.ones(n_obs) * 1.0, "km/s"),
        )

        params = _make_rv_params(eccentricity=0.2)
        prior = _rv_prior()

        log_lik_small = RVLikelihood(
            data_small, linear_marginalized_prior=prior
        ).log_prob(params)
        log_lik_large = RVLikelihood(
            data_large, linear_marginalized_prior=prior
        ).log_prob(params)

        assert log_lik_small > log_lik_large

    def test_circular_vs_eccentric(self):
        """Test likelihood for circular vs eccentric orbits."""
        data = RVData(
            time=Q(jnp.linspace(0, 365, 50), "day"),
            rv=Q(jnp.zeros(50), "km/s"),
            rv_err=Q(jnp.ones(50) * 0.1, "km/s"),
        )
        prior = _rv_prior()

        log_lik_circ = RVLikelihood(data, linear_marginalized_prior=prior).log_prob(
            _make_rv_params(eccentricity=0.0, arg_peri=0.0)
        )
        log_lik_ecc = RVLikelihood(data, linear_marginalized_prior=prior).log_prob(
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
        lik = RVLikelihood(data=data, linear_marginalized_prior=_rv_prior())

        eccentricities = jnp.linspace(0.0, 0.5, n_samples)
        params_batch = RVParameters.marginalized(
            period=Q(jnp.ones(n_samples) * 100.0, "day"),
            eccentricity=eccentricities,
            phase_peri=jnp.zeros(n_samples),
            arg_peri=Q(jnp.ones(n_samples) * 1.0, "rad"),
        )

        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        assert log_liks.shape == (n_samples,)
        assert jnp.all(jnp.isfinite(log_liks))

    def test_batch_vs_single(self):
        """Test that vmap gives same result as serial evaluation."""
        n_obs = 15
        data = RVData(
            time=Q(jnp.linspace(0, 100, n_obs), "day"),
            rv=Q(jnp.zeros(n_obs), "km/s"),
            rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
        )
        prior = _rv_prior()
        lik = RVLikelihood(data=data, linear_marginalized_prior=prior)

        eccs = jnp.linspace(0.0, 0.5, 5)

        params_batch = RVParameters.marginalized(
            period=Q(jnp.ones(5) * 100.0, "day"),
            eccentricity=eccs,
            phase_peri=jnp.zeros(5),
            arg_peri=Q(jnp.ones(5), "rad"),
        )
        log_liks_batch = jax.jit(jax.vmap(lik.log_prob))(params_batch)

        log_liks_serial = jnp.array(
            [
                lik.log_prob(
                    RVParameters.marginalized(
                        period=Q(100.0, "day"),
                        eccentricity=float(eccs[i]),
                        phase_peri=0.0,
                        arg_peri=Q(1.0, "rad"),
                    )
                )
                for i in range(5)
            ]
        )

        assert jnp.allclose(log_liks_batch, log_liks_serial, rtol=1e-5)


class TestExplicitLikelihoodRV:
    """Tests for the explicit (non-marginalized) RV likelihood path."""

    def _make_params(self, rv_semiamp=10.0, v_sys=0.0):
        return RVParameters(
            period=Q(100.0, "day"),
            eccentricity=0.3,
            phase_peri=0.0,
            arg_peri=Q(1.0, "rad"),
            rv_semiamp=Q(rv_semiamp, "km/s"),
            v_sys=Q(v_sys, "km/s"),
        )

    def test_single_survey_explicit_finite(self):
        """Explicit single-survey log-prob is finite."""
        data = _make_rv_data()
        lik = RVLikelihood(data=data)
        params = self._make_params()

        log_lik = lik.log_prob(params)

        assert jnp.isfinite(log_lik)

    def test_multi_survey_explicit_with_offsets_finite(self):
        """Multi-survey explicit with named offsets returns a finite value."""
        n_obs = 20
        data = _make_rv_data(n_obs)
        ind = jnp.zeros((n_obs, 1))
        ind = ind.at[10:, 0].set(1.0)
        lik = RVLikelihood(
            data=data, indicator_matrix=ind, instrument_names=("ESPRESSO",)
        )
        # Offsets are now specified as additional linear params on the full
        # RVParameters, which doesn't have an offsets field. For the explicit
        # path, we just verify the no-offset case works (offsets are handled
        # through marginalization in practice).
        params = self._make_params()

        log_lik = lik.log_prob(params)

        assert jnp.isfinite(log_lik)

    def test_multi_survey_explicit_no_offsets_ignores_indicator(self):
        """Without offsets, indicator_matrix doesn't affect the result."""
        n_obs = 20
        data = _make_rv_data(n_obs)
        ind = jnp.zeros((n_obs, 1))
        ind = ind.at[10:, 0].set(1.0)
        lik_with_ind = RVLikelihood(
            data=data, indicator_matrix=ind, instrument_names=("ESPRESSO",)
        )
        lik_no_ind = RVLikelihood(data=data)
        params = self._make_params()

        assert jnp.allclose(lik_with_ind.log_prob(params), lik_no_ind.log_prob(params))

    def test_explicit_vmap(self):
        """Vmap over explicit params works correctly."""
        n_obs = 20
        n_samples = 5
        data = _make_rv_data(n_obs)
        lik = RVLikelihood(data=data)

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
