"""Unit tests for GaiaAstrometryModel."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import GaiaAstrometryData
from harv.models.astrometry import GaiaAstrometryModel


def _make_astro_data(n_obs=50):
    return GaiaAstrometryData(
        time=Q(jnp.linspace(0, 1000, n_obs), "day"),
        al_position=Q(jnp.zeros(n_obs), "mas"),
        al_position_err=Q(jnp.ones(n_obs) * 0.1, "mas"),
        scan_angle=Q(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad"),
        parallax_factor=jnp.ones(n_obs) * 0.5,
        t_ref=Q(0.0, "day"),
    )


def _astro_prior():
    return {
        "ra0": dist.Normal(0.0, 1000.0),
        "dec0": dist.Normal(0.0, 1000.0),
        "pmra": dist.Normal(0.0, 1000.0),
        "pmdec": dist.Normal(0.0, 1000.0),
        "parallax": dist.Normal(0.0, 1000.0),
        "semi_major_axis": dist.Normal(0.0, 1000.0),
    }


def _nl_values():
    return {
        "period": Q(100.0, "day"),
        "eccentricity": 0.3,
        "phase_peri": 0.0,
        "arg_peri": Q(1.0, "rad"),
        "lon_asc_node": Q(2.0, "rad"),
        "cos_i": 0.5,
    }


class TestGaiaAstrometryModelBasic:
    def test_construction(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(data=data)
        assert model.data is data

    def test_param_names(self):
        model = GaiaAstrometryModel(data=_make_astro_data())
        assert set(model._all_nonlinear_names()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
            "lon_asc_node",
            "cos_i",
        }
        assert set(model._all_linear_names()) == {
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "semi_major_axis",
        }

    def test_obs_unit(self):
        model = GaiaAstrometryModel(data=_make_astro_data())
        assert model._obs_unit() == "mas"

    def test_design_matrix_shape(self):
        data = _make_astro_data(n_obs=15)
        model = GaiaAstrometryModel(data=data)
        X = model._base_design_matrix(_nl_values())
        assert X.shape == (15, 6)


class TestGaiaAstrometryModelMarginalized:
    def test_marginalized_is_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(data=data, linear_prior=_astro_prior())
        ll = model.log_prob(_nl_values())
        assert jnp.isfinite(ll)

    def test_marginalized_jit(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(data=data, linear_prior=_astro_prior())
        nl = _nl_values()

        @jax.jit
        def fn():
            return model.log_prob(nl)

        ll = fn()
        assert jnp.isfinite(ll)


class TestGaiaAstrometryModelSampleConditional:
    def test_sample_returns_all_linear(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(data=data, linear_prior=_astro_prior())
        key = jax.random.key(42)
        samples = model.sample_conditional_linear(_nl_values(), key)
        expected = {"ra0", "dec0", "pmra", "pmdec", "parallax", "semi_major_axis"}
        assert set(samples.keys()) == expected

    def test_sample_values_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(data=data, linear_prior=_astro_prior())
        key = jax.random.key(0)
        samples = model.sample_conditional_linear(_nl_values(), key)
        for name, val in samples.items():
            assert jnp.isfinite(val), f"{name} is not finite"
