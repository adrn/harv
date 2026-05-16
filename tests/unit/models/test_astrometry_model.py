"""Unit tests for GaiaAstrometryModel."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

from harv.data import GaiaAstrometryData
from harv.kepler.orbits import thiele_innes_ABFG
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry


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
        model = GaiaAstrometryModel()
        assert model.parameterization is not None

    def test_param_names(self):
        model = GaiaAstrometryModel()
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
        model = GaiaAstrometryModel()
        assert model._obs_unit(_make_astro_data()) == "mas"

    def test_design_matrix_shape(self):
        data = _make_astro_data(n_obs=15)
        model = GaiaAstrometryModel()
        X = model._base_design_matrix(_nl_values(), data)
        assert X.shape == (15, 6)


class TestGaiaAstrometryModelMarginalized:
    def test_marginalized_is_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        ll = model.log_prob(_nl_values(), data, linear_prior=_astro_prior())
        assert jnp.isfinite(ll)

    def test_marginalized_jit(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        nl = _nl_values()

        @jax.jit
        def fn():
            return model.log_prob(nl, data, linear_prior=_astro_prior())

        ll = fn()
        assert jnp.isfinite(ll)


class TestGaiaAstrometryModelSampleConditional:
    def test_sample_returns_all_linear(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        key = jax.random.key(42)
        samples = model.sample_conditional_linear(
            _nl_values(), key, data, linear_prior=_astro_prior()
        )
        expected = {"ra0", "dec0", "pmra", "pmdec", "parallax", "semi_major_axis"}
        assert set(samples.keys()) == expected

    def test_sample_values_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        key = jax.random.key(0)
        samples = model.sample_conditional_linear(
            _nl_values(), key, data, linear_prior=_astro_prior()
        )
        for name, val in samples.items():
            assert jnp.isfinite(val), f"{name} is not finite"


class TestGaiaAstrometryModelThieleInnes:
    """Tests for GaiaAstrometryModel with ThieleInnesGaiaAstrometry."""

    def _ti_prior(self):
        return {
            "ra0": dist.Normal(0.0, 1000.0),
            "dec0": dist.Normal(0.0, 1000.0),
            "pmra": dist.Normal(0.0, 1000.0),
            "pmdec": dist.Normal(0.0, 1000.0),
            "parallax": dist.Normal(0.0, 1000.0),
            "ti_A": dist.Normal(0.0, 10.0),
            "ti_B": dist.Normal(0.0, 10.0),
            "ti_F": dist.Normal(0.0, 10.0),
            "ti_G": dist.Normal(0.0, 10.0),
        }

    def _ti_nl_values(self):
        return {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
        }

    def test_construction(self):
        p = ThieleInnesGaiaAstrometry(a_floor=0.01)
        model = GaiaAstrometryModel(parameterization=p)
        assert isinstance(model.parameterization, ThieleInnesGaiaAstrometry)

    def test_param_names(self):
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(a_floor=0.01)
        )
        assert set(model._all_nonlinear_names()) == {
            "period",
            "eccentricity",
            "phase_peri",
        }
        assert set(model._all_linear_names()) == {
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "ti_A",
            "ti_B",
            "ti_F",
            "ti_G",
        }

    def test_design_matrix_shape(self):
        data = _make_astro_data(n_obs=20)
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(a_floor=0.01)
        )
        X = model._base_design_matrix(self._ti_nl_values(), data=data)
        assert X.shape == (20, 9)

    def test_linear_param_units(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(a_floor=0.01)
        )
        units = model._linear_param_units(data)
        assert units["pmra"] == "mas/yr"
        assert units["ti_A"] == "mas"
        assert units["ti_B"] == "mas"
        assert units["ti_F"] == "mas"
        assert units["ti_G"] == "mas"

    def test_log_prob_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )
        ll = model.log_prob(
            self._ti_nl_values(), data=data, linear_prior=self._ti_prior()
        )
        assert jnp.isfinite(ll)

    def test_log_prob_jit(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )
        nl = self._ti_nl_values()

        @jax.jit
        def fn():
            return model.log_prob(nl, data=data, linear_prior=self._ti_prior())

        ll = fn()
        assert jnp.isfinite(ll)

    def test_orbit_contribution_matches_standard(self):
        """TI and Standard models produce identical orbit contributions."""
        a0 = 1.5
        arg_peri, lon_asc_node, cos_i = 0.8, 1.1, 0.6
        A, B, F, G = thiele_innes_ABFG(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )

        data = _make_astro_data(n_obs=20)

        nl_std = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(arg_peri, "rad"),
            "lon_asc_node": Q(lon_asc_node, "rad"),
            "cos_i": cos_i,
        }
        nl_ti = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
        }

        model_std = GaiaAstrometryModel()
        model_ti = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )

        X_std = model_std._base_design_matrix(nl_std, data=data)
        X_ti = model_ti._base_design_matrix(nl_ti, data=data)

        # Standard orbit column (col 5): scaled by a0 externally
        orbit_std = X_std[:, 5] * a0

        # TI orbit: X_ti cols 5-8 @ [A, B, F, G] * a0
        orbit_ti = (
            X_ti[:, 5] * A * a0
            + X_ti[:, 6] * B * a0
            + X_ti[:, 7] * F * a0
            + X_ti[:, 8] * G * a0
        )

        assert jnp.allclose(orbit_std, orbit_ti, atol=1e-5)
