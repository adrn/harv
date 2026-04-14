"""Unit tests for the Model class."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxt import Q

from harv.data import GaiaAstrometryData, RVData, SourceData
from harv.distributions import QuantityDistribution as QD
from harv.likelihood.composite import CompositeLikelihood
from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    MarginalizedParameters,
    RVParameters,
)
from harv.likelihood.rv import RVLikelihood
from harv.model import Model
from harv.samplers import RejectionPrior

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rv_data():
    return RVData(
        time=Q(jnp.linspace(0, 200, 20), "day"),
        rv=Q(jnp.sin(jnp.linspace(0, 4 * jnp.pi, 20)) * 10.0, "km/s"),
        rv_err=Q(jnp.ones(20) * 0.5, "km/s"),
    )


@pytest.fixture
def rv_prior():
    return RejectionPrior.default_rv(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


@pytest.fixture
def rv_model(rv_prior, rv_data):
    return Model(rv_prior, rv_data)


@pytest.fixture
def astro_data():
    n = 30
    rng = jax.random.key(42)
    k1, k2, k3 = jax.random.split(rng, 3)
    return GaiaAstrometryData(
        time=Q(jnp.linspace(0, 1000, n), "day"),
        al_position=Q(jax.random.normal(k1, (n,)) * 0.5, "mas"),
        al_position_err=Q(jnp.ones(n) * 0.1, "mas"),
        scan_angle=Q(jax.random.uniform(k2, (n,)) * 2 * jnp.pi, "rad"),
        parallax_factor=jax.random.uniform(k3, (n,), minval=-0.5, maxval=0.5),
    )


@pytest.fixture
def astro_prior():
    return RejectionPrior.default_gaia_astrometry(
        period_min=Q(50.0, "day"),
        period_max=Q(2000.0, "day"),
        sigma_a0=Q(5.0, "AU"),
        sigma_parallax=Q(10.0, "mas"),
        sigma_pos=Q(100.0, "mas"),
        sigma_vtan=Q(50.0, "km/s"),
    )


@pytest.fixture
def astro_model(astro_prior, astro_data):
    return Model(astro_prior, astro_data)


@pytest.fixture
def combined_data(rv_data, astro_data):
    return SourceData(gaia=astro_data, spec=rv_data)


@pytest.fixture
def combined_prior():
    nonlinear = {
        "period": QD(dist.LogUniform(50.0, 2000.0), "day"),
        "eccentricity": dist.Beta(0.867, 3.03),
        "phase_peri": dist.Uniform(0.0, 1.0),
        "cos_i": dist.Uniform(-1.0, 1.0),
        "arg_peri": QD(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        "lon_asc_node": QD(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
    }
    linear = {
        "ra0": QD(dist.Normal(0.0, 1000.0), "mas"),
        "dec0": QD(dist.Normal(0.0, 1000.0), "mas"),
        "pmra": QD(dist.Normal(0.0, 1000.0), "mas/yr"),
        "pmdec": QD(dist.Normal(0.0, 1000.0), "mas/yr"),
        "parallax": QD(dist.Normal(0.0, 1000.0), "mas"),
        "semi_major_axis": QD(dist.Normal(0.0, 1000.0), "mas"),
        "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
        "v_sys": QD(dist.Normal(0.0, 100.0), "km/s"),
    }
    return RejectionPrior(nonlinear_priors=nonlinear, linear_prior=linear)


@pytest.fixture
def combined_model(combined_prior, combined_data):
    return Model(combined_prior, combined_data)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestModelConstruction:
    """Tests for Model construction and computed attributes."""

    def test_rv_model_attributes(self, rv_model):
        assert rv_model.data_type == "rv"
        assert rv_model.time_unit == "d"
        assert isinstance(rv_model.likelihood, RVLikelihood)
        assert rv_model.full_cls == (RVParameters,)
        assert rv_model.t_ref is not None

    def test_astro_model_attributes(self, astro_model):
        assert astro_model.data_type == "astrometry"
        assert astro_model.time_unit == "d"
        assert isinstance(astro_model.likelihood, GaiaAstrometryLikelihood)
        assert astro_model.full_cls == (GaiaAstrometryParameters,)

    def test_combined_model_attributes(self, combined_model):
        assert combined_model.data_type == "combined"
        assert isinstance(combined_model.likelihood, CompositeLikelihood)
        assert combined_model.full_cls == (GaiaAstrometryParameters, RVParameters)

    def test_linear_param_units_rv(self, rv_model):
        units = rv_model.linear_param_units
        assert "rv_semiamp" in units
        assert "v_sys" in units

    def test_all_linear_names_rv(self, rv_model):
        names = rv_model.all_linear_names
        assert names == ("rv_semiamp", "v_sys")

    def test_all_linear_names_astro(self, astro_model):
        names = astro_model.all_linear_names
        assert "ra0" in names
        assert "parallax" in names
        assert "semi_major_axis" in names

    def test_all_linear_names_combined(self, combined_model):
        names = combined_model.all_linear_names
        # Should include both astro and RV linear params
        assert "ra0" in names
        assert "rv_semiamp" in names

    def test_unsupported_data_type_raises(self, rv_prior):
        with pytest.raises(TypeError, match="Unsupported data type"):
            Model(rv_prior, "not_data")

    def test_missing_prior_params_raises(self, rv_data):
        # A prior missing a required nonlinear param should fail
        incomplete_prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(dist.LogUniform(2.0, 1000.0), "day"),
                # Missing: eccentricity, phase_peri, arg_peri
            },
            linear_prior={
                "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
                "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
            },
        )
        with pytest.raises(ValueError, match="Prior missing required"):
            Model(incomplete_prior, rv_data)


# ---------------------------------------------------------------------------
# log_prob tests
# ---------------------------------------------------------------------------


class TestLogProb:
    """Tests for Model.log_prob."""

    def test_rv_log_prob_marginalized(self, rv_model):
        """Marginalized log_prob returns a finite scalar."""
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
        }
        lp = rv_model.log_prob(values)
        assert jnp.isfinite(lp)
        assert lp.shape == ()

    def test_rv_log_prob_full(self, rv_model):
        """Full (non-marginalized) log_prob returns a finite scalar."""
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
            "rv_semiamp": Q(10.0, "km/s"),
            "v_sys": Q(5.0, "km/s"),
        }
        lp = rv_model.log_prob(values, marginalize=False)
        assert jnp.isfinite(lp)
        assert lp.shape == ()

    def test_astro_log_prob_marginalized(self, astro_model):
        """Astrometry marginalized log_prob returns finite scalar."""
        values = {
            "period": Q(200.0, "day"),
            "eccentricity": Q(0.2, ""),
            "phase_peri": Q(0.5, ""),
            "arg_peri": Q(1.0, "rad"),
            "cos_i": Q(0.5, ""),
            "lon_asc_node": Q(2.0, "rad"),
            # parallax is explicitly sampled (HalfNormal prior)
            "parallax": Q(5.0, "mas"),
            # semi_major_axis is period-dependent (callable prior)
            "semi_major_axis": Q(0.5, "mas"),
        }
        lp = astro_model.log_prob(values)
        assert jnp.isfinite(lp)

    def test_combined_log_prob_marginalized(self, combined_model):
        """Combined model marginalized log_prob returns finite scalar."""
        values = {
            "period": Q(200.0, "day"),
            "eccentricity": Q(0.2, ""),
            "phase_peri": Q(0.5, ""),
            "arg_peri": Q(1.0, "rad"),
            "cos_i": Q(0.5, ""),
            "lon_asc_node": Q(2.0, "rad"),
            "parallax": Q(5.0, "mas"),
            "semi_major_axis": Q(0.5, "mas"),
        }
        lp = combined_model.log_prob(values)
        assert jnp.isfinite(lp)

    def test_missing_param_raises(self, rv_model):
        """Missing a required nonlinear parameter should raise."""
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            # Missing phase_peri and arg_peri
        }
        with pytest.raises(ValueError, match="Missing required nonlinear"):
            rv_model.log_prob(values)

    def test_missing_linear_param_full_raises(self, rv_model):
        """Missing linear param with marginalize=False should raise."""
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
            "rv_semiamp": Q(10.0, "km/s"),
            # Missing v_sys
        }
        with pytest.raises(ValueError, match="Missing required linear"):
            rv_model.log_prob(values, marginalize=False)

    def test_different_params_give_different_values(self, rv_model):
        """Different parameter values should produce different log_prob."""
        values1 = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
        }
        values2 = {
            "period": Q(50.0, "day"),
            "eccentricity": Q(0.1, ""),
            "phase_peri": Q(0.8, ""),
            "arg_peri": Q(0.5, "rad"),
        }
        lp1 = rv_model.log_prob(values1)
        lp2 = rv_model.log_prob(values2)
        assert not jnp.allclose(lp1, lp2)


# ---------------------------------------------------------------------------
# build_params tests
# ---------------------------------------------------------------------------


class TestBuildParams:
    """Tests for Model.build_params."""

    def test_rv_build_params_marginalized(self, rv_model):
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
        }
        params = rv_model.build_params(values)
        assert isinstance(params, MarginalizedParameters)
        assert params.source_cls is RVParameters
        assert len(params.marginalized_names) > 0

    def test_rv_build_params_full(self, rv_model):
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
            "rv_semiamp": Q(10.0, "km/s"),
            "v_sys": Q(5.0, "km/s"),
        }
        params = rv_model.build_params(values, marginalize=False)
        assert isinstance(params, MarginalizedParameters)
        assert params.marginalized_names == ()

    def test_combined_build_params_returns_dict(self, combined_model):
        values = {
            "period": Q(200.0, "day"),
            "eccentricity": Q(0.2, ""),
            "phase_peri": Q(0.5, ""),
            "arg_peri": Q(1.0, "rad"),
            "cos_i": Q(0.5, ""),
            "lon_asc_node": Q(2.0, "rad"),
            "parallax": Q(5.0, "mas"),
            "semi_major_axis": Q(0.5, "mas"),
        }
        params = combined_model.build_params(values)
        assert isinstance(params, dict)
        assert "astro" in params
        assert "rv" in params
        assert isinstance(params["astro"], MarginalizedParameters)
        assert isinstance(params["rv"], MarginalizedParameters)


# ---------------------------------------------------------------------------
# sample_conditional_linear tests
# ---------------------------------------------------------------------------


class TestSampleConditionalLinear:
    """Tests for Model.sample_conditional_linear."""

    def test_rv_sample_linear(self, rv_model):
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
        }
        key = jax.random.key(0)
        linear = rv_model.sample_conditional_linear(values, key)
        assert "rv_semiamp" in linear
        assert "v_sys" in linear


# ---------------------------------------------------------------------------
# JAX compatibility tests
# ---------------------------------------------------------------------------


class TestJAXCompat:
    """Tests for JIT and vmap compatibility."""

    def test_jit_log_prob(self, rv_model):
        """log_prob works under eqx.filter_jit."""
        values = {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
        }
        jit_lp = eqx.filter_jit(rv_model.log_prob)(values)
        eager_lp = rv_model.log_prob(values)
        assert jnp.allclose(jit_lp, eager_lp)

    def test_vmap_log_prob(self, rv_model):
        """log_prob works under jax.vmap over parameter batches."""
        batch_values = {
            "period": Q(jnp.array([80.0, 100.0, 120.0]), "day"),
            "eccentricity": Q(jnp.array([0.1, 0.3, 0.5]), ""),
            "phase_peri": Q(jnp.array([0.0, 0.1, 0.2]), ""),
            "arg_peri": Q(jnp.array([1.0, 1.5, 2.0]), "rad"),
        }
        lps = jax.vmap(rv_model.log_prob)(batch_values)
        assert lps.shape == (3,)
        assert jnp.all(jnp.isfinite(lps))
