"""Unit tests for Samples container."""

import jax
import jax.numpy as jnp
import pytest
from unxt import Quantity

from harv.likelihood.params import (
    GaiaAstrometryParameters,
    RVParameters,
)
from harv.samplers.samples import Samples


def _make_astro_samples() -> Samples:
    """Helper: astrometry Samples with 3 draws."""
    nonlinear = {
        "period": Quantity(jnp.array([10.0, 100.0, 1000.0]), "day"),
        "eccentricity": Quantity(jnp.array([0.1, 0.2, 0.3]), ""),
        "phase_peri": Quantity(jnp.array([0.0, 0.25, 0.5]), ""),
        "cos_i": Quantity(jnp.array([0.5, 0.6, 0.7]), ""),
        "arg_peri": Quantity(jnp.array([0.5, 1.0, 1.5]), "rad"),
        "lon_asc_node": Quantity(jnp.array([1.0, 2.0, 3.0]), "rad"),
    }
    linear = {
        "ra0": Quantity(jnp.array([10.0, 11.0, 12.0]), "mas"),
        "dec0": Quantity(jnp.array([20.0, 21.0, 22.0]), "mas"),
        "pmra": Quantity(jnp.array([5.0, 6.0, 7.0]), "mas/yr"),
        "pmdec": Quantity(jnp.array([-3.0, -2.0, -1.0]), "mas/yr"),
        "parallax": Quantity(jnp.array([10.0, 11.0, 12.0]), "mas"),
        "semi_major_axis": Quantity(jnp.array([1.0, 1.1, 1.2]), "mas"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        orbit_cls=GaiaAstrometryParameters,
        full_cls=(GaiaAstrometryParameters,),
        data_type="astrometry",
        metadata={"t_ref": 0.0},
    )


class TestSamplesCreation:
    """Tests for creating Samples objects."""

    def test_basic_creation(self):
        """Test creating a basic Samples object."""
        samples = _make_astro_samples()

        assert samples.n_samples == 3
        assert samples.data_type == "astrometry"

    def test_n_samples_property(self):
        """Test that n_samples returns correct value."""
        nonlinear = {
            "period": Quantity(jnp.array([100.0, 200.0]), "day"),
            "eccentricity": Quantity(jnp.array([0.1, 0.2]), ""),
            "phase_peri": Quantity(jnp.array([0.0, 0.5]), ""),
            "arg_peri": Quantity(jnp.array([1.0, 2.0]), "rad"),
        }
        linear = {
            "K": Quantity(jnp.array([1.0, 3.0]), "km/s"),
            "v0": Quantity(jnp.array([2.0, 4.0]), "km/s"),
        }

        samples = Samples(
            nonlinear=nonlinear,
            linear=linear,
            orbit_cls=RVParameters,
            full_cls=(RVParameters,),
            data_type="rv",
            metadata={},
        )

        assert samples.n_samples == 2
        assert len(samples) == 2


class TestSamplesAccess:
    """Tests for accessing parameters from Samples."""

    @pytest.fixture
    def astrometry_samples(self):
        """Create sample astrometry Samples object."""
        return _make_astro_samples()

    def test_getitem_nonlinear(self, astrometry_samples):
        """Test accessing nonlinear parameters returns Quantity."""
        ecc = astrometry_samples["eccentricity"]
        assert isinstance(ecc, Quantity)
        assert ecc.shape == (3,)
        assert jnp.allclose(ecc.value, jnp.array([0.1, 0.2, 0.3]))

    def test_getitem_linear(self, astrometry_samples):
        """Test accessing linear parameters."""
        parallax = astrometry_samples["parallax"]
        assert isinstance(parallax, Quantity)
        assert parallax.shape == (3,)

    def test_getitem_period_and_log_period(self, astrometry_samples):
        """Test period (stored) and log_period (derived)."""
        period = astrometry_samples["period"]
        assert isinstance(period, Quantity)
        assert jnp.allclose(period.value, jnp.array([10.0, 100.0, 1000.0]))

        log_period = astrometry_samples["log_period"]
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(log_period, expected)

    def test_getitem_invalid_key(self, astrometry_samples):
        """Test that invalid key raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            _ = astrometry_samples["invalid_param"]

    def test_keys_method(self, astrometry_samples):
        """Test keys() returns all parameter names."""
        keys = astrometry_samples.keys()
        assert isinstance(keys, list)
        assert "period" in keys
        assert "log_period" in keys
        assert "eccentricity" in keys
        assert "parallax" in keys
        # 6 nonlinear + 6 linear + 3 derived (log_period, t_peri, inclination) = 15
        assert len(keys) == 15

    def test_unit_conversion_angles(self):
        """Test that angles are returned with correct units."""
        nonlinear = {
            "period": Quantity(jnp.array([100.0]), "day"),
            "eccentricity": Quantity(jnp.array([0.1]), ""),
            "phase_peri": Quantity(jnp.array([0.0]), ""),
            "arg_peri": Quantity(jnp.array([1.57]), "rad"),
            "lon_asc_node": Quantity(jnp.array([3.14]), "rad"),
        }
        linear = {
            "K": Quantity(jnp.array([1.0]), "km/s"),
            "v0": Quantity(jnp.array([2.0]), "km/s"),
        }

        samples = Samples(
            nonlinear=nonlinear,
            linear=linear,
            orbit_cls=RVParameters,
            full_cls=(RVParameters,),
            data_type="rv",
            metadata={},
        )

        arg_peri = samples["arg_peri"]
        assert isinstance(arg_peri, Quantity)
        assert str(arg_peri.unit) == "rad"


class TestSamplesRepr:
    """Tests for Samples string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        nonlinear = {
            "period": Quantity(jnp.array([100.0, 200.0]), "day"),
            "eccentricity": Quantity(jnp.array([0.1, 0.2]), ""),
            "phase_peri": Quantity(jnp.array([0.0, 0.5]), ""),
            "arg_peri": Quantity(jnp.array([1.0, 2.0]), "rad"),
        }
        linear = {
            "K": Quantity(jnp.array([1.0, 3.0]), "km/s"),
            "v0": Quantity(jnp.array([2.0, 4.0]), "km/s"),
        }

        samples = Samples(
            nonlinear=nonlinear,
            linear=linear,
            orbit_cls=RVParameters,
            full_cls=(RVParameters,),
            data_type="rv",
            metadata={},
        )

        repr_str = repr(samples)
        assert "n_samples=2" in repr_str
        assert "data_type='rv'" in repr_str
        assert "parameters=" in repr_str


class TestSamplesJAX:
    """Tests for JAX compatibility."""

    def test_pytree_flatten_unflatten(self):
        """Test that Samples is a valid JAX pytree."""
        samples = _make_astro_samples()
        leaves, treedef = jax.tree_util.tree_flatten(samples)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed.n_samples == samples.n_samples
        assert jnp.allclose(
            reconstructed["eccentricity"].value, samples["eccentricity"].value
        )
