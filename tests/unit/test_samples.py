"""Unit tests for Samples container."""

import jax
import jax.numpy as jnp
import pytest
from unxt import Quantity

from harv.likelihood._params import (
    GaiaAstrometryFullParameters,
    GaiaAstrometryOrbitParameters,
    RVFullParameters,
    RVOrbitParameters,
)
from harv.samplers.samples import Samples


def _make_astro_samples():
    """Helper: astrometry Samples with 3 draws."""
    nonlinear = {
        "period": jnp.array([10.0, 100.0, 1000.0]),  # days
        "eccentricity": jnp.array([0.1, 0.2, 0.3]),
        "phase_peri": jnp.array([0.0, 0.25, 0.5]),
        "cos_i": jnp.array([0.5, 0.6, 0.7]),
        "arg_peri": jnp.array([0.5, 1.0, 1.5]),
        "lon_asc_node": jnp.array([1.0, 2.0, 3.0]),
    }
    linear = jnp.array(
        [
            [10.0, 20.0, 5.0, -3.0, 10.0, 1.0],
            [11.0, 21.0, 6.0, -2.0, 11.0, 1.1],
            [12.0, 22.0, 7.0, -1.0, 12.0, 1.2],
        ]
    )
    return Samples(
        _nonlinear=nonlinear,
        _linear=linear,
        _orbit_cls=GaiaAstrometryOrbitParameters,
        _full_cls=(GaiaAstrometryFullParameters,),
        _linear_param_units=("mas", "mas", "mas/yr", "mas/yr", "mas", "mas"),
        _time_unit="day",
        _metadata={"t_ref": 0.0},
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
            "period": jnp.array([100.0, 200.0]),
            "eccentricity": jnp.array([0.1, 0.2]),
            "phase_peri": jnp.array([0.0, 0.5]),
            "arg_peri": jnp.array([1.0, 2.0]),
        }
        linear = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _orbit_cls=RVOrbitParameters,
            _full_cls=(RVFullParameters,),
            _linear_param_units=("km/s", "km/s"),
            _time_unit="day",
            _metadata={},
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
        """Test accessing nonlinear parameters."""
        ecc = astrometry_samples["eccentricity"]
        assert isinstance(ecc, jnp.ndarray)
        assert ecc.shape == (3,)
        assert jnp.allclose(ecc, jnp.array([0.1, 0.2, 0.3]))

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
            "period": jnp.array([100.0]),
            "eccentricity": jnp.array([0.1]),
            "phase_peri": jnp.array([0.0]),
            "arg_peri": jnp.array([1.57]),  # ~π/2 radians
            "lon_asc_node": jnp.array([3.14]),  # ~π radians
        }
        linear = jnp.array([[1.0, 2.0]])

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _orbit_cls=RVOrbitParameters,
            _full_cls=(RVFullParameters,),
            _linear_param_units=("km/s", "km/s"),
            _time_unit="day",
            _metadata={},
        )

        arg_peri = samples["arg_peri"]
        assert isinstance(arg_peri, Quantity)
        assert str(arg_peri.unit) == "rad"


class TestSamplesRepr:
    """Tests for Samples string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        nonlinear = {
            "period": jnp.array([100.0, 200.0]),
            "eccentricity": jnp.array([0.1, 0.2]),
            "phase_peri": jnp.array([0.0, 0.5]),
            "arg_peri": jnp.array([1.0, 2.0]),
        }
        linear = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _orbit_cls=RVOrbitParameters,
            _full_cls=(RVFullParameters,),
            _linear_param_units=("km/s", "km/s"),
            _time_unit="day",
            _metadata={},
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
            reconstructed["eccentricity"], samples["eccentricity"]
        )
