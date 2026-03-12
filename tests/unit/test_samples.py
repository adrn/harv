"""Unit tests for Samples container."""

import jax.numpy as jnp
import pytest
from unxt import Quantity

from harv.samplers.samples import Samples


class TestSamplesCreation:
    """Tests for creating Samples objects."""

    def test_basic_creation(self):
        """Test creating a basic Samples object."""
        nonlinear = {
            "log_period": jnp.array([1.0, 2.0, 3.0]),
            "eccentricity": jnp.array([0.1, 0.2, 0.3]),
            "phase_peri": jnp.array([0.0, 0.5, 1.0]),
            "cos_i": jnp.array([0.5, 0.6, 0.7]),
            "lon_asc_node": jnp.array([1.0, 2.0, 3.0]),
        }
        linear = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _linear_param_names=(
                "alpha_0",
                "delta_0",
                "mu_alpha",
                "mu_delta",
                "parallax",
                "semimajor_axis",
            ),
            _data_type="astrometry",
            _metadata={"t_ref": 0.0},
        )

        assert samples.n_samples == 3
        assert samples.data_type == "astrometry"

    def test_n_samples_property(self):
        """Test that n_samples returns correct value."""
        nonlinear = {
            "log_period": jnp.array([1.0, 2.0]),
            "eccentricity": jnp.array([0.1, 0.2]),
            "phase_peri": jnp.array([0.0, 0.5]),
        }
        linear = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _linear_param_names=("K", "v0"),
            _data_type="rv",
            _metadata={},
        )

        assert samples.n_samples == 2
        assert len(samples) == 2


class TestSamplesAccess:
    """Tests for accessing parameters from Samples."""

    @pytest.fixture
    def astrometry_samples(self):
        """Create sample astrometry Samples object."""
        nonlinear = {
            "log_period": jnp.array([1.0, 2.0, 3.0]),
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
            _linear_param_names=(
                "alpha_0",
                "delta_0",
                "mu_alpha",
                "mu_delta",
                "parallax",
                "semimajor_axis",
            ),
            _data_type="astrometry",
            _metadata={"t_ref": 0.0},
        )

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

    def test_getitem_derived_period(self, astrometry_samples):
        """Test accessing derived period."""
        period = astrometry_samples["period"]
        assert isinstance(period, Quantity)
        # log_period = [1, 2, 3] -> period = [10, 100, 1000] days
        expected = jnp.array([10.0, 100.0, 1000.0])
        assert jnp.allclose(period.value, expected)

    def test_getitem_invalid_key(self, astrometry_samples):
        """Test that invalid key raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            _ = astrometry_samples["invalid_param"]

    def test_keys_method(self, astrometry_samples):
        """Test keys() returns all parameter names."""
        keys = astrometry_samples.keys()
        assert isinstance(keys, list)
        assert "log_period" in keys
        assert "eccentricity" in keys
        assert "parallax" in keys
        # 6 nonlinear + 6 linear + 3 derived (period, t_peri, inclination) = 15
        assert len(keys) == 15

    def test_unit_conversion_angles(self):
        """Test that angles are returned with correct units."""
        nonlinear = {
            "log_period": jnp.array([1.0]),
            "eccentricity": jnp.array([0.1]),
            "phase_peri": jnp.array([0.0]),
            "arg_peri": jnp.array([1.57]),  # ~π/2 radians
            "lon_asc_node": jnp.array([3.14]),  # ~π radians
        }
        linear = jnp.array([[1.0, 2.0]])

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _linear_param_names=("K", "v0"),
            _data_type="rv",
            _metadata={},
        )

        arg_peri = samples["arg_peri"]
        assert isinstance(arg_peri, Quantity)
        assert arg_peri.unit == "rad"


class TestSamplesRepr:
    """Tests for Samples string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        nonlinear = {
            "log_period": jnp.array([1.0, 2.0]),
            "eccentricity": jnp.array([0.1, 0.2]),
            "phase_peri": jnp.array([0.0, 0.5]),
        }
        linear = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        samples = Samples(
            _nonlinear=nonlinear,
            _linear=linear,
            _linear_param_names=("K", "v0"),
            _data_type="rv",
            _metadata={},
        )

        repr_str = repr(samples)
        assert "n_samples=2" in repr_str
        assert "data_type='rv'" in repr_str
        assert "parameters=" in repr_str
