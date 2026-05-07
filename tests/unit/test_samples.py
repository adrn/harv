"""Unit tests for Samples container."""

import jax
import jax.numpy as jnp
import pytest
from unxt import Q

from harv.kepler.orbits import astrometric_orbit_at_times, rv_at_times
from harv.samplers.samples import Samples


def _make_astro_samples() -> Samples:
    """Helper: astrometry Samples with 3 draws."""
    nonlinear = {
        "period": Q(jnp.array([10.0, 100.0, 1000.0]), "day"),
        "eccentricity": Q(jnp.array([0.1, 0.2, 0.3]), ""),
        "phase_peri": Q(jnp.array([0.0, 0.25, 0.5]), ""),
        "cos_i": Q(jnp.array([0.5, 0.6, 0.7]), ""),
        "arg_peri": Q(jnp.array([0.5, 1.0, 1.5]), "rad"),
        "lon_asc_node": Q(jnp.array([1.0, 2.0, 3.0]), "rad"),
    }
    linear = {
        "ra0": Q(jnp.array([10.0, 11.0, 12.0]), "mas"),
        "dec0": Q(jnp.array([20.0, 21.0, 22.0]), "mas"),
        "pmra": Q(jnp.array([5.0, 6.0, 7.0]), "mas/yr"),
        "pmdec": Q(jnp.array([-3.0, -2.0, -1.0]), "mas/yr"),
        "parallax": Q(jnp.array([10.0, 11.0, 12.0]), "mas"),
        "semi_major_axis": Q(jnp.array([1.0, 1.1, 1.2]), "mas"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="gaia_astro",
        metadata={"t_ref": 0.0},
    )


class TestSamplesCreation:
    """Tests for creating Samples objects."""

    def test_basic_creation(self):
        """Test creating a basic Samples object."""
        samples = _make_astro_samples()

        assert samples.n_samples == 3
        assert samples.data_type == "gaia_astro"

    def test_n_samples_property(self):
        """Test that n_samples returns correct value."""
        nonlinear = {
            "period": Q(jnp.array([100.0, 200.0]), "day"),
            "eccentricity": Q(jnp.array([0.1, 0.2]), ""),
            "phase_peri": Q(jnp.array([0.0, 0.5]), ""),
            "arg_peri": Q(jnp.array([1.0, 2.0]), "rad"),
        }
        linear = {
            "rv_semiamp": Q(jnp.array([1.0, 3.0]), "km/s"),
            "v_sys": Q(jnp.array([2.0, 4.0]), "km/s"),
        }

        samples = Samples(
            nonlinear=nonlinear,
            linear=linear,
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
        """Test accessing nonlinear parameters returns Q."""
        ecc = astrometry_samples["eccentricity"]
        assert isinstance(ecc, Q)
        assert ecc.shape == (3,)
        assert jnp.allclose(ecc.value, jnp.array([0.1, 0.2, 0.3]))

    def test_getitem_linear(self, astrometry_samples):
        """Test accessing linear parameters."""
        parallax = astrometry_samples["parallax"]
        assert isinstance(parallax, Q)
        assert parallax.shape == (3,)

    def test_getitem_period_and_log_period(self, astrometry_samples):
        """Test period (stored) and log_period (derived)."""
        period = astrometry_samples["period"]
        assert isinstance(period, Q)
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
            "period": Q(jnp.array([100.0]), "day"),
            "eccentricity": Q(jnp.array([0.1]), ""),
            "phase_peri": Q(jnp.array([0.0]), ""),
            "arg_peri": Q(jnp.array([1.57]), "rad"),
            "lon_asc_node": Q(jnp.array([3.14]), "rad"),
        }
        linear = {
            "rv_semiamp": Q(jnp.array([1.0]), "km/s"),
            "v_sys": Q(jnp.array([2.0]), "km/s"),
        }

        samples = Samples(
            nonlinear=nonlinear,
            linear=linear,
            data_type="rv",
            metadata={},
        )

        arg_peri = samples["arg_peri"]
        assert isinstance(arg_peri, Q)
        assert str(arg_peri.unit) == "rad"


class TestSamplesRepr:
    """Tests for Samples string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        nonlinear = {
            "period": Q(jnp.array([100.0, 200.0]), "day"),
            "eccentricity": Q(jnp.array([0.1, 0.2]), ""),
            "phase_peri": Q(jnp.array([0.0, 0.5]), ""),
            "arg_peri": Q(jnp.array([1.0, 2.0]), "rad"),
        }
        linear = {
            "rv_semiamp": Q(jnp.array([1.0, 3.0]), "km/s"),
            "v_sys": Q(jnp.array([2.0, 4.0]), "km/s"),
        }

        samples = Samples(
            nonlinear=nonlinear,
            linear=linear,
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


class TestSamplesToArviz:
    """Tests for Samples.to_arviz chain-reshape behaviour."""

    def test_single_chain_shape(self):
        """Without num_chains metadata, to_arviz produces (1, n_samples) arrays."""
        pytest.importorskip("arviz")
        samples = Samples(
            nonlinear={
                "period": Q(jnp.array([100.0, 101.0, 99.5, 100.5]), "day"),
                "eccentricity": Q(jnp.array([0.1, 0.15, 0.12, 0.11]), ""),
                "phase_peri": Q(jnp.array([0.3, 0.31, 0.29, 0.28]), ""),
                "arg_peri": Q(jnp.array([1.0, 1.1, 0.9, 1.2]), "rad"),
            },
            linear={
                "rv_semiamp": Q(jnp.array([10.0, 11.0, 9.5, 10.5]), "km/s"),
                "v_sys": Q(jnp.array([5.0, 5.1, 4.9, 5.2]), "km/s"),
            },
            data_type="rv",
            metadata={},
        )
        idata = samples.to_arviz(["period", "eccentricity"])
        period_arr = idata.posterior["period"].values
        assert period_arr.shape == (1, 4)

    def test_multi_chain_shape(self):
        """With num_chains=2 in metadata, to_arviz produces (2, n_per_chain) arrays."""
        pytest.importorskip("arviz")
        # 6 samples = 2 chains x 3 draws
        samples = Samples(
            nonlinear={
                "period": Q(jnp.arange(6, dtype=float) + 100.0, "day"),
                "eccentricity": Q(jnp.full(6, 0.1), ""),
                "phase_peri": Q(jnp.full(6, 0.3), ""),
                "arg_peri": Q(jnp.full(6, 1.0), "rad"),
            },
            linear={
                "rv_semiamp": Q(jnp.full(6, 10.0), "km/s"),
                "v_sys": Q(jnp.full(6, 0.0), "km/s"),
            },
            data_type="rv",
            metadata={"num_chains": 2},
        )
        idata = samples.to_arviz(["period"])
        period_arr = idata.posterior["period"].values
        assert period_arr.shape == (2, 3)

    def test_indivisible_falls_back_to_one_chain(self):
        """When n_samples % num_chains != 0, fall back to a single chain."""
        pytest.importorskip("arviz")
        samples = Samples(
            nonlinear={
                "period": Q(jnp.array([100.0, 101.0, 99.5]), "day"),
                "eccentricity": Q(jnp.full(3, 0.1), ""),
                "phase_peri": Q(jnp.full(3, 0.3), ""),
                "arg_peri": Q(jnp.full(3, 1.0), "rad"),
            },
            linear={
                "rv_semiamp": Q(jnp.full(3, 10.0), "km/s"),
                "v_sys": Q(jnp.full(3, 0.0), "km/s"),
            },
            data_type="rv",
            metadata={"num_chains": 2},  # 3 % 2 != 0
        )
        with pytest.warns(UserWarning, match="not divisible by num_chains"):
            idata = samples.to_arviz(["period"])
        period_arr = idata.posterior["period"].values
        assert period_arr.shape == (1, 3)


def _make_rv_samples_with_signs(K_values: list[float]) -> Samples:
    """RV-only Samples with controllable rv_semiamp signs."""
    n = len(K_values)
    return Samples(
        nonlinear={
            "period": Q(jnp.full(n, 100.0), "day"),
            "eccentricity": Q(jnp.full(n, 0.3), ""),
            "phase_peri": Q(jnp.full(n, 0.25), ""),
            "arg_peri": Q(jnp.linspace(0.5, 2.5, n), "rad"),
        },
        linear={
            "rv_semiamp": Q(jnp.asarray(K_values), "km/s"),
            "v_sys": Q(jnp.full(n, 5.0), "km/s"),
        },
        data_type="rv",
        metadata={"t_ref": Q(0.0, "day")},
    )


def _make_astro_samples_with_signs(a_values: list[float]) -> Samples:
    """Astrometry-only Samples with controllable semi_major_axis signs."""
    n = len(a_values)
    return Samples(
        nonlinear={
            "period": Q(jnp.full(n, 300.0), "day"),
            "eccentricity": Q(jnp.full(n, 0.3), ""),
            "phase_peri": Q(jnp.full(n, 0.1), ""),
            "arg_peri": Q(jnp.linspace(0.5, 2.5, n), "rad"),
            "cos_i": Q(jnp.full(n, 0.4), ""),
            "lon_asc_node": Q(jnp.full(n, 0.8), "rad"),
        },
        linear={
            "ra0": Q(jnp.zeros(n), "mas"),
            "dec0": Q(jnp.zeros(n), "mas"),
            "pmra": Q(jnp.zeros(n), "mas/yr"),
            "pmdec": Q(jnp.zeros(n), "mas/yr"),
            "parallax": Q(jnp.full(n, 5.0), "mas"),
            "semi_major_axis": Q(jnp.asarray(a_values), "mas"),
        },
        data_type="gaia_astro",
        metadata={"t_ref": Q(0.0, "day")},
    )


def _make_joint_samples_with_signs(
    K_values: list[float], a_values: list[float]
) -> Samples:
    """Joint RV+astrometry Samples with controllable K and a signs."""
    assert len(K_values) == len(a_values)
    n = len(K_values)
    return Samples(
        nonlinear={
            "period": Q(jnp.full(n, 300.0), "day"),
            "eccentricity": Q(jnp.full(n, 0.3), ""),
            "phase_peri": Q(jnp.full(n, 0.1), ""),
            "arg_peri": Q(jnp.linspace(0.5, 2.5, n), "rad"),
            "cos_i": Q(jnp.full(n, 0.4), ""),
            "lon_asc_node": Q(jnp.full(n, 0.8), "rad"),
        },
        linear={
            "rv_semiamp": Q(jnp.asarray(K_values), "km/s"),
            "v_sys": Q(jnp.zeros(n), "km/s"),
            "ra0": Q(jnp.zeros(n), "mas"),
            "dec0": Q(jnp.zeros(n), "mas"),
            "pmra": Q(jnp.zeros(n), "mas/yr"),
            "pmdec": Q(jnp.zeros(n), "mas/yr"),
            "parallax": Q(jnp.full(n, 5.0), "mas"),
            "semi_major_axis": Q(jnp.asarray(a_values), "mas"),
        },
        data_type="joint",
        metadata={"t_ref": Q(0.0, "day")},
    )


class TestSamplesWrapAngles:
    """Tests for Samples.wrap_angles."""

    def test_rv_only_flips_negative_K(self):
        """Negative rv_semiamp entries flip to positive; arg_peri shifts by pi."""
        samples = _make_rv_samples_with_signs([-10.0, 12.0, -8.0, 5.0])
        old_omega = samples["arg_peri"].value
        wrapped = samples.wrap_angles()
        new_K = wrapped["rv_semiamp"].value
        new_omega = wrapped["arg_peri"].value
        assert bool((new_K >= 0).all())
        # Flipped indices: 0, 2.  Positive entries unchanged.
        assert jnp.allclose(new_K, jnp.array([10.0, 12.0, 8.0, 5.0]))
        expected_omega = jnp.where(
            jnp.array([True, False, True, False]),
            jnp.mod(old_omega + jnp.pi, 2 * jnp.pi),
            old_omega,
        )
        assert jnp.allclose(new_omega, expected_omega)

    def test_astrometry_only_flips_negative_a(self):
        """When no rv_semiamp present, fall back to semi_major_axis trigger."""
        samples = _make_astro_samples_with_signs([-1.5, 2.0, -0.5])
        old_omega = samples["arg_peri"].value
        wrapped = samples.wrap_angles()
        new_a = wrapped["semi_major_axis"].value
        new_omega = wrapped["arg_peri"].value
        assert bool((new_a >= 0).all())
        assert jnp.allclose(new_a, jnp.array([1.5, 2.0, 0.5]))
        expected_omega = jnp.where(
            jnp.array([True, False, True]),
            jnp.mod(old_omega + jnp.pi, 2 * jnp.pi),
            old_omega,
        )
        assert jnp.allclose(new_omega, expected_omega)

    def test_joint_flips_both_K_and_a_together(self):
        """In a joint fit, K<0 implies a flip on both K and a."""
        # K and a both negative for first two, both positive for last two.
        samples = _make_joint_samples_with_signs(
            K_values=[-10.0, -8.0, 5.0, 7.0],
            a_values=[-2.0, -1.5, 1.0, 1.2],
        )
        wrapped = samples.wrap_angles()
        assert jnp.allclose(
            wrapped["rv_semiamp"].value, jnp.array([10.0, 8.0, 5.0, 7.0])
        )
        assert jnp.allclose(
            wrapped["semi_major_axis"].value, jnp.array([2.0, 1.5, 1.0, 1.2])
        )

    def test_rv_invariance(self):
        """rv_at_times produces identical signals before and after wrap_angles."""
        samples = _make_rv_samples_with_signs([-10.0, 12.0, -8.0])
        wrapped = samples.wrap_angles()
        times = Q(jnp.linspace(0.0, 300.0, 25), "day")
        for i in range(samples.n_samples):
            kwargs_orig = {
                "period": samples["period"][i],
                "eccentricity": samples["eccentricity"][i],
                "t_peri": samples["t_peri"][i],
                "arg_peri": samples["arg_peri"][i],
                "rv_semiamp": samples["rv_semiamp"][i],
                "v_sys": samples["v_sys"][i],
            }
            kwargs_wrap = {
                "period": wrapped["period"][i],
                "eccentricity": wrapped["eccentricity"][i],
                "t_peri": wrapped["t_peri"][i],
                "arg_peri": wrapped["arg_peri"][i],
                "rv_semiamp": wrapped["rv_semiamp"][i],
                "v_sys": wrapped["v_sys"][i],
            }
            rv_orig = rv_at_times(times, **kwargs_orig)
            rv_wrap = rv_at_times(times, **kwargs_wrap)
            assert jnp.allclose(rv_orig.value, rv_wrap.value, atol=1e-10)

    def test_astrometry_invariance(self):
        """astrometric_orbit_at_times unchanged after wrap_angles."""
        samples = _make_astro_samples_with_signs([-1.5, 2.0, -0.5])
        wrapped = samples.wrap_angles()
        times = Q(jnp.linspace(0.0, 600.0, 25), "day")
        for i in range(samples.n_samples):
            kwargs_orig = {
                "period": samples["period"][i],
                "eccentricity": samples["eccentricity"][i],
                "t_peri": samples["t_peri"][i],
                "arg_peri": samples["arg_peri"][i],
                "cos_i": samples["cos_i"][i],
                "lon_asc_node": samples["lon_asc_node"][i],
                "semi_major_axis": samples["semi_major_axis"][i],
            }
            kwargs_wrap = {
                "period": wrapped["period"][i],
                "eccentricity": wrapped["eccentricity"][i],
                "t_peri": wrapped["t_peri"][i],
                "arg_peri": wrapped["arg_peri"][i],
                "cos_i": wrapped["cos_i"][i],
                "lon_asc_node": wrapped["lon_asc_node"][i],
                "semi_major_axis": wrapped["semi_major_axis"][i],
            }
            dra_orig, ddec_orig = astrometric_orbit_at_times(times, **kwargs_orig)
            dra_wrap, ddec_wrap = astrometric_orbit_at_times(times, **kwargs_wrap)
            assert jnp.allclose(dra_orig.value, dra_wrap.value, atol=1e-10)
            assert jnp.allclose(ddec_orig.value, ddec_wrap.value, atol=1e-10)

    def test_no_op_when_all_positive(self):
        """wrap_angles is a no-op when no entries need flipping."""
        samples = _make_rv_samples_with_signs([10.0, 12.0, 8.0])
        wrapped = samples.wrap_angles()
        assert jnp.allclose(wrapped["rv_semiamp"].value, samples["rv_semiamp"].value)
        assert jnp.allclose(wrapped["arg_peri"].value, samples["arg_peri"].value)

    def test_no_op_when_arg_peri_absent(self):
        """wrap_angles is a no-op when arg_peri is missing from nonlinear."""
        samples = Samples(
            nonlinear={
                "period": Q(jnp.array([100.0, 200.0]), "day"),
                "eccentricity": Q(jnp.array([0.1, 0.2]), ""),
                "phase_peri": Q(jnp.array([0.0, 0.5]), ""),
            },
            linear={
                "rv_semiamp": Q(jnp.array([-1.0, 3.0]), "km/s"),
                "v_sys": Q(jnp.array([2.0, 4.0]), "km/s"),
            },
            data_type="rv",
            metadata={},
        )
        wrapped = samples.wrap_angles()
        assert wrapped is samples

    def test_wrapped_omega_is_in_canonical_range(self):
        """Shifted arg_peri values land in [0, 2*pi)."""
        # arg_peri values near 2*pi-epsilon -> +pi -> wrap into low end of range.
        samples = Samples(
            nonlinear={
                "period": Q(jnp.array([100.0, 100.0]), "day"),
                "eccentricity": Q(jnp.array([0.1, 0.1]), ""),
                "phase_peri": Q(jnp.array([0.0, 0.0]), ""),
                "arg_peri": Q(jnp.array([6.0, 0.5]), "rad"),
            },
            linear={
                "rv_semiamp": Q(jnp.array([-1.0, -1.0]), "km/s"),
                "v_sys": Q(jnp.array([0.0, 0.0]), "km/s"),
            },
            data_type="rv",
            metadata={},
        )
        wrapped = samples.wrap_angles()
        new_omega = wrapped["arg_peri"].value
        assert bool(((new_omega >= 0) & (new_omega < 2 * jnp.pi)).all())

    def test_sb2_flips_both_K_with_shared_arg_peri(self):
        """SB2 namespaced rv_semiamps flip in lockstep with the shared omega.

        Trigger is the FIRST rv_semiamp-suffixed key (insertion order).
        For samples where it's negative, both K_primary and K_secondary
        are flipped and arg_peri is shifted by pi; positive-trigger samples
        are untouched.
        """
        old_omega_val = jnp.array([1.0, 1.0, 1.0, 1.0])
        samples = Samples(
            nonlinear={
                "period": Q(jnp.full(4, 100.0), "day"),
                "eccentricity": Q(jnp.full(4, 0.3), ""),
                "phase_peri": Q(jnp.full(4, 0.25), ""),
                "arg_peri": Q(old_omega_val, "rad"),
            },
            linear={
                "primary.rv_semiamp": Q(jnp.array([-10.0, 12.0, -8.0, 5.0]), "km/s"),
                "secondary.rv_semiamp": Q(jnp.array([+5.0, -3.0, +4.0, -2.0]), "km/s"),
                "v_sys": Q(jnp.zeros(4), "km/s"),
            },
            data_type="joint",
            metadata={},
        )
        wrapped = samples.wrap_angles()

        # Trigger is primary.rv_semiamp -> negative at indices 0 and 2.
        flip_mask = jnp.array([True, False, True, False])

        # Primary K is non-negative everywhere after the wrap.
        new_K1 = wrapped["primary.rv_semiamp"].value
        assert bool((new_K1 >= 0).all())
        assert jnp.allclose(new_K1, jnp.array([10.0, 12.0, 8.0, 5.0]))

        # Secondary K: flipped sign on the same indices as primary.
        new_K2 = wrapped["secondary.rv_semiamp"].value
        assert jnp.allclose(new_K2, jnp.array([-5.0, -3.0, -4.0, -2.0]))

        # arg_peri shifted by pi at the flipped indices, untouched otherwise.
        new_omega = wrapped["arg_peri"].value
        expected_omega = jnp.where(
            flip_mask, jnp.mod(old_omega_val + jnp.pi, 2 * jnp.pi), old_omega_val
        )
        assert jnp.allclose(new_omega, expected_omega)

        # v_sys (shared, unrelated) is unchanged.
        assert jnp.allclose(wrapped["v_sys"].value, jnp.zeros(4))

    def test_joint_sb2_with_semi_major_axis(self):
        """Hypothetical SB2 + astrometry: K's and a all flip together."""
        samples = Samples(
            nonlinear={
                "period": Q(jnp.full(3, 300.0), "day"),
                "eccentricity": Q(jnp.full(3, 0.3), ""),
                "phase_peri": Q(jnp.full(3, 0.1), ""),
                "arg_peri": Q(jnp.linspace(0.5, 2.5, 3), "rad"),
                "cos_i": Q(jnp.full(3, 0.4), ""),
                "lon_asc_node": Q(jnp.full(3, 0.8), "rad"),
            },
            linear={
                "primary.rv_semiamp": Q(jnp.array([-10.0, 8.0, -6.0]), "km/s"),
                "secondary.rv_semiamp": Q(jnp.array([+4.0, -3.0, +2.0]), "km/s"),
                "semi_major_axis": Q(jnp.array([1.5, -2.0, 0.5]), "mas"),
                "v_sys": Q(jnp.zeros(3), "km/s"),
            },
            data_type="joint",
            metadata={},
        )
        wrapped = samples.wrap_angles()

        # Trigger is primary.rv_semiamp -> negative at indices 0 and 2.
        # Both K's AND semi_major_axis flip on those indices regardless of
        # whether semi_major_axis was already positive (it shares the same
        # arg_peri as the K's).
        assert jnp.allclose(
            wrapped["primary.rv_semiamp"].value, jnp.array([10.0, 8.0, 6.0])
        )
        assert jnp.allclose(
            wrapped["secondary.rv_semiamp"].value, jnp.array([-4.0, -3.0, -2.0])
        )
        assert jnp.allclose(
            wrapped["semi_major_axis"].value, jnp.array([-1.5, -2.0, -0.5])
        )

    def test_per_component_arg_peri_raises(self):
        """Multiple per-component arg_peri keys aren't supported (yet)."""
        samples = Samples(
            nonlinear={
                "period": Q(jnp.full(2, 100.0), "day"),
                "eccentricity": Q(jnp.full(2, 0.3), ""),
                "phase_peri": Q(jnp.full(2, 0.25), ""),
                "primary.arg_peri": Q(jnp.array([1.0, 2.0]), "rad"),
                "secondary.arg_peri": Q(jnp.array([1.0, 2.0]), "rad"),
            },
            linear={
                "primary.rv_semiamp": Q(jnp.array([-10.0, 5.0]), "km/s"),
                "secondary.rv_semiamp": Q(jnp.array([4.0, -3.0]), "km/s"),
                "v_sys": Q(jnp.zeros(2), "km/s"),
            },
            data_type="joint",
            metadata={},
        )
        with pytest.raises(NotImplementedError, match="multiple per-component"):
            samples.wrap_angles()
