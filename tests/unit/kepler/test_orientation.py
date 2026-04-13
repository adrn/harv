"""Unit tests for :mod:`harv.kepler.orientation`."""

import jax
import pytest
import quaxed.numpy as jnp
from jax import config as jax_config
from unxt import Q, ustrip

from harv.kepler.orientation import KeplerianOrientation

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    """Parametrize tests over float32 and float64."""
    original_value = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", request.param == "float64")
    yield request.param
    jax_config.update("jax_enable_x64", original_value)


# =============================================================================
# Helpers
# =============================================================================


def _make_orientation(
    arg_peri: float = 0.5,
    lon_asc_node: float = 1.2,
    inclination: float = 0.8,
) -> KeplerianOrientation:
    """Create a KeplerianOrientation from angles in radians."""
    return KeplerianOrientation.from_angles(
        arg_peri=Q(arg_peri, "rad"),
        lon_asc_node=Q(lon_asc_node, "rad"),
        inclination=Q(inclination, "rad"),
    )


def _check_thiele_innes_round_trip(
    orientation: KeplerianOrientation,
    semi_major_axis: Q,
    rtol: float = 1e-10,
    atol: float = 1e-8,
) -> None:
    """Check that Thiele-Innes constants round-trip correctly."""
    ti_constants = orientation.thiele_innes_constants(semi_major_axis)
    roundtrip_orientation, roundtrip_a = KeplerianOrientation.from_thiele_innes(
        *ti_constants
    )

    # Check semi-major axis is preserved
    assert jnp.allclose(
        ustrip("au", roundtrip_a), ustrip("au", semi_major_axis), rtol=rtol, atol=atol
    )

    # Compare rotation matrices -- more robust than comparing angles since
    # there can be angle ambiguities giving the same rotation.
    match_primary = jnp.allclose(
        roundtrip_orientation.rotation_matrix,
        orientation.rotation_matrix,
        rtol=rtol,
        atol=atol,
    )

    # Symmetric solution: Omega -> Omega+pi, omega -> omega+pi (T-I invariant under this transform)
    sym_orientation = KeplerianOrientation.from_angles(
        arg_peri=orientation.arg_peri + Q(jnp.pi, "rad"),
        lon_asc_node=orientation.lon_asc_node + Q(jnp.pi, "rad"),
        inclination=orientation.inclination,
    )
    match_sym = jnp.allclose(
        roundtrip_orientation.rotation_matrix,
        sym_orientation.rotation_matrix,
        rtol=rtol,
        atol=atol,
    )

    assert (
        match_primary or match_sym
    ), "Recovered orientation does not match original or symmetric solution."

    # Check Thiele-Innes constants are preserved
    roundtrip_constants = roundtrip_orientation.thiele_innes_constants(roundtrip_a)
    assert jnp.allclose(
        jnp.stack([ustrip("AU", val) for val in ti_constants]),
        jnp.stack([ustrip("AU", val) for val in roundtrip_constants]),
        rtol=rtol,
        atol=atol,
    )


# =============================================================================
# Basic construction tests
# =============================================================================


class TestKeplerianOrientationConstruction:
    def test_default_is_identity(self) -> None:
        """Default orientation has zero angles (identity rotation)."""
        o = KeplerianOrientation()
        assert jnp.allclose(o.sin_arg_peri, 0.0)
        assert jnp.allclose(o.cos_arg_peri, 1.0)
        assert jnp.allclose(o.sin_lon_asc_node, 0.0)
        assert jnp.allclose(o.cos_lon_asc_node, 1.0)
        assert jnp.allclose(o.sin_i, 0.0)
        assert jnp.allclose(o.cos_i, 1.0)

    def test_rotation_matrix_identity(self) -> None:
        """Default rotation matrix is the identity matrix."""
        R = KeplerianOrientation().rotation_matrix
        assert jnp.allclose(R, jnp.eye(3), atol=1e-12)

    def test_from_angles_stores_sin_cos(self) -> None:
        """from_angles correctly stores sin/cos pairs."""
        o = KeplerianOrientation.from_angles(
            arg_peri=Q(jnp.pi / 2, "rad"),
        )
        assert jnp.allclose(o.sin_arg_peri, 1.0, atol=1e-7)
        assert jnp.allclose(o.cos_arg_peri, 0.0, atol=1e-7)

    def test_angle_properties_round_trip(self) -> None:
        """from_angles -> .arg_peri/.lon_asc_node/.inclination recovers inputs."""
        w, W, i_ = 0.7, 1.3, 0.4
        o = _make_orientation(w, W, i_)
        assert jnp.allclose(ustrip("rad", o.arg_peri), w, atol=1e-6)
        assert jnp.allclose(ustrip("rad", o.lon_asc_node), W, atol=1e-6)
        assert jnp.allclose(ustrip("rad", o.inclination), i_, atol=1e-6)

    def test_converter_accepts_quantity(self) -> None:
        """Sin/cos fields accept dimensionless Q values."""
        o = KeplerianOrientation(
            sin_arg_peri=Q(0.0, ""),
            cos_arg_peri=Q(1.0, ""),
        )
        assert jnp.allclose(o.sin_arg_peri, 0.0)


# =============================================================================
# Rotation matrix properties
# =============================================================================


class TestRotationMatrix:
    def test_orthogonality(self) -> None:
        """R @ R.T == I for a non-trivial orientation."""
        R = _make_orientation().rotation_matrix
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-6)

    def test_determinant_is_one(self) -> None:
        """Rotation matrix has determinant +1."""
        R = _make_orientation(0.3, 2.1, 1.0).rotation_matrix
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-6)


# =============================================================================
# JAX compatibility
# =============================================================================


class TestJAXCompat:
    def test_pytree_flatten_unflatten(self) -> None:
        """KeplerianOrientation round-trips through pytree flatten/unflatten."""
        o = _make_orientation()
        leaves, treedef = jax.tree.flatten(o)
        o2 = treedef.unflatten(leaves)
        assert jnp.allclose(o.rotation_matrix, o2.rotation_matrix)

    def test_jit_rotation_matrix(self) -> None:
        """jax.jit can compute the rotation matrix."""
        o = _make_orientation()
        R = jax.jit(lambda o: o.rotation_matrix)(o)
        assert R.shape == (3, 3)
        assert jnp.allclose(R, o.rotation_matrix)

    def test_jit_thiele_innes(self) -> None:
        """jax.jit can compute Thiele-Innes constants."""
        o = _make_orientation()
        a = Q(3.0, "AU")

        @jax.jit
        def f(o, a):
            return o.thiele_innes_constants(a)

        result = f(o, a)
        assert len(result) == 4

    def test_vmap_over_angles(self) -> None:
        """Vmap over batched KeplerianOrientation (different arg_peri per element)."""
        angles = [0.0, 0.5, 1.2, 2.5]
        orientations = [
            KeplerianOrientation.from_angles(
                arg_peri=Q(w, "rad"),
                lon_asc_node=Q(1.0, "rad"),
                inclination=Q(0.4, "rad"),
            )
            for w in angles
        ]
        orientations_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *orientations)

        result = jax.vmap(lambda o: o.rotation_matrix)(orientations_batched)
        assert result.shape == (4, 3, 3)

        # Each batched result should match the direct scalar computation
        for i, o in enumerate(orientations):
            assert jnp.allclose(result[i], o.rotation_matrix, atol=1e-7)

    def test_vmap_over_inclination_thiele_innes(self) -> None:
        """Vmap over Thiele-Innes constants across different inclinations."""
        inclinations = [0.1, 0.5, 1.0, 1.5]
        a = Q(3.0, "AU")
        orientations = [
            KeplerianOrientation.from_angles(
                arg_peri=Q(0.5, "rad"),
                lon_asc_node=Q(1.0, "rad"),
                inclination=Q(i, "rad"),
            )
            for i in inclinations
        ]
        orientations_batched = jax.tree.map(lambda *xs: jnp.stack(xs), *orientations)

        result = jax.vmap(lambda o: jnp.stack(list(o.thiele_innes_constants(a))))(
            orientations_batched
        )
        assert result.shape == (4, 4)  # 4 inclinations, 4 T-I constants (stripped)


# =============================================================================
# Thiele-Innes round-trip tests
# =============================================================================


@pytest.mark.parametrize(
    ("arg_peri", "lon_asc_node", "inclination", "semi_major_axis"),
    [
        (0.0, 0.0, 0.0, 5.0),  # zero angles
        (1.5, 2.0, jnp.pi / 2, 3.5),  # max inclination
        (2 * jnp.pi - 0.01, 2 * jnp.pi - 0.02, jnp.pi / 2 - 0.001, 2.8),  # near 2pi
        (1.2, 0.5, 0.3, 4.0),  # arbitrary
        (0.8, 2.3, 2.5, 7.2),  # inclination > pi/2
    ],
)
def test_thiele_innes_round_trip_edge_cases(
    arg_peri: float,
    lon_asc_node: float,
    inclination: float,
    semi_major_axis: float,
    dtype: str,
) -> None:
    """Test Thiele-Innes round-trip for edge cases and typical values."""
    rtol = 5e-4 if dtype == "float32" else 1e-6
    orientation = KeplerianOrientation.from_angles(
        arg_peri=Q(arg_peri, "rad"),
        lon_asc_node=Q(lon_asc_node, "rad"),
        inclination=Q(inclination, "rad"),
    )
    _check_thiele_innes_round_trip(orientation, Q(semi_major_axis, "AU"), rtol=rtol)


@pytest.mark.parametrize(
    ("seed", "incl_range"),
    [
        (42, "low"),
        (123, "low"),
        (456, "low"),
        (42, "high"),
        (99, "high"),
    ],
)
def test_thiele_innes_round_trip_random(seed: int, incl_range: str, dtype: str) -> None:
    """Test Thiele-Innes round-trip with random angles at various inclinations."""
    if incl_range == "low" and dtype == "float32":
        rtol, atol = 2e-2, 2e-2
    elif dtype == "float32":
        rtol, atol = 5e-4, 1e-8
    else:
        rtol, atol = 1e-6, 1e-8

    key = jax.random.key(seed)
    random_vals = jax.random.uniform(key, shape=(4,))

    arg_peri = random_vals[0] * 2 * jnp.pi
    lon_asc_node = random_vals[1] * 2 * jnp.pi
    if incl_range == "low":
        inclination = random_vals[2] * jnp.pi / 2
    else:
        inclination = jnp.pi / 2 + random_vals[2] * jnp.pi / 2
    semi_major_axis = Q(jnp.asarray(1.0 + random_vals[3] * 10), "AU")

    orientation = KeplerianOrientation.from_angles(
        arg_peri=Q(arg_peri, "rad"),
        lon_asc_node=Q(lon_asc_node, "rad"),
        inclination=Q(inclination, "rad"),
    )
    _check_thiele_innes_round_trip(orientation, semi_major_axis, rtol=rtol, atol=atol)
