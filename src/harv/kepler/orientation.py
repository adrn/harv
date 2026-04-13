"""Keplerian orbit implementation with units support and JAX compatibility."""

from typing import cast

import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import Q, ustrip

from harv.custom_types import ScalarFloat, ScalarQAngle, ScalarQLength, float_converter
from harv.kepler.orbits import thiele_innes_ABFG


class KeplerianOrientation(eqx.Module):
    """Orientation of a Keplerian orbit in 3D space.

    Stores the three Euler angles that define how the orbital plane
    is oriented relative to the observer's reference frame:
    - Inclination (i): tilt of orbital plane from sky plane
    - Longitude of ascending node (Omega): where orbit crosses sky plane
    - Argument of pericenter (omega): orientation of ellipse within orbital plane

    Angles are stored as sin/cos pairs for numerical stability.
    """

    # sin/cos of argument of pericenter (omega)
    sin_arg_peri: ScalarFloat = eqx.field(default=0.0, converter=float_converter)
    cos_arg_peri: ScalarFloat = eqx.field(default=1.0, converter=float_converter)

    # sin/cos of longitude of ascending node (Omega)
    sin_lon_asc_node: ScalarFloat = eqx.field(default=0.0, converter=float_converter)
    cos_lon_asc_node: ScalarFloat = eqx.field(default=1.0, converter=float_converter)

    # sin/cos of inclination (i)
    sin_i: ScalarFloat = eqx.field(default=0.0, converter=float_converter)
    cos_i: ScalarFloat = eqx.field(default=1.0, converter=float_converter)

    def __check_init__(self) -> None:
        x = jnp.array(self.sin_arg_peri**2 + self.cos_arg_peri**2)
        eqx.error_if(
            x,
            jnp.logical_not(
                jnp.isclose(
                    x,
                    1.0,
                    atol=jnp.finfo(float).eps,  # type: ignore[no-untyped-call]
                )
            ),
            "Argument of pericenter sin/cos values are not normalized",
        )

        x = jnp.array(self.sin_lon_asc_node**2 + self.cos_lon_asc_node**2)
        eqx.error_if(
            x,
            jnp.logical_not(
                jnp.isclose(
                    x,
                    1.0,
                    atol=jnp.finfo(float).eps,  # type: ignore[no-untyped-call]
                )
            ),
            "Longitude of ascending node sin/cos values are not normalized",
        )

        x = jnp.array(self.sin_i**2 + self.cos_i**2)
        eqx.error_if(
            x,
            jnp.logical_not(
                jnp.isclose(
                    x,
                    1.0,
                    atol=jnp.finfo(float).eps,  # type: ignore[no-untyped-call]
                )
            ),
            "Inclination sin/cos values are not normalized",
        )

    @classmethod
    def from_angles(
        cls,
        /,
        arg_peri: ScalarQAngle = Q.from_(0.0, "rad"),
        lon_asc_node: ScalarQAngle = Q.from_(0.0, "rad"),
        inclination: ScalarQAngle = Q.from_(0.0, "rad"),
    ) -> "KeplerianOrientation":
        """Construct from angle values."""
        return cls(
            sin_arg_peri=jnp.sin(Q.from_(arg_peri)),
            cos_arg_peri=jnp.cos(Q.from_(arg_peri)),
            sin_lon_asc_node=jnp.sin(Q.from_(lon_asc_node)),
            cos_lon_asc_node=jnp.cos(Q.from_(lon_asc_node)),
            sin_i=jnp.sin(Q.from_(inclination)),
            cos_i=jnp.cos(Q.from_(inclination)),
        )

    @classmethod
    def from_thiele_innes(
        cls,
        A: ScalarQLength | ScalarQAngle,
        B: ScalarQLength | ScalarQAngle,
        F: ScalarQLength | ScalarQAngle,
        G: ScalarQLength | ScalarQAngle,
    ) -> tuple["KeplerianOrientation", ScalarQLength | ScalarQAngle]:
        """Construct orientation from Thiele-Innes constants.

        Inverts the Thiele-Innes constants to recover (omega, Omega, i, a).
        This loosely follows Appendix A of https://arxiv.org/abs/2206.05726
        but my convention for angles is different. For this implementation:

        0 < i < pi/2
        0 < omega < 2pi
        0 < Omega < 2pi

        whereas the paper linked above assumes:

        0 < i < pi
        0 < omega < 2pi
        0 < Omega < pi

        Returns
        -------
        orientation
            KeplerianOrientation object
        semi_major_axis
            Recovered semi-major axis
        """
        # A: Q[Any] = Q.from_(A)
        # B: Q[Any] = Q.from_(B)
        # F: Q[Any] = Q.from_(F)
        # G: Q[Any] = Q.from_(G)

        u_ = (A**2 + B**2 + F**2 + G**2) / 2.0
        v_ = A * G - B * F

        inner_tmp = (u_ + v_) * (u_ - v_)
        # Guard against small negative from roundoff
        inner = jnp.where(inner_tmp < 0.0, Q.from_(0.0, inner_tmp.unit), inner_tmp)
        a = jnp.sqrt(u_ + jnp.sqrt(inner))

        # From algebraic manipulation of T-I
        cos_i = ustrip("", v_ / a**2)
        cos_i = jnp.clip(cos_i, -1.0, 1.0)

        sin_i_squared = 1.0 - cos_i**2
        sin_i = jnp.sqrt(jnp.where(sin_i_squared < 0.0, 0.0, sin_i_squared))
        i = jnp.arctan2(sin_i, cos_i)  # i in [0, pi]

        # Sums & differences of angles (Binnendijk paper referenced in above paper)
        # sp = (omega + Omega), sm = (omega - Omega)
        # Using the identities:
        # A + G = a(1 + cos(i)) cos(omega + Omega)
        # B - F = a(1 + cos(i)) sin(omega + Omega)
        # A - G = a(1 - cos(i)) cos(omega - Omega)
        # B + F = -a(1 - cos(i)) sin(omega - Omega)
        sp = ustrip("rad", jnp.arctan2(B - F, A + G))
        sm = ustrip("rad", jnp.arctan2(-(B + F), A - G))

        # Normalize sp to [0, 2pi) for consistency
        sp = jnp.mod(sp, 2 * jnp.pi)
        # Keep sm in [-pi, pi] as returned by arctan2

        # Compute omega and Omega, then normalize to [0, 2pi)
        omega = jnp.mod(0.5 * (sp + sm), 2 * jnp.pi)
        Omega = jnp.mod(0.5 * (sp - sm), 2 * jnp.pi)

        return (
            cls.from_angles(
                arg_peri=Q.from_(omega, "rad"),
                lon_asc_node=Q.from_(Omega, "rad"),
                inclination=Q.from_(i, "rad"),
            ),
            cast("ScalarQLength | ScalarQAngle", a),
        )

    @property
    def arg_peri(self) -> ScalarQAngle:
        """Argument of pericenter (omega)."""
        return Q.from_(jnp.arctan2(self.sin_arg_peri, self.cos_arg_peri), "rad")

    @property
    def lon_asc_node(self) -> ScalarQAngle:
        """Longitude of ascending node (Omega)."""
        return Q.from_(jnp.arctan2(self.sin_lon_asc_node, self.cos_lon_asc_node), "rad")

    @property
    def inclination(self) -> ScalarQAngle:
        """Inclination (i)."""
        return Q.from_(jnp.arctan2(self.sin_i, self.cos_i), "rad")

    @property
    def rotation_matrix(self) -> jax.Array:
        """Compute rotation matrix from orbital plane to observer frame.

        Returns the rotation matrix R such that:
        r_observer_frame = R @ r_orbital_frame

        The rotation is composed of three sequential rotations:
        1. R_z(omega): Rotate by argument of pericenter, omega, in orbital plane
        2. R_x(i): Rotate by inclination, i, to tilt orbital plane
        3. R_z(Omega): Rotate by longitude of ascending node, Omega, on sky plane

        The full rotation matrix is therefore:
        R = R_z(Omega) @ R_x(i) @ R_z(omega)

        We build the matrix directly from the sin/cos pairs for numerical stability and
        speed, but using the notation below, it is equivalent to:

        R1 = jnp.array([[c_w, -s_w, 0], [s_w, c_w, 0], [0, 0, 1]])
        R2 = jnp.array([[1., 0, 0], [0, c_i, -s_i], [0, s_i, c_i]])
        R3 = jnp.array([[c_W, -s_W, 0], [s_W, c_W, 0], [0, 0, 1]])
        R = R3 @ R2 @ R1

        Or, alternatively:
        omega = arg_peri.to_value("rad")
        Omega = lon_asc_node.to_value("rad")
        i = inclination.to_value("rad")
        R = Rotation.from_euler('ZXZ', [Omega, i, omega]).as_matrix()

        """
        s_w = self.sin_arg_peri
        c_w = self.cos_arg_peri
        s_W = self.sin_lon_asc_node
        c_W = self.cos_lon_asc_node
        s_i = self.sin_i
        c_i = self.cos_i

        # Write out all terms explicitly (for speed)
        r11 = c_W * c_w - s_W * c_i * s_w
        r12 = -c_W * s_w - s_W * c_i * c_w
        r13 = s_W * s_i
        r21 = s_W * c_w + c_W * c_i * s_w
        r22 = -s_W * s_w + c_W * c_i * c_w
        r23 = -c_W * s_i
        r31 = s_i * s_w
        r32 = s_i * c_w
        r33 = c_i

        return jnp.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    def thiele_innes_constants(
        self, semi_major_axis: ScalarQLength | ScalarQAngle | None = None
    ) -> tuple[ScalarQLength | ScalarQAngle | jax.Array, ...]:
        """Compute Thiele-Innes constants (A, B, F, G).

        These constants linearize the relationship between orbital position
        and sky-plane projection. See Appendix A of: https://arxiv.org/abs/2206.05726

        Parameters
        ----------
        semi_major_axis
            Semi-major axis of the orbit

        Returns
        -------
        A, B, F, G
            The four Thiele-Innes constants.
        """
        a: jax.Array | ScalarQLength | ScalarQAngle = (
            jnp.array(1.0) if semi_major_axis is None else Q.from_(semi_major_axis)
        )

        A, B, F, G = thiele_innes_ABFG(
            self.cos_arg_peri,
            self.sin_arg_peri,
            self.cos_lon_asc_node,
            self.sin_lon_asc_node,
            self.cos_i,
        )

        return cast(
            "tuple[ScalarQLength | ScalarQAngle | jax.Array, ...]",
            (a * A, a * B, a * F, a * G),
        )
