"""Custom types used in harv."""

from typing import Any, Literal

import jax
import numpy as np
from jaxtyping import Float, Int, Real
from unxt import Quantity
from unxt.quantity import AllowValue, ustrip

Angle = Literal["angle"]
AngularSpeed = Literal["angular speed"]
Length = Literal["length"]
Mass = Literal["mass"]
Speed = Literal["speed"]
Time = Literal["time"]
Dimless = Literal["dimensionless"]

ScalarQAngle = Real[Quantity["angle"], ""]
ScalarQAngularSpeed = Real[Quantity["angular speed"], ""]
ScalarQDimless = Real[Quantity["dimensionless"], ""]
ScalarQLength = Real[Quantity["length"], ""]
ScalarQMass = Real[Quantity["mass"], ""]
ScalarQSpeed = Real[Quantity["speed"], ""]
ScalarQTime = Real[Quantity["time"], ""]

NAngle = Real[Quantity["angle"], "n"]
NDimless = Real[Quantity["dimensionless"], "n"]
NTime = Real[Quantity["time"], "n"]
NVelocity = Real[Quantity["speed"], "n"]
NFloatArray = Float[jax.Array, "n"]
NIntArray = Int[jax.Array, "n"]

Vec3QLength = Real[Quantity["length"], "3"]
Vec3QSpeed = Real[Quantity["speed"], "3"]

ScalarFloat = Float[jax.Array, ""] | np.floating[Any] | float | int | ScalarQDimless


def float_converter(x: ScalarFloat) -> Float[jax.Array, ""]:
    """Converter for dimensionless scalar float fields.

    Strips units from a dimensionless quantity or passes through plain scalars,
    always producing a 0-d JAX array. Use as an ``eqx.field`` converter wherever
    a dimensionless scalar is stored internally as a plain JAX float.
    """
    return ustrip(AllowValue, "", x)
