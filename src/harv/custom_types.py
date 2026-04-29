"""Custom types used in harv."""

from typing import Any, Literal

import jax
import numpy as np
from jaxtyping import Float, Int, Real
from unxt import AbstractQuantity, Q
from unxt.quantity import AllowValue, ustrip

Angle = Literal["angle"]
AngularSpeed = Literal["angular_speed"]
Length = Literal["length"]
Mass = Literal["mass"]
Speed = Literal["speed"]
Time = Literal["time"]
Dimless = Literal["dimensionless"]

QAngle = Q["angle"]
QAngularSpeed = Q["angular_speed"]
QDimless = Q["dimensionless"]
QLength = Q["length"]
QMass = Q["mass"]
QSpeed = Q["speed"]
QTime = Q["time"]

ScalarQAngle = Real[Q["angle"], ""]
ScalarQAngularSpeed = Real[Q["angular_speed"], ""]
ScalarQDimless = Real[Q["dimensionless"], ""]
ScalarQLength = Real[Q["length"], ""]
ScalarQMass = Real[Q["mass"], ""]
ScalarQSpeed = Real[Q["speed"], ""]
ScalarQTime = Real[Q["time"], ""]
ScalarQAny = Real[AbstractQuantity, ""]

NAngle = Real[Q["angle"], "n"]
NDimless = Real[Q["dimensionless"], "n"]
NTime = Real[Q["time"], "n"]
NVelocity = Real[Q["speed"], "n"]
NFloatArray = Float[jax.Array, "n"]
NIntArray = Int[jax.Array, "n"]
NQAny = Real[AbstractQuantity, ""]

Vec3QLength = Real[Q["length"], "3"]
Vec3QSpeed = Real[Q["speed"], "3"]

BatchVec3QLength = Real[Q["length"], "3 *batch"]
BatchVec3QSpeed = Real[Q["speed"], "3 *batch"]

BatchQAngle = Real[Q["angle"], "*batch"]
BatchQAngularSpeed = Real[Q["angular_speed"], "*batch"]
BatchQDimless = Real[Q["dimensionless"], "*batch"]
BatchQLength = Real[Q["length"], "*batch"]
BatchQSpeed = Real[Q["speed"], "*batch"]
BatchQTime = Real[Q["time"], "*batch"]
BatchFloat = Float[jax.Array, "*batch"] | BatchQDimless
BatchQAny = Real[AbstractQuantity, "*batch"]

# Set of all Batch-level Quantity types that carry physical dimensions.
# Used by AbstractParameters.__init_subclass__ to auto-detect which fields
# require a QDistribution prior.
DIMENSIONED_BATCH_TYPES: frozenset[Any] = frozenset(
    {
        BatchQAngle,
        BatchQAngularSpeed,
        BatchQLength,
        BatchQSpeed,
        BatchQTime,
    }
)

ScalarFloat = Float[jax.Array, ""] | np.floating[Any] | float | int | ScalarQDimless


def float_converter(x: ScalarFloat) -> Float[jax.Array, ""]:
    """Converter for dimensionless scalar float fields.

    Strips units from a dimensionless quantity or passes through plain scalars,
    always producing a 0-d JAX array. Use as an ``eqx.field`` converter wherever
    a dimensionless scalar is stored internally as a plain JAX float.
    """
    return ustrip(AllowValue, "", x)
