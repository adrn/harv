"""Custom types used in harv."""

from typing import Literal

import jax
from jaxtyping import Float, Int, Real
from unxt import Quantity

Angle = Literal["angle"]
Length = Literal["length"]
Mass = Literal["mass"]
Speed = Literal["speed"]
Time = Literal["time"]
Dimless = Literal["dimensionless"]

NAngle = Real[Angle, "n"]
NTime = Real[Time, "n"]
NVelocity = Real[Speed, "n"]
NFloatArray = Float[jax.Array, "n"]
NIntArray = Int[jax.Array, "n"]

DimlessValue = Quantity[Dimless] | jax.Array | float
