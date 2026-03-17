"""Custom types used in harv."""

from __future__ import annotations

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

NAngle = Real[Quantity[Angle], "n"]
NTime = Real[Quantity[Time], "n"]
NVelocity = Real[Quantity[Speed], "n"]
NFloatArray = Float[jax.Array, "n"]
NIntArray = Int[jax.Array, "n"]

DimlessValue = Quantity[Dimless] | jax.Array | float
