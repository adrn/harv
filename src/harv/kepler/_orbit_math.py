"""Low-level orbit math building blocks.

Canonical implementations of core orbit computations used by ``harv.kepler``,
``harv.likelihood``, and ``harv.simulate``.

``mean_anomaly`` and ``true_anomaly_from_mean`` accept and return
:class:`unxt.Quantity` objects so that callers never need to strip units
themselves. ``rv_shape`` and ``thiele_innes_ABFG`` remain pure functions on raw
JAX arrays because their inputs are always already dimensionless.
"""

__all__ = (
    "mean_anomaly",
    "rv_shape",
    "thiele_innes_ABFG",
    "true_anomaly_from_mean",
)

import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import Quantity
from unxt.quantity import ustrip

from harv.custom_types import (
    BatchableFloat,
    BatchableQAngle,
    BatchableQTime,
    ScalarFloat,
    ScalarQTime,
)


def mean_anomaly(dt: BatchableQTime, period: ScalarQTime) -> BatchableQAngle:
    """Compute mean anomaly from elapsed time and period.

    ``M = 2π · dt / period``, returned as a :class:`~unxt.Quantity` with angle
    units (radians).
    """
    return Quantity.from_(ustrip("", 2 * jnp.pi * dt / period), "rad")


def true_anomaly_from_mean(
    M: BatchableQAngle, eccentricity: ScalarFloat
) -> tuple[BatchableFloat, BatchableFloat]:
    """Solve Kepler's equation: mean anomaly → (sin f, cos f).

    Wraps ``jaxoplanet.core.kepler.kepler``. The mean anomaly is stripped to
    radians internally.
    """
    return kepler(ustrip("rad", M), eccentricity)


def rv_shape(
    sin_f: BatchableFloat,
    cos_f: BatchableFloat,
    eccentricity: ScalarFloat,
    arg_peri: ScalarFloat,
) -> BatchableFloat:
    """RV shape function: cos(ω + f) + e·cos(ω).

    Returns the dimensionless RV amplitude factor for each observation.
    ``arg_peri`` is in radians.
    """
    cos_wf = jnp.cos(arg_peri) * cos_f - jnp.sin(arg_peri) * sin_f
    return cos_wf + eccentricity * jnp.cos(arg_peri)


def thiele_innes_ABFG(
    cos_arg_peri: ScalarFloat,
    sin_arg_peri: ScalarFloat,
    cos_lon_asc_node: ScalarFloat,
    sin_lon_asc_node: ScalarFloat,
    cos_i: ScalarFloat,
) -> tuple[ScalarFloat, ScalarFloat, ScalarFloat, ScalarFloat]:
    """Compute unit Thiele-Innes constants (A, B, F, G).

    Returns the constants with an implicit semi-major axis of 1. Multiply each
    by ``a`` to recover the physical Thiele-Innes constants.

    See Eq. A.1 of https://arxiv.org/abs/2206.05726.
    """
    A = cos_arg_peri * cos_lon_asc_node - sin_arg_peri * sin_lon_asc_node * cos_i
    B = cos_arg_peri * sin_lon_asc_node + sin_arg_peri * cos_lon_asc_node * cos_i
    F = -(sin_arg_peri * cos_lon_asc_node + cos_arg_peri * sin_lon_asc_node * cos_i)
    G = -(sin_arg_peri * sin_lon_asc_node - cos_arg_peri * cos_lon_asc_node * cos_i)
    return A, B, F, G
