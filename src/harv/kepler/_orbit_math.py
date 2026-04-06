"""Low-level orbit math building blocks.

Pure functions on raw JAX arrays — no Quantity, no eqx.Module. These are the
canonical implementations of the core orbit computations used by ``harv.kepler``,
``harv.likelihood``, and ``harv.simulate``. Higher-level wrappers handle unit
stripping and Quantity construction.

All inputs and outputs are dimensionless JAX arrays (or Python floats that JAX
will promote). Callers must ensure consistent units before stripping.
"""

__all__ = (
    "mean_anomaly",
    "rv_shape",
    "thiele_innes_ABFG",
    "true_anomaly_from_mean",
)

from typing import Any

import jax.numpy as jnp
from jaxoplanet.core.kepler import kepler

#: Permissive array-like type so callers don't need explicit casts from ustrip etc.
ArrayLike = Any


def mean_anomaly(dt: ArrayLike, period: ArrayLike) -> Any:
    """Compute mean anomaly from elapsed time and period.

    ``M = 2π · dt / period``. Inputs must be in consistent units (both in days,
    both in years, etc.) — the ratio is dimensionless.
    """
    return 2 * jnp.pi * dt / period


def true_anomaly_from_mean(M: ArrayLike, eccentricity: ArrayLike) -> tuple[Any, Any]:
    """Solve Kepler's equation: mean anomaly → (sin f, cos f).

    Wraps ``jaxoplanet.core.kepler.kepler``.
    """
    return kepler(M, eccentricity)


def rv_shape(
    sin_f: ArrayLike,
    cos_f: ArrayLike,
    eccentricity: ArrayLike,
    arg_peri: ArrayLike,
) -> Any:
    """RV shape function: cos(ω + f) + e·cos(ω).

    Returns the dimensionless RV amplitude factor for each observation.
    ``arg_peri`` is in radians.
    """
    cos_wf = jnp.cos(arg_peri) * cos_f - jnp.sin(arg_peri) * sin_f
    return cos_wf + eccentricity * jnp.cos(arg_peri)


def thiele_innes_ABFG(
    cos_arg_peri: ArrayLike,
    sin_arg_peri: ArrayLike,
    cos_lon_asc_node: ArrayLike,
    sin_lon_asc_node: ArrayLike,
    cos_i: ArrayLike,
) -> tuple[Any, Any, Any, Any]:
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
