"""Helper functions shared across the likelihood modules."""

from __future__ import annotations

__all__ = ["_solve_kepler"]

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import ustrip

if TYPE_CHECKING:
    from harv.data import AbstractData
    from harv.likelihood._params import AbstractBaseKeplerParameters


def _solve_kepler(
    data: AbstractData,
    params: AbstractBaseKeplerParameters,
) -> tuple[jax.Array, jax.Array]:
    """Solve Kepler's equation; return (sin_f, cos_f)."""
    t_peri = params.phase_peri * params.period
    dt = data.time - t_peri
    M = 2 * jnp.pi * ustrip("", dt / params.period)
    return kepler(M, params.eccentricity)
