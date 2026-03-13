"""Helpers for Keplerian orbits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from unxt import Quantity, ustrip

if TYPE_CHECKING:
    from jaxtyping import Array, Float

    from harv.custom_types import Time


def compute_true_anomaly_components(
    time: Quantity[Time],
    period: Quantity[Time],
    eccentricity: Float[Array, ""],
    t_peri: Quantity[Time],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute true anomaly at given times.

    Parameters
    ----------
    time
        Observation times, shape (n,)
    period
        Orbital period
    eccentricity
        Orbital eccentricity
    t_peri
        Time of pericenter passage

    Returns
    -------
    sin_f, cos_f
        True anomaly components, each shape (n,)
    """
    M = ustrip("", 2 * jnp.pi * (time - t_peri) / period)
    return kepler(M, eccentricity)
