"""Helpers for Keplerian orbits."""

import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from jaxtyping import Array, Float
from unxt import ustrip

from harv.custom_types import ScalarFloat, ScalarQTime


def compute_true_anomaly_components(
    time: ScalarQTime,
    period: ScalarQTime,
    eccentricity: ScalarFloat,
    t_peri: ScalarQTime,
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
