"""Helper for Keplerian orbits."""

from jaxtyping import Array, Float

from harv.custom_types import ScalarFloat, ScalarQTime
from harv.kepler._orbit_math import mean_anomaly, true_anomaly_from_mean


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
    M = mean_anomaly(time - t_peri, period)
    return true_anomaly_from_mean(M, eccentricity)
