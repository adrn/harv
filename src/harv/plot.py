"""Plotting utilities."""

__all__ = ("get_t_grid",)

import numpy as np
from unxt import Quantity, ustrip

from .custom_types import BatchQTime


def get_t_grid(
    times: BatchQTime,
    period: Quantity["time"],
    *,
    span_factor: float = 0.1,
    n_points_per_period: int = 64,
    max_t_grid: int | None = None,
) -> Quantity["time"]:
    """Dense time grid spanning the observation baseline with a small buffer.

    Generates a regular grid of times suitable for plotting model orbits over
    data. The grid resolution adapts to the orbital period so that fast orbits
    are well-resolved while long-period orbits don't create excessive grids.

    Parameters
    ----------
    times : Quantity["time"]
        Observation times.
    period : Quantity["time"]
        Orbital period (scalar).  Used to set the grid spacing as
        ``period / n_points_per_period``.
    span_factor : float, optional
        Fractional buffer added to each side of the observation baseline.
        Default: 0.1 (10 % on each side).
    n_points_per_period : int, optional
        Number of grid points per orbital period.  Default: 64.
    max_t_grid : int or None, optional
        Maximum number of grid points.  If the computed grid would exceed
        this, the spacing is coarsened.

    Returns
    -------
    t_grid : Quantity["time"]
        Regular time grid spanning the buffered observation range.

    Examples
    --------
    >>> from unxt import Quantity
    >>> times = Quantity([0.0, 50.0, 100.0], "day")
    >>> t_grid = get_t_grid(times, Quantity(30.0, "day"))
    >>> len(t_grid) > 0
    True
    """
    time_unit = str(times.unit)
    t_vals = np.asarray(times.value)
    t_min, t_max = float(t_vals.min()), float(t_vals.max())
    w = t_max - t_min

    p_val = float(ustrip(time_unit, period))
    dt = p_val / n_points_per_period

    n_grid = w / dt if dt > 0 else 1
    if max_t_grid is not None and n_grid > max_t_grid:
        dt = w / max_t_grid

    grid = np.arange(
        t_min - w * span_factor / 2,
        t_max + w * span_factor / 2 + dt,
        dt,
    )
    return Quantity(grid, time_unit)
