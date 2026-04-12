"""Plotting utilities."""

__all__ = ("get_t_grid",)

from typing import Any

import numpy as np
from unxt import Q, ustrip

from .custom_types import BatchQTime

_DEFAULT_ERRORBAR_STYLE: dict[str, Any] = {
    "linestyle": "none",
    "marker": "o",
    "markersize": 4.0,
    "elinewidth": 1.0,
    "capsize": 0,
    "color": "k",
    "ecolor": "#666666",
    "zorder": 10,
}


def _plot_timeseries_errorbar(
    ax: Any,
    time: Any,
    obs: Any,
    obs_err: Any,
    *,
    time_unit: str,
    obs_unit: str,
    t_ref: Any | None = None,
    relative_to_t_ref: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_labels: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot observation vs time as error bars (internal helper).

    Parameters
    ----------
    ax
        Matplotlib axes to draw on.
    time, obs, obs_err
        Q arrays for time, observation, and uncertainty.
    time_unit, obs_unit
        Unit strings for axes.
    t_ref
        Reference epoch (Q or None).
    relative_to_t_ref
        Subtract ``t_ref`` from times before plotting.
    xlabel, ylabel
        Axis label overrides.
    add_labels
        Whether to set axis labels.
    **kwargs
        Forwarded to ``ax.errorbar()``, overriding defaults.
    """
    t = np.asarray(ustrip(time_unit, time))
    if relative_to_t_ref and t_ref is not None:
        t = t - float(ustrip(time_unit, t_ref))

    style = {**_DEFAULT_ERRORBAR_STYLE, **kwargs}

    ax.errorbar(
        t,
        np.asarray(ustrip(obs_unit, obs)),
        yerr=np.asarray(ustrip(obs_unit, obs_err)),
        **style,
    )

    if add_labels:
        if xlabel is None:
            xlabel = (
                f"Time $-$ t_ref [{time_unit}]"
                if relative_to_t_ref
                else f"Time [{time_unit}]"
            )
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    return ax


def get_t_grid(
    times: BatchQTime,
    period: Q["time"],
    *,
    span_factor: float = 0.1,
    n_points_per_period: int = 64,
    max_t_grid: int | None = None,
) -> Q["time"]:
    """Dense time grid spanning the observation baseline with a small buffer.

    Generates a regular grid of times suitable for plotting model orbits over
    data. The grid resolution adapts to the orbital period so that fast orbits
    are well-resolved while long-period orbits don't create excessive grids.

    Parameters
    ----------
    times : Q["time"]
        Observation times.
    period : Q["time"]
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
    t_grid : Q["time"]
        Regular time grid spanning the buffered observation range.

    Examples
    --------
    >>> from unxt import Q
    >>> times = Q([0.0, 50.0, 100.0], "day")
    >>> t_grid = get_t_grid(times, Q(30.0, "day"))
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
    return Q(grid, time_unit)
