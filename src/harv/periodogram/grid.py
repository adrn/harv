"""Frequency-grid construction for periodograms.

TODO: to review

See ``docs/spec.md``, "Periodogram and interim period priors".
"""

__all__ = ("frequency_grid",)

import math

import quaxed.numpy as jnp
from unxt import Q, ustrip

from harv.custom_types import NFrequency, ScalarQTime
from harv.data.containers import AbstractDatasetContainer
from harv.data.datasets import AbstractData


def _data_t_span(data: AbstractData | AbstractDatasetContainer, unit: str) -> float:
    """Total time baseline spanned by all observations, in ``unit``."""
    datasets = (
        list(data.values()) if isinstance(data, AbstractDatasetContainer) else [data]
    )
    t_min = min(float(jnp.min(ustrip(unit, d.time))) for d in datasets)
    t_max = max(float(jnp.max(ustrip(unit, d.time))) for d in datasets)
    return t_max - t_min


def frequency_grid(
    data: AbstractData | AbstractDatasetContainer | None = None,
    *,
    period_min: ScalarQTime,
    period_max: ScalarQTime | None = None,
    t_span: ScalarQTime | None = None,
    samples_per_peak: int = 5,
    max_period_factor: float = 1.0,
    n_grid: int | None = None,
) -> NFrequency:
    """Build a frequency grid, uniform in frequency, for a periodogram.

    The grid spans ``[1/period_max, 1/period_min]`` with spacing
    ``1 / (samples_per_peak * t_span)`` (the natural periodogram peak width is
    ``1/t_span`` in frequency), unless ``n_grid`` is given explicitly.

    Exactly one of ``data`` or ``t_span`` must be provided; for dataset
    containers the baseline spans all contained datasets.

    .. note:: To guarantee an identical prior pytree structure across a
       population of sources (so the sampler JIT-compiles once), pass the same
       ``period_min``, ``period_max``, and ``n_grid`` for every source instead
       of deriving the grid size from each source's baseline.

    Parameters
    ----------
    data
        Observations used to derive the time baseline. Mutually exclusive with
        ``t_span``.
    period_min
        Shortest trial period (sets the highest frequency). Its unit defines
        the unit of the returned grid (``1/unit``).
    period_max
        Longest trial period. Defaults to ``max_period_factor * t_span``.
    t_span
        Time baseline. Mutually exclusive with ``data``.
    samples_per_peak
        Grid oversampling factor per periodogram peak width. Default: 5.
    max_period_factor
        Sets the default ``period_max`` as a multiple of ``t_span``.
    n_grid
        Explicit number of grid points, overriding the spacing rule.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.periodogram import frequency_grid
    >>> f = frequency_grid(
    ...     t_span=Q(1000.0, "day"), period_min=Q(10.0, "day"), n_grid=101
    ... )
    >>> f.shape, str(f.unit)
    ((101,), '1 / d')
    """
    if (data is None) == (t_span is None):
        raise TypeError("Exactly one of data or t_span must be provided")

    unit = str(period_min.unit)
    p_min = float(ustrip(unit, period_min))
    if p_min <= 0:
        raise ValueError("period_min must be positive")

    span = (
        _data_t_span(data, unit)
        if data is not None
        # t_span is not None here -- guaranteed by the exactly-one check above:
        else float(ustrip(unit, t_span))  # ty: ignore[no-matching-overload]
    )
    if span <= 0:
        raise ValueError("The data time baseline (t_span) must be positive")

    p_max = (
        float(ustrip(unit, period_max))
        if period_max is not None
        else max_period_factor * span
    )
    if p_max <= p_min:
        raise ValueError("period_max must be greater than period_min")

    f_min = 1.0 / p_max
    f_max = 1.0 / p_min
    if n_grid is None:
        df = 1.0 / (samples_per_peak * span)
        n_grid = math.ceil((f_max - f_min) / df) + 1
    if n_grid < 2:
        raise ValueError("The frequency grid must have at least 2 points")

    return Q(jnp.linspace(f_min, f_max, n_grid), f"1/({unit})")
