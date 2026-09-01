"""Builders mapping a periodogram onto an interim period prior.

An **interim prior** is the prior actually used to generate a given source's
samples. Here it is periodogram-informed, and therefore *different for every
source*: mass is concentrated where that source's data say a period is
plausible, which is what buys the rejection sampler its acceptance rate. The
price is bookkeeping — population-level (hierarchical) inference must divide
each source's interim prior back out, sample by sample, which is what
:func:`attach_interim_period_prior` records.

Both builders return a `~harv.distributions.QuantityDistribution` wrapping a
:class:`~harv.stats.LogGridDensity`, which drops directly into the ``period=``
override of any ``default_prior(...)`` (or into
``HarvPrior.nonlinear_priors["period"]``).

Both also mix in a log-uniform "floor" of weight ``floor`` (λ). That keeps the
interim prior positive across the whole period domain, so no region the
population prior cares about has zero proposal density, and it bounds the
importance weights relative to a log-uniform interim prior by ``1/floor``. See
``docs/spec.md``, "Periodogram and interim period priors".

Densities here are always **per unit ln-period** (``d(ln P)``), the measure in
which a log-uniform prior is flat and which is invariant under a change of the
time unit. :class:`~harv.stats.LogGridDensity` stores its knots in that measure
too.
"""

__all__ = (
    "LN_INTERIM_PERIOD_PRIOR_KEY",
    "attach_interim_period_prior",
    "peak_period_prior",
    "tempered_period_prior",
)

import warnings

import numpy as np
import quaxed.numpy as jnp
from unxt import Q, ustrip

from harv.custom_types import ScalarQFrequency, ScalarQTime
from harv.distributions import QuantityDistribution
from harv.periodogram.core import PeriodogramResult
from harv.samplers.samples import Samples
from harv.stats import LogGridDensity

LN_INTERIM_PERIOD_PRIOR_KEY = "ln_interim_period_prior"
# Reserved Samples column name for the per-sample interim period prior
# log-density (per unit ln-period; see attach_interim_period_prior).


def _validate_floor(floor: float) -> None:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    if floor == 0.0:
        warnings.warn(
            "floor=0 removes the log-uniform mixture component; the interim "
            "prior then lacks full period support, which voids the validity "
            "guarantee for hierarchical importance reweighting.",
            UserWarning,
            stacklevel=3,
        )


def _assemble_knots(
    result: PeriodogramResult,
    period_min: ScalarQTime | None,
    period_max: ScalarQTime | None,
    unit: str | None,
) -> tuple[np.ndarray, np.ndarray, str, tuple[float, float]]:
    """Map the periodogram grid to ascending ln-period knots on the domain.

    Returns ``(ln_period, delta, unit, (ln_p_min, ln_p_max))``: the knot
    positions in ``ln(P / unit)``, the ``delta_ln_likelihood`` value at each
    knot, the time unit they are expressed in, and the domain endpoints.

    The periodogram grid is uniform in *frequency* and descending in period, so
    it is reversed here to ascend in ln-period. The requested domain must lie
    within the computed grid: the periodogram is evidence only where it was
    evaluated, so a domain reaching past the grid would be shaped by a Δ nobody
    computed. Host-side NumPy — builders run once per source, eagerly.

    Raises
    ------
    ValueError
        If ``period_max <= period_min``, or if the requested domain reaches
        outside the periodogram grid.
    """
    unit = str(result.period.unit) if unit is None else unit
    p_grid = np.asarray(ustrip(unit, result.period), dtype=np.float64)[::-1]
    delta = np.asarray(result.delta_ln_likelihood, dtype=np.float64)[::-1]
    ln_grid = np.log(p_grid)

    ln_p_min = (
        float(np.log(ustrip(unit, period_min)))
        if period_min is not None
        else ln_grid[0]
    )
    ln_p_max = (
        float(np.log(ustrip(unit, period_max)))
        if period_max is not None
        else ln_grid[-1]
    )
    if ln_p_max <= ln_p_min:
        raise ValueError("period_max must be greater than period_min")

    # The domain must sit inside the grid, but the natural call passes the same
    # period_min/period_max that built the grid, and those round-trip through
    # 1/f and log() -- landing within ~1e-7 of the request in float32, exactly
    # on it in x64. So allow a slack far above that round-trip yet far below any
    # real overreach (which misses by O(0.1-10) in ln-period), then clip.
    tol = 1e-6
    if ln_p_min < ln_grid[0] - tol or ln_p_max > ln_grid[-1] + tol:
        raise ValueError(
            f"The requested prior domain "
            f"[{np.exp(ln_p_min):g}, {np.exp(ln_p_max):g}] {unit} reaches outside "
            f"the periodogram grid "
            f"[{np.exp(ln_grid[0]):g}, {np.exp(ln_grid[-1]):g}] {unit}, where the "
            "periodogram provides no evidence. Recompute the periodogram over the "
            "wider range, or narrow period_min/period_max to the grid."
        )
    ln_p_min = max(ln_p_min, ln_grid[0])
    ln_p_max = min(ln_p_max, ln_grid[-1])

    # Endpoint knots are interpolated onto the domain edges; every other knot is
    # a grid point. Both endpoints are inside the grid, so no extrapolation. The
    # same tolerance excludes a grid knot sitting within a hair of an endpoint,
    # which would otherwise duplicate it and break strict monotonicity.
    interior = (ln_grid > ln_p_min + tol) & (ln_grid < ln_p_max - tol)
    knots = np.concatenate([[ln_p_min], ln_grid[interior], [ln_p_max]])
    vals = np.concatenate(
        [
            [np.interp(ln_p_min, ln_grid, delta)],
            delta[interior],
            [np.interp(ln_p_max, ln_grid, delta)],
        ]
    )
    return knots, vals, unit, (ln_p_min, ln_p_max)


def _to_prior(
    ln_period: np.ndarray, density: np.ndarray, unit: str
) -> QuantityDistribution:
    """Wrap knots and a density per unit ln-period as a period prior."""
    with np.errstate(divide="ignore"):  # density == 0 -> log_density == -inf is valid
        log_density = np.log(density)
    return QuantityDistribution(
        LogGridDensity(jnp.asarray(ln_period), jnp.asarray(log_density)), unit
    )


def tempered_period_prior(
    result: PeriodogramResult,
    *,
    beta: float = 1.0,
    floor: float = 0.1,
    period_min: ScalarQTime | None = None,
    period_max: ScalarQTime | None = None,
    unit: str | None = None,
) -> QuantityDistribution:
    """Interim period prior from the tempered periodogram.

    The density per unit ln-period is
    ``(1 - floor) * exp(beta * delta_ln_likelihood) / Z + floor * log-uniform``.
    ``beta = 0`` reduces to an exact log-uniform prior on
    ``[period_min, period_max]``;
    ``beta = 1`` treats the periodogram as a likelihood times log-uniform.

    Parameters
    ----------
    result
        Output of :func:`~harv.periodogram.periodogram`.
    beta
        Tempering exponent (>= 0). Smaller values are more amplitude-agnostic.
    floor
        Weight λ of the log-uniform mixture component (support guarantee).
    period_min, period_max
        Domain of the prior. Defaults to the periodogram grid range, and must
        lie within it — the periodogram is evidence only where it was
        evaluated. Narrow the domain to focus the prior; to widen it, recompute
        the periodogram over the wider range.
    unit
        Time unit of the returned prior. Defaults to the periodogram's unit.

    Examples
    --------
    >>> from unxt import Q
    >>> import harv.models as hm
    >>> import harv.periodogram as hp
    >>> from harv.simulate import simulate_rv_sb1_data
    >>> data, _ = simulate_rv_sb1_data(seed=1, n_obs=40, period=Q(30.0, "day"))
    >>> fourier_prior = hm.FourierRV(n_terms=2).default_prior(
    ...     period_min=Q(5.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_amp=Q(30.0, "km/s"),
    ...     sigma_v0=Q(10.0, "km/s"),
    ... )
    >>> result = hp.periodogram(data, prior=fourier_prior, period_min=Q(5.0, "day"))
    >>> prior = hp.tempered_period_prior(result, beta=1.0, floor=0.2)
    >>> str(prior.unit)
    'd'
    """
    _validate_floor(floor)
    if beta < 0:
        raise ValueError("beta must be non-negative")

    ln_period, delta, unit, (ln_p_min, ln_p_max) = _assemble_knots(
        result, period_min, period_max, unit
    )
    w = np.exp(beta * (delta - np.max(delta)))
    density = (1.0 - floor) * w / np.trapezoid(w, ln_period) + floor / (
        ln_p_max - ln_p_min
    )
    return _to_prior(ln_period, density, unit)


def _select_peaks(
    ln_period: np.ndarray,
    delta: np.ndarray,
    height_drop: float,
    peak_width: float,
    max_peaks: int,
) -> np.ndarray:
    """Strict local maxima within ``height_drop`` of the best, after suppression.

    A candidate is a strict local maximum whose ``delta_ln_likelihood`` is
    within ``height_drop`` (nats) of the global maximum — a *relative*
    criterion, so it is scale-invariant across data types (RV periodograms
    reach hundreds of nats; astrometry, where the orbit is a small
    perturbation on the marginalized 5-parameter astrometric signal, only a
    few). Real periodograms carry many spurious local maxima riding on alias
    and noise structure; candidates within one ``peak_width`` (in frequency) of
    a stronger kept peak are suppressed, and at most the ``max_peaks``
    strongest survivors are returned. The global maximum always qualifies, so
    at least one peak is always found (unless the periodogram is perfectly
    flat).
    """
    cut = float(np.max(delta)) - height_drop
    is_peak = (
        (delta[1:-1] > delta[:-2]) & (delta[1:-1] > delta[2:]) & (delta[1:-1] >= cut)
    )
    candidates = np.arange(1, ln_period.shape[0] - 1)[is_peak]
    kept: list[int] = []
    frequency = np.exp(-ln_period)
    for i in candidates[np.argsort(delta[candidates])[::-1]]:
        if all(abs(frequency[i] - frequency[j]) > peak_width for j in kept):
            kept.append(int(i))
        if len(kept) == max_peaks:
            break
    return np.asarray(sorted(kept), dtype=int)


def peak_period_prior(
    result: PeriodogramResult,
    *,
    height_drop: float = 10.0,
    max_peaks: int = 8,
    peak_width: ScalarQFrequency | None = None,
    floor: float = 0.1,
    period_min: ScalarQTime | None = None,
    period_max: ScalarQTime | None = None,
    unit: str | None = None,
) -> QuantityDistribution:
    """Interim period prior from periodogram peaks, equal weights.

    Strict local maxima of ``delta_ln_likelihood`` within ``height_drop`` nats
    of the global maximum each receive a top-hat in ln-period of full frequency
    width ``peak_width`` (default ``1/t_span``, the natural periodogram peak
    width) and **equal mass** ``1/n_peaks`` regardless of peak amplitude — the
    amplitude-agnostic alternative to :func:`tempered_period_prior`. Candidate
    maxima within one peak width of a stronger peak are suppressed, and at most
    the ``max_peaks`` strongest peaks are kept (this bounds the mass dilution:
    each kept peak carries at least ``(1 - floor) / max_peaks``). The peak
    mixture is combined with a log-uniform floor of weight ``floor``.

    The ``height_drop`` criterion is *relative* to the best peak, so it works
    across data types without tuning: RV periodograms span hundreds of nats,
    while astrometry periodograms — where the orbit is a small perturbation on
    the marginalized astrometric signal — span only a few.

    Parameters
    ----------
    result
        Output of :func:`~harv.periodogram.periodogram`.
    height_drop
        A local maximum counts as a peak if its ``delta_ln_likelihood`` is at
        least ``max(delta_ln_likelihood) - height_drop`` (a log-likelihood
        ratio relative to the best peak, in nats). Larger values admit weaker
        secondary peaks / aliases.
    max_peaks
        Maximum number of peaks kept (strongest first, after suppression).
    peak_width
        Full width of each peak's top-hat, in frequency units.
    floor, period_min, period_max, unit
        As in :func:`tempered_period_prior`.
    """
    _validate_floor(floor)
    if max_peaks < 1:
        raise ValueError("max_peaks must be at least 1")
    if height_drop <= 0:
        raise ValueError("height_drop must be positive")
    ln_period, delta, unit, (ln_p_min, ln_p_max) = _assemble_knots(
        result, period_min, period_max, unit
    )

    if peak_width is None:
        width = 1.0 / float(ustrip(unit, result.t_span))
    else:
        width = float(ustrip(f"1/({unit})", peak_width))

    peak_idx = _select_peaks(ln_period, delta, height_drop, width, max_peaks)

    if peak_idx.size == 0:
        warnings.warn(
            "The periodogram has no interior local maximum (it is flat or "
            "monotonic); falling back to a pure log-uniform interim period "
            "prior.",
            UserWarning,
            stacklevel=2,
        )
        peak_density = np.full_like(ln_period, 1.0 / (ln_p_max - ln_p_min))
    else:
        peak_density = np.zeros_like(ln_period)
        for i in peak_idx:
            ln_p_peak = ln_period[i]
            # |d ln P| = |df| / f: a full width `width` in frequency at
            # f_peak = exp(-ln_p_peak) is a half-width (width/2) * exp(ln_p_peak)
            # in ln-period. Clamp to the local knot spacing so every top-hat
            # covers at least one segment.
            half_width = 0.5 * width * np.exp(ln_p_peak)
            half_width = max(
                half_width,
                ln_period[i] - ln_period[i - 1],
                ln_period[i + 1] - ln_period[i],
            )
            peak_density += np.where(
                np.abs(ln_period - ln_p_peak) <= half_width,
                1.0 / (2.0 * half_width),
                0.0,
            )
        peak_density /= peak_idx.size

    density = (1.0 - floor) * peak_density + floor / (ln_p_max - ln_p_min)
    return _to_prior(ln_period, density, unit)


def attach_interim_period_prior(
    samples: Samples,
    period_prior: QuantityDistribution,
    *,
    name: str = LN_INTERIM_PERIOD_PRIOR_KEY,
) -> Samples:
    """Record each sample's interim period prior log-density on ``samples``.

    The *interim prior* is the period prior these samples were actually drawn
    under. When it is periodogram-informed it differs from source to source, so
    population-level (hierarchical) inference has to divide it back out per
    sample: the per-source estimator is importance sampling with the interim
    prior as its proposal. This function evaluates that proposal density at
    every retained sample and stores it, so the population step can just read
    the column.

    The stored value is the log-density **per unit ln-period**,
    ``period_prior.log_prob(P) + ln(P / unit)``, which is invariant under a
    change of the prior's time unit. Population densities must be converted to
    the same measure before forming weight ratios: a density per unit ``log10``
    period adds ``ln(ln 10)``; a density per unit period in unit ``u``
    subtracts ``ln(P / u)``.

    The column is added to ``samples.nonlinear`` as a dimensionless extra
    parameter, so it flows through ``pad_and_stack_samples`` and
    ``to_hdf5``/``from_hdf5`` unchanged.

    Works with any scalar-unit period prior (e.g. ``QD(LogUniform, "day")``),
    not only the grid priors built here — the classic shared-prior case is just
    the special case where every source has the same interim prior. Returns a
    new ``Samples``; the input is unchanged.
    """
    unit = period_prior.unit
    if not isinstance(unit, str):
        raise TypeError("period_prior must have a single scalar unit")
    p = ustrip(unit, samples["period"])
    ln_interim = period_prior.distribution.log_prob(p) + jnp.log(p)
    return Samples(
        nonlinear={**samples.nonlinear, name: Q(ln_interim, "")},
        linear=samples.linear,
        data_type=samples.data_type,
        metadata=samples.metadata,
        linear_extension_names=samples.linear_extension_names,
        ln_likelihood=samples.ln_likelihood,
        ln_prior=samples.ln_prior,
    )
