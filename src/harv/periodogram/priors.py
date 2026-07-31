"""Builders mapping a periodogram onto an interim period prior.

TODO: to review

Both builders return a `~harv.distributions.QuantityDistribution` wrapping a
:class:`~harv.periodogram.LogGridDensity`, which drops directly into the
``period=`` override of any ``default_prior(...)`` (or into
``HarvPrior.nonlinear_priors["period"]``).

For probabilistic rigor in downstream hierarchical reweighting, both builders
mix in a log-uniform "floor" of weight ``floor`` (λ): the interim prior then
has full support on ``[period_min, period_max]`` and the importance weights
relative to a log-uniform interim prior are bounded by ``1/floor``. See
``docs/spec.md``, "Periodogram and interim period priors".
"""

__all__ = (
    "LN_PINT_PERIOD_KEY",
    "attach_ln_pint",
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
from harv.periodogram.distribution import LogGridDensity
from harv.samplers.samples import Samples

LN_PINT_PERIOD_KEY = "ln_pint_period"
# Reserved Samples column name for the per-sample interim period prior
# log-density (per unit ln-period; see attach_ln_pint).

_EDGE_EPS_FACTOR = 1e-3


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

    Returns ``(u_knots, delta_knots, unit, (u_lo, u_hi))``. Where the
    requested domain extends beyond the computed grid, the delta-ln-likelihood values
    continue flat at 0 (a short transition knot is inserted at the grid edge),
    so a tempered prior is floor-like there. Host-side NumPy — builders run
    once per source, eagerly.
    """
    unit = str(result.period.unit) if unit is None else unit
    p_grid = np.asarray(ustrip(unit, result.period), dtype=np.float64)[::-1]
    delta = np.asarray(result.delta_ln_likelihood, dtype=np.float64)[::-1]
    u = np.log(p_grid)

    u_lo = float(np.log(ustrip(unit, period_min))) if period_min is not None else u[0]
    u_hi = float(np.log(ustrip(unit, period_max))) if period_max is not None else u[-1]
    if u_hi <= u_lo:
        raise ValueError("period_max must be greater than period_min")

    eps = _EDGE_EPS_FACTOR * float(np.min(np.diff(u)))

    knots = [np.array([u_lo])]
    vals = [np.array([np.interp(u_lo, u, delta, left=0.0, right=0.0)])]
    if u_lo < u[0] - 2 * eps:
        # Flat (Δ = 0) extension below the grid, transitioning at the edge:
        knots.append(np.array([u[0] - eps]))
        vals.append(np.array([0.0]))
    interior = (u > u_lo) & (u < u_hi)
    knots.append(u[interior])
    vals.append(delta[interior])
    if u_hi > u[-1] + 2 * eps:
        knots.append(np.array([u[-1] + eps]))
        vals.append(np.array([0.0]))
    knots.append(np.array([u_hi]))
    vals.append(np.array([np.interp(u_hi, u, delta, left=0.0, right=0.0)]))
    return np.concatenate(knots), np.concatenate(vals), unit, (u_lo, u_hi)


def _to_prior(u: np.ndarray, rho: np.ndarray, unit: str) -> QuantityDistribution:
    with np.errstate(divide="ignore"):  # rho == 0 -> log_density == -inf is valid
        log_density = np.log(rho)
    return QuantityDistribution(
        LogGridDensity(jnp.asarray(u), jnp.asarray(log_density)), unit
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
        Domain of the prior. Defaults to the periodogram grid range; may
        extend beyond it (delta-ln-likelihood continues flat at 0 there).
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

    u, delta, unit, (u_lo, u_hi) = _assemble_knots(result, period_min, period_max, unit)
    w = np.exp(beta * (delta - np.max(delta)))
    rho = (1.0 - floor) * w / np.trapezoid(w, u) + floor / (u_hi - u_lo)
    return _to_prior(u, rho, unit)


def _select_peaks(
    u: np.ndarray,
    delta: np.ndarray,
    height_drop: float,
    pw: float,
    max_peaks: int,
) -> np.ndarray:
    """Strict local maxima within ``height_drop`` of the best, after suppression.

    A candidate is a strict local maximum whose ``delta_ln_likelihood`` is
    within ``height_drop`` (nats) of the global maximum — a *relative*
    criterion, so it is scale-invariant across data types (RV periodograms
    reach hundreds of nats; astrometry, where the orbit is a small
    perturbation on the marginalized 5-parameter astrometric signal, only a
    few). Real periodograms carry many spurious local maxima riding on alias
    and noise structure; candidates within one peak width (``pw``, in
    frequency) of a stronger kept peak are suppressed, and at most the
    ``max_peaks`` strongest survivors are returned. The global maximum always
    qualifies, so at least one peak is always found (unless the periodogram is
    perfectly flat).
    """
    cut = float(np.max(delta)) - height_drop
    is_peak = (
        (delta[1:-1] > delta[:-2]) & (delta[1:-1] > delta[2:]) & (delta[1:-1] >= cut)
    )
    candidates = np.arange(1, u.shape[0] - 1)[is_peak]
    kept: list[int] = []
    f = np.exp(-u)
    for i in candidates[np.argsort(delta[candidates])[::-1]]:
        if all(abs(f[i] - f[j]) > pw for j in kept):
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
    u, delta, unit, (u_lo, u_hi) = _assemble_knots(result, period_min, period_max, unit)

    if peak_width is None:
        pw = 1.0 / float(ustrip(unit, result.t_span))
    else:
        pw = float(ustrip(f"1/({unit})", peak_width))

    peak_idx = _select_peaks(u, delta, height_drop, pw, max_peaks)

    if peak_idx.size == 0:
        warnings.warn(
            "The periodogram has no interior local maximum (it is flat or "
            "monotonic); falling back to a pure log-uniform interim period "
            "prior.",
            UserWarning,
            stacklevel=2,
        )
        rho_peaks = np.full_like(u, 1.0 / (u_hi - u_lo))
    else:
        rho_peaks = np.zeros_like(u)
        for i in peak_idx:
            u_p = u[i]
            # |du| = |df| / f: a full width pw in frequency at f_p = exp(-u_p)
            # is a half-width (pw/2) * exp(u_p) in ln-period. Clamp to the
            # local knot spacing so every top-hat covers at least one segment.
            h = 0.5 * pw * np.exp(u_p)
            h = max(h, u[i] - u[i - 1], u[i + 1] - u[i])
            rho_peaks += np.where(np.abs(u - u_p) <= h, 1.0 / (2.0 * h), 0.0)
        rho_peaks /= peak_idx.size

    rho = (1.0 - floor) * rho_peaks + floor / (u_hi - u_lo)
    return _to_prior(u, rho, unit)


def attach_ln_pint(
    samples: Samples,
    period_prior: QuantityDistribution,
    *,
    name: str = LN_PINT_PERIOD_KEY,
) -> Samples:
    """Attach the per-sample interim period prior log-density to ``samples``.

    The stored value is the log-density **per unit ln-period**,
    ``period_prior.log_prob(P) + ln(P / unit)``, which is invariant under a
    change of the prior's time unit. (Density per unit ``log10`` period adds
    ``ln(ln 10)``; density per unit period in unit ``u`` subtracts
    ``ln(P / u)``.)

    The column is added to ``samples.nonlinear`` as a dimensionless extra
    parameter, so it flows through ``pad_and_stack_samples`` and
    ``to_hdf5``/``from_hdf5`` unchanged — exactly what the population-level
    reweighting step needs to divide out per-source interim priors.

    Works with any scalar-unit period prior (e.g. ``QD(LogUniform, "day")``),
    not only the grid priors built here. Returns a new ``Samples``; the input
    is unchanged.
    """
    unit = period_prior.unit
    if not isinstance(unit, str):
        raise TypeError("period_prior must have a single scalar unit")
    p = ustrip(unit, samples["period"])
    ln_pint = period_prior.distribution.log_prob(p) + jnp.log(p)
    return Samples(
        nonlinear={**samples.nonlinear, name: Q(ln_pint, "")},
        linear=samples.linear,
        data_type=samples.data_type,
        metadata=samples.metadata,
        linear_extension_names=samples.linear_extension_names,
        ln_likelihood=samples.ln_likelihood,
        ln_prior=samples.ln_prior,
    )
