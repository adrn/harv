"""Per-source summary statistics, computed while the data and samples are in memory.

See ``packages/hq/docs/spec.md``, "Summary statistics" and "Weighted samples".
Every statistic comes from harv's ``Samples`` methods; hq adds only the
weighted percentile and the equal-weight resample that top-K samples need.
"""

__all__ = ("summary_stats", "weighted_resample")

from typing import Any, cast

import jax
import numpy as np
from unxt import AbstractQuantity

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import Samples

_PERCENTILES = (16, 50, 84)


def _normalized_weights(samples: Samples) -> np.ndarray:
    weights = np.asarray(samples.weight, dtype=float)
    total = weights.sum()
    if not (np.isfinite(total) and total > 0):
        msg = "the samples carry no finite posterior weight to normalize"
        raise ValueError(msg)
    return weights / total


def weighted_resample(samples: Samples, key: jax.Array, n: int) -> Samples:
    """Draw ``n`` equal-weight samples, with replacement, in proportion to weight.

    Parameters
    ----------
    samples
        Weighted (top-K rejection) samples; uses ``Samples.weight``.
    key
        PRNG key.
    n
        Number of draws.

    Returns
    -------
        The resampled ``Samples``.

    Raises
    ------
    ValueError
        If the weights do not sum to a positive finite number.
    """
    weights = _normalized_weights(samples)
    index = jax.random.choice(key, len(weights), shape=(n,), replace=True, p=weights)
    return samples[np.asarray(index)]


def _weighted_percentiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Zero-weight samples carry no posterior mass; left in, they would anchor
    # the interpolation and pull percentiles toward them.
    values, weights = values[weights > 0], weights[weights > 0]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights  # midpoint rule
    return np.interp(np.asarray(_PERCENTILES) / 100, cdf, values)


def _value_and_unit(value: Any) -> tuple[np.ndarray, str]:
    if isinstance(value, AbstractQuantity):
        return np.asarray(value.value, dtype=float), str(value.unit)
    return np.asarray(value, dtype=float), ""


def summary_stats(
    samples: Samples,
    data: RVData | GaiaAstrometryData,
    *,
    resample_key: jax.Array | None = None,
    min_evidence_ess: float | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The spec's summary-statistic columns for one source.

    Parameters
    ----------
    samples
        The source's posterior samples, with ``ln_likelihood`` and
        ``ln_prior`` (for the MAP sample).
    data
        The source's data (for period unimodality and phase coverage).
    resample_key
        Pass it for weighted (top-K rejection) samples: percentiles then use
        ``Samples.weight``, and ``period_unimodal`` uses an equal-weight
        resample drawn with this key. ``None`` for equal-weight (MCMC)
        samples.
    min_evidence_ess
        When given, adds ``well_resolved`` from
        ``Samples.acceptance_diagnostics`` (rejection only).

    Returns
    -------
        The statistics as a flat dict of Python scalars, and the unit of each
        statistic that has one.
    """
    stats: dict[str, Any] = {}
    units: dict[str, str] = {}

    if min_evidence_ess is not None:
        diagnostics = samples.acceptance_diagnostics(min_evidence_ess=min_evidence_ess)
        stats["well_resolved"] = bool(diagnostics["well_resolved"])

    weighted = resample_key is not None
    if weighted:
        equal = weighted_resample(samples, resample_key, samples.n_samples)
        weights = _normalized_weights(samples)
    else:
        equal = samples
    stats["period_unimodal"] = bool(equal.period_unimodal(data))

    # map_sample() without return_index returns a Samples; harv types it as a union.
    map_sample = cast("Samples", samples.map_sample())
    stats["max_phase_gap"] = float(map_sample.max_phase_gap(data)[0])
    stats["phase_coverage"] = float(map_sample.phase_coverage(data)[0])
    stats["periods_spanned"] = float(map_sample.periods_spanned(data)[0])

    for name in samples.keys():  # noqa: SIM118 -- Samples.keys() adds derived keys
        values, unit = _value_and_unit(samples[name])
        map_value, _ = _value_and_unit(map_sample[name])
        percentiles = (
            _weighted_percentiles(values, weights)
            if weighted
            else np.percentile(values, _PERCENTILES)
        )
        columns = {f"map_{name}": map_value[0]} | {
            f"{name}_p{q}": p for q, p in zip(_PERCENTILES, percentiles, strict=True)
        }
        for column, value in columns.items():
            stats[column] = float(value)
            if unit:
                units[column] = unit
    return stats, units
