"""Process one source with the rejection sampler.

See ``packages/hq/docs/spec.md``, "``run_rejection``". The result is returned
as a plain-data :class:`~harv_hq.results.Payload` so that later phases can
hand it across processes.
"""

__all__ = ("process_rejection",)

import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import RejectionSampler
from harv_hq.config import RejectionConfig
from harv_hq.ids import source_key, stable_hash
from harv_hq.results import Payload, samples_table
from harv_hq.summary_stats import summary_stats


def process_rejection(
    source_id: Any,
    data: RVData | GaiaAstrometryData,
    *,
    sampler: RejectionSampler,
    prior_cache: Path,
    config: RejectionConfig,
    seed: int,
) -> Payload:
    """Run the rejection sampler on one source and summarize the result.

    Any exception is caught and returned as a ``failed`` payload with its
    traceback; harv's warnings (e.g. under-resolution) are recorded in the
    ``warnings`` column instead of printed.

    Parameters
    ----------
    source_id
        The source's ID.
    data
        Its prepared data.
    sampler
        The run's ``RejectionSampler``, built once per process.
    prior_cache
        Path to the shared prior cache.
    config
        The ``[rejection]`` settings.
    seed
        The run seed. The source's rejection key is split in two: one half
        drives the sampler, the other the equal-weight resample.

    Returns
    -------
        The source's payload.
    """
    started = datetime.now(UTC)
    clock = time.perf_counter()
    row: dict[str, Any] = {
        "source_id": source_id,
        "seed_hash": stable_hash(source_id),
        "n_obs": int(data.n_obs),
    }
    sampler_key, resample_key = jax.random.split(
        source_key(seed, source_id, "rejection")
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            samples = sampler.run_with_samples(
                data,
                prior_cache,
                key=sampler_key,
                top_k=config.top_k,
                ignore_non_finite=config.ignore_non_finite,
                randomize_prior_order=config.randomize_prior_order,
            )
            stats, stat_units = summary_stats(
                samples,
                data,
                weighted=True,
                resample_key=resample_key,
                min_evidence_ess=config.min_evidence_ess,
            )
            table, sample_units, structure, metadata = samples_table(
                samples, source_id, stage="rejection"
            )
        except Exception:  # noqa: BLE001 -- recorded as this source's failure
            failure = traceback.format_exc()
        else:
            failure = None

    row["warnings"] = "\n".join(str(w.message) for w in caught)
    row["started"] = started
    row["finished"] = datetime.now(UTC)
    row["wall_time_s"] = time.perf_counter() - clock
    if failure is not None:
        return Payload(row={**row, "status": "failed", "error": failure})
    return Payload(
        row={**row, "status": "ok", "error": "", **metadata, **stats},
        row_units=stat_units,
        samples=table,
        samples_units=sample_units,
        structure=structure,
        metadata_keys=tuple(metadata),
    )
