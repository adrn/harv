"""Compaction: merge a stage's result parts into a few large ones.

See ``packages/hq/docs/spec.md``, "Compaction". Compaction keeps the newest
row per source, writes it through the normal part protocol, verifies the new
parts, and only then deletes its inputs. It needs no lock: a crash at any step
leaves a readable (possibly duplicated) stage, and rerunning finishes the job.
"""

__all__ = ("CompactReport", "compact")

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harv_hq._parquet_io import (
    field_units,
    groups_spanning,
    read_metadata,
    row_group_starts,
)
from harv_hq.results import (
    SAMPLES_SUFFIX,
    SOURCES_SUFFIX,
    Payload,
    ResultRecord,
    ResultsIndex,
    read_source_rows,
    write_part,
)

logger = logging.getLogger(__name__)

COMPACTED_PREFIX = "c-"
# Assigned again by write_part, so dropped from the rows carried over.
_ASSIGNED = ("samples_row_start", "n_samples")


@final
@dataclass(frozen=True)
class CompactReport:
    """What one compaction did."""

    n_input_parts: int
    n_sources: int
    n_dropped_rows: int
    n_output_parts: int


def compact(
    stage_dir: Path,
    *,
    stage: str,
    expected_provenance: dict[str, Any] | None,
    compact_n_sources: int,
) -> CompactReport:
    """Rewrite a stage's parts as parts holding only the newest row per source.

    Parameters
    ----------
    stage_dir
        ``results/<stage>``.
    stage
        ``"rejection"`` or ``"mcmc"``.
    expected_provenance
        The current provenance; every input part must match it (``None``:
        not checked).
    compact_n_sources
        Sources per output part.

    Returns
    -------
        Counts of input parts, sources kept, superseded rows dropped, and
        output parts. All zero when there is nothing to do.

    Raises
    ------
    ProvenanceError
        If an input part was built from different inputs.
    ValueError
        If input parts hold ``Samples`` of different structure.
    RuntimeError
        If the new parts do not hold exactly the selected rows (nothing is
        deleted then).
    """
    # 1. Snapshot: only parts committed now are inputs.
    snapshot = sorted(Path(stage_dir).glob(f"*{SOURCES_SUFFIX}"))
    # 2-3. Check provenance and structure, and select the newest row per source.
    index = ResultsIndex.build(
        stage_dir, expected_provenance=expected_provenance, paths=snapshot
    )
    already = all(p.name.startswith(COMPACTED_PREFIX) for p in snapshot)
    if not snapshot or (already and index.n_rows == len(index.records)):
        return CompactReport(0, 0, 0, 0)
    selected_ids = sorted(index.records)
    selected = [index.records[sid] for sid in selected_ids]
    metadata = {
        "provenance": expected_provenance,
        "stage": stage,
        "samples_metadata_keys": _metadata_keys(snapshot),
        **(index.structure or {}),
    }

    # 4. Write blocks of compact_n_sources, in source_id order.
    run_id = uuid.uuid4().hex[:8]
    outputs = []
    for k, start in enumerate(range(0, len(selected), compact_n_sources)):
        block = selected[start : start + compact_n_sources]
        stem = f"{COMPACTED_PREFIX}{run_id}-{k:04d}"
        outputs.append(write_part(stage_dir, stem, _payloads(block), metadata))

    # 5. Verify, then 6. delete the inputs.
    _verify(outputs, {(sid, index.records[sid].finished) for sid in selected_ids})
    _delete_inputs(snapshot)
    report = CompactReport(
        n_input_parts=len(snapshot),
        n_sources=len(selected),
        n_dropped_rows=index.n_rows - len(selected),
        n_output_parts=len(outputs),
    )
    logger.info("%s: compacted %s", stage, report)
    return report


def _metadata_keys(paths: list[Path]) -> list[str]:
    """The union of the inputs' ``samples_metadata_keys``, in first-seen order."""
    keys: list[str] = []
    for path in paths:
        for key in read_metadata(path).get("samples_metadata_keys", []):
            if key not in keys:
                keys.append(key)
    return keys


def _payloads(block: list[ResultRecord]) -> list[Payload]:
    """The block's sources rows and sample ranges, as payloads to rewrite."""
    rows, row_units = read_source_rows(block)
    rows = rows.drop_columns([c for c in _ASSIGNED if c in rows.column_names])
    samples, samples_units = _read_sample_ranges(block)
    return [
        Payload(
            row={k: v for k, v in row.items() if v is not None},
            row_units=row_units,
            samples=samples[i],
            samples_units=samples_units,
        )
        for i, row in enumerate(rows.to_pylist())
    ]


def _read_sample_ranges(
    block: list[ResultRecord],
) -> tuple[list[pa.Table | None], dict[str, str]]:
    """Each record's samples rows, decoding each input's needed row groups once."""
    by_file: dict[Path, list[int]] = {}
    for position, record in enumerate(block):
        if record.n_samples:
            by_file.setdefault(record.samples_path, []).append(position)
    samples: list[pa.Table | None] = [None] * len(block)
    units: dict[str, str] = {}
    for path, positions in by_file.items():
        parquet_file = pq.ParquetFile(path)
        starts = row_group_starts(parquet_file)
        groups = sorted(
            {
                group
                for p in positions
                for group in groups_spanning(
                    starts, block[p].samples_row_start, block[p].n_samples
                )
            }
        )
        table = parquet_file.read_row_groups(groups)
        # Where each group read begins within ``table``.
        sizes = np.diff(starts)[groups]
        offset = dict(zip(groups, np.cumsum(sizes) - sizes, strict=True))
        for p in positions:
            record = block[p]
            group = int(np.searchsorted(starts, record.samples_row_start, "right")) - 1
            first = offset[group] + record.samples_row_start - int(starts[group])
            samples[p] = table.slice(int(first), record.n_samples)
        units.update(field_units(parquet_file.schema_arrow))
    return samples, units


def _verify(outputs: list[Path], wanted: set[tuple[Any, Any]]) -> None:
    """Raise unless the new parts hold exactly the selected rows, once each."""
    written = [
        (row["source_id"], row["finished"])
        for path in outputs
        for row in pq.read_table(path, columns=["source_id", "finished"]).to_pylist()
    ]
    if set(written) != wanted or len(written) != len(wanted):
        msg = (
            "compaction wrote rows that differ from the selected ones; "
            "inputs were left in place"
        )
        raise RuntimeError(msg)


def _delete_inputs(snapshot: list[Path]) -> None:
    """Remove each input part, sources file first; files already gone are fine."""
    for sources_path in snapshot:
        sources_path.unlink(missing_ok=True)
        stem = sources_path.name.removesuffix(SOURCES_SUFFIX)
        sources_path.with_name(stem + SAMPLES_SUFFIX).unlink(missing_ok=True)
