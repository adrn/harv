"""Compaction: merge a stage's result parts into a few large ones.

See ``packages/hq/docs/spec.md``, "Compaction". Compaction keeps the newest
row per source, writes it through the normal part protocol, verifies the new
parts, and only then deletes its inputs. It needs no lock: a crash at any step
leaves a readable (possibly duplicated) stage, and rerunning finishes the job.
"""

__all__ = ("ORPHAN_AGE_S", "CompactReport", "compact")

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harv_hq._parquet_io import field_units, row_group_starts
from harv_hq.results import (
    SAMPLES_SUFFIX,
    SOURCES_SUFFIX,
    ResultRecord,
    ResultsIndex,
    rescanning,
    write_part_tables,
)

logger = logging.getLogger(__name__)

COMPACTED_PREFIX = "c-"
# A samples file without its sources file is an orphan: a crash between a
# part's two renames, or between compaction's two deletions. It might also be a
# writer between its renames right now, so only orphans this old are removed.
# ponytail: an age heuristic; a writer's gap between renames is seconds.
ORPHAN_AGE_S = 3600.0

_Units = tuple[dict[str, str], dict[str, str]]


@final
@dataclass(frozen=True)
class CompactReport:
    """What one compaction did."""

    n_input_parts: int
    n_sources: int
    n_dropped_rows: int
    n_output_parts: int
    n_orphans_removed: int = 0


def compact(
    stage_dir: Path,
    *,
    stage: str,
    expected_provenance: dict[str, Any] | None,
    compact_n_sources: int,
) -> CompactReport:
    """Rewrite a stage's parts as parts holding only the newest row per source.

    Starts over if a concurrent compaction removes one of its inputs.

    Parameters
    ----------
    stage_dir
        ``results/<stage>``.
    stage
        ``"rejection"`` or ``"mcmc"``.
    expected_provenance
        What every input part must match (``None``: not checked). The outputs
        record the inputs' own provenance, not this.
    compact_n_sources
        Sources per output part.

    Returns
    -------
        Counts of input parts, sources kept, superseded rows dropped, output
        parts, and orphaned samples files removed (see ``ORPHAN_AGE_S``).

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
    return rescanning(
        lambda: _compact(
            Path(stage_dir),
            stage=stage,
            expected_provenance=expected_provenance,
            compact_n_sources=compact_n_sources,
        )
    )


def _compact(
    stage_dir: Path,
    *,
    stage: str,
    expected_provenance: dict[str, Any] | None,
    compact_n_sources: int,
) -> CompactReport:
    # 1. Snapshot: only parts committed now are inputs, and only orphans that
    # are already old are removed.
    snapshot = sorted(stage_dir.glob(f"*{SOURCES_SUFFIX}"))
    orphans = _old_orphans(stage_dir)
    # 2-3. Check provenance and structure, and select the newest row per source.
    index = ResultsIndex.build(
        stage_dir, expected_provenance=expected_provenance, paths=snapshot
    )
    already = all(p.name.startswith(COMPACTED_PREFIX) for p in snapshot)
    if not snapshot or (already and index.n_rows == len(index.records)):
        return CompactReport(0, 0, 0, 0, _delete_orphans(orphans))
    metadata: dict[str, Any] = {
        "stage": stage,
        "samples_metadata_keys": list(index.metadata_keys),
        **(index.structure or {}),
    }
    if index.provenance is not None:
        # The newest input's record: the samples were made then, not now.
        metadata["provenance"] = index.provenance

    # 4. Read each input once, in snapshot order, filling output parts of
    # compact_n_sources sources (rows sorted by source_id within each part).
    by_file: dict[Path, list[tuple[Any, ResultRecord]]] = {}
    for sid, record in index.records.items():
        by_file.setdefault(record.sources_path, []).append((sid, record))
    run_id = uuid.uuid4().hex[:8]
    outputs: list[Path] = []
    block = _Block()
    for path in snapshot:
        for source_id, row, samples, units in _read_input(path, by_file.get(path, [])):
            block.add(source_id, row, samples, units)
            if len(block) == compact_n_sources:
                outputs.append(block.write(stage_dir, run_id, len(outputs), metadata))
                block = _Block()
    if len(block):
        outputs.append(block.write(stage_dir, run_id, len(outputs), metadata))

    # 5. Verify, then 6. delete the inputs.
    _verify(outputs, {(sid, r.finished) for sid, r in index.records.items()})
    _delete_inputs(snapshot)
    report = CompactReport(
        n_input_parts=len(snapshot),
        n_sources=len(index.records),
        n_dropped_rows=index.n_rows - len(index.records),
        n_output_parts=len(outputs),
        n_orphans_removed=_delete_orphans(orphans),
    )
    logger.info("%s: compacted %s", stage, report)
    return report


@final
class _Block:
    """One output part being filled: sources rows and their samples."""

    def __init__(self) -> None:
        self.entries: list[tuple[Any, pa.Table, pa.Table | None]] = []
        self.sources_units: dict[str, str] = {}
        self.samples_units: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def add(
        self, source_id: Any, row: pa.Table, samples: pa.Table | None, units: _Units
    ) -> None:
        self.entries.append((source_id, row, samples))
        self.sources_units.update(units[0])
        self.samples_units.update(units[1])

    def write(
        self, stage_dir: Path, run_id: str, k: int, metadata: dict[str, Any]
    ) -> Path:
        entries = sorted(self.entries, key=lambda e: e[0])
        # ponytail: concatenates one-row tables (about a second per 100k
        # sources); take() per input file if that ever shows in a profile.
        sources = pa.concat_tables([e[1] for e in entries], promote_options="default")
        n = np.array([0 if e[2] is None else e[2].num_rows for e in entries])
        sources = _set_column(sources, "n_samples", pa.array(n, pa.int64()))
        sources = _set_column(
            sources, "samples_row_start", pa.array(np.cumsum(n) - n, pa.int64())
        )
        tables = [e[2] for e in entries if e[2] is not None]
        return write_part_tables(
            stage_dir,
            f"{COMPACTED_PREFIX}{run_id}-{k:04d}",
            sources,
            pa.concat_tables(tables) if tables else None,
            sources_units=self.sources_units,
            samples_units=self.samples_units,
            metadata=metadata,
        )


def _set_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, values)


def _read_input(
    sources_path: Path, selected: list[tuple[Any, ResultRecord]]
) -> list[tuple[Any, pa.Table, pa.Table | None, _Units]]:
    """The selected sources of one input part: rows, samples, and units."""
    if not selected:
        return []
    selected = sorted(selected, key=lambda e: e[1].row)
    records = [record for _, record in selected]
    rows = pq.read_table(sources_path).take([r.row for r in records])
    samples, samples_units = _read_samples(records)
    units = (field_units(rows.schema), samples_units)
    return [
        (sid, rows.slice(i, 1), samples[i], units)
        for i, (sid, _) in enumerate(selected)
    ]


def _read_samples(
    records: list[ResultRecord],
) -> tuple[list[pa.Table | None], dict[str, str]]:
    """Each record's samples, from one input, one row group at a time.

    Only the needed rows of each row group are kept (copied out with
    ``take``), so memory holds one decoded row group plus what is kept.
    """
    samples: list[pa.Table | None] = [None] * len(records)
    with_samples = [i for i, r in enumerate(records) if r.n_samples]
    if not with_samples:
        return samples, {}
    parquet_file = pq.ParquetFile(records[with_samples[0]].samples_path)
    starts = row_group_starts(parquet_file)
    first = np.array([records[i].samples_row_start for i in with_samples])
    end = first + np.array([records[i].n_samples for i in with_samples])
    order = np.argsort(first)
    first, end = first[order], end[order]
    owners = np.asarray(with_samples)[order]
    pieces: dict[int, list[pa.Table]] = {int(i): [] for i in owners}
    for group in range(parquet_file.num_row_groups):
        lo_row, hi_row = int(starts[group]), int(starts[group + 1])
        # Records overlapping this row group (ranges are disjoint and sorted).
        lo = int(np.searchsorted(end, lo_row, side="right"))
        hi = int(np.searchsorted(first, hi_row, side="left"))
        if lo >= hi:
            continue
        a = np.maximum(first[lo:hi], lo_row) - lo_row
        b = np.minimum(end[lo:hi], hi_row) - lo_row
        wanted = np.concatenate([np.arange(x, y) for x, y in zip(a, b, strict=True)])
        taken = parquet_file.read_row_group(group).take(wanted)
        offset = 0
        for owner, size in zip(owners[lo:hi], b - a, strict=True):
            pieces[int(owner)].append(taken.slice(offset, int(size)))
            offset += int(size)
    for owner, parts in pieces.items():
        samples[owner] = pa.concat_tables(parts)
    return samples, field_units(parquet_file.schema_arrow)


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


def _sources_twin(samples_path: Path) -> Path:
    stem = samples_path.name.removesuffix(SAMPLES_SUFFIX)
    return samples_path.with_name(stem + SOURCES_SUFFIX)


def _old_orphans(stage_dir: Path) -> list[Path]:
    """Samples files with no sources file, last written over ``ORPHAN_AGE_S`` ago."""
    cutoff = time.time() - ORPHAN_AGE_S
    orphans = []
    for path in stage_dir.glob(f"*{SAMPLES_SUFFIX}"):
        if _sources_twin(path).exists():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                orphans.append(path)
        except FileNotFoundError:
            pass  # a concurrent compaction just deleted it
    return orphans


def _delete_orphans(orphans: list[Path]) -> int:
    """Remove orphans still without a sources file; returns how many this call did."""
    removed = 0
    for path in orphans:
        if _sources_twin(path).exists():
            continue
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass  # removed by a concurrent compaction
    return removed


def _delete_inputs(snapshot: list[Path]) -> None:
    """Remove each input part, sources file first; files already gone are fine."""
    for sources_path in snapshot:
        sources_path.unlink(missing_ok=True)
        stem = sources_path.name.removesuffix(SOURCES_SUFFIX)
        sources_path.with_name(stem + SAMPLES_SUFFIX).unlink(missing_ok=True)
