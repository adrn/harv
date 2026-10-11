"""Result parts: writing them, indexing them, and reading samples back.

See ``packages/hq/docs/spec.md``, "Results", "Resume", and "Loading results".
A process buffers per-source results and writes them as immutable *parts*: a
``.samples.parquet`` (one row per sample) and then a ``.sources.parquet`` (one
row per source), each renamed into place. The sources file is the commit, so
every reader starts from the sources files and ignores anything else.
"""

__all__ = (
    "PartWriter",
    "Payload",
    "ResultRecord",
    "ResultsIndex",
    "read_source_samples",
    "samples_table",
    "supersede",
)

import logging
import os
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harv.samplers import SampleColumns, Samples
from harv_hq._parquet_io import field_units, read_metadata, read_rows, write_parquet
from harv_hq.provenance import check_provenance

logger = logging.getLogger(__name__)

SOURCES_SUFFIX = ".sources.parquet"
SAMPLES_SUFFIX = ".samples.parquet"
# Samples-table columns hq adds; a model parameter may not use these names.
_BOOKKEEPING_COLUMNS = ("source_id", "sample_index", "chain", "weight")
_STRUCTURE_KEYS = (
    "model_type",
    "linear_extension_names",
    "nonlinear_names",
    "linear_names",
)
SAMPLES_ROW_GROUP_SIZE = 131_072
_INDEX_COLUMNS = ["source_id", "status", "finished", "samples_row_start", "n_samples"]


@final
@dataclass(frozen=True)
class Payload:
    """One source's result, as plain host data (picklable across processes).

    ``row`` is the source's sources-table row, without ``samples_row_start``
    and ``n_samples``, which the writer assigns. ``samples`` is ``None`` for a
    failed source.
    """

    row: dict[str, Any]
    row_units: dict[str, str] = field(default_factory=dict)
    samples: pa.Table | None = None
    samples_units: dict[str, str] = field(default_factory=dict)
    structure: dict[str, Any] | None = None
    metadata_keys: tuple[str, ...] = ()


def samples_table(
    samples: Samples, source_id: Any, *, stage: str
) -> tuple[pa.Table, dict[str, str], dict[str, Any], dict[str, Any]]:
    """Flatten one source's ``Samples`` into its samples-table rows.

    Built on ``Samples.to_columns``; adds ``source_id``, ``sample_index``,
    ``chain`` (MCMC, when ``num_chains`` is in the metadata), and ``weight``
    (rejection).

    Parameters
    ----------
    samples
        A flat (1-D) ``Samples``.
    source_id
        The source's ID.
    stage
        ``"rejection"`` or ``"mcmc"``.

    Returns
    -------
        The table, the units of its columns, the run-wide structure
        (``model_type``, ``linear_extension_names``, ``nonlinear_names``,
        ``linear_names``), and the source's ``Samples.metadata``.

    Raises
    ------
    ValueError
        If ``samples`` is batched, or a parameter is named like a column hq
        adds (``source_id``, ``sample_index``, ``chain``, ``weight``).
    """
    if samples.batch_shape:
        msg = f"result samples must be flat, got batch shape {samples.batch_shape}"
        raise ValueError(msg)
    cols = samples.to_columns()
    if clash := sorted(set(cols.columns) & set(_BOOKKEEPING_COLUMNS)):
        msg = f"parameter names {clash} clash with columns hq adds to the results"
        raise ValueError(msg)
    n = samples.n_samples
    columns: dict[str, Any] = {
        "source_id": [source_id] * n,
        "sample_index": np.arange(n, dtype=np.int32),
    }
    if stage == "mcmc" and "num_chains" in cols.metadata:
        per_chain = n // int(cols.metadata["num_chains"])
        columns["chain"] = (np.arange(n) // per_chain).astype(np.int16)
    columns.update(
        {k: np.asarray(v, dtype=np.float64) for k, v in cols.columns.items()}
    )
    if stage == "rejection":
        columns["weight"] = np.asarray(samples.weight, dtype=np.float64)
    structure = {
        "model_type": cols.model_type,
        "linear_extension_names": list(cols.linear_extension_names),
        "nonlinear_names": list(cols.nonlinear_names),
        "linear_names": list(cols.linear_names),
    }
    return pa.table(columns), dict(cols.units), structure, dict(cols.metadata)


@final
class PartWriter:
    """Buffer per-source payloads and write them as result parts.

    A part is written when ``flush_n_sources`` sources are buffered, when
    ``flush_seconds`` have passed since the first buffered one, and on
    :meth:`close`. Payloads that never reach a flush (a hard crash) are lost;
    nothing already written is touched.

    Parameters
    ----------
    stage_dir
        ``results/<stage>``; created if needed.
    stage
        ``"rejection"`` or ``"mcmc"``.
    shard
        This process's slice ``(i, N)``; it names the parts.
    provenance
        The provenance record every part carries.
    flush_n_sources, flush_seconds
        The ``[results]`` flush limits.
    structure
        The ``Samples`` structure of parts already in ``stage_dir``
        (:attr:`ResultsIndex.structure`), so a resumed run cannot append parts
        with a different schema.
    """

    def __init__(
        self,
        stage_dir: Path,
        *,
        stage: str,
        shard: tuple[int, int],
        provenance: dict[str, Any],
        flush_n_sources: int,
        flush_seconds: float,
        structure: dict[str, Any] | None = None,
    ) -> None:
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.shard = shard
        self.provenance = provenance
        self.flush_n_sources = flush_n_sources
        self.flush_seconds = flush_seconds
        self._buffer: list[Payload] = []
        self._buffer_started = 0.0
        self._structure = structure
        self._metadata_keys: list[str] = []

    def add(self, payload: Payload) -> None:
        """Buffer one source's result, and write a part if a limit is reached."""
        if payload.structure is not None:
            if self._structure is None:
                self._structure = payload.structure
            elif payload.structure != self._structure:
                msg = (
                    "every source in a run must share one Samples structure; got "
                    f"{payload.structure} after {self._structure}"
                )
                raise ValueError(msg)
            for key in payload.metadata_keys:
                if key not in self._metadata_keys:
                    self._metadata_keys.append(key)
        if not self._buffer:
            self._buffer_started = time.monotonic()
        self._buffer.append(payload)
        if (
            len(self._buffer) >= self.flush_n_sources
            or time.monotonic() - self._buffer_started >= self.flush_seconds
        ):
            self.flush()

    def flush(self) -> None:
        """Write the buffered results as one part (no-op when empty).

        The samples file is written first and the sources file second; the
        sources file's rename is the commit.
        """
        if not self._buffer:
            return
        i, n = self.shard
        stem = f"{i:04d}-of-{n:04d}-{uuid.uuid4().hex[:8]}"
        rows: list[dict[str, Any]] = []
        tables: list[pa.Table] = []
        row_units: dict[str, str] = {}
        samples_units: dict[str, str] = {}
        row_start = 0
        for payload in self._buffer:
            n = 0 if payload.samples is None else payload.samples.num_rows
            rows.append({**payload.row, "samples_row_start": row_start, "n_samples": n})
            row_units.update(payload.row_units)
            if payload.samples is not None:
                tables.append(payload.samples)
                samples_units.update(payload.samples_units)
            row_start += n

        metadata = {
            "provenance": self.provenance,
            "stage": self.stage,
            "shard": list(self.shard),
            "samples_metadata_keys": self._metadata_keys,
            **(self._structure or {}),
        }
        if tables:
            write_parquet(
                self.stage_dir / (stem + SAMPLES_SUFFIX),
                pa.concat_tables(tables),
                units=samples_units,
                metadata=metadata,
                row_group_size=SAMPLES_ROW_GROUP_SIZE,
            )
        sources = _rows_to_columns(rows)
        sources_path = self.stage_dir / (stem + SOURCES_SUFFIX)
        write_parquet(
            sources_path,
            sources,
            units={k: v for k, v in row_units.items() if k in sources.column_names},
            metadata=metadata,
        )
        logger.info("%s: wrote part %s (%d sources)", self.stage, stem, len(rows))
        self._buffer = []

    def close(self) -> None:
        """Write anything still buffered."""
        self.flush()


def _rows_to_columns(rows: list[dict[str, Any]]) -> pa.Table:
    """Rows with differing keys (failed sources lack statistics) as one table."""
    names: list[str] = []
    for row in rows:
        names.extend(k for k in row if k not in names)
    return pa.table({name: [row.get(name) for row in rows] for name in names})


@final
@dataclass(frozen=True)
class ResultRecord:
    """Where one source's newest result for a stage lives."""

    status: str
    finished: datetime
    sources_path: Path
    row: int
    samples_row_start: int
    n_samples: int

    @property
    def samples_path(self) -> Path:
        """The part's samples file."""
        name = self.sources_path.name.removesuffix(SOURCES_SUFFIX) + SAMPLES_SUFFIX
        return self.sources_path.with_name(name)


@final
class ResultsIndex:
    """The newest result per source for one stage, built from its sources files.

    Parameters
    ----------
    records
        Source ID to its newest :class:`ResultRecord`.
    structure
        The ``Samples`` structure the parts share (``None`` when no part has
        samples).
    """

    def __init__(
        self,
        records: dict[Any, ResultRecord],
        structure: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.structure = structure

    @classmethod
    def build(
        cls, stage_dir: Path, *, expected_provenance: dict[str, Any] | None = None
    ) -> "ResultsIndex":
        """Index every committed part in ``stage_dir``.

        Parameters
        ----------
        stage_dir
            ``results/<stage>``; a missing directory gives an empty index.
        expected_provenance
            When given, every part's provenance is checked against it.

        Returns
        -------
            The index; for a source with several rows, the newest
            ``finished`` wins, and ties go to the part stem that sorts first.

        Raises
        ------
        ProvenanceError
            If a part was built from different inputs.
        ValueError
            If two parts hold ``Samples`` of different structure.
        """
        records: dict[Any, ResultRecord] = {}
        structure: dict[str, Any] | None = None
        if not Path(stage_dir).is_dir():
            return cls(records)
        for path in sorted(Path(stage_dir).glob(f"*{SOURCES_SUFFIX}")):
            metadata = read_metadata(path)
            if expected_provenance is not None:
                found = metadata.get("provenance", {})
                check_provenance(found, expected_provenance, path=path)
            if "model_type" in metadata:  # failed-only parts carry no structure
                part = {k: metadata[k] for k in _STRUCTURE_KEYS}
                if structure is None:
                    structure = part
                elif part != structure:
                    msg = (
                        f"{path} holds Samples structured as {part}, but earlier "
                        f"parts hold {structure}; rerun with --overwrite"
                    )
                    raise ValueError(msg)
            rows = pq.read_table(path, columns=_INDEX_COLUMNS).to_pylist()
            for row, r in enumerate(rows):
                current = records.get(r["source_id"])
                if current is None or r["finished"] > current.finished:
                    records[r["source_id"]] = ResultRecord(
                        status=r["status"],
                        finished=r["finished"],
                        sources_path=path,
                        row=row,
                        samples_row_start=r["samples_row_start"],
                        n_samples=r["n_samples"],
                    )
        return cls(records, structure)

    def done(self) -> set[Any]:
        """Source IDs whose newest result is ``ok``; anything else is rerun."""
        return {sid for sid, record in self.records.items() if record.status == "ok"}

    def status_counts(self) -> dict[str, int]:
        """Number of sources per status."""
        return dict(Counter(record.status for record in self.records.values()))


def read_source_samples(record: ResultRecord) -> Samples:
    """Rebuild one source's ``Samples`` from its part.

    Reads only the row groups of the samples file that hold the source's
    rows, and the source's row of the sources file for its metadata.

    Parameters
    ----------
    record
        The source's :class:`ResultRecord` (status ``ok``).

    Returns
    -------
        The source's ``Samples``, equal to what the sampler returned.

    Raises
    ------
    ValueError
        If the record is not an ``ok`` result.
    """
    if record.status != "ok":
        msg = f"no samples for a {record.status!r} result"
        raise ValueError(msg)
    samples_file = pq.ParquetFile(record.samples_path)
    rows = read_rows(samples_file, record.samples_row_start, record.n_samples)
    meta = read_metadata(record.samples_path)
    units = field_units(samples_file.schema_arrow)
    source_row = read_rows(
        pq.ParquetFile(record.sources_path), record.row, 1
    ).to_pylist()[0]
    metadata = {
        key: source_row[key]
        for key in meta["samples_metadata_keys"]
        if source_row.get(key) is not None
    }
    names = (*meta["nonlinear_names"], *meta["linear_names"])
    columns = {
        name: rows.column(name).to_numpy()
        for name in (*names, "ln_likelihood", "ln_prior")
        if name in rows.column_names
    }
    return Samples.from_columns(
        SampleColumns(
            columns=columns,
            units={name: units.get(name, "") for name in names},
            nonlinear_names=tuple(meta["nonlinear_names"]),
            linear_names=tuple(meta["linear_names"]),
            model_type=meta["model_type"],
            linear_extension_names=tuple(meta["linear_extension_names"]),
            metadata=metadata,
        )
    )


def supersede(stage_dir: Path) -> None:
    """Move a stage's results to ``results/superseded-<timestamp>-<stage>``.

    Used by ``--overwrite``; nothing is deleted.

    Parameters
    ----------
    stage_dir
        ``results/<stage>``.
    """
    if not stage_dir.is_dir() or not any(stage_dir.iterdir()):
        return
    # Microseconds, so two overwrites in one second get separate archives.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    target = stage_dir.with_name(f"superseded-{stamp}-{stage_dir.name}")
    shutil.move(os.fspath(stage_dir), os.fspath(target))
    logger.info("moved %s to %s", stage_dir, target)
