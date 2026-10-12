"""``Run``: one run directory, and the pipeline stages as methods.

See ``packages/hq/docs/spec.md``, "Stages" and "Public API". Each CLI
subcommand is a thin wrapper around one method here.
"""

__all__ = ("Run", "SourceResult")

import functools
import json
import logging
import os
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, final

import h5py
import pyarrow as pa

# pyarrow.compute generates its functions at import, so ty cannot see them.
import pyarrow.compute as pc
import pyarrow.parquet as pq

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import Samples, make_prior_cache
from harv_hq._model_file import ModelFile
from harv_hq._parquet_io import read_metadata
from harv_hq.compact import CompactReport, compact
from harv_hq.config import Config, ConfigError
from harv_hq.execution import Progress, Setup, mpi_comm, run_pool, run_serial
from harv_hq.ids import prior_cache_key
from harv_hq.mcmc import SELECTION_COLUMNS, mcmc_setup, select_sources
from harv_hq.prepare import PreparedData, prepare
from harv_hq.provenance import (
    MCMC_TABLES,
    PRIOR_CACHE_TABLES,
    REJECTION_TABLES,
    ProvenanceError,
    check_provenance,
    make_provenance,
)
from harv_hq.rejection import rejection_setup
from harv_hq.results import (
    SOURCES_SUFFIX,
    PartWriter,
    Payload,
    ResultsIndex,
    read_source_samples,
    rescanning,
    supersede,
)
from harv_hq.summarize import SUMMARY_FILE, summarize

logger = logging.getLogger("harv_hq")

PRIOR_CACHE_FILE = "prior_cache.h5"
STAGES = ("rejection", "mcmc")
_PROVENANCE_ATTR = "hq.provenance"


@final
@dataclass(frozen=True)
class SourceResult:
    """Everything hq holds for one source (see ``Run.load_source``)."""

    source_id: Any
    data: RVData | GaiaAstrometryData
    rejection: Samples | None
    mcmc: Samples | None
    rejection_status: str
    mcmc_status: str
    summary: dict[str, Any] | None


@final
@dataclass(frozen=True)
class _StageWork:
    """What one process of a stage runs: its slice, inputs, and setup."""

    ids: list[Any]  # the slice, in processing order
    read: Callable[[list[Any]], Mapping[Any, Any]]  # each source's task input
    setup: Setup
    provenance: dict[str, Any]  # what the stage's parts record


@final
class Run:
    """A run directory: its configuration, model file, and stages.

    Parameters
    ----------
    run_dir
        The directory holding ``hq.toml``.

    Raises
    ------
    ConfigError
        If ``hq.toml`` is invalid.
    """

    def __init__(self, run_dir: str | os.PathLike) -> None:
        self.config = Config.from_file(Path(run_dir) / "hq.toml")
        self._model_file: ModelFile | None = None
        self._prepared: PreparedData | None = None
        self._indexes: dict[str, ResultsIndex] = {}

    @property
    def run_dir(self) -> Path:
        """The run directory."""
        return self.config.run_dir

    @property
    def model_file(self) -> ModelFile:
        """The run's model file, imported on first use."""
        if self._model_file is None:
            self._model_file = ModelFile.load(self.config.run.model_file)
        return self._model_file

    def prepare(self, *, overwrite: bool = False) -> None:
        """Prepare per-source data from the input table (``hq prepare``).

        Parameters
        ----------
        overwrite
            Replace existing prepared data.
        """
        prepare(
            self.config, select_rows=self.model_file.select_rows, overwrite=overwrite
        )
        self._prepared = None

    def make_prior_cache(self, *, overwrite: bool = False) -> None:
        """Build the shared prior cache (``hq prior-cache``).

        Draws ``[prior_cache] n_samples`` samples from the model file's prior
        with ``harv.samplers.make_prior_cache``, then records the provenance
        (with a fresh ``prior_cache_id``) in the HDF5 root attribute
        ``hq.provenance``. Written to a temporary file and renamed.

        Parameters
        ----------
        overwrite
            Replace an existing cache (which invalidates every result built
            on it).

        Raises
        ------
        FileExistsError
            If the cache exists and ``overwrite`` is false.
        """
        path = self.run_dir / PRIOR_CACHE_FILE
        if path.exists() and not overwrite:
            msg = (
                f"{path} exists; pass overwrite=True (hq prior-cache --overwrite) "
                "to rebuild it"
            )
            raise FileExistsError(msg)
        prior, model = self.model_file.setup(self.config.run.kind)
        tmp = path.with_name(path.name + ".tmp")
        try:
            make_prior_cache(
                prior,
                model,
                self.config.prior_cache.n_samples,
                tmp,
                key=prior_cache_key(self.config.run.seed),
                batch_size=self.config.prior_cache.batch_size,
            )
            record = make_provenance(
                self.config,
                tables=PRIOR_CACHE_TABLES,
                prior_cache_id=uuid.uuid4().hex,
            )
            with h5py.File(tmp, "a") as f:
                f.attrs[_PROVENANCE_ATTR] = json.dumps(record)
            tmp.replace(path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        logger.info(
            "prior-cache: wrote %d samples to %s",
            self.config.prior_cache.n_samples,
            path,
        )

    def run_rejection(
        self,
        *,
        shard: tuple[int, int] = (0, 1),
        workers: int = 1,
        mpi: bool = False,
        overwrite: bool = False,
        compact: bool = True,
    ) -> None:
        """Run the rejection sampler on every source in a slice (``hq run``).

        Checks the prepared data and the prior cache against the current
        ``hq.toml`` and model file, skips sources already done (see the
        spec's "Resume"), and writes result parts as it goes. Anything still
        buffered is written when the loop ends, including when it is
        interrupted by an exception. See the spec's "Execution modes".

        Parameters
        ----------
        shard
            This process's slice ``(i, N)``.
        workers
            Local worker processes; above 1, sources run on a spawned pool
            and this process writes the parts. Combines with ``mpi`` (a pool
            per rank, e.g. one rank per node).
        mpi
            Process slice ``rank/size`` of ``MPI.COMM_WORLD`` (needs
            ``mpi4py``); excludes ``shard``.
        overwrite
            Move existing rejection results to
            ``results/superseded-<timestamp>-rejection/`` and start over.
        compact
            Compact the stage at the end when this invocation covered all of
            it (slice ``0/1``, or MPI on rank 0); explicit shards never do.

        Raises
        ------
        ValueError
            If ``mpi`` is combined with ``shard``, or ``workers < 1``.
        FileNotFoundError
            If the data are not prepared or the prior cache is not built.
        ProvenanceError
            If an input or an existing part was built from different inputs.
        """
        self._run_stage(
            "rejection",
            shard=shard,
            workers=workers,
            mpi=mpi,
            overwrite=overwrite,
            compact=compact,
        )

    def run_mcmc(
        self,
        *,
        shard: tuple[int, int] = (0, 1),
        workers: int = 1,
        mpi: bool = False,
        overwrite: bool = False,
        compact: bool = True,
    ) -> None:
        """Run MCMC follow-up on the selected sources in a slice (``hq mcmc``).

        Selects sources from ``summary.parquet`` by the ``[mcmc]`` rule or the
        model file's ``select_for_mcmc``, and starts each source's chains
        from its rejection samples. Otherwise it runs like
        :meth:`run_rejection`: the same slicing, resume, execution modes,
        part writing, and compaction, into ``results/mcmc/``. See the spec's
        "MCMC follow-up".

        Parameters
        ----------
        shard, workers, mpi, compact
            As for :meth:`run_rejection`.
        overwrite
            Move existing MCMC results to
            ``results/superseded-<timestamp>-mcmc/`` and start over.

        Raises
        ------
        ConfigError
            If ``hq.toml`` has no ``[mcmc]`` table.
        ValueError
            If ``mpi`` is combined with ``shard``, or ``workers < 1``.
        FileNotFoundError
            If the data are not prepared or ``summary.parquet`` does not
            exist.
        ProvenanceError
            If an input or an existing part was built from different inputs,
            or ``summary.parquet`` does not hold the current rejection
            results.
        """
        self._run_stage(
            "mcmc",
            shard=shard,
            workers=workers,
            mpi=mpi,
            overwrite=overwrite,
            compact=compact,
        )

    def _run_stage(
        self,
        stage: str,
        *,
        shard: tuple[int, int],
        workers: int,
        mpi: bool,
        overwrite: bool,
        compact: bool,
    ) -> None:
        if workers < 1:
            msg = f"workers must be at least 1, got {workers}"
            raise ValueError(msg)
        comm = None
        if mpi:
            if shard != (0, 1):
                msg = "mpi=True takes its slice from the MPI rank; omit shard"
                raise ValueError(msg)
            comm = mpi_comm()
            shard = (comm.Get_rank(), comm.Get_size())
        try:
            part_provenance = self._run_slice(stage, shard, workers, overwrite, comm)
            # The whole stage ran here (MPI: rank 0, after the final barrier).
            # Checked against what this run wrote with, so an edit to hq.toml
            # or the cache during the run cannot fail it at the very end.
            if compact and shard[0] == 0 and (comm is not None or shard[1] == 1):
                self._compact(stage, part_provenance)
        except BaseException:
            # A rank that stops early would leave the others waiting at a
            # barrier forever; nothing above it can recover, so end them all.
            if comm is not None:
                logger.exception("%s: rank %d failed; aborting", stage, shard[0])
                comm.Abort(1)
            raise

    def _run_slice(
        self,
        stage: str,
        shard: tuple[int, int],
        workers: int,
        overwrite: bool,
        comm: Any,
    ) -> dict[str, Any]:
        i, n = shard
        work = self._stage_work(stage, shard, workers)
        stage_dir = self.run_dir / "results" / stage
        # Under MPI only rank 0 moves results aside, and every rank waits for
        # it before reading what is done.
        if overwrite and (comm is None or i == 0):
            supersede(stage_dir)
        if comm is not None:
            comm.Barrier()
        existing = ResultsIndex.build(stage_dir, expected_provenance=work.provenance)
        done = existing.done()
        todo = [sid for sid in work.ids if sid not in done]
        log_handler, log_level = self._log_to_file(f"{stage}-{i:04d}-of-{n:04d}.log")
        logger.info(
            "%s: slice %d/%d, %d sources to run (%d already done)",
            stage,
            i,
            n,
            len(todo),
            len(done),
        )
        try:
            inputs = work.read(todo)
            writer = PartWriter(
                stage_dir,
                stage=stage,
                shard=shard,
                provenance=work.provenance,
                flush_n_sources=self.config.results.flush_n_sources,
                flush_seconds=self.config.results.flush_seconds,
                structure=existing.structure,
            )
            progress = Progress(
                stage, len(todo), ranks=n if comm is not None and i == 0 else 1
            )

            def sink(payload: Payload) -> None:
                writer.add(payload)
                progress.tick()

            try:
                if workers > 1:
                    run_pool(todo, inputs, work.setup, sink, workers=workers)
                else:
                    run_serial(todo, inputs, work.setup, sink)
            finally:
                writer.close()
        finally:
            # Parts may have been written even if the loop raised.
            self._indexes.pop(stage, None)
            logger.removeHandler(log_handler)
            logger.setLevel(log_level)
            log_handler.close()
        if comm is not None:
            comm.Barrier()
        return work.provenance

    def _stage_work(
        self, stage: str, shard: tuple[int, int], workers: int
    ) -> _StageWork:
        """Check a stage's inputs, and say what slice ``shard`` of it runs."""
        if stage == "rejection":
            prepared, provenance = self._inputs(need_cache=True)
            return _StageWork(
                ids=prepared.slice_ids(shard),
                read=prepared.read,
                setup=functools.partial(
                    rejection_setup, self.config, self.run_dir / PRIOR_CACHE_FILE
                ),
                provenance=provenance["rejection"],
            )
        if self.config.mcmc is None:
            msg = f"{self.config.run_dir / 'hq.toml'} has no [mcmc] table"
            raise ConfigError(msg)
        prepared, provenance = self._inputs()
        rejection = ResultsIndex.build(
            self.run_dir / "results" / "rejection",
            expected_provenance=provenance["rejection"],
        )
        hook = self.model_file.select_for_mcmc
        summary = self._current_summary(
            provenance["rejection"],
            rejection,
            columns=None if hook else list(SELECTION_COLUMNS),
        )
        selected = select_sources(summary, self.config.mcmc, hook)
        records = rejection.records

        def read(ids: list[Any]) -> dict[Any, Any]:
            data = prepared.read(ids)
            return {sid: (data[sid], records[sid]) for sid in data}

        return _StageWork(
            ids=prepared.slice_ids(shard, among=selected & rejection.done()),
            read=read,
            # Pool workers import the model file themselves; this process
            # reuses the one it already imported.
            setup=functools.partial(
                mcmc_setup, self.config, None if workers > 1 else self.model_file
            ),
            provenance=provenance["mcmc"],
        )

    def _current_summary(
        self,
        expected: dict[str, Any],
        rejection: ResultsIndex,
        *,
        columns: list[str] | None,
    ) -> pa.Table:
        """``summary.parquet``'s ``columns``, refused unless it is current.

        Current means built from the current inputs and holding every
        rejection result: none finished after the summary was started, and
        its ``ok`` sources are exactly the rejection stage's.
        """
        path = self.run_dir / SUMMARY_FILE
        if not path.exists():
            msg = f"{self.run_dir} has no {SUMMARY_FILE}; run hq summarize first"
            raise FileNotFoundError(msg)
        found = read_metadata(path).get("provenance") or {}
        check_provenance(found, expected, path=path)
        summary = pq.read_table(path, columns=columns)
        ok = summary.filter(pc.equal(summary["rejection_status"], "ok"))  # ty: ignore[unresolved-attribute]
        newest = max((r.finished for r in rejection.records.values()), default=None)
        created = found.get("created")
        if (
            newest is not None
            and (created is None or newest > datetime.fromisoformat(created))
        ) or set(ok["source_id"].to_pylist()) != rejection.done():
            msg = (
                f"{path} does not hold the current rejection results; "
                "rerun hq summarize"
            )
            raise ProvenanceError(msg)
        return summary

    def compact(self, stage: str = "rejection") -> CompactReport:
        """Merge a stage's result parts (``hq compact``); see the spec's "Compaction".

        Parameters
        ----------
        stage
            ``"rejection"`` or ``"mcmc"``.

        Returns
        -------
            What was merged.

        Raises
        ------
        ProvenanceError
            If a part was built from different inputs.
        """
        return self._compact(stage, self._stage_provenance(stage))

    def _compact(self, stage: str, expected: dict[str, Any]) -> CompactReport:
        report = compact(
            self.run_dir / "results" / stage,
            stage=stage,
            expected_provenance=expected,
            compact_n_sources=self.config.results.compact_n_sources,
        )
        self._indexes.pop(stage, None)
        return report

    def summarize(self) -> Path:
        """Write ``summary.parquet`` (``hq summarize``).

        Returns
        -------
            Its path.

        Raises
        ------
        ProvenanceError
            If a result part was built from different inputs.
        """
        path = summarize(
            self.run_dir,
            expected=self._inputs()[1],
        )
        logger.info("summarize: wrote %s", path)
        return path

    def status(self) -> dict[str, dict[str, int]]:
        """Sources per status for each stage (``hq status``).

        Returns
        -------
            ``stage -> status -> count`` over the prepared sources. Rejection
            also counts ``pending`` sources, prepared but without a result.

        Raises
        ------
        ProvenanceError
            If a stage's parts were built from different inputs (e.g. before
            ``hq prepare --overwrite``), rather than counting stale results.
        """
        stage_dirs = {stage: self.run_dir / "results" / stage for stage in STAGES}
        if any(any(d.glob(f"*{SOURCES_SUFFIX}")) for d in stage_dirs.values()):
            prepared, provenance = self._inputs()
        else:
            # Nothing to check, and the prior cache may not exist yet.
            prepared, provenance = PreparedData(self.run_dir), {}
        ids = set(prepared.source_ids)
        counts: dict[str, dict[str, int]] = {}
        for stage, stage_dir in stage_dirs.items():
            records = ResultsIndex.build(
                stage_dir, expected_provenance=provenance.get(stage)
            ).records
            counts[stage] = dict(
                Counter(r.status for sid, r in records.items() if sid in ids)
            )
            if stage == "rejection":
                counts[stage]["pending"] = len(ids - records.keys())
        return counts

    def load_source(self, source_id: Any) -> SourceResult:
        """Everything hq holds for one source.

        Parameters
        ----------
        source_id
            The source's ID.

        Returns
        -------
            Its data, its rejection and MCMC samples (``None`` when absent),
            their statuses (``"pending"`` before a stage has a result), and
            its summary row (``None`` until ``summarize`` has run).
        """
        if self._prepared is None:
            self._prepared = PreparedData(self.run_dir)
        data = self._prepared.read([source_id])
        key, source_data = next(iter(data.items()))
        samples: dict[str, Samples | None] = {}
        statuses: dict[str, str] = {}
        for stage in STAGES:
            record = self._index(stage).records.get(key)
            statuses[stage] = "pending" if record is None else record.status
            samples[stage] = (
                self._read_samples(stage, key) if statuses[stage] == "ok" else None
            )
        return SourceResult(
            source_id=key,
            data=source_data,
            rejection=samples["rejection"],
            mcmc=samples["mcmc"],
            rejection_status=statuses["rejection"],
            mcmc_status=statuses["mcmc"],
            summary=self._summary_row(key),
        )

    def _inputs(
        self, *, need_cache: bool = False
    ) -> tuple[PreparedData, dict[str, dict[str, Any]]]:
        """The checked prepared data, and what each stage's parts must match.

        Only ``hq run`` reads the prior cache (``need_cache``). Later stages
        check it when it exists; after it is deleted they take its ID from
        the rejection results, whose parts must all agree on it.
        """
        prepared = PreparedData(self.run_dir)
        current = make_provenance(self.config, data_id=prepared.data_id)
        check_provenance(prepared.provenance, current, path=prepared.path)
        if need_cache or (self.run_dir / PRIOR_CACHE_FILE).exists():
            cache_id = self._checked_prior_cache(current)["prior_cache_id"]
        else:
            recorded = ResultsIndex.build(self.run_dir / "results" / "rejection")
            cache_id = (recorded.provenance or {}).get("prior_cache_id")
        stage_tables = {"rejection": REJECTION_TABLES, "mcmc": MCMC_TABLES}
        provenance = {
            stage: make_provenance(
                self.config,
                tables=tables,
                data_id=prepared.data_id,
                prior_cache_id=cache_id,
            )
            for stage, tables in stage_tables.items()
        }
        return prepared, provenance

    def _stage_provenance(self, stage: str) -> dict[str, Any]:
        """What a stage's parts must match."""
        return self._inputs()[1][stage]

    def _summary_row(self, source_id: Any) -> dict[str, Any] | None:
        path = self.run_dir / SUMMARY_FILE
        if not path.exists():
            return None
        rows = pq.read_table(path, filters=[("source_id", "==", source_id)])
        return rows.to_pylist()[0] if rows.num_rows else None

    def _index(self, stage: str) -> ResultsIndex:
        if stage not in self._indexes:
            self._indexes[stage] = ResultsIndex.build(self.run_dir / "results" / stage)
        return self._indexes[stage]

    def _read_samples(self, stage: str, source_id: Any) -> Samples:
        def read() -> Samples:
            try:
                return read_source_samples(self._index(stage).records[source_id])
            except FileNotFoundError:
                # Merged away by compaction since the index was built.
                self._indexes.pop(stage, None)
                raise

        return rescanning(read)

    def _checked_prior_cache(self, expected: dict[str, Any]) -> dict[str, Any]:
        """The prior cache's provenance, checked against ``expected``."""
        path = self.run_dir / PRIOR_CACHE_FILE
        if not path.exists():
            msg = f"{self.run_dir} has no prior cache; run hq prior-cache first"
            raise FileNotFoundError(msg)
        with h5py.File(path, "r") as f:
            found = json.loads(f.attrs[_PROVENANCE_ATTR])
        # The cache does not depend on the prepared data, so data_id is not checked.
        check_provenance(
            found, {k: v for k, v in expected.items() if k != "data_id"}, path=path
        )
        return found

    def _log_to_file(self, name: str) -> tuple[logging.Handler, int]:
        """Also log this stage to ``logs/<name>``; returns the handler and old level."""
        log_dir = self.run_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        handler = logging.FileHandler(log_dir / name)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        previous = logger.level
        if previous == logging.NOTSET or previous > logging.INFO:
            logger.setLevel(logging.INFO)
        return handler, previous
