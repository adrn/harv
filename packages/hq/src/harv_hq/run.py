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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import h5py
import pyarrow.parquet as pq

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import Samples, make_prior_cache
from harv_hq._model_file import ModelFile
from harv_hq.compact import CompactReport, compact
from harv_hq.config import Config
from harv_hq.execution import Progress, mpi_comm, run_pool, run_serial
from harv_hq.ids import prior_cache_key
from harv_hq.prepare import PreparedData, prepare
from harv_hq.provenance import (
    PRIOR_CACHE_TABLES,
    REJECTION_TABLES,
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
            part_provenance = self._run_rejection(shard, workers, overwrite, comm)
            # The whole stage ran here (MPI: rank 0, after the final barrier).
            # Checked against what this run wrote with, so an edit to hq.toml
            # or the cache during the run cannot fail it at the very end.
            if compact and shard[0] == 0 and (comm is not None or shard[1] == 1):
                self._compact("rejection", part_provenance)
        except BaseException:
            # A rank that stops early would leave the others waiting at a
            # barrier forever; nothing above it can recover, so end them all.
            if comm is not None:
                logger.exception("rejection: rank %d failed; aborting", shard[0])
                comm.Abort(1)
            raise

    def _run_rejection(
        self, shard: tuple[int, int], workers: int, overwrite: bool, comm: Any
    ) -> dict[str, Any]:
        i, n = shard
        prepared, cache_path, part_provenance = self._rejection_inputs()
        stage_dir = self.run_dir / "results" / "rejection"
        # Under MPI only rank 0 moves results aside, and every rank waits for
        # it before reading what is done.
        if overwrite and (comm is None or i == 0):
            supersede(stage_dir)
        if comm is not None:
            comm.Barrier()
        existing = ResultsIndex.build(stage_dir, expected_provenance=part_provenance)
        done = existing.done()
        todo = [sid for sid in prepared.slice_ids(shard) if sid not in done]
        log_handler, log_level = self._log_to_file(f"rejection-{i:04d}-of-{n:04d}.log")
        logger.info(
            "rejection: slice %d/%d, %d sources to run (%d already done)",
            i,
            n,
            len(todo),
            len(done),
        )
        try:
            data = prepared.read(todo)
            setup = functools.partial(rejection_setup, self.config, cache_path)
            writer = PartWriter(
                stage_dir,
                stage="rejection",
                shard=shard,
                provenance=part_provenance,
                flush_n_sources=self.config.results.flush_n_sources,
                flush_seconds=self.config.results.flush_seconds,
                structure=existing.structure,
            )
            progress = Progress(
                "rejection", len(todo), ranks=n if comm is not None and i == 0 else 1
            )

            def sink(payload: Payload) -> None:
                writer.add(payload)
                progress.tick()

            try:
                if workers > 1:
                    run_pool(todo, data, setup, sink, workers=workers)
                else:
                    run_serial(todo, data, setup, sink)
            finally:
                writer.close()
        finally:
            # Parts may have been written even if the loop raised.
            self._indexes.pop("rejection", None)
            logger.removeHandler(log_handler)
            logger.setLevel(log_level)
            log_handler.close()
        if comm is not None:
            comm.Barrier()
        return part_provenance

    def compact(self, stage: str = "rejection") -> CompactReport:
        """Merge a stage's result parts (``hq compact``); see the spec's "Compaction".

        Parameters
        ----------
        stage
            ``"rejection"`` (``"mcmc"`` arrives with MCMC follow-up).

        Returns
        -------
            What was merged.

        Raises
        ------
        ProvenanceError
            If a part was built from different inputs.
        """
        return self._compact(stage, self._stage_provenance(stage))

    def _compact(self, stage: str, expected: dict[str, Any] | None) -> CompactReport:
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
            expected={stage: self._stage_provenance(stage) for stage in STAGES},
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
        prepared = set(PreparedData(self.run_dir).source_ids)
        counts: dict[str, dict[str, int]] = {}
        for stage in STAGES:
            stage_dir = self.run_dir / "results" / stage
            has_parts = any(stage_dir.glob(f"*{SOURCES_SUFFIX}"))
            records = ResultsIndex.build(
                stage_dir,
                expected_provenance=self._stage_provenance(stage)
                if has_parts
                else None,
            ).records
            counts[stage] = dict(
                Counter(r.status for sid, r in records.items() if sid in prepared)
            )
            if stage == "rejection":
                counts[stage]["pending"] = len(prepared - records.keys())
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

    def _rejection_inputs(self) -> tuple[PreparedData, Path, dict[str, Any]]:
        """The checked prepared data and prior cache, and parts' provenance."""
        prepared = PreparedData(self.run_dir)
        current = make_provenance(self.config, data_id=prepared.data_id)
        check_provenance(prepared.provenance, current, path=prepared.path)
        cache_path, cache_provenance = self._checked_prior_cache(current)
        part = make_provenance(
            self.config,
            tables=REJECTION_TABLES,
            data_id=prepared.data_id,
            prior_cache_id=cache_provenance["prior_cache_id"],
        )
        return prepared, cache_path, part

    def _stage_provenance(self, stage: str) -> dict[str, Any] | None:
        """What a stage's parts must match (MCMC parts: not checked yet)."""
        return self._rejection_inputs()[2] if stage == "rejection" else None

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

    def _checked_prior_cache(
        self, expected: dict[str, Any]
    ) -> tuple[Path, dict[str, Any]]:
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
        return path, found

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
