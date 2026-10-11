"""``Run``: one run directory, and the pipeline stages as methods.

See ``packages/hq/docs/spec.md``, "Stages" and "Public API". Each CLI
subcommand is a thin wrapper around one method here.
"""

__all__ = ("Run", "SourceResult")

import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import h5py

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import RejectionSampler, Samples, make_prior_cache
from harv_hq._model_file import ModelFile
from harv_hq.config import Config
from harv_hq.ids import prior_cache_key
from harv_hq.prepare import PreparedData, prepare
from harv_hq.provenance import check_provenance, make_provenance
from harv_hq.rejection import process_rejection
from harv_hq.results import PartWriter, ResultsIndex, read_source_samples, supersede

logger = logging.getLogger("harv_hq")

PRIOR_CACHE_FILE = "prior_cache.h5"
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
            record = make_provenance(self.config, prior_cache_id=uuid.uuid4().hex)
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
        overwrite: bool = False,
    ) -> None:
        """Run the rejection sampler on every source in a slice (``hq run``).

        Checks the prepared data and the prior cache against the current
        ``hq.toml`` and model file, skips sources already done (see the
        spec's "Resume"), and writes result parts as it goes. Anything still
        buffered is written when the loop ends, including when it is
        interrupted by an exception.

        Parameters
        ----------
        shard
            This process's slice ``(i, N)``.
        overwrite
            Move existing rejection results to
            ``results/superseded-<timestamp>-rejection/`` and start over.

        Raises
        ------
        FileNotFoundError
            If the data are not prepared or the prior cache is not built.
        ProvenanceError
            If an input or an existing part was built from different inputs.
        """
        prepared = PreparedData(self.run_dir)
        expected = make_provenance(self.config, data_id=prepared.data_id)
        check_provenance(prepared.provenance, expected, path=prepared.path)
        cache_path, cache_provenance = self._checked_prior_cache(expected)

        stage_dir = self.run_dir / "results" / "rejection"
        if overwrite:
            supersede(stage_dir)
        part_provenance = {
            **expected,
            "prior_cache_id": cache_provenance["prior_cache_id"],
        }
        done = ResultsIndex.build(stage_dir, expected_provenance=part_provenance).done()
        todo = [sid for sid in prepared.slice_ids(shard) if sid not in done]
        i, n = shard
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
            prior, model = self.model_file.setup(self.config.run.kind)
            rejection = self.config.rejection
            sampler = RejectionSampler(
                prior,
                model,
                batch_size=rejection.batch_size,
                min_evidence_ess=rejection.min_evidence_ess,
            )
            writer = PartWriter(
                stage_dir,
                stage="rejection",
                shard=shard,
                provenance=part_provenance,
                flush_n_sources=self.config.results.flush_n_sources,
                flush_seconds=self.config.results.flush_seconds,
            )
            try:
                for source_id in todo:
                    writer.add(
                        process_rejection(
                            source_id,
                            data[source_id],
                            sampler=sampler,
                            prior_cache=cache_path,
                            config=rejection,
                            seed=self.config.run.seed,
                        )
                    )
            finally:
                writer.close()
        finally:
            logger.removeHandler(log_handler)
            logger.setLevel(log_level)
            log_handler.close()
        self._indexes.pop("rejection", None)

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
        for stage in ("rejection", "mcmc"):
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
            summary=None,
        )

    def _index(self, stage: str) -> ResultsIndex:
        if stage not in self._indexes:
            self._indexes[stage] = ResultsIndex.build(self.run_dir / "results" / stage)
        return self._indexes[stage]

    def _read_samples(self, stage: str, source_id: Any) -> Samples:
        try:
            return read_source_samples(self._index(stage).records[source_id])
        except FileNotFoundError:
            # The part was merged away (compaction) since the index was built.
            self._indexes.pop(stage, None)
            return read_source_samples(self._index(stage).records[source_id])

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
