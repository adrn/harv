"""MCMC follow-up: select sources, and process one with ``NumpyroSampler``.

See ``packages/hq/docs/spec.md``, "MCMC follow-up (``run_mcmc``)". Chains
start from an equal-weight resample of the source's rejection samples.
"""

__all__ = (
    "SELECTION_COLUMNS",
    "MCMCInput",
    "RejectionSamples",
    "convergence",
    "mcmc_setup",
    "process_mcmc",
    "select_sources",
)

import functools
import importlib
from collections.abc import Callable
from typing import Any, final

import jax
import numpy as np
import pyarrow as pa

# pyarrow.compute generates its functions at import, so ty cannot see them.
import pyarrow.compute as pc

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import NumpyroSampler, Samples
from harv_hq._model_file import ModelFile
from harv_hq.config import Config, MCMCConfig
from harv_hq.ids import source_key
from harv_hq.results import (
    Payload,
    ResultRecord,
    ResultsIndex,
    process_source,
    read_source_samples,
    rescanning,
)
from harv_hq.summary_stats import summary_stats, weighted_resample

# One task: the source's data and where its rejection samples live. Workers
# read the samples themselves, so the parent holds only the index.
MCMCInput = tuple[RVData | GaiaAstrometryData, ResultRecord]
# The summary columns the [mcmc] rule reads; with a select_for_mcmc hook,
# every column is read.
SELECTION_COLUMNS = (
    "source_id",
    "rejection_status",
    "well_resolved",
    "rejection_period_unimodal",
)
# Summary rows turned into dicts at a time for the select_for_mcmc hook.
_HOOK_BATCH_ROWS = 65_536


def select_sources(
    summary: pa.Table,
    config: MCMCConfig,
    hook: Callable[[dict[str, Any]], bool] | None = None,
) -> set[Any]:
    """The source IDs that go to MCMC follow-up.

    Only sources whose rejection result is ``ok`` can be selected: their
    rejection samples start the chains. The rule reads the rejection-stage
    values (``well_resolved``, ``rejection_period_unimodal``), so a converged
    MCMC result never changes the selection.

    Parameters
    ----------
    summary
        The ``summary.parquet`` table; the rule needs only
        :data:`SELECTION_COLUMNS`.
    config
        The ``[mcmc]`` settings (``select``, ``require_unimodal``).
    hook
        The model file's ``select_for_mcmc(row)``; when given, it replaces
        the rule and receives each ``ok`` source's summary row as a dict.

    Returns
    -------
        The selected IDs.

    Examples
    --------
    >>> import pyarrow as pa
    >>> from harv_hq.config import MCMCConfig
    >>> summary = pa.table({
    ...     "source_id": ["a", "b", "c"],
    ...     "rejection_status": ["ok", "ok", "failed"],
    ...     "well_resolved": [False, True, None],
    ...     "rejection_period_unimodal": [True, True, None],
    ... })
    >>> select_sources(summary, MCMCConfig())
    {'a'}
    >>> sorted(select_sources(summary, MCMCConfig(select="all")))
    ['a', 'b']
    """
    if hook is not None:
        return {
            row["source_id"]
            for batch in summary.to_batches(max_chunksize=_HOOK_BATCH_ROWS)
            for row in batch.to_pylist()
            if row["rejection_status"] == "ok" and hook(row)
        }

    def column(name: str) -> pa.ChunkedArray:
        # All-null columns (no ok rejection result yet) are stored untyped.
        return summary[name].cast(pa.bool_())

    keep = pc.equal(summary["rejection_status"], "ok")  # ty: ignore[unresolved-attribute]
    if config.select == "under_resolved":
        keep = pc.and_(keep, pc.invert(column("well_resolved")))  # ty: ignore[unresolved-attribute]
    if config.require_unimodal:
        keep = pc.and_(keep, column("rejection_period_unimodal"))  # ty: ignore[unresolved-attribute]
    # Nulls in the mask drop their rows.
    return set(summary.filter(keep)["source_id"].to_pylist())


def convergence(samples: Samples) -> tuple[float, float]:
    """``r_hat_max`` and ``ess_bulk_min`` over every parameter of MCMC samples.

    Any non-finite draw (NaN or infinite, in any parameter) makes both NaN.
    Otherwise a parameter whose draws are all equal has no defined R-hat and
    is skipped; if every parameter is, both are NaN. NaN fails every
    threshold.

    Parameters
    ----------
    samples
        MCMC samples, chains recorded by ``metadata["num_chains"]``.

    Returns
    -------
        The largest rank-normalized R-hat and the smallest bulk ESS.

    Examples
    --------
    >>> import numpy as np
    >>> from unxt import Q
    >>> from harv.samplers import Samples
    >>> draws = np.random.default_rng(0).normal(size=2000)
    >>> samples = Samples(
    ...     nonlinear={"period": Q(draws, "day"), "e": Q(np.zeros(2000), "")},
    ...     linear={},
    ...     model_type="RVModel",
    ...     metadata={"num_chains": 2},
    ... )
    >>> r_hat, ess = convergence(samples)  # the constant "e" is skipped
    >>> bool(r_hat < 1.01), bool(ess > 1000)
    (True, True)
    """
    names = []
    for name in (*samples.nonlinear, *samples.linear):
        value = samples[name]
        draws = np.asarray(getattr(value, "value", value), dtype=float)
        if not np.isfinite(draws).all():
            return float("nan"), float("nan")
        if not np.all(draws == draws[0]):
            names.append(name)
    if not names:
        return float("nan"), float("nan")
    # Imported here, not at the top: arviz pulls in matplotlib, pandas and
    # xarray, nearly doubling the import time of every hq process, and only
    # the MCMC stage uses it (harv's Samples.to_arviz imports it lazily too).
    az = importlib.import_module("arviz")
    idata = samples.to_arviz(names)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_hat = [float(v) for v in az.rhat(idata).data_vars.values()]
        ess = [float(v) for v in az.ess(idata, method="bulk").data_vars.values()]
    return float(np.max(r_hat)), float(np.min(ess))


@final
class RejectionSamples:
    """Read sources' rejection samples, following them through compaction.

    The records come from an index built when the stage started. If a
    compaction of the rejection stage has since removed a part, the stage is
    indexed again and the read retried (spec, "Loading results"); later reads
    use the new index.
    """

    def __init__(self) -> None:
        self._index: ResultsIndex | None = None

    def __call__(self, source_id: Any, record: ResultRecord) -> Samples:
        """The rejection ``Samples`` of ``source_id``, last known at ``record``."""

        def read() -> Samples:
            current = record if self._index is None else self._index.records[source_id]
            try:
                return read_source_samples(current)
            except FileNotFoundError:
                self._index = ResultsIndex.build(record.sources_path.parent)
                raise

        return rescanning(read)


def mcmc_setup(
    config: Config, model_file: ModelFile | None = None
) -> Callable[[Any, MCMCInput], Payload]:
    """Build a run's per-source MCMC function, once per process.

    Calls the model file's ``make_setup()`` and builds one
    ``NumpyroSampler``. Pass it with :func:`functools.partial` as an
    execution driver's ``setup``.

    Parameters
    ----------
    config
        The run's configuration; ``config.mcmc`` must be set.
    model_file
        The model file, when this process has already imported it; imported
        from ``config`` otherwise (pool workers).

    Returns
    -------
        ``process(source_id, (data, rejection_record)) -> Payload``.
    """
    if config.mcmc is None:
        msg = "mcmc_setup needs an [mcmc] table"
        raise ValueError(msg)
    if model_file is None:
        model_file = ModelFile.load(config.run.model_file)
    prior, model = model_file.setup(config.run.kind)
    return functools.partial(
        process_mcmc,
        sampler=NumpyroSampler(prior, model),
        read_rejection=RejectionSamples(),
        config=config.mcmc,
        seed=config.run.seed,
    )


def process_mcmc(
    source_id: Any,
    inputs: MCMCInput,
    *,
    sampler: NumpyroSampler,
    read_rejection: Callable[[Any, ResultRecord], Samples],
    config: MCMCConfig,
    seed: int,
) -> Payload:
    """Run MCMC on one source, started from its rejection samples.

    Failures and warnings are recorded as :func:`~harv_hq.results.process_source`
    describes; a failed source gets ``mcmc_status = "failed"``.

    Parameters
    ----------
    source_id
        The source's ID.
    inputs
        Its prepared data and the record of its rejection result.
    sampler
        The run's ``NumpyroSampler``, built once per process.
    read_rejection
        Reads a source's rejection samples (a :class:`RejectionSamples`).
    config
        The ``[mcmc]`` settings.
    seed
        The run seed. The source's MCMC key is split in two: one half drives
        the sampler, the other the equal-weight resample the chains start from.

    Returns
    -------
        The source's payload, with ``mcmc_status``, ``r_hat_max``, and
        ``ess_bulk_min``.
    """
    data, rejection = inputs
    sampler_key, init_key = jax.random.split(source_key(seed, source_id, "mcmc"))

    def compute() -> tuple[Samples, dict[str, Any], dict[str, str]]:
        init = weighted_resample(
            read_rejection(source_id, rejection), init_key, config.num_chains
        )
        samples = sampler.run(
            data,
            key=sampler_key,
            init_samples=init,
            num_chains=config.num_chains,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            chain_method=config.chain_method,
            return_logprobs=True,
            # One bar per source would flood a pipeline's logs.
            mcmc_kwargs={"progress_bar": False},
        )
        r_hat_max, ess_bulk_min = convergence(samples)
        converged = (
            r_hat_max <= config.max_r_hat and ess_bulk_min >= config.min_ess_bulk
        )
        stats, units = summary_stats(samples, data)
        columns = {
            "mcmc_status": "converged" if converged else "not_converged",
            "r_hat_max": r_hat_max,
            "ess_bulk_min": ess_bulk_min,
            **stats,
        }
        return samples, columns, units

    return process_source(
        source_id,
        data,
        stage="mcmc",
        compute=compute,
        failed_columns={"mcmc_status": "failed"},
    )
