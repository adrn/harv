"""MCMC follow-up: select sources, and process one with ``NumpyroSampler``.

See ``packages/hq/docs/spec.md``, "MCMC follow-up (``run_mcmc``)". Chains
start from an equal-weight resample of the source's rejection samples.
"""

__all__ = (
    "MCMCInput",
    "convergence",
    "mcmc_setup",
    "process_mcmc",
    "select_sources",
)

import functools
from collections.abc import Callable
from typing import Any

import arviz as az
import jax
import numpy as np
import pyarrow as pa

from harv.data import GaiaAstrometryData, RVData
from harv.samplers import NumpyroSampler, Samples
from harv_hq._model_file import ModelFile
from harv_hq.config import Config, MCMCConfig
from harv_hq.ids import source_key
from harv_hq.results import Payload, ResultRecord, process_source, read_source_samples
from harv_hq.summary_stats import summary_stats, weighted_resample

# One task: the source's data and where its rejection samples live. Workers
# read the samples themselves, so the parent holds only the index.
MCMCInput = tuple[RVData | GaiaAstrometryData, ResultRecord]


def select_sources(
    summary: pa.Table,
    config: MCMCConfig,
    hook: Callable[[dict[str, Any]], bool] | None = None,
) -> set[Any]:
    """The source IDs that go to MCMC follow-up.

    Only sources whose rejection result is ``ok`` can be selected: their
    rejection samples start the chains.

    Parameters
    ----------
    summary
        The ``summary.parquet`` table.
    config
        The ``[mcmc]`` settings (``select``, ``require_unimodal``).
    hook
        The model file's ``select_for_mcmc(row)``; when given, it replaces
        the rule and receives each ``ok`` source's summary row as a dict.

    Returns
    -------
        The selected IDs.
    """

    def rule(row: dict[str, Any]) -> bool:
        return (config.select == "all" or row["well_resolved"] is False) and (
            not config.require_unimodal or row["period_unimodal"] is True
        )

    if hook is None:
        hook = rule
        summary = summary.select(
            ["source_id", "rejection_status", "well_resolved", "period_unimodal"]
        )
    return {
        row["source_id"]
        for row in summary.to_pylist()
        if row["rejection_status"] == "ok" and hook(row)
    }


def convergence(samples: Samples) -> tuple[float, float]:
    """``r_hat_max`` and ``ess_bulk_min`` over every parameter of MCMC samples.

    A parameter constant over every draw has no defined R-hat and is skipped;
    if every parameter is, both are NaN (which fails any threshold).

    Parameters
    ----------
    samples
        MCMC samples, chains recorded by ``metadata["num_chains"]``.

    Returns
    -------
        The largest rank-normalized R-hat and the smallest bulk ESS.
    """
    idata = samples.to_arviz([*samples.nonlinear, *samples.linear])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_hat = np.array([float(v) for v in az.rhat(idata).data_vars.values()])
        ess = np.array(
            [float(v) for v in az.ess(idata, method="bulk").data_vars.values()]
        )
    finite = np.isfinite(r_hat)
    if not finite.any():
        return float("nan"), float("nan")
    return float(r_hat[finite].max()), float(ess[finite].min())


def mcmc_setup(config: Config) -> Callable[[Any, MCMCInput], Payload]:
    """Build a run's per-source MCMC function, once per process.

    Imports the model file, calls ``make_setup()``, and builds one
    ``NumpyroSampler``. Pass it with :func:`functools.partial` as an
    execution driver's ``setup``.

    Parameters
    ----------
    config
        The run's configuration; ``config.mcmc`` must be set.

    Returns
    -------
        ``process(source_id, (data, rejection_record)) -> Payload``.
    """
    if config.mcmc is None:
        msg = "mcmc_setup needs an [mcmc] table"
        raise ValueError(msg)
    prior, model = ModelFile.load(config.run.model_file).setup(config.run.kind)
    return functools.partial(
        process_mcmc,
        sampler=NumpyroSampler(prior, model),
        config=config.mcmc,
        seed=config.run.seed,
    )


def process_mcmc(
    source_id: Any,
    inputs: MCMCInput,
    *,
    sampler: NumpyroSampler,
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
            read_source_samples(rejection), init_key, config.num_chains
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
