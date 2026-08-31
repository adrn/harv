"""Scaling benchmarks for :class:`harv.samplers.rejection.RejectionSampler`.

One test, parametrized over the grid in :mod:`grid`. Run with ``--bench``; see
``docs/running-benchmarks.md``.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import pytest
from grid import TOP_K, Cell, build_data, build_prior_and_model

# The sampler warns when `logZ_int_ess < 3` -- the library did not resolve this
# posterior. Some cells (small M, sharply peaked likelihood) will trip it by
# construction, and `filterwarnings = ["error"]` is set repo-wide. These
# benchmarks measure throughput, not resolution quality, so the warning is noise
# here. It is emphatically *not* noise in the test suite, which is why this is
# scoped to this module.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_sampler(cell: Cell) -> tuple[Any, Any]:
    from harv.samplers.rejection import RejectionSampler

    prior, model = build_prior_and_model(cell.parameterization)
    data = build_data(cell.model_kind, cell.n_obs)
    return RejectionSampler(prior, model, batch_size=cell.batch_size), data


def test_run_with_samples(
    benchmark: Any,
    cell: Cell,
    prior_cache: Any,
    rounds: int,
    device_info: dict[str, Any],
    grid_mode: str,
) -> None:
    """Time one ``run_with_samples`` call for this grid cell."""
    sampler, data = _make_sampler(cell)

    def call() -> Any:
        # block_until_ready is load-bearing on GPU: JAX dispatch is async, so
        # without it the timer measures how fast we can enqueue work. `Samples`
        # is an eqx.Module and therefore a pytree, so this blocks on every leaf.
        return jax.block_until_ready(
            sampler.run_with_samples(
                data,
                prior_cache,
                top_k=TOP_K,
                seed=0,
                randomize_prior_order=False,
                # Uniform across cells, so they stay comparable. Required for
                # EcoswEsinwRV, whose default prior puts ~21% of draws outside the
                # unit disk (e >= 1) where the K prior is NaN; without it the
                # recorded evidence metadata is NaN. See docs/sharp-bits.md.
                ignore_non_finite=True,
            )
        )

    # Cold call = compile + execute, measured against a cleared cache. This is
    # the number docs/spec.md ("First-call JIT compile time has not been
    # systematically benchmarked") is missing; `cold - warm median` estimates it.
    jax.clear_caches()
    t0 = time.perf_counter()
    result = call()
    cold_seconds = time.perf_counter() - t0

    n_returned = int(result.n_samples)
    assert n_returned == TOP_K, f"top_k contract broken: got {n_returned} rows"

    benchmark.extra_info.update(cell.asdict() | device_info)
    benchmark.extra_info["grid_mode"] = grid_mode
    benchmark.extra_info["cold_seconds"] = cold_seconds
    benchmark.extra_info["weight_captured"] = float(
        result.metadata.get("weight_captured", float("nan"))
    )
    benchmark.extra_info["evidence_ess"] = float(
        result.metadata.get("logZ_int_ess", float("nan"))
    )

    # pedantic: no calibration, explicit warmup. The automatic mode would size
    # rounds from a sub-second probe, which is meaningless for calls that take
    # seconds and whose first call is dominated by compilation.
    benchmark.pedantic(call, rounds=rounds, iterations=1, warmup_rounds=1)
