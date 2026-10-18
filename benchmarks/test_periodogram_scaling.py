"""Scaling benchmarks for :func:`harv.periodogram.periodogram`.

One test, parametrized over the grid in :mod:`periodogram_grid`. Run with
``--bench``; see ``docs/benchmarks/running-the-benchmarks.md``.
"""

import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from grid import build_data
from periodogram_grid import (
    PERIOD_MAX,
    PERIOD_MIN,
    TIME_SPAN,
    PeriodogramCell,
    build_prior,
)

# Small-n_obs cells trip the "fewer than two observations per column" warning by
# design, and `filterwarnings = ["error"]` is set repo-wide.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

EAGER_ROUNDS = 3


def test_periodogram(
    benchmark: Any,
    pgram_cell: PeriodogramCell,
    rounds: int,
    device_info: dict[str, Any],
    grid_mode: str,
) -> None:
    """Time one jitted ``periodogram`` call for this grid cell."""
    from harv.periodogram import frequency_grid, periodogram

    cell = pgram_cell
    data = build_data(cell.model_kind, cell.n_obs)
    freq = frequency_grid(
        time_span=TIME_SPAN,
        period_min=PERIOD_MIN,
        period_max=PERIOD_MAX,
        n_grid=cell.n_grid,
    )
    prior, prior_params = build_prior(cell)

    def run(d: Any, f: Any) -> Any:
        return periodogram(
            d, f, prior=prior, n_terms=cell.n_terms, prior_params=prior_params
        ).delta_ln_likelihood

    jitted = jax.jit(run)

    # Cold = compile + execute against a cleared cache, as in the rejection harness.
    jax.clear_caches()
    t0 = time.perf_counter()
    delta = jax.block_until_ready(jitted(data, freq))
    cold_seconds = time.perf_counter() - t0
    assert delta.shape == (cell.n_grid,)
    assert bool(jnp.isfinite(delta).all())

    # Calling periodogram() directly, unjitted, is what a one-off user does. It is
    # dominated by op-by-op dispatch rather than arithmetic, so it gets recorded
    # alongside rather than plotted against the jitted numbers.
    jax.block_until_ready(run(data, freq))
    eager = []
    for _ in range(EAGER_ROUNDS):
        t0 = time.perf_counter()
        jax.block_until_ready(run(data, freq))
        eager.append(time.perf_counter() - t0)

    benchmark.extra_info.update(cell.asdict() | device_info)
    benchmark.extra_info["grid_mode"] = grid_mode
    benchmark.extra_info["cold_seconds"] = cold_seconds
    benchmark.extra_info["eager_seconds"] = statistics.median(eager)

    benchmark.pedantic(
        lambda: jax.block_until_ready(jitted(data, freq)),
        rounds=rounds,
        iterations=1,
        warmup_rounds=1,
    )
