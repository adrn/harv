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
    BASELINE,
    PERIOD_MAX,
    PERIOD_MIN,
    TIME_SPAN,
    PeriodogramCell,
)
from unxt import Q

import harv.models as hm
from harv.periodogram import frequency_grid, periodogram

# Small-n_obs cells trip the observations-per-column warning by design, and
# `filterwarnings = ["error"]` is set repo-wide. Only that warning is ignored.
pytestmark = pytest.mark.filterwarnings(
    "ignore:n_terms=.* observations per linear column:UserWarning"
)

EAGER_ROUNDS = 3


def build_prior(cell: PeriodogramCell) -> tuple[Any, dict[str, Any] | None]:
    """Return ``(prior, prior_params)`` to pass to ``periodogram``."""
    if cell.prior == "profile":
        return False, None

    bounds = {"period_min": PERIOD_MIN, "period_max": PERIOD_MAX}
    period_dependent = cell.prior == "period_dependent"
    param = getattr(hm, cell.model)(n_terms=cell.n_terms)

    if cell.model_kind == "rv":
        amp = (
            {"sigma_K0": Q(30.0, "km/s"), "period_ref": Q(365.25, "day")}
            if period_dependent
            else {"sigma_amp": Q(30.0, "km/s")}
        )
        return param.default_prior(**bounds, **amp, sigma_v0=Q(50.0, "km/s")), None

    amp = (
        {"sigma_a0": Q(1.0, "AU"), "period_ref": Q(365.25, "day")}
        if period_dependent
        else {"sigma_amp": Q(10.0, "mas")}
    )
    prior = param.default_prior(
        **bounds,
        **amp,
        sigma_pos=Q(1e3, "mas"),
        sigma_pm=Q(100.0, "mas/yr"),
        sigma_parallax=Q(100.0, "mas"),
    )
    # The period-dependent Gaia prior needs a parallax to turn sigma_a0 into an angle.
    return prior, {"parallax": Q(10.0, "mas")} if period_dependent else None


def test_periodogram(
    benchmark: Any,
    pgram_cell: PeriodogramCell,
    rounds: int,
    device_info: dict[str, Any],
    grid_mode: str,
) -> None:
    """Time one jitted ``periodogram`` call for this grid cell."""
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
        )

    jitted = jax.jit(lambda d, f: run(d, f).delta_ln_likelihood)

    # Cold = compile + execute against a cleared cache, as in the rejection harness.
    jax.clear_caches()
    t0 = time.perf_counter()
    delta = jax.block_until_ready(jitted(data, freq))
    cold_seconds = time.perf_counter() - t0
    assert delta.shape == (cell.n_grid,)
    assert bool(jnp.isfinite(delta).all())

    # Profile mode lowers n_terms when the trial model would be underdetermined,
    # so record what was actually timed, not what was asked for. n_terms is a
    # static field, so eval_shape reads it without running the scan.
    n_terms_effective = jax.eval_shape(run, data, freq).n_terms

    benchmark.extra_info.update(cell.asdict() | device_info)
    benchmark.extra_info["grid_mode"] = grid_mode
    benchmark.extra_info["cold_seconds"] = cold_seconds
    benchmark.extra_info["n_terms_effective"] = n_terms_effective

    # Calling periodogram() directly, unjitted, is what a one-off user does: per-op
    # dispatch outside the scan, plus the scan itself. The report only shows it at
    # each series' baseline point, so it is only measured there.
    baseline = (BASELINE.n_terms, BASELINE.n_obs, BASELINE.n_grid)
    if (cell.n_terms, cell.n_obs, cell.n_grid) == baseline:
        jax.block_until_ready(run(data, freq))  # warm the scan's compile cache
        eager = []
        for _ in range(EAGER_ROUNDS):
            t0 = time.perf_counter()
            jax.block_until_ready(run(data, freq))
            eager.append(time.perf_counter() - t0)
        benchmark.extra_info["eager_seconds"] = statistics.median(eager)

    benchmark.pedantic(
        lambda: jax.block_until_ready(jitted(data, freq)),
        rounds=rounds,
        iterations=1,
        warmup_rounds=1,
    )
