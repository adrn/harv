# Running the benchmarks

How to regenerate {doc}`benchmarks`. This is a developer page — the benchmarks
are **not** part of the test suite and never run on CI.

## TL;DR

```bash
# One-time: install the benchmark tooling
uv sync --group bench

# Confirm JAX actually sees the GPU. If this prints only CpuDevice, the "gpu" run
# below silently produces a SECOND CPU dataset -- see "Check the device first".
uv run --no-sync python -c "import jax; print(jax.devices())"

# Verify the harness works (~15 s, 6 tiny cells, exercises the whole pipeline).
# Writes outside benchmarks/results/ so it cannot contaminate a real run, and
# renders to its own file so it cannot overwrite the committed docs page.
HARV_NO_TYPECHECK=1 uv run --no-sync pytest benchmarks/ --bench --bench-smoke \
    --benchmark-json=benchmarks/smoke/smoke.json
uv run --no-sync python benchmarks/report.py \
    --results benchmarks/smoke --out benchmarks/smoke/benchmarks.md --include-smoke

# The real thing, on the workstation (hours per device).
# report.py merges EVERY *.json in benchmarks/results/, so clear out old runs first.
rm -f benchmarks/results/*.json
HARV_NO_TYPECHECK=1 JAX_PLATFORMS=cpu uv run --no-sync pytest benchmarks/ --bench \
    --benchmark-json=benchmarks/results/cpu.json
HARV_NO_TYPECHECK=1 uv run --no-sync pytest benchmarks/ --bench \
    --benchmark-json=benchmarks/results/gpu.json

# Regenerate the docs page + figures from whatever is in benchmarks/results/
uv run --no-sync python benchmarks/report.py

git add benchmarks/results docs/benchmarks.md docs/_static/benchmarks
```

## Check the device first

`jax` falls back to CPU when a CUDA-enabled `jaxlib` is not installed, printing only
a warning:

```
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not
installed. Falling back to cpu.
```

Nothing in the filename makes a run a GPU run — `report.py` labels each device from
`jax.devices()[0]`. So a fallback means `gpu.json` is a second CPU dataset under the
same device label as `cpu.json`, and one would overwrite the other. `report.py` now
refuses that case by name rather than silently dropping half the data, but the fix is
to install CUDA-enabled jaxlib (`jax[cuda12]`) before spending hours on the run.

Run the one-liner above and check for a `CudaDevice` before starting.

## Why the incantation looks like that

`JAX_PLATFORMS=cpu`
: Forces the CPU run. Without it JAX picks the GPU when one is visible. The GPU
run just omits it. harv has no device-placement code of its own, so this
environment variable is the entire device-selection mechanism.

`HARV_NO_TYPECHECK=1`
: Skips the beartype/jaxtyping import hooks in the root `conftest.py`. Those hooks
only fire at JAX *trace* time, so they do not affect warm timings — but they do
inflate the first-call compile number the report attributes to compilation.
Leave it unset for the ordinary test suite, which wants the checks.

`--bench`
: Required. Without it every benchmark skips, so a stray `pytest benchmarks/`
cannot start a multi-hour run by accident.

`--benchmark-json=...`
: pytest-benchmark prints a table but does not persist anything by default. The
JSON is what `report.py` consumes; without this flag a finished run leaves
nothing behind. pytest-benchmark writes it from `pytest_sessionfinish` — *after*
every benchmark has run — so `benchmarks/conftest.py` creates the parent
directory and checks it is writable at startup. Otherwise a typo'd path would
discard hours of compute at the very end.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--bench` | off | Required opt-in. Everything skips without it. |
| `--bench-smoke` | off | 6 tiny cells that exercise tables, slope fitting, and plotting in ~15 s. |
| `--bench-full` | off | Full cartesian product (~650 cells) instead of the star design. |
| `--bench-rounds N` | 5 | Timed rounds per cell, after one warmup round. `3` cuts roughly a third off the wall clock. |
| `--bench-cache-dir DIR` | pytest temp | Where HDF5 prior caches live. Point it somewhere persistent to reuse them between runs. |

Useful pytest selectors: `-k StandardRV` runs one parameterization,
`-k "n64 and memory"` one slice of the grid.

## What is measured

One `RejectionSampler.run_with_samples(...)` call per grid cell, with `top_k=256`.
`top_k` is deliberate: it gives a **static** output shape, so the `jax.vmap` over
the conditional-linear solve is traced once instead of once per distinct
acceptance count. Under plain rejection, timings would partly measure
recompilation driven by the data rather than by the axis under test.

Each cell records a cold call against a cleared JIT cache, then a warmup round,
then `--bench-rounds` timed rounds. Every timed call is wrapped in
`jax.block_until_ready` — on GPU, JAX dispatch is asynchronous, and without it
the timer would measure how fast Python can enqueue work.

float64 is pinned on (`benchmarks/conftest.py` sets it before harv is imported).
harv does not enable x64 itself and {doc}`sharp-bits` makes it the user's job, but
every tutorial turns it on, and float32 changes the sampler's *arithmetic* rather
than just its precision — a float32 timing would not describe how anyone runs harv.

## The grid

A **star** design: each curve varies one axis with the rest pinned to the baseline
(`n_obs=64`, `M=1e6`, `batch_size=1e5`, in-memory cache, `top_k=256`).

| Curve | Varies | Over |
|---|---|---|
| `n_obs` | 8 … 256 observations | all 6 parameterizations |
| `n_prior_samples` | 1e4 … 1e7 | all 6 parameterizations |
| `batch_size` | 1e4, 1e5, 1e6 | `StandardRV`, `StandardGaiaAstrometry` at M = 1e6, 1e7 |
| `backend` | in-memory vs HDF5 | `StandardRV`, `StandardGaiaAstrometry` at M = 1e6, 1e7 |

66 measurements after deduplication (curves share baseline points; they are timed
once and reported by all curves that contain them). `--bench-full` expands to the
full product when a curve looks like it is hiding an interaction.

Axes, baselines, and the simulated systems are defined in `benchmarks/grid.py` —
that file is the source of truth, this table is a summary.

## Cost and resources

Expect **1–3 hours per device**, dominated by the ~16 cells at `M = 1e7`.

Memory: prior libraries are built once per parameterization at that
parameterization's largest required size and sliced down, with an
`lru_cache(maxsize=1)` so only one is resident. At `M = 1e7` a Gaia library is
roughly 500 MB. On GPU it lives in device memory.

Disk: the HDF5 caches for the `backend` curve need ~2 GB. They go to a pytest temp
directory unless `--bench-cache-dir` says otherwise.

## Why this never runs on CI

Three independent reasons, so that no single mistake exposes CI to a multi-hour job:

1. `benchmarks/` is outside `testpaths = ["README.md", "tests", "src"]`, and CI runs
   bare `pytest` with no path arguments.
2. `pytest-benchmark` lives in the `bench` dependency group. CI installs
   `uv sync --no-default-groups --group test`, so the plugin is not even present.
3. `--bench` is required; without it every benchmark skips.

To confirm belt 1 still holds after touching pytest config:

```bash
uv run --no-sync pytest --collect-only -q | grep -c benchmarks/   # must print 0
```

## Adding a parameterization or an axis

Both live in `benchmarks/grid.py`:

- **New parameterization**: add its name to `PARAMETERIZATIONS` and a `case` to
  `build_prior_and_model`. The Fourier parameterizations have no data-driven prior
  defaults by design, so every scale must be passed explicitly.
- **New axis**: add the values, add a curve to `curve_definitions`, and add the
  axis to `AXIS_LABELS` / `NUMERIC_AXES` in `benchmarks/report.py`. Numeric axes
  get a fitted log-log slope column and a figure; categorical ones get a table.

Run `--bench-smoke` after either — it is the check that catches a broken
`report.py` before, rather than after, a three-hour run.
