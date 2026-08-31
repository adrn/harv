# Benchmarks

:::{warning}
**These are smoke-test numbers, not real results.** They come from a
`--bench-smoke` run: a handful of tiny cells at `M = 1e4` whose only
job is to prove the harness and this report generator work end to end.
The timings are meaningless as a description of harv's performance.
Replace them with a real run -- see {doc}`running-benchmarks`.
:::

Wall-clock scaling of harv's `RejectionSampler`
across model parameterizations, dataset sizes, prior library sizes, and
`batch_size`. Every number here is measured, not modelled.

:::{note}
This page is generated from committed benchmark results and is **not**
rebuilt when the docs build — ReadTheDocs has no GPU. To regenerate it, see
{doc}`running-benchmarks`.
:::

## Run metadata

| device | date | platform | jax | harv | float64 | top_k |
|---|---|---|---|---|---|---|
| CPU (Apple M4 Max) | 2026-08-31 | Darwin / py3.12.10 | 0.8.1 | 0.0.2.dev62+g73df930e8.d20260831 | yes | 256 |

All cells use `top_k` selection, which gives a static output shape so that
timings are not contaminated by recompilation as the accepted-sample count
changes (see the Top-K selection section of the design spec).

## Scaling with number of observations

Varying number of observations.

**CPU (Apple M4 Max)** — median wall time

| parameterization | 8 | 16 | 32 | slope |
|---|---|---|---|---|
| StandardGaiaAstrometry M=10,000 batch=10,000 | 8.8 ms | 10.7 ms | 14.5 ms | +0.36 |
| StandardRV M=10,000 batch=10,000 | 6.3 ms | 7.2 ms | 9.2 ms | +0.27 |

`slope` is the exponent of a log-log least-squares fit: 1.0 is linear
in number of observations, 0.0 is free.

![Scaling with number of observations](_static/benchmarks/n_obs.png)

## First-call compile cost

Median of `cold - warm` across every cell, where the cold call runs against
a cleared JIT cache. This is compilation plus one execution minus one
execution, so it estimates compile time alone.

| parameterization | CPU (Apple M4 Max) |
|---|---|
| StandardGaiaAstrometry | 1.15 s |
| StandardRV | 954.6 ms |

Compile cost is paid once per distinct input shape. In a population loop
over many sources with identical data shapes it is amortized to nothing;
for a single one-off fit it can dominate.
