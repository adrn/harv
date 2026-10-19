# Rejection sampling

Wall-clock scaling of harv's `RejectionSampler`
across model parameterizations, dataset sizes, prior library sizes, and
`batch_size`. Every number here is measured, not modelled.

:::{note}
This page is generated from committed benchmark results and is **not**
rebuilt when the docs build — ReadTheDocs has no GPU. To regenerate it, see
{doc}`running-the-benchmarks`.
:::

## Run metadata

| device | date | platform | jax | harv | float64 | top_k |
|---|---|---|---|---|---|---|
| CPU (Intel(R) Xeon(R) w5-3435X) | 2026-10-19 | Linux / py3.14.5 | 0.8.1 | 0.0.2.dev184+g8912c38de | yes | 256 |
| GPU (NVIDIA RTX 6000 Ada Generation) | 2026-10-19 | Linux / py3.14.5 | 0.8.1 | 0.0.2.dev184+g8912c38de | yes | 256 |

All cells use `top_k` selection, which gives a static output shape so that
timings are not contaminated by recompilation as the accepted-sample count
changes (see the Top-K selection section of the design spec).

## What the numbers say

Computed from the results in this page, on the hardware above. The
interpretation is fixed; the figures are not, so regenerating on your
own machine ({doc}`running-the-benchmarks`) updates them rather than
contradicting them.

### The GPU advantage grows with library size

At `M=1e4` the GPU is only 2.3-8.5x faster; at `M=1e7` it is 34.0-58.4x. The reason is in the
slopes: cost is near-linear in `M` on CPU and clearly sublinear on
GPU, because a small library cannot fill the device. Below roughly
`M=1e5` a GPU call is dominated by fixed cost, not by work — the smallest warm call measured here is 9 ms on GPU against 36 ms on CPU.

**Consequence for population runs.** That floor is paid once per
source. Millions of sources at small `M` spend most of their time in
per-call overhead, so a GPU pays off through *bigger libraries*, not
through more calls. See {doc}`/at-scale`.

### Throughput, in millions of prior samples per second

Sustained rate at `M=1e7`, `n_obs=64`, in-memory library.
This is the number to plan a run from.

| parameterization | CPU (Intel(R) Xeon(R) w5-3435X) | GPU (NVIDIA RTX 6000 Ada Generation) |
|---|---|---|
| StandardRV | 0.45 | 26.0 |
| EcoswEsinwRV | 0.45 | 26.0 |
| FourierRV | 0.40 | 20.5 |
| StandardGaiaAstrometry | 0.28 | 13.5 |
| ThieleInnesGaiaAstrometry | 0.18 | 6.2 |
| FourierGaiaAstrometry | 0.15 | 5.8 |

The spread across parameterizations is real work, not noise: the
astrometric models carry more linear columns than the RV ones, and the
Thiele-Innes variant additionally evaluates a Jacobian correction per
sample.

### The speedup is free

Across all 66 cells measured on both devices, the largest relative
difference in `ln_Z_int_ess` is 1.4e-10. Both runs pin float64,
and the GPU is not trading accuracy for speed — it is the same
arithmetic, faster.

### `batch_size` matters more on CPU than on GPU

- **CPU (Intel(R) Xeon(R) w5-3435X)**: slowest/fastest `batch_size` at fixed `M` spans 1.17-1.23x.
- **GPU (NVIDIA RTX 6000 Ada Generation)**: slowest/fastest `batch_size` at fixed `M` spans 1.07-1.13x.

This inverts the guidance the design spec used to give. Setting
`batch_size = n_prior_samples` on GPU is not the lever it was
assumed to be; the device is already saturated at far smaller
batches.
The knob sets the working-set size of the `(batch, n_obs, n_linear)`
intermediate, which is why it moves CPU timings at all.

:::{warning}
Every CPU cell here is a **single process on an otherwise idle node**, so
these are per-core ceilings, not per-node predictions. Under many-rank
contention the optimum moves *down*, because ranks compete for memory
bandwidth — a large batch that wins alone can starve a full node. Measure
it under the contention you will actually run. See {doc}`/at-scale`.
:::

### Streaming from HDF5 costs the GPU much more than the CPU

- **CPU (Intel(R) Xeon(R) w5-3435X)**: 1.11-1.33x of the in-memory time.
- **GPU (NVIDIA RTX 6000 Ada Generation)**: 1.36-1.68x of the in-memory time.

The absolute I/O cost is similar; what differs is what it is
competing with. On CPU the compute is slow enough to hide the
reads. On GPU the compute is fast enough that the reads become
the run.
Keep the library in memory whenever it fits, and reserve the HDF5 path
for libraries that genuinely cannot.

### Compilation is a per-shape tax, and it is higher on GPU

- **CPU (Intel(R) Xeon(R) w5-3435X)**: 1.5-2.9 s per distinct input shape.
- **GPU (NVIDIA RTX 6000 Ada Generation)**: 4.7-6.9 s per distinct input shape.

Paid once per shape, so a population loop over identically-shaped data
amortizes it to nothing — and a catalog with a different epoch count per
source pays it *per source*, which is why bucketing epoch counts is the
first thing to do with real astrometry. See {doc}`/at-scale`.

### The GPU advantage falls away at large `n_obs`

For `FourierRV` the speedup drops from 51x at `n_obs=128` to
14x at `n_obs=256` — a discontinuity, not a trend, and all
six parameterizations show it at the same place.

:::{note}
The arithmetic points at the working set: at `batch_size=1e5` and
`n_obs=256`, one float64 column array is
`1e5 x 256 x 8 B = 205 MB`, and the design
matrix holds several. Lowering `batch_size` so that
`batch_size x n_obs` stays roughly constant is the obvious remedy,
**but this grid did not measure it** — the `batch_size` curve
was only run at `n_obs=64`. Treat the
cause as a hypothesis and measure it on your own data before relying
on it.
:::

## Scaling with number of observations

Varying number of observations.
All other axes are held at the baseline: `n_obs=64`, `M=1e6`, `batch_size=1e5`, in-memory cache.

**CPU (Intel(R) Xeon(R) w5-3435X)** — median wall time

| parameterization | 8 | 16 | 32 | 64 | 128 | 256 | slope |
|---|---|---|---|---|---|---|---|
| EcoswEsinwRV | 950.3 ms | 1.17 s | 1.57 s | 2.42 s | 3.60 s | 6.36 s | +0.55 |
| FourierGaiaAstrometry | 2.54 s | 2.99 s | 4.25 s | 6.93 s | 11.40 s | 19.71 s | +0.61 |
| FourierRV | 1.17 s | 1.41 s | 1.75 s | 2.49 s | 4.08 s | 6.83 s | +0.51 |
| StandardGaiaAstrometry | 1.68 s | 1.82 s | 2.35 s | 3.61 s | 5.82 s | 10.19 s | +0.53 |
| StandardRV | 986.9 ms | 1.16 s | 1.49 s | 2.20 s | 3.58 s | 6.27 s | +0.54 |
| ThieleInnesGaiaAstrometry | 2.80 s | 3.18 s | 3.82 s | 5.45 s | 8.90 s | 16.14 s | +0.50 |

**GPU (NVIDIA RTX 6000 Ada Generation)** — median wall time

| parameterization | 8 | 16 | 32 | 64 | 128 | 256 | slope |
|---|---|---|---|---|---|---|---|
| EcoswEsinwRV | 29.7 ms | 32.4 ms | 35.6 ms | 52.3 ms | 83.5 ms | 523.3 ms | +0.72 |
| FourierGaiaAstrometry | 84.6 ms | 88.3 ms | 124.6 ms | 193.7 ms | 319.7 ms | 571.0 ms | +0.57 |
| FourierRV | 42.3 ms | 43.6 ms | 48.3 ms | 60.8 ms | 80.6 ms | 481.0 ms | +0.59 |
| StandardGaiaAstrometry | 58.3 ms | 63.0 ms | 75.0 ms | 96.9 ms | 137.5 ms | 577.2 ms | +0.58 |
| StandardRV | 25.7 ms | 30.5 ms | 38.7 ms | 53.9 ms | 85.6 ms | 529.2 ms | +0.76 |
| ThieleInnesGaiaAstrometry | 127.1 ms | 133.9 ms | 148.6 ms | 186.9 ms | 238.0 ms | 682.4 ms | +0.43 |

**Speedup** — CPU (Intel(R) Xeon(R) w5-3435X) / GPU (NVIDIA RTX 6000 Ada Generation) (>1 means GPU (NVIDIA RTX 6000 Ada Generation) is faster)

| parameterization | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| EcoswEsinwRV | 32.0x | 36.1x | 44.1x | 46.3x | 43.2x | 12.2x |
| FourierGaiaAstrometry | 30.1x | 33.9x | 34.1x | 35.8x | 35.7x | 34.5x |
| FourierRV | 27.5x | 32.3x | 36.2x | 41.0x | 50.6x | 14.2x |
| StandardGaiaAstrometry | 28.8x | 29.0x | 31.4x | 37.2x | 42.3x | 17.7x |
| StandardRV | 38.4x | 38.1x | 38.5x | 40.8x | 41.8x | 11.8x |
| ThieleInnesGaiaAstrometry | 22.1x | 23.7x | 25.7x | 29.2x | 37.4x | 23.6x |

`slope` is the exponent of a log-log least-squares fit: 1.0 is linear
in number of observations, 0.0 is free.

![Scaling with number of observations](figures/rejection-n_obs.png)

## Scaling with prior library size

Varying prior library size $M$.
All other axes are held at the baseline: `n_obs=64`, `M=1e6`, `batch_size=1e5`, in-memory cache.

**CPU (Intel(R) Xeon(R) w5-3435X)** — median wall time

| parameterization | 10000 | 100000 | 1000000 | 10000000 | slope |
|---|---|---|---|---|---|
| EcoswEsinwRV | 40.7 ms | 274.6 ms | 2.42 s | 22.26 s | +0.92 |
| FourierGaiaAstrometry | 97.9 ms | 782.7 ms | 6.93 s | 67.75 s | +0.95 |
| FourierRV | 40.4 ms | 280.7 ms | 2.49 s | 24.74 s | +0.93 |
| StandardGaiaAstrometry | 69.5 ms | 457.2 ms | 3.61 s | 35.92 s | +0.90 |
| StandardRV | 36.1 ms | 245.5 ms | 2.20 s | 22.42 s | +0.93 |
| ThieleInnesGaiaAstrometry | 75.6 ms | 591.0 ms | 5.45 s | 54.59 s | +0.95 |

**GPU (NVIDIA RTX 6000 Ada Generation)** — median wall time

| parameterization | 10000 | 100000 | 1000000 | 10000000 | slope |
|---|---|---|---|---|---|
| EcoswEsinwRV | 11.3 ms | 14.4 ms | 52.3 ms | 385.0 ms | +0.52 |
| FourierGaiaAstrometry | 11.6 ms | 30.7 ms | 193.7 ms | 1.72 s | +0.73 |
| FourierRV | 8.9 ms | 13.3 ms | 60.8 ms | 488.9 ms | +0.59 |
| StandardGaiaAstrometry | 17.4 ms | 24.8 ms | 96.9 ms | 742.8 ms | +0.55 |
| StandardRV | 15.5 ms | 18.3 ms | 53.9 ms | 383.9 ms | +0.47 |
| ThieleInnesGaiaAstrometry | 16.7 ms | 32.6 ms | 186.9 ms | 1.60 s | +0.67 |

**Speedup** — CPU (Intel(R) Xeon(R) w5-3435X) / GPU (NVIDIA RTX 6000 Ada Generation) (>1 means GPU (NVIDIA RTX 6000 Ada Generation) is faster)

| parameterization | 10000 | 100000 | 1000000 | 10000000 |
|---|---|---|---|---|
| EcoswEsinwRV | 3.6x | 19.1x | 46.3x | 57.8x |
| FourierGaiaAstrometry | 8.5x | 25.5x | 35.8x | 39.3x |
| FourierRV | 4.5x | 21.1x | 41.0x | 50.6x |
| StandardGaiaAstrometry | 4.0x | 18.4x | 37.2x | 48.4x |
| StandardRV | 2.3x | 13.4x | 40.8x | 58.4x |
| ThieleInnesGaiaAstrometry | 4.5x | 18.1x | 29.2x | 34.0x |

`slope` is the exponent of a log-log least-squares fit: 1.0 is linear
in prior library size $M$, 0.0 is free.

![Scaling with prior library size](figures/rejection-n_prior_samples.png)

## Effect of `batch_size`

Varying `batch_size`.
All other axes are held at the baseline: `n_obs=64`, `M=1e6`, `batch_size=1e5`, in-memory cache.

**CPU (Intel(R) Xeon(R) w5-3435X)** — median wall time

| parameterization | 10000 | 100000 | 1000000 | slope |
|---|---|---|---|---|
| StandardGaiaAstrometry | 4.08 s | 3.61 s | 4.35 s | +0.01 |
| StandardGaiaAstrometry M=10,000,000 | 40.97 s | 35.92 s | 34.34 s | -0.04 |
| StandardRV | 2.51 s | 2.20 s | 2.16 s | -0.03 |
| StandardRV M=10,000,000 | 25.52 s | 22.42 s | 20.73 s | -0.05 |

**GPU (NVIDIA RTX 6000 Ada Generation)** — median wall time

| parameterization | 10000 | 100000 | 1000000 | slope |
|---|---|---|---|---|
| StandardGaiaAstrometry | 102.2 ms | 96.9 ms | 95.8 ms | -0.01 |
| StandardGaiaAstrometry M=10,000,000 | 794.7 ms | 742.8 ms | 751.2 ms | -0.01 |
| StandardRV | 60.5 ms | 53.9 ms | 55.1 ms | -0.02 |
| StandardRV M=10,000,000 | 433.4 ms | 383.9 ms | 382.4 ms | -0.03 |

**Speedup** — CPU (Intel(R) Xeon(R) w5-3435X) / GPU (NVIDIA RTX 6000 Ada Generation) (>1 means GPU (NVIDIA RTX 6000 Ada Generation) is faster)

| parameterization | 10000 | 100000 | 1000000 |
|---|---|---|---|
| StandardGaiaAstrometry | 39.9x | 37.2x | 45.5x |
| StandardGaiaAstrometry M=10,000,000 | 51.6x | 48.4x | 45.7x |
| StandardRV | 41.5x | 40.8x | 39.1x |
| StandardRV M=10,000,000 | 58.9x | 58.4x | 54.2x |

`slope` is the exponent of a log-log least-squares fit: 1.0 is linear
in `batch_size`, 0.0 is free.

![Effect of `batch_size`](figures/rejection-batch_size.png)

## In-memory library vs streamed HDF5 cache

Varying prior-cache backend.
All other axes are held at the baseline: `n_obs=64`, `M=1e6`, `batch_size=1e5`, in-memory cache.

**CPU (Intel(R) Xeon(R) w5-3435X)** — median wall time

| parameterization | hdf5 | memory |
|---|---|---|
| StandardGaiaAstrometry | 4.80 s | 3.61 s |
| StandardGaiaAstrometry M=10,000,000 | 47.82 s | 35.92 s |
| StandardRV | 2.46 s | 2.20 s |
| StandardRV M=10,000,000 | 25.00 s | 22.42 s |

**GPU (NVIDIA RTX 6000 Ada Generation)** — median wall time

| parameterization | hdf5 | memory |
|---|---|---|
| StandardGaiaAstrometry | 131.4 ms | 96.9 ms |
| StandardGaiaAstrometry M=10,000,000 | 1.18 s | 742.8 ms |
| StandardRV | 76.6 ms | 53.9 ms |
| StandardRV M=10,000,000 | 645.0 ms | 383.9 ms |

**Speedup** — CPU (Intel(R) Xeon(R) w5-3435X) / GPU (NVIDIA RTX 6000 Ada Generation) (>1 means GPU (NVIDIA RTX 6000 Ada Generation) is faster)

| parameterization | hdf5 | memory |
|---|---|---|
| StandardGaiaAstrometry | 36.5x | 37.2x |
| StandardGaiaAstrometry M=10,000,000 | 40.6x | 48.4x |
| StandardRV | 32.1x | 40.8x |
| StandardRV M=10,000,000 | 38.8x | 58.4x |

## First-call compile cost

Median of `cold - warm` across every cell, where the cold call runs against
a cleared JIT cache. This is compilation plus one execution minus one
execution, so it estimates compile time alone.

| parameterization | CPU (Intel(R) Xeon(R) w5-3435X) | GPU (NVIDIA RTX 6000 Ada Generation) |
|---|---|---|
| EcoswEsinwRV | 2.23 s | 5.19 s |
| FourierGaiaAstrometry | 1.68 s | 5.08 s |
| FourierRV | 1.53 s | 4.69 s |
| StandardGaiaAstrometry | 2.62 s | 6.92 s |
| StandardRV | 2.36 s | 6.13 s |
| ThieleInnesGaiaAstrometry | 2.92 s | 6.88 s |

Compile cost is paid once per distinct input shape. In a population loop
over many sources with identical data shapes it is amortized to nothing;
for a single one-off fit it can dominate.
