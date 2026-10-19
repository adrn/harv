# Periodograms

Wall-clock cost of one `harv.periodogram.periodogram` call for a single
source, under `jax.jit`, by model, prior mode, number of observations,
number of trial frequencies, and number of harmonics.

:::{note}
Generated from committed results; not rebuilt with the docs. See
{doc}`running-the-benchmarks`.
:::

## Run metadata

| device | date | platform | jax | harv | float64 |
|---|---|---|---|---|---|
| CPU (Apple M4 Max) | 2026-10-18 | Darwin / py3.14.8 | 0.8.1 | 0.0.2.dev181+gba91d27bb.d20261018 | yes |

`flat` is a fixed Gaussian amplitude prior, `period_dependent` a
`LinearPriorCallable` resolved at every trial period, and `profile` is
`prior=False` (amplitudes maximized). `eager` is the median of calling
`periodogram` directly, without `jax.jit`, at the baseline value of the
axis, after a first call has compiled the scan: that is what each call
in a plain Python loop over same-shape sources costs. The difference
from the jitted time is per-op dispatch of the code around the scan.
`slope` is the log-log exponent of the
jitted times. † marks a profile-mode cell where `periodogram` lowered
`n_terms` to keep the trial model overdetermined, so it timed fewer
harmonics than the cell asked for; those cells are left out of `slope`
and the figures.

## Scaling with number of observations

Held at `n_grid=10,000`, `n_terms=2`.

**CPU (Apple M4 Max)**

| series | 8 | 16 | 32 | 64 | 128 | 256 | slope | eager @ 64 |
|---|---|---|---|---|---|---|---|---|
| FourierRV flat | 3.0 ms | 4.6 ms | 8.4 ms | 14.4 ms | 26.4 ms | 50.6 ms | 0.82 | 20.6 ms |
| FourierRV period_dependent | 3.2 ms | 5.0 ms | 8.8 ms | 14.6 ms | 27.3 ms | 51.2 ms | 0.80 | 22.9 ms |
| FourierRV profile | 26.7 ms † | 46.1 ms | 50.8 ms | 60.0 ms | 81.3 ms | 139.6 ms | 0.39 | 63.6 ms |
| FourierGaiaAstrometry flat | 6.3 ms | 7.9 ms | 13.5 ms | 29.1 ms | 47.3 ms | 83.6 ms | 0.79 | 42.8 ms |
| FourierGaiaAstrometry period_dependent | 8.6 ms | 9.3 ms | 14.3 ms | 30.5 ms | 51.2 ms | 91.9 ms | 0.73 | 45.6 ms |
| FourierGaiaAstrometry profile | 61.4 ms † | 95.0 ms † | 220.2 ms | 266.0 ms | 360.9 ms | 517.2 ms | 0.41 | 269.1 ms |

![number of observations](figures/periodogram-n_obs.png)

## Scaling with number of trial frequencies

Held at `n_obs=64`, `n_terms=2`.

**CPU (Apple M4 Max)**

| series | 1,000 | 10,000 | 100,000 | slope | eager @ 10,000 |
|---|---|---|---|---|---|
| FourierRV flat | 1.7 ms | 14.4 ms | 108.3 ms | 0.90 | 20.6 ms |
| FourierRV period_dependent | 1.8 ms | 14.6 ms | 108.7 ms | 0.89 | 22.9 ms |
| FourierRV profile | 6.2 ms | 60.0 ms | 581.8 ms | 0.99 | 63.6 ms |
| FourierGaiaAstrometry flat | 3.6 ms | 29.1 ms | 269.0 ms | 0.94 | 42.8 ms |
| FourierGaiaAstrometry period_dependent | 3.1 ms | 30.5 ms | 283.1 ms | 0.98 | 45.6 ms |
| FourierGaiaAstrometry profile | 27.5 ms | 266.0 ms | 2.60 s | 0.99 | 269.1 ms |

![number of trial frequencies](figures/periodogram-n_grid.png)

## Scaling with number of Fourier harmonics `n_terms`

Held at `n_grid=10,000`, `n_obs=64`.

**CPU (Apple M4 Max)**

| series | 1 | 2 | 3 | 4 | slope | eager @ 2 |
|---|---|---|---|---|---|---|
| FourierRV flat | 6.3 ms | 14.4 ms | 18.2 ms | 21.1 ms | 0.88 | 20.6 ms |
| FourierRV period_dependent | 6.5 ms | 14.6 ms | 18.6 ms | 21.3 ms | 0.87 | 22.9 ms |
| FourierRV profile | 34.7 ms | 60.0 ms | 93.2 ms | 156.1 ms | 1.05 | 63.6 ms |
| FourierGaiaAstrometry flat | 20.9 ms | 29.1 ms | 36.4 ms | 50.3 ms | 0.60 | 42.8 ms |
| FourierGaiaAstrometry period_dependent | 19.6 ms | 30.5 ms | 36.0 ms | 49.5 ms | 0.64 | 45.6 ms |
| FourierGaiaAstrometry profile | 153.5 ms | 266.0 ms | 385.3 ms | 547.4 ms | 0.90 | 269.1 ms |

![number of Fourier harmonics `n_terms`](figures/periodogram-n_terms.png)

## First-call compile cost

Median of `cold - warm` across every cell, the cold call against a cleared
JIT cache. Paid once per distinct data shape, whether or not the caller
wraps `periodogram` in `jax.jit`.

| device | compile |
|---|---|
| CPU (Apple M4 Max) | 161.6 ms |
