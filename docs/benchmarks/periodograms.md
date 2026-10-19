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
| CPU (Intel(R) Xeon(R) w5-3435X) | 2026-10-19 | Linux / py3.14.5 | 0.8.1 | 0.0.2.dev184+g8912c38de | yes |
| GPU (NVIDIA RTX 6000 Ada Generation) | 2026-10-19 | Linux / py3.14.5 | 0.8.1 | 0.0.2.dev184+g8912c38de | yes |

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

**CPU (Intel(R) Xeon(R) w5-3435X)**

| series | 8 | 16 | 32 | 64 | 128 | 256 | slope | eager @ 64 |
|---|---|---|---|---|---|---|---|---|
| FourierRV flat | 12.6 ms | 16.3 ms | 17.6 ms | 31.7 ms | 52.7 ms | 97.9 ms | 0.59 | 44.7 ms |
| FourierRV period_dependent | 12.7 ms | 13.6 ms | 17.3 ms | 28.3 ms | 54.6 ms | 101.6 ms | 0.62 | 50.2 ms |
| FourierRV profile | 30.4 ms † | 66.8 ms | 80.8 ms | 99.1 ms | 149.8 ms | 236.0 ms | 0.45 | 112.9 ms |
| FourierGaiaAstrometry flat | 21.1 ms | 28.9 ms | 59.9 ms | 107.4 ms | 190.4 ms | 340.0 ms | 0.83 | 138.1 ms |
| FourierGaiaAstrometry period_dependent | 53.9 ms | 50.0 ms | 65.3 ms | 115.4 ms | 205.0 ms | 352.6 ms | 0.58 | 149.9 ms |
| FourierGaiaAstrometry profile | 95.2 ms † | 152.9 ms † | 308.3 ms | 363.2 ms | 491.6 ms | 734.5 ms | 0.42 | 380.5 ms |

**GPU (NVIDIA RTX 6000 Ada Generation)**

| series | 8 | 16 | 32 | 64 | 128 | 256 | slope | eager @ 64 |
|---|---|---|---|---|---|---|---|---|
| FourierRV flat | 0.6 ms | 0.6 ms | 0.7 ms | 0.7 ms | 0.9 ms | 5.3 ms | 0.52 | 19.2 ms |
| FourierRV period_dependent | 0.6 ms | 0.6 ms | 0.7 ms | 0.8 ms | 0.9 ms | 5.4 ms | 0.50 | 25.8 ms |
| FourierRV profile | 12.0 ms † | 17.2 ms | 18.7 ms | 8.64 s | 8.58 s | 8.55 s | 2.68 | 8.66 s |
| FourierGaiaAstrometry flat | 1.1 ms | 1.1 ms | 1.6 ms | 2.1 ms | 3.7 ms | 6.3 ms | 0.52 | 46.3 ms |
| FourierGaiaAstrometry period_dependent | 1.8 ms | 1.9 ms | 2.2 ms | 2.9 ms | 4.3 ms | 7.1 ms | 0.39 | 46.8 ms |
| FourierGaiaAstrometry profile | 27.6 ms † | 33.0 ms † | 52.5 ms | 15.27 s | 15.79 s | 16.25 s | 2.49 | 16.13 s |

![number of observations](figures/periodogram-n_obs.png)

## Scaling with number of trial frequencies

Held at `n_obs=64`, `n_terms=2`.

**CPU (Intel(R) Xeon(R) w5-3435X)**

| series | 1,000 | 10,000 | 100,000 | slope | eager @ 10,000 |
|---|---|---|---|---|---|
| FourierRV flat | 3.3 ms | 31.7 ms | 234.9 ms | 0.92 | 44.7 ms |
| FourierRV period_dependent | 4.4 ms | 28.3 ms | 236.4 ms | 0.86 | 50.2 ms |
| FourierRV profile | 9.6 ms | 99.1 ms | 947.8 ms | 1.00 | 112.9 ms |
| FourierGaiaAstrometry flat | 11.2 ms | 107.4 ms | 868.3 ms | 0.95 | 138.1 ms |
| FourierGaiaAstrometry period_dependent | 10.9 ms | 115.4 ms | 906.9 ms | 0.96 | 149.9 ms |
| FourierGaiaAstrometry profile | 41.4 ms | 363.2 ms | 3.54 s | 0.97 | 380.5 ms |

**GPU (NVIDIA RTX 6000 Ada Generation)**

| series | 1,000 | 10,000 | 100,000 | slope | eager @ 10,000 |
|---|---|---|---|---|---|
| FourierRV flat | 0.3 ms | 0.7 ms | 5.3 ms | 0.62 | 19.2 ms |
| FourierRV period_dependent | 0.3 ms | 0.8 ms | 5.7 ms | 0.62 | 25.8 ms |
| FourierRV profile | 866.2 ms | 8.64 s | 86.37 s | 1.00 | 8.66 s |
| FourierGaiaAstrometry flat | 0.9 ms | 2.1 ms | 19.1 ms | 0.66 | 46.3 ms |
| FourierGaiaAstrometry period_dependent | 0.6 ms | 2.9 ms | 27.4 ms | 0.84 | 46.8 ms |
| FourierGaiaAstrometry profile | 1.58 s | 15.27 s | 152.70 s | 0.99 | 16.13 s |

![number of trial frequencies](figures/periodogram-n_grid.png)

## Scaling with number of Fourier harmonics `n_terms`

Held at `n_grid=10,000`, `n_obs=64`.

**CPU (Intel(R) Xeon(R) w5-3435X)**

| series | 1 | 2 | 3 | 4 | slope | eager @ 2 |
|---|---|---|---|---|---|---|
| FourierRV flat | 16.7 ms | 31.7 ms | 55.1 ms | 59.2 ms | 0.97 | 44.7 ms |
| FourierRV period_dependent | 16.5 ms | 28.3 ms | 57.8 ms | 57.6 ms | 0.98 | 50.2 ms |
| FourierRV profile | 52.4 ms | 99.1 ms | 156.5 ms | 215.1 ms | 1.02 | 112.9 ms |
| FourierGaiaAstrometry flat | 55.8 ms | 107.4 ms | 138.4 ms | 176.0 ms | 0.82 | 138.1 ms |
| FourierGaiaAstrometry period_dependent | 69.2 ms | 115.4 ms | 102.5 ms | 140.6 ms | 0.45 | 149.9 ms |
| FourierGaiaAstrometry profile | 199.6 ms | 363.2 ms | 579.1 ms | 834.5 ms | 1.02 | 380.5 ms |

**GPU (NVIDIA RTX 6000 Ada Generation)**

| series | 1 | 2 | 3 | 4 | slope | eager @ 2 |
|---|---|---|---|---|---|---|
| FourierRV flat | 0.5 ms | 0.7 ms | 1.0 ms | 1.4 ms | 0.69 | 19.2 ms |
| FourierRV period_dependent | 0.5 ms | 0.8 ms | 1.1 ms | 1.6 ms | 0.76 | 25.8 ms |
| FourierRV profile | 6.25 s | 8.64 s | 11.09 s | 12.38 s | 0.50 | 8.66 s |
| FourierGaiaAstrometry flat | 1.4 ms | 2.1 ms | 2.6 ms | 3.6 ms | 0.64 | 46.3 ms |
| FourierGaiaAstrometry period_dependent | 1.6 ms | 2.9 ms | 4.9 ms | 6.6 ms | 1.04 | 46.8 ms |
| FourierGaiaAstrometry profile | 13.07 s | 15.27 s | 19.14 s | 23.65 s | 0.42 | 16.13 s |

![number of Fourier harmonics `n_terms`](figures/periodogram-n_terms.png)

## First-call compile cost

Median of `cold - warm` across every cell, the cold call against a cleared
JIT cache. Paid once per distinct data shape, whether or not the caller
wraps `periodogram` in `jax.jit`.

| device | compile |
|---|---|
| CPU (Intel(R) Xeon(R) w5-3435X) | 357.4 ms |
| GPU (NVIDIA RTX 6000 Ada Generation) | 943.0 ms |
