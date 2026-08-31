# Running harv at scale

Guidance for population-scale rejection sampling: one shared prior library, one
`run_with_samples` call per source, thousands to millions of sources.

This page has two halves. **Making it fast** is about hardware and settings.
**Making it right** is about the failure modes that only appear at scale — every
one of them silent, several of them flattering.

:::{important}
Two independent sources of evidence are quoted here and they are never mixed:

- **Benchmark**, meaning {doc}`benchmarks` — an Intel Xeon w5-3435X (32 cores)
  and an NVIDIA RTX 6000 Ada, float64, `top_k`, synthetic data.
- **Production**, meaning one campaign over a simulated *Gaia* DR4 catalog
  (5.7 M stars × 3 populations, 44–298 along-scan epochs each) on a different
  cluster.

Neither predicts your machine. The last section says what to measure first.
:::

## The shape of the cost

Per-source wall time is, to a good approximation,

$$t \approx M \times N_\text{obs} \times c_\text{model}$$

with $M$ the prior library size and $N_\text{obs}$ the **padded** epoch count.
Both scalings are measured: near-linear in $M$ on CPU (log-log slope 0.89–0.94),
and linear in the padded epoch count in production.

`c_model` varies more than people expect. Benchmark throughput at $M = 10^7$, in
millions of prior samples per second:

| parameterization | CPU | GPU | ratio |
|---|---|---|---|
| `StandardRV` | 0.50 | 26.0 | 52× |
| `EcoswEsinwRV` | 0.50 | 25.9 | 52× |
| `FourierRV` | 0.46 | 20.4 | 44× |
| `StandardGaiaAstrometry` | 0.31 | 13.4 | 43× |
| `ThieleInnesGaiaAstrometry` | 0.21 | 6.2 | 30× |
| `FourierGaiaAstrometry` | 0.17 | 5.8 | 34× |

The astrometric models carry more linear columns; `ThieleInnesGaiaAstrometry`
additionally evaluates a Jacobian correction per sample, and costs ~4× a plain
RV model for the same library.

## Budgeting a run

Benchmark seconds per source at `n_obs=64`, in-memory library, and what that
implies for 10⁶ sources:

| model | $M$ | s/source CPU | s/source GPU | 10⁶ sources |
|---|---|---|---|---|
| `StandardRV` | 10⁶ | 2.19 | 0.055 | 610 CPU-h / 15 GPU-h |
| `StandardRV` | 10⁷ | 20.06 | 0.385 | 5,571 CPU-h / 107 GPU-h |
| `StandardGaiaAstrometry` | 10⁶ | 3.24 | 0.096 | 900 CPU-h / 27 GPU-h |
| `StandardGaiaAstrometry` | 10⁷ | 32.37 | 0.746 | 8,992 CPU-h / 207 GPU-h |
| `ThieleInnesGaiaAstrometry` | 10⁶ | 4.76 | 0.187 | 1,321 CPU-h / 52 GPU-h |
| `ThieleInnesGaiaAstrometry` | 10⁷ | 47.67 | 1.608 | 13,240 CPU-h / 447 GPU-h |

The single most useful conversion: **one RTX 6000 Ada is worth about one full
32-core node.** At $M = 10^7$ with `ThieleInnesGaiaAstrometry`, the GPU does 0.62
sources/s and 32 single-threaded CPU processes do 0.67 — and that CPU figure
ignores contention, which production found to be severe. In practice the one card
wins.

Scale those numbers by `padded_epochs / 64` for astrometry. Multiply by nothing
for CPU-node counts until you have measured under contention.

## Choosing a hardware path

**harv has no multi-device support.** There is no device placement, no sharding,
no `pmap` anywhere in the package — `harv/samplers/rejection.py` carries a
`shard_map` TODO and that is all. Scaling out means running one *process* per
shard of sources. The work is embarrassingly parallel and you supply the
parallelism.

::::{tab-set}

:::{tab-item} Laptop or one workstation CPU
**Use it to get the pipeline correct, not to size the run.**

Production sized two jobs from a laptop and was wrong by ~8×; the second burned
~4,000 core-hours and produced nothing. An M4 Max performance core measured
**6.3× a Zen 4 core** on harv's kernel at matched epoch counts — not the 1.5–2×
you would guess. Clock and core width explain maybe half of it. XLA CPU threading
was ruled out (`OMP_NUM_THREADS=1` moved the laptop time by 2%).

Budget: tens of thousands of sources at $M = 10^6$, overnight.
:::

:::{tab-item} One GPU
**The best price/performance step available, and the easiest to get wrong.**

- Keep the library **in device memory**. Streaming the same library from HDF5
  costs 1.35–1.76× on GPU against 1.05–1.22× on CPU: the compute is fast enough
  that I/O becomes the run.
- Spend your budget on a **larger library**, not more calls. The GPU advantage
  grows with $M$ — 3.1–8.3× at $M = 10^4$, 29.6–52.1× at $M = 10^7$ — because a
  small library cannot fill the device. Below about $M = 10^5$ you are paying a
  fixed per-call cost (~9 ms measured) rather than doing work.
- `batch_size` is **not** the lever: 1.04–1.13× across the whole sweep.
- Watch large epoch counts. The measured GPU advantage collapses from ~38× at
  `n_obs=128` to ~10× at `n_obs=256`. See "Large epoch counts" below.

Budget: ~10⁶ sources at $M = 10^7$ in 100–450 GPU-hours, model depending.
:::

:::{tab-item} Many CPU nodes (MPI)
**Bandwidth-bound, and the tuning inverts.**

- **Benchmark `batch_size` under contention, not alone.** Production measured
  `1e4` fastest by 3% single-threaded, and `1e3` **21% faster** at 64 ranks/node
  — while the node running `1e4` was so bandwidth-starved it completed no work
  unit in two hours. A single-core benchmark gets the *sign* wrong.
- Size work units so each rank gets **≥4**, and assign cost-aware. Measured
  allocation used: contiguous slices 57%, strided 52%, cost-aware (LPT) at 1.9
  units/rank ~77%, at 3.8 units/rank ~95%. Striding fixes correlation, not
  variance; with ~2 units per rank there is nothing to average and the slowest
  rank is whoever drew the most expensive unit. When one unit costs more than a
  rank's fair share, no scheduler can help.
- Cost is predictable before any fitting — sum the padded epoch counts, which the
  input catalog already carries. No trial run needed.
:::

:::{tab-item} Many GPU nodes
**One rank per GPU, and the bottleneck moves to your own plumbing.**

Everything from the single-GPU tab applies per rank, plus:

- One library per rank, resident on that rank's device. At $M = 10^7$ a Gaia
  library is roughly 0.5–1 GB, so this is comfortable on modern cards and is what
  keeps you off the HDF5 penalty.
- **Compilation is now a real line item.** Each distinct input shape costs
  4.1–6.2 s on GPU against 1.5–2.7 s on CPU, per rank. With nine epoch buckets
  that is ~45 s per rank of pure compile — trivial across a long run, dominant if
  your work units are short.
- Per-source overhead (~9 ms) sets a floor no library size can amortize: 10⁶
  sources cannot take less than ~3 GPU-hours, whatever else you do.

Rough scaling from the benchmark, `ThieleInnesGaiaAstrometry` at $M = 10^7$:
5.7 M sources take ~2,550 GPU-h — 320 h on 8 GPUs, 40 h on 64.
:::

::::

## Settings that matter

`batch_size`
: Sets the working-set size of the `(batch, n_obs, n_linear)` intermediate.
Nearly irrelevant on GPU (1.04–1.13×), worth 1.55–1.84× on a single CPU core,
and **inverts under contention** (see the MPI tab). The design spec's old advice
to use `batch_size = n_prior_samples` on GPU is not supported by measurement.

Library backend
: In-memory `Samples` when it fits, HDF5 when it does not. `make_prior_cache`
writes a library larger than RAM and `run_with_samples` streams it in contiguous
slices. The penalty is small on CPU and substantial on GPU.

Library size $M$
: Production found **recovery saturates around $M = 10^6$**: 10⁵ → 28.8%,
10⁶ → 35.5%, 10⁷ → 36.1%. Ten times the samples bought 0.6 points, because past
the knee the limit is the prior and the data. But ESS keeps scaling linearly with
$M$, so if you need posterior *widths* rather than point estimates the trade is
different. Decide which you need before buying an order of magnitude.

Period prior width
: Every decade the prior covers costs sampling density in the decades that
matter. Production narrowed 7.8 → 4.0 decades and gained the equivalent of
**about one order of magnitude of $M$** for period accuracy, losing only 0.6% of
high-SNR systems outside the new range. A 5.5-year, ~80-epoch baseline can
constrain roughly two decades. Consider a periodogram-informed prior
({doc}`tutorials/rv/6-periodogram-prior`).

Selection policy
: `top_k` for population work. It returns a fixed number of rows per source, so
results form a rectangular table and the `jax.vmap` over the conditional-linear
solve is traced once instead of once per distinct acceptance count.

## Per data type

### Radial velocities

Few epochs, cheap per source, no padding problem in practice. `StandardRV` and
`EcoswEsinwRV` cost the same. Multi-instrument offsets are an extension
(`MultiSurveyOffset`) and add linear columns, not nonlinear dimensions — cheap.

:::{warning}
`EcoswEsinwRV`'s default prior puts independent `Uniform(-1, 1)` on `ecosw` and
`esinw`, whose support is a square rather than the unit disk, so ~21% of draws
have $e \ge 1$. Pass `ignore_non_finite=True` or every evidence statistic returns
`NaN`. See {doc}`sharp-bits`.
:::

### Gaia epoch astrometry

Epoch counts vary per source (44–298 in DR4), and **harv compiles per epoch
count.** A catalog of 17 M systems would compile 17 M times.

Pad each source up to one of a handful of bucket sizes. Padding is provably
neutral — same samples, same weights, same ESS — **provided all three of these
hold**:

1. Padded rows get a **large finite** uncertainty, never `inf`. Infinity zeroes
   the χ² contribution as intended, but the Gaussian normalization's $\log\sigma$
   term diverges and `logZ` returns `-inf`.
2. `t_ref` is passed **explicitly**, computed from the real epochs. harv defaults
   it to `mean(time)`, so padding otherwise drags the model's time origin.
3. The parameterization is built from the **unpadded** data.
   `ThieleInnesGaiaAstrometry.from_data` sets `a_floor = med(σ_AL)/√N`, and padded
   rows move both the median and $N$.

The consequence people miss: **cost now scales with the bucket, not with the
source.** Production's first benchmark used a 108-epoch system and reported
2.5 s/system; the catalog's mean bucket made the real figure 25 s. Any
single-source timing is unrepresentative unless you know where that source sits
in the bucket distribution.

### Large epoch counts

The benchmark GPU advantage falls from ~38× at `n_obs=128` to ~10× at `n_obs=256`
— a discontinuity, and all six parameterizations show it at the same place. At
`batch_size=1e5` and `n_obs=256` one float64 column array is 205 MB and the
design matrix holds several, so a working-set limit is the obvious suspect, and
holding `batch_size × n_obs` roughly constant is the obvious remedy.

:::{note}
That remedy is **untested** — the benchmark's `batch_size` sweep only ran at
`n_obs=64`. If your epochs run to the hundreds, measure it yourself before
relying on it. It is one short job ({doc}`running-benchmarks`).
:::

______________________________________________________________________

## Making it right

### Weighted output is the easiest thing to get silently wrong

`top_k` returns importance weights normalized over the **whole library**, so they
sum to `weight_captured`, not to 1. Every average over them must renormalize.
This is the failure mode most likely to reach a published number unnoticed.

**Store `ln_likelihood`, never `weight`.** `Samples.weight` is derived, and a
strong detection spans ~10⁻¹³⁰ — which carries no information below ~10⁻³⁸ in
float32 and stores as exactly `0.0` below ~10⁻⁴⁵. Production stored weights and
paid 70 GB of mostly zeros *and* a floor in every figure that read them. Storing
`ln_likelihood` is lossless (it is O(10²–10³) nats), smaller, and reconstructs
the weights exactly.

**Colour weighted-sample plots by $\ln(w/w_\text{best})$ and cut on cumulative
mass.** Top-K keeps its draws by *rank*, not by merit, so at ESS ≈ 1–8 a handful
carry the mass and the rest are prior draws that happened to rank highest.
Plotting them all alike produces a picture of the **prior** with the real draws
hidden underneath. That reached a full production run before anyone noticed.

### ESS is a resolution diagnostic, not a quality metric

For a strongly detected system **ESS ≈ 1 is arithmetic, not failure**: the Δχ²
between the best prior draw and the second-best runs into the hundreds, so
$e^{-\Delta\chi^2/2}$ annihilates everything else regardless of how the prior is
written. Neither tightening the parallax prior across four orders of magnitude
nor reparameterizing moved it.

Worse for intuition: at fixed likelihood specification, **recovery falls as ESS
rises.** A well-constrained period gives a sharp posterior, which a fixed-size
library resolves with *fewer* effective samples. Production's best-recovering
period bin had median ESS 1.7; the worst had 7.7.

That relation holds *at* fixed likelihood specification and **not across it**.
Change the weights on the data and ESS tells you about the likelihood instead:
correcting under-reported uncertainties raised median ESS 15.8 → 39.8 *and*
lowered recovery 46.7% → 44.1%, where the ESS rise is correct calibration and the
recovery drop is the removal of over-detection. Compare ESS across arms only when
the likelihood is the thing you changed, and say so.

Report `logZ_int_ess` (did the library sample this posterior) and
`weight_captured` (was `top_k` big enough) as **separate** diagnostics, and let
neither be read as "is this answer good".

### Calibrate your uncertainties; do not fit a jitter for them

Reported per-epoch uncertainties are often smaller than the true scatter — real
*Gaia* included, because the published formal error omits calibration terms that
are genuinely there. Production's median ratio $r = \sigma_\text{true}/\sigma_\text{reported}$
was 1.276 with a tail to 11.5.

**This lands in an exponent.** χ² scales as $1/\sigma^2$ and the weight is
$e^{-\Delta\chi^2/2}$, so under-reported errors sharpen the contrast between
library draws by $r^2 = 1.63$ *in the exponent*. The sampler is not merely wrong;
it is overconfident by construction — and the sharpening favours orbit-over-null
as well as orbit-over-orbit, so **a mis-specified likelihood biases toward
claiming companions.** Anyone benchmarking a method on optimistic errors will
over-report its performance, and the direction is always flattering.

Four arms at matched $M = 10^7$:

| arm | ESS | railed | recovered |
|---|---|---|---|
| reported (status quo) | 15.8 | 12.7% | 46.7% |
| **corrected weights** | **39.8** | 16.4% | **44.1%** |
| reported + fitted jitter | 27.7 | 21.4% | 40.5% |
| corrected + fitted jitter | 33.4 | 20.1% | 41.3% |

Correcting the weights beats learning the jitter on both axes and costs nothing.
Fitted jitter loses three ways: it is a **nonlinear** parameter, so effective
resolution per dimension falls from $M^{1/3} \approx 215$ to $M^{1/4} \approx 56$
at $M = 10^7$; it **competes with the orbit**, since an orbit the library cannot
match exactly is cheaper to explain as noise (railing 12.7% → 21.4%); and it has
a **floor made of library coarseness** — run on data with exact uncertainties,
where it should return zero, it returned 0.0150 mas, 34% of the noise scale.

A shared library makes it worse: `sigma_reported` varies star to star but one
library serves every source, so the jitter prior must be absolute and broad
enough for the whole catalog, wasting resolution on every individual star.

**Measure $r$ first.** The reduced chi-square of a no-signal fit *is* $r^2$ — one
least-squares per source, no modelling. If $r \neq 1$, correct the weights from an
independent noise estimate (for *Gaia*, DR3's `astrometric_excess_noise`, which
does not depend on the orbit being fitted). Reserve a fitted jitter for when no
independent estimate exists, and budget a nonlinear dimension when you do.

**Run the control that catches the failure mode:** fit a jitter on
correctly-weighted data, where it should return ~0. If it does not, it is
absorbing your library's inability to match the signal, and it will look like a
better fit while making detection worse. That arm is invisible in recovery and
rail rate alone — you have to read the jitter posterior itself.

### The amplitude prior sets your detection threshold (astrometry)

`sigma_a0` is the width of the Gaussian prior on the orbit's astrometric
amplitude. harv scales it as $(P/P_0)^{2/3}\varpi$, so the Occam penalty it
imposes **grows with period** — falling on real orbits and barely at all on the
no-orbit solution. Set it too wide and the evidence prefers "no companion" for
genuinely detectable systems.

Production had it ~4,900× too wide. The symptom was not a warning but a
**collapse to the shortest period in the prior**, where the amplitude is forced to
zero and the model reduces to a five-parameter astrometric fit. That was **65% of
all recovery failures**, initially misdiagnosed as aliasing. A `sigma_a0` sweep at
fixed library size moved overall recovery 36.1% → 44.9%.

Set it as the physical quantity it is — the largest companion you expect. At the
reference period a companion of mass $m$ around a star of mass $M$ displaces the
photocentre by $a_0 \approx m/M^{2/3}$, so one companion-mass ceiling gives the
right scale for every star. It is free to compute per source: `sigma_a0` shapes
only the analytically marginalized priors, never enters the shared library, and
triggers no extra compile.

Do **not** pick it by maximizing recovery against known truth — that is fitting
to the answer. Use a sweep as evidence that a physically-motivated value is in
the right regime, not as the source of the number.

`log_uniform_in_a=True` is **not** the lever. It sets $m = 4$ in
$-m\ln(a_0 + a_\text{floor})$, which rewards *small* $a_0$ more strongly and makes
the collapse worse.

### Bin completeness on detectable SNR, never on recorded SNR

Position, proper motion and parallax are free parameters, so whatever part of an
orbit those five columns reproduce is **subtracted along with them**. An orbit
whose period is comparable to the observing span is largely a straight line plus
a slow curve — which is exactly what proper motion is. Measured by exact
projection, the median orbit in production's catalog kept ~60% of its amplitude,
and 18.6% kept under 25%, with cases down to 5.4%.

Binning the same rail fractions on recorded versus detectable SNR:

| SNR bin | railed, binned on recorded SNR | railed, binned on detectable SNR |
|---|---|---|
| 5–10 | 50.2% | 21–35% |
| 10–20 | 20.4% | **0.0%** |
| 20–40 | 2.4% | **0.0%** |
| >40 | 0.0% | **0.0%** |

Railing is exactly zero above detectable SNR 10, in every arm of a four-arm
sweep. Read on the recorded axis, the same data appears to show a detection
threshold sitting *above* the selection cut; that conclusion was an artefact.
The residual in the 10–20 and 20–40 bins was long-period systems whose signal the
astrometric solution had already eaten — correctly reported as non-detections,
filed under the wrong signal strength.

Compute, per source, the fraction of injected signal surviving projection onto
the astrometric basis, and bin every completeness and failure-rate figure on
`SNR × that fraction`. It is one small least-squares per source (~95 core-hours
over 17.2 M systems) and without it a long-period non-detection is
indistinguishable from a prior pathology. Keep the recorded SNR for the
*selection* cut, where a cheap a-priori proxy is the right thing, and never for
an axis.

### Report failures in categories, not as one number

A single "recovery rate" conflates three unrelated things:

- **no detection** (collapsed to the prior floor) — a prior problem
- **outside the searched range** — impossible by construction, not a failure
- **wrong period** — the only real failure

Splitting them turned an opaque 31.6% into a diagnosis. Bin recovery by injected
period *and* by SNR: the period profile shows the recoverable window (for a
5.5-year baseline, roughly 0.1–10 yr) and the SNR profile shows where the prior's
threshold sits.

### Top-K posterior widths are not error bars

At ESS ≈ 1 the weighted standard deviation of a parameter collapses toward zero —
a non-trivial share of sources report exactly `0.0` — because one draw holds all
the weight. The pull against injected truth is then wildly overdispersed. Point
estimates from a top-K run are usable; their spreads are not, and that is what an
MCMC second pass is for.

### Refit both sides of any Δχ²

harv returns the marginalized linear parameters as a *draw* from their
conditional Gaussian, not as its mean. Score that drawn θ against a
least-squares no-orbit fit and you are comparing a posterior sample with an
optimum: production measured Δχ² = −13 on a railed system, an apparently
*negative* improvement from adding a companion. Refit both models at their own
maxima and the nesting guarantee holds.

______________________________________________________________________

## Before you commit an allocation

Each of these is one short job. Together they would have saved production two
failed runs, about a week, and one confidently wrong conclusion.

1. **Warm seconds per source** on the target hardware, at the target library
   size, **with the node full**.
2. **Peak RSS per rank** at the target library size → ranks per node. Production's
   memory model under-predicted by 1.5× at $M = 10^6$.
3. **`batch_size` under contention** — not on an idle core.
4. **$r$**, the reduced chi-square of a no-signal fit → are your reported
   uncertainties the ones the data were generated with?
5. **The surviving-signal fraction** per source → the axis for every completeness
   figure.
6. **Rail fraction against detectable SNR** → is the amplitude prior setting your
   detection threshold?
7. **Units per rank ≥ 4**, with cost-aware assignment.

## Operational notes

- **Set `jax_enable_x64` at package import**, not in one module. At float32 the
  marginalized log-likelihoods over a wide period prior all underflow to `-inf`,
  ESS returns `NaN`, and top-K then returns arbitrary rows. It is silent garbage,
  not an error — a production diagnostic script imported two modules but not the
  one setting the flag and produced a convincing false alarm.
- **Suppress the per-call warnings.** harv's non-Gaussian-prior notice and its
  under-resolution warning are both correct and both fire once per source. Use
  `warnings.catch_warnings` around the loop.
- **harv imports matplotlib**, so on a cluster with node-local cache directories
  every rank rebuilds the font list. Point `MPLCONFIGDIR` at a pre-built shared
  cache.
- **Print progress inside a work unit.** Production printed only on unit
  completion and a unit was hours long; a failing 32-node job was undiagnosable
  from its log for two hours.
- **Report peak RSS**, especially at $M = 10^7$ where the library dominates.
