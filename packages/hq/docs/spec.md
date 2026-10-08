# harv-hq — Design Specification

**hq** (PyPI distribution `harv-hq`, import package `harv_hq`) is the companion
package to harv for running it on large samples. It takes a survey catalog of
radial velocities or Gaia epoch astrometry, prepares per-source data, runs harv
on every source, stores the results in a standard on-disk format, and serves a
local web viewer for exploring them.

This document is the authoritative source of truth for hq. It plays the same
role for hq that `docs/spec.md` plays for harv: code, docstrings, and tests
follow it, and any public API must be documented here first. Where this
document refers to harv behavior (samplers, `Samples`, priors), harv's
`docs/spec.md` is normative and this document does not restate it.

______________________________________________________________________

## Scope

hq handles everything around a single harv fit:

1. **Catalog preparation.** Read survey tables (one row per observation),
   apply column mappings, units, time conversions and row selections, group
   the rows by source, and write a prepared per-source data file.
1. **Pipelining.** Build one shared prior cache, run the rejection sampler on
   every source, select sources for MCMC follow-up and run it, all resumable
   and parallelizable over shards, a local process pool, or MPI.
1. **Results storage.** Write per-source posterior samples, per-source run
   status, and run provenance in a standard format, and reduce them to one
   summary table.
1. **Exploration.** Serve a local web app with a catalog view (scatter plots
   of any catalog or summary column, lasso selection, a table of the
   selection) and a per-source page looked up by catalog source ID.

### Supported data

| Run kind          | Input rows                                                                    | harv model            |
| ----------------- | ----------------------------------------------------------------------------- | --------------------- |
| `rv`              | RV epochs from one instrument of any survey (APOGEE, SDSS-V, DESI, Gaia, ...) | `RVModel`             |
| `gaia_astrometry` | Gaia epoch astrometry (along-scan positions)                                  | `GaiaAstrometryModel` |

A run is one kind, read from one input table, and each source's data is a
single `RVData` or `GaiaAstrometryData`. Multi-instrument RV and joint RV +
astrometry runs are planned (see "Planned features").

### Non-goals

- hq never re-implements orbit math, likelihoods, priors, or sampling. It
  calls harv's public API only. If hq needs something harv cannot do, the
  capability is added to harv (and its spec) first.
- hq does not do barycentric corrections. Input times must already be
  barycentric; hq only converts their time *scale* and format.
- hq does not choose priors. The prior and model are defined by the user in
  Python (see "The model file").

### Lessons carried over

hq replaces the HQ pipeline built on The Joker (`hq-thejoker`) and generalizes
the `phobos` project. The design fixes the specific problems found in both:

| Problem in earlier pipelines                                     | hq rule                                                                    |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Failed sources only appear in logs and are silently retried      | Every source gets a recorded status (`ok` or `failed`) and error           |
| Stage completion means "a file exists"; stale results undetected | Every output records provenance hashes; mismatches refuse to resume        |
| One shared results file, merged by callbacks and `atexit` hooks  | One results file per process, written and flushed per source               |
| Per-source scans of the full input table                         | One group-by at prepare time; workers read prepared per-source data        |
| Config keys silently ignored                                     | Unknown config keys are an error                                           |
| Survey-specific columns hard-coded (`APOGEE_ID`)                 | All columns, units and IDs come from the config                            |
| Prior constants and data paths in Python module constants        | Run settings in `hq.toml`, model and prior in `prior.py`, both snapshotted |

______________________________________________________________________

## Run directory

Everything about a run lives in one directory. Paths in the config are
relative to it.

```
run/
├── hq.toml                         # run configuration
├── prior.py                        # the model file: prior + model
├── data.h5                         # prepared per-source data       (hq prepare)
├── catalog.parquet                 # one row per prepared source     (hq prepare)
├── prior_cache.h5                  # shared prior library            (hq prior-cache)
├── results/
│   ├── rejection-0003-of-0016.h5   # one file per process            (hq run)
│   └── mcmc-0003-of-0016.h5        #                                 (hq mcmc)
├── summary.parquet                 # one row per source              (hq summarize)
└── logs/
    └── rejection-0003-of-0016.log  # one log per process
```

`hq init RUN_DIR` creates the directory with a template `hq.toml` and a
template `prior.py` for each run kind.

______________________________________________________________________

## Configuration (`hq.toml`)

The run configuration is a TOML file parsed with the standard-library
`tomllib`. It is loaded into a frozen `harv_hq.Config` object. Validation is
strict: unknown keys, wrong types, and missing required keys raise
`harv_hq.ConfigError` naming the offending key and table. Keys marked
"required" have no default.

### `[run]`

| Key          | Type  | Default      | Description                                              |
| ------------ | ----- | ------------ | -------------------------------------------------------- |
| `name`       | `str` | required     | Run name, recorded in provenance and shown in the viewer |
| `kind`       | `str` | required     | `"rv"` or `"gaia_astrometry"`                            |
| `seed`       | `int` | required     | Root seed; every PRNG key in the run derives from it     |
| `model_file` | `str` | `"prior.py"` | Path to the model file                                   |

### `[data]`

The input table, one row per observation, read with
`astropy.table.Table.read`. Required.

Shared keys:

| Key           | Type                 | Default  | Description                                                            |
| ------------- | -------------------- | -------- | ---------------------------------------------------------------------- |
| `file`        | `str`                | required | Input file path                                                        |
| `format`      | `str \| None`        | `None`   | astropy format string; `None` lets astropy infer it from the extension |
| `hdu`         | `int \| str \| None` | `None`   | FITS HDU, for FITS inputs                                              |
| `source_id`   | `str`                | required | Column holding the source identifier                                   |
| `time`        | `str`                | required | Column holding barycentric observation times                           |
| `time_format` | `str`                | `"jd"`   | astropy `Time` format of the time column (`"jd"`, `"mjd"`, ...)        |
| `time_scale`  | `str`                | `"tdb"`  | astropy `Time` scale of the time column; converted to TCB              |

Every row is treated as coming from one instrument. A table that mixes
instruments (e.g. APOGEE's two telescopes) must be cut to one with
`select_rows` until multi-instrument support exists.

Keys for `kind = "rv"`:

| Key       | Type  | Default  | Description                                  |
| --------- | ----- | -------- | -------------------------------------------- |
| `rv`      | `str` | required | RV column                                    |
| `rv_err`  | `str` | required | RV uncertainty column                        |
| `rv_unit` | `str` | `"km/s"` | Unit applied when the column carries no unit |

Keys for `kind = "gaia_astrometry"`:

| Key                | Type  | Default  | Description                                  |
| ------------------ | ----- | -------- | -------------------------------------------- |
| `al_position`      | `str` | required | Along-scan position column                   |
| `al_position_err`  | `str` | required | Along-scan uncertainty column                |
| `scan_angle`       | `str` | required | Scan angle column                            |
| `parallax_factor`  | `str` | required | Along-scan parallax factor column            |
| `al_position_unit` | `str` | `"mas"`  | Unit applied when the column carries no unit |
| `scan_angle_unit`  | `str` | `"deg"`  | Unit applied when the column carries no unit |

A column that carries an astropy unit keeps it, and the `*_unit` key is
ignored for that column. Units are converted to `unxt.Quantity` when the harv
data objects are built.

### `[prepare]`

| Key         | Type  | Default | Description                                        |
| ----------- | ----- | ------- | -------------------------------------------------- |
| `min_n_obs` | `int` | `3`     | Sources with fewer usable observations are dropped |

Row-level quality cuts (flags, error limits, finite checks beyond the
built-in ones) are not expressed in TOML. They go in the optional
`select_rows` hook in the model file.

### `[catalog]`

Optional. A source-level table (one row per source) whose columns are joined
onto the prepared sources for the viewer, e.g. photometry for a CMD.

| Key         | Type                 | Default  | Description                                        |
| ----------- | -------------------- | -------- | -------------------------------------------------- |
| `file`      | `str`                | required | Catalog file, read with `astropy.table.Table.read` |
| `format`    | `str \| None`        | `None`   | astropy format string                              |
| `hdu`       | `int \| str \| None` | `None`   | FITS HDU                                           |
| `source_id` | `str`                | required | Column matching the data table's source IDs        |
| `columns`   | `list[str] \| None`  | `None`   | Columns to keep; `None` keeps every scalar column  |

### `[prior_cache]`

| Key          | Type  | Default   | Description                                   |
| ------------ | ----- | --------- | --------------------------------------------- |
| `n_samples`  | `int` | required  | Size of the shared prior library              |
| `batch_size` | `int` | `100_000` | Forwarded to `harv.samplers.make_prior_cache` |

### `[rejection]`

| Key                     | Type    | Default   | Description                                                 |
| ----------------------- | ------- | --------- | ----------------------------------------------------------- |
| `top_k`                 | `int`   | required  | Forwarded to `RejectionSampler.run_with_samples(top_k=...)` |
| `batch_size`            | `int`   | `100_000` | `RejectionSampler.batch_size`                               |
| `min_evidence_ess`      | `float` | `3.0`     | `RejectionSampler.min_evidence_ess`                         |
| `ignore_non_finite`     | `bool`  | `false`   | Forwarded to `run_with_samples`                             |
| `randomize_prior_order` | `bool`  | `true`    | Forwarded to `run_with_samples`                             |

### `[mcmc]`

Optional. Without it, `hq mcmc` raises `ConfigError`.

| Key                | Type    | Default            | Description                                                     |
| ------------------ | ------- | ------------------ | --------------------------------------------------------------- |
| `select`           | `str`   | `"under_resolved"` | Selection rule; see "MCMC follow-up"                            |
| `require_unimodal` | `bool`  | `true`             | Only select sources whose rejection period samples are unimodal |
| `num_chains`       | `int`   | `4`                | Forwarded to `NumpyroSampler.run`                               |
| `num_warmup`       | `int`   | `1000`             | Forwarded to `NumpyroSampler.run`                               |
| `num_samples`      | `int`   | `1000`             | Forwarded to `NumpyroSampler.run`                               |
| `chain_method`     | `str`   | `"sequential"`     | Forwarded to `NumpyroSampler.run`                               |
| `max_r_hat`        | `float` | `1.05`             | Convergence threshold for `mcmc_status`                         |
| `min_ess_bulk`     | `float` | `400.0`            | Convergence threshold for `mcmc_status`                         |

`num_*` names follow harv's numpyro-passthrough exception to the `n_*` rule.

### `[serve]`

| Key    | Type  | Default       | Description  |
| ------ | ----- | ------------- | ------------ |
| `host` | `str` | `"127.0.0.1"` | Bind address |
| `port` | `int` | `8000`        | Port         |

### Example

```toml
[run]
name = "apogee-dr17-binaries"
kind = "rv"
seed = 42

[data]
file = "allVisit-dr17.fits"
source_id = "APOGEE_ID"
time = "JD"
time_format = "jd"
time_scale = "tdb"
rv = "VHELIO"
rv_err = "VRELERR"

[prepare]
min_n_obs = 4

[catalog]
file = "allStar-dr17.fits"
source_id = "APOGEE_ID"
columns = ["TEFF", "LOGG", "M_H", "J", "K"]

[prior_cache]
n_samples = 100_000_000

[rejection]
top_k = 512

[mcmc]
select = "under_resolved"
```

______________________________________________________________________

## The model file (`prior.py`)

The prior and model are defined in Python, in the file named by
`run.model_file`. hq imports it once per process with `importlib` and calls
the functions below. The file is part of the run's provenance: its sha256 is
recorded in every output, and outputs built from a different version of the
file are refused (see "Provenance").

### Required function

```python
def make_setup() -> tuple[HarvPrior, RVModel | GaiaAstrometryModel]:
    """The prior and model for the run."""
```

One prior and one model serve the whole run: `hq prior-cache` passes them to
`harv.samplers.make_prior_cache`, and `hq run`, `hq mcmc` and the viewer use
them for every source. Each process calls `make_setup` once. The model must
match `run.kind` (`RVModel` for `rv`, `GaiaAstrometryModel` for
`gaia_astrometry`), checked once with a `ConfigError` on mismatch.

### Optional functions

```python
def select_rows(table: astropy.table.Table) -> numpy.ndarray:
    """Boolean mask of input rows to keep, at prepare time."""
```

Built-in cuts that always apply first: finite time, observation, and
uncertainty, and uncertainty strictly positive.

```python
def select_for_mcmc(row: dict[str, Any]) -> bool:
    """Whether a source goes to MCMC follow-up, given its summary row."""
```

When defined, it replaces the `[mcmc] select` / `require_unimodal` rule.

### Template

`hq init` writes this template for `kind = "rv"`:

```python
from unxt import Q

import harv.models as hm


def make_setup():
    prior = hm.StandardRV().default_prior(
        period_min=Q(2, "day"),
        period_max=Q(4096, "day"),
        sigma_K0=Q(30, "km/s"),
        sigma_v0=Q(100, "km/s"),
    )
    return prior, hm.RVModel()
```

______________________________________________________________________

## Source IDs

Source IDs are whatever the input tables use: 19-digit Gaia `source_id`
integers, APOGEE `2MASS`-style strings, DESI `TARGETID`s.

- The ID's type (`int64` or `str`) is taken from the `[data]` table. The
  `[catalog]` table's ID column is cast to that type before matching, and a
  failed cast raises `ConfigError`.
- Tables (`catalog.parquet`, `summary.parquet`) keep the native type in a
  `source_id` column.
- HDF5 groups are named `str(source_id)`. An ID containing `/` or equal to
  `"."` is rejected at prepare time with a `ValueError` naming it.
- In the viewer, IDs appear URL-encoded in paths (`+` in APOGEE IDs).

### Per-source randomness

Each source's PRNG key is derived from the run seed and the ID alone, so it
does not depend on shard layout, process count, or execution order:

```python
key = jax.random.fold_in(jax.random.key(seed), stable_hash(source_id))
```

`stable_hash` is the first 4 bytes of the sha256 of `str(source_id)`, read as
an unsigned 32-bit integer. Stage keys are split from it with
`jax.random.fold_in(key, stage)`, with `stage` 0 for rejection and 1 for MCMC.
The prior cache uses `jax.random.fold_in(jax.random.key(seed), 2**32 - 1)`.

______________________________________________________________________

## Stages

Each stage is a public method on `harv_hq.Run` (see "Public API"), and each
CLI subcommand is a thin wrapper around one method. Stages run in this order:

| CLI               | Method                 | Reads                                                  | Writes                       |
| ----------------- | ---------------------- | ------------------------------------------------------ | ---------------------------- |
| `hq init RUN_DIR` | `harv_hq.init_run`     | nothing                                                | `hq.toml`, `prior.py`        |
| `hq prepare`      | `Run.prepare`          | `[data]`, `[catalog]` inputs                           | `data.h5`, `catalog.parquet` |
| `hq prior-cache`  | `Run.make_prior_cache` | `prior.py`                                             | `prior_cache.h5`             |
| `hq run`          | `Run.run_rejection`    | `data.h5`, `prior_cache.h5`                            | `results/rejection-*.h5`     |
| `hq summarize`    | `Run.summarize`        | `catalog.parquet`, `results/*.h5`                      | `summary.parquet`            |
| `hq mcmc`         | `Run.run_mcmc`         | `data.h5`, `summary.parquet`, `results/rejection-*.h5` | `results/mcmc-*.h5`          |
| `hq summarize`    | `Run.summarize`        | (again, to add MCMC columns)                           | `summary.parquet`            |
| `hq status`       | `Run.status`           | `results/*.h5`                                         | nothing (prints counts)      |
| `hq serve`        | `harv_hq.create_app`   | everything above                                       | nothing                      |

Every CLI subcommand takes `--run-dir` (default: the current directory).

### `prepare`

1. Read the `[data]` table, keeping only the configured columns.
1. Apply the built-in cuts and then `select_rows`, if defined.
1. Convert times to TCB with astropy `Time`, and store them as MJD (TCB) in
   days.
1. Sort all rows by `(source_id, time)` once, and split at ID boundaries. No
   step scans the table per source.
1. Drop sources with fewer than `min_n_obs` observations, and record the
   number dropped in the log.
1. Write `data.h5` and `catalog.parquet`.

`prepare` refuses to overwrite an existing `data.h5` unless `overwrite=True`
(`--overwrite`), because doing so invalidates every downstream output.

#### `data.h5` layout

```
/                          attrs: hq_version, harv_version, data_id (uuid4), kind,
                                  config_sha256, created
/index/source_id           (n_sources,)  int64 or variable-length str
/index/n_obs               (n_sources,)  int64
/sources/<id>/
    time                   float64, attrs: unit="day", format="mjd", scale="tcb"
    rv, rv_err             float64, attrs: unit           (rv)
    al_position, al_position_err, scan_angle, parallax_factor
                           float64, attrs: unit           (gaia_astrometry)
```

`harv_hq.read_source(data_file, source_id) -> RVData | GaiaAstrometryData`
rebuilds the harv data object. `time_ref` is left to harv's default (the mean
time).

#### `catalog.parquet`

One row per prepared source: `source_id`, `n_obs`, `time_baseline` (days),
then the `[catalog]` columns joined on `source_id` (left join: prepared
sources with no catalog row get nulls). Catalog columns that collide with
these names raise `ConfigError`.

### `make_prior_cache`

Calls `make_setup()`, then
`harv.samplers.make_prior_cache(prior, model, n_samples, path, key=..., batch_size=...)`.
hq adds provenance attrs to the file's root after harv writes it.

### `run_rejection`

Each process calls `make_setup()` once and builds one
`RejectionSampler(prior, model, batch_size=..., min_evidence_ess=...)`. Then,
for each source in its slice (see "Execution modes") that is not already done:

1. `data = read_source(...)`.
1. `samples = sampler.run_with_samples(data, prior_cache, key=..., top_k=..., ignore_non_finite=..., randomize_prior_order=...)`.
   `top_k` forces `return_logprobs` and `return_evidence_stats`, so every
   result carries `ln_likelihood`, `ln_prior`, and the evidence metadata.
1. Write the result and status (see "Results").

harv's under-resolution `UserWarning` is captured per source with
`warnings.catch_warnings(record=True)` and stored as the source's `warnings`
attr, not printed. Any exception is caught, its traceback is stored, and the
source is marked `failed`; the run continues.

Within a slice, sources are processed in ascending `n_obs`, so consecutive
sources share array shapes and reuse JIT compilations.

### MCMC follow-up (`run_mcmc`)

Reads `summary.parquet` (which must be current, see "Provenance") to select
sources. Each process calls `make_setup()` once and builds one
`NumpyroSampler(prior, model)`. Then, for each selected source in its slice:

1. `data = read_source(...)`.
1. Load the source's rejection `Samples` and pass its equal-weight resample
   (see "Weighted samples") as `init_samples`, so chains start at draws in
   proportion to their posterior weight.
1. `sampler.run(data, init_samples=..., key=..., num_chains=..., num_warmup=..., num_samples=..., chain_method=..., return_logprobs=True)`.
1. Compute `r_hat_max` and `ess_bulk_min` over all sampled parameters from
   `samples.to_arviz()` (`arviz.rhat`, `arviz.ess`).
1. Write the result with `mcmc_status`:

| `mcmc_status`     | Meaning                                                     |
| ----------------- | ----------------------------------------------------------- |
| `"converged"`     | `r_hat_max <= max_r_hat` and `ess_bulk_min >= min_ess_bulk` |
| `"not_converged"` | Ran, but failed either threshold                            |
| `"failed"`        | Raised; traceback stored                                    |

Selection rules (`[mcmc] select`):

| Rule               | Selects sources whose rejection run...                      |
| ------------------ | ----------------------------------------------------------- |
| `"under_resolved"` | has `well_resolved == False` (evidence ESS below threshold) |
| `"all"`            | finished with status `ok`                                   |

With `require_unimodal = true`, a source must also have
`period_unimodal == True`. A multimodal, under-resolved source needs a larger
prior library, not MCMC; rerunning those is planned (see "Planned
features"). `select_for_mcmc` in the model file overrides both keys.

`arviz` is a required dependency of hq because convergence diagnostics are
part of the MCMC result.

______________________________________________________________________

## Execution modes

`hq run` and `hq mcmc` partition the work the same way. Sources are ordered by
`(n_obs, source_id)`, and slice `i` of `N` is `order[i::N]`. Round-robin
assignment balances total `n_obs` across slices without a cost model, and
each slice stays sorted by `n_obs`. The slice depends only on `data.h5`, `i`,
and `N`.

| Mode                     | Invocation                          | Slice               | Output file                     |
| ------------------------ | ----------------------------------- | ------------------- | ------------------------------- |
| Single process (default) | `hq run`                            | `0/1`               | `rejection-0000-of-0001.h5`     |
| Explicit shard           | `hq run --shard 3/16`               | `3/16`              | `rejection-0003-of-0016.h5`     |
| Local pool               | `hq run --workers 8 [--shard 3/16]` | the process's slice | one file, written by the parent |
| MPI                      | `mpirun -n 64 hq run --mpi`         | `rank/size`         | one file per rank               |

- **Explicit shards** are for job arrays and task launchers (SLURM
  `--array=0-15` with `--shard $SLURM_ARRAY_TASK_ID/16`, disBatch). Shards are
  fully independent processes.
- **Local pool** runs a `ProcessPoolExecutor` with the `spawn` start method.
  Each worker is pinned to one CPU thread (BLAS and XLA environment variables
  set before JAX is imported), so `W` workers use `W` cores. The worker initializer imports the model file and opens
  `data.h5` and the prior cache once. Workers return results as plain NumPy
  payloads, and only the parent process writes to HDF5.
- **MPI** uses `mpi4py` (the `harv-hq[mpi]` extra), imported lazily so a
  missing install raises `ImportError` naming `hq run --mpi`. Rank `r` of `n`
  processes slice `r/n` and writes its own file, so ranks do no I/O on each
  other's files and need no communication beyond a final barrier. Rank 0
  additionally logs aggregate progress. `--mpi` is mutually exclusive with
  `--shard` and `--workers`.
- GPU runs use the single-process or explicit-shard modes, one process per
  device.

### Resume

Resume does not depend on shard layout. Before processing its slice, a
process reads the source IDs and statuses from **every** file matching
`results/<stage>-*.h5`. A source is done if any file records it with status
`ok`. A source recorded as `failed` is retried only with
`--retry-failed`. `--overwrite` ignores existing results and writes fresh
files (existing files for the same stage are moved to
`results/superseded-<timestamp>/`, never deleted).

So a run started as `--shard i/16` can be finished with `--mpi` on 64 ranks,
or with a single process, without recomputing finished sources. When a source
appears in more than one file, the most recent `finished` timestamp wins.

A results file that cannot be opened (e.g. truncated by a crash mid-write) is
renamed to `<name>.corrupt` with a logged warning, and its sources are treated
as not done.

______________________________________________________________________

## Results

### Per-process results file

```
results/rejection-0003-of-0016.h5
/                          attrs: provenance (see below), stage, shard="3/16"
/<id>/                     attrs: status, error, warnings, seed_hash, n_obs,
                                  started, finished, wall_time_s
/<id>/samples/             Samples.to_hdf5 layout (absent unless status == "ok")
    nonlinear/<param>      attrs: unit
    linear/<param>         attrs: unit
    ln_likelihood, ln_prior
    metadata/              attrs: model_type, linear_extension_names, ...
```

| Attr                        | Type    | Description                                       |
| --------------------------- | ------- | ------------------------------------------------- |
| `status`                    | `str`   | `"ok"` or `"failed"`                              |
| `error`                     | `str`   | Full traceback for `failed`, else `""`            |
| `warnings`                  | `str`   | Newline-joined captured warning messages, or `""` |
| `seed_hash`                 | `int`   | `stable_hash(source_id)`                          |
| `n_obs`                     | `int`   | Observations used                                 |
| `started`, `finished`       | `str`   | ISO 8601 UTC timestamps                           |
| `wall_time_s`               | `float` | Wall time for this source                         |
| `mcmc_status`               | `str`   | MCMC files only; see "MCMC follow-up"             |
| `r_hat_max`, `ess_bulk_min` | `float` | MCMC files only                                   |

The `samples/` group is written by harv's `Samples.to_hdf5`, so
`Samples.from_hdf5` reads it back. **This requires a harv change:**
`Samples.to_hdf5` and `Samples.from_hdf5` must accept an open `h5py.Group` in
addition to a filename. The change and its spec entry land in harv before hq's
results writer.

The file is opened in append mode, each source's group is written completely,
and the file is flushed before the next source starts. A crash therefore loses
at most the source in progress (or, in the worst case, the file; see
"Resume").

`harv_hq.Run.load_source(source_id) -> SourceResult` returns the prepared
data, the rejection and MCMC `Samples` (or `None`), their statuses,
and the summary row, regardless of which files hold them.

### Provenance

Every output file (`data.h5`, `prior_cache.h5`, each results file) carries
these root attrs:

| Attr                                        | Description                                           |
| ------------------------------------------- | ----------------------------------------------------- |
| `hq_version`, `harv_version`, `jax_version` | Package versions                                      |
| `config_sha256`                             | sha256 of `hq.toml` bytes                             |
| `model_file_sha256`                         | sha256 of the model file bytes (absent on `data.h5`)  |
| `data_id`                                   | The `data_id` of the `data.h5` used                   |
| `prior_cache_sha256`                        | sha256 of the prior cache's root attrs (results only) |

A stage refuses to append to, or build on, outputs whose `config_sha256`,
`model_file_sha256`, or `data_id` differ from the current ones, raising
`harv_hq.ProvenanceError` that names the mismatched field and file. The fix is
`--overwrite` on that stage. Version differences are logged but not refused.
`summary.parquet` stores the same fields in its Parquet key-value metadata,
and `run_mcmc` refuses a summary that is older than any rejection results
file.

### Summary table (`summary.parquet`)

`Run.summarize()` reduces all results files to one row per prepared source
(sources not yet run have `rejection_status = "pending"`). Columns:

| Column(s)                                                                                              | Source                                                       |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `source_id`, `n_obs`, `time_baseline`                                                                  | `catalog.parquet`                                            |
| `rejection_status`, `rejection_error`                                                                  | results attrs                                                |
| `n_prior_samples`, `ln_Z_int`, `ln_Z_int_mcse`, `ln_Z_int_ess`, `max_ln_likelihood`, `weight_captured` | `Samples.metadata`                                           |
| `well_resolved`                                                                                        | `Samples.acceptance_diagnostics(min_evidence_ess=...)`       |
| `period_unimodal`                                                                                      | `Samples.period_unimodal(data)` on the equal-weight resample |
| `max_phase_gap`, `phase_coverage`, `periods_spanned`                                                   | the corresponding `Samples` methods, at the MAP sample       |
| `map_<param>`                                                                                          | `Samples.map_sample()`, every parameter in `Samples.keys()`  |
| `<param>_p16`, `<param>_p50`, `<param>_p84`                                                            | weighted percentiles (see below)                             |
| `mcmc_status`, `r_hat_max`, `ess_bulk_min`                                                             | MCMC results attrs; `"not_selected"` / null when absent      |
| `final`                                                                                                | `"mcmc"` if `mcmc_status == "converged"`, else `"rejection"` |

Values with units are stored in the unit of the samples and the unit is
recorded in the column's Parquet field metadata (`unit`). Every source shares
one model, so every `ok` source has the same parameter columns; sources
without results get nulls.

The `<param>_p*` columns describe the `final` sample set.

### Weighted samples

Top-K rejection samples are weighted (harv spec, "Top-K selection"); MCMC
samples are equal-weight. hq treats the weights as follows.

- Percentiles use `Samples.weight` renormalized to sum to one.
- Diagnostics that assume equal-weight draws (`period_unimodal`, MCMC
  initialization) use an *equal-weight resample*: `top_k` indices drawn with
  replacement in proportion to the renormalized weights, using the source's
  rejection key. Without it, the low-weight tail of a top-K set would make
  nearly every source look multimodal.
- Weighted percentiles on a truncated top-K set are biased when
  `weight_captured` is well below one. The column is reported so the user can
  cut on it.

______________________________________________________________________

## Web viewer

`hq serve` starts a local web app for exploring a run. It is built with
FastAPI and Jinja2 templates, served by uvicorn, and plots in the browser with
Plotly.js loaded from a CDN. There is no frontend build step.

```python
harv_hq.create_app(run_dir: str | os.PathLike) -> fastapi.FastAPI
```

At startup the app loads `hq.toml`, the model file, `catalog.parquet` and
`summary.parquet` (joined on `source_id`) into memory once. It opens `data.h5`
and the results files read-only, and reads per-source groups on request. The
summary is reloaded when its file modification time changes. The server binds
`127.0.0.1` by default; for a run on a remote machine, use an SSH tunnel
(`ssh -L 8000:localhost:8000 host`).

### Pages

#### Catalog explorer (`/`)

- A scatter plot (Plotly `scattergl`) of any two numeric columns from the
  joined catalog + summary table, with optional color by a third column and
  per-axis log scale and reversal (for magnitudes in a CMD).
- Filters: a status filter (`ok`, `failed`, ...), and numeric range filters
  on any column.
- Box or lasso selection fills a sortable table below the plot. The table
  shows `source_id` plus user-chosen columns, and each row links to the
  source page.
- A search box opens the source page for an exact source ID.
- The chosen axes, filters and columns are kept in the URL query string so a
  view can be bookmarked or shared.

#### Source page (`/source/{source_id}`)

- Data panel. RV: RV versus time with posterior orbit curves for a subset of
  samples, and a phase-fold toggle (folded on the selected sample nearest the
  median period). Gaia astrometry: the sky-plane orbit and along-scan
  residuals versus time.
- A posterior scatter matrix (Plotly `splom`) with a parameter picker.
  Selecting points in it limits the orbit curves to those samples.
- A toggle between the rejection and MCMC sample sets when both exist.
- The source's summary row, status, error traceback, captured warnings, and
  catalog columns.

Orbit curves are computed on request, not stored. The server uses the model
from `make_setup` and evaluates
`RVModel.predict_at_times` or `GaiaAstrometryModel.predict_orbit_sky` under
`jax.vmap` on a grid from `harv.plot.get_time_grid`, capped at 4096 points.
At most 256 samples are sent to the browser, chosen by weight for top-K
rejection samples and uniformly for MCMC.

### JSON API

| Route                                       | Returns                                                          |
| ------------------------------------------- | ---------------------------------------------------------------- |
| `GET /api/columns`                          | Column names, dtypes, and units of the joined table              |
| `GET /api/catalog?columns=a,b,c&filter=...` | The requested columns as arrays, for all rows passing the filter |
| `POST /api/rows`                            | Body `{source_ids, columns}`; the requested rows                 |
| `GET /api/source/{source_id}`               | Data, samples, curves, summary row, statuses                     |

Non-finite values are encoded as JSON `null`. `/api/catalog` returns whole
columns, so the catalog view scales to roughly 10^6 sources before payload
size dominates; larger runs should filter first.

______________________________________________________________________

## Public API

```python
import harv_hq

harv_hq.init_run(run_dir, *, kind)          # writes hq.toml + prior.py templates
run = harv_hq.Run(run_dir)                  # loads and validates hq.toml
run.config                                  # harv_hq.Config (frozen)
run.prepare(*, overwrite=False)
run.make_prior_cache(*, overwrite=False)
run.run_rejection(*, shard=(0, 1), workers=1, mpi=False,
                  overwrite=False, retry_failed=False)
run.summarize()
run.run_mcmc(*, shard=(0, 1), workers=1, mpi=False,
             overwrite=False, retry_failed=False)
run.status() -> dict[str, dict[str, int]]   # stage -> status -> count
run.load_source(source_id) -> harv_hq.SourceResult

harv_hq.read_source(data_file, source_id) -> RVData | GaiaAstrometryData
harv_hq.create_app(run_dir) -> fastapi.FastAPI

# Exceptions
harv_hq.ConfigError, harv_hq.ProvenanceError
```

`SourceResult` is a frozen dataclass with fields `source_id`, `data`
(`RVData | GaiaAstrometryData`), `rejection` and `mcmc` (`Samples | None`),
`rejection_status` and `mcmc_status` (`str`), and `summary`
(`dict[str, Any] | None`).

`Run`, `Config`, and `SourceResult` are `@final`, following the
abstract-final pattern; hq has no abstract bases in this version.

The CLI entry point is the `hq` console script, built with `argparse`.

______________________________________________________________________

## Dependencies

| Dependency                      | Used for                                    |
| ------------------------------- | ------------------------------------------- |
| `harv` (same version)           | everything model-related                    |
| `astropy`                       | reading input tables, time-scale conversion |
| `h5py`                          | `data.h5` and results files                 |
| `pyarrow`                       | `catalog.parquet`, `summary.parquet`        |
| `arviz`                         | MCMC convergence diagnostics                |
| `fastapi`, `uvicorn`, `jinja2`  | the web viewer                              |
| `mpi4py` (extra `harv-hq[mpi]`) | `--mpi`                                     |

`hq` pins `harv` to its own version (they are released in lockstep from one
tag).

______________________________________________________________________

## Testing

- Unit tests mirror `src/harv_hq/` under `packages/hq/tests/unit/`.
- An end-to-end test builds a small run from simulated data
  (`harv.simulate.simulate_rv_sb1_data` written out as a survey-style table,
  and `simulate_gaia_epoch_astrometry`), runs every stage with a small
  prior cache, checks statuses, resume after a simulated crash, a change of
  shard count between invocations, and provenance refusals.
- The viewer is tested with FastAPI's `TestClient` against that run.
- MPI is tested only when `mpi4py` is installed, with `mpirun -n 2`.

______________________________________________________________________

## Planned features

- **Multi-instrument RV.** Several input tables, or an instrument column, per
  run, combined per source into a `SourceData` keyed by instrument and fit
  with harv's `MultiSurveyOffset`. Each source has its own instrument set, so
  the model file's contract would split into a cache setup and a per-source
  setup that returns `(prior, model, data)`. The offsets are Gaussian linear
  parameters, analytically marginalized, so the prior cache stays shared. hq
  would provide a helper that builds the stacked data and the offset
  extension from one `SourceData.indicator_data_by_type` call, so their row
  order agrees, with the reference instrument chosen from a configured
  preference order.
- **SB2 runs** (`kind = "sb2"`), needed for SDSS-V, where double-lined
  sources are identified in advance and routed to their own run. Each source
  becomes a `SystemData(primary=RVData, secondary=RVData)` fit with
  `JointModel.for_sb2` and `default_sb2_prior` (both already in harv; the
  shared prior cache works because `make_prior_cache` accepts a
  `JointModel`). Open questions: how the input table encodes the two
  components (a component column with one row per component-epoch, or paired
  `rv`/`rv2` columns per epoch), how component-qualified parameter names
  (`primary.rv_semiamp`) appear as summary columns, a derived mass-ratio
  column, and a two-component RV panel in the viewer.
- **Joint RV + Gaia astrometry runs**, with `kind = "joint"` and
  `JointModel.for_rv_and_gaia`.
- **Iterative rejection reruns** for under-resolved, multimodal sources (a
  larger or per-source prior library), replacing old HQ's `rerun_thejoker`.
- **Per-source periodogram interim priors** as an alternative to the shared
  cache, once harv's planned prior-cache resampling utility exists.
- **Null-model evidence**: a constant-RV (or 5-parameter astrometric) evidence
  column per source, so `ln_Z_int` can be compared against "no companion".
- **Dynamic MPI load balancing** (a master–worker queue) if static
  round-robin slices prove unbalanced in practice.
- **Batched evaluation over sources**, adopting harv's planned padded batch
  inference ("Batch inference over many datasets" in harv's spec) once it
  lands.
- **Export** of per-source samples to FITS for data releases.
