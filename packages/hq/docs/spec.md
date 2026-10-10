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
   status and summary statistics, and run provenance as Parquet tables, and
   reduce them to one summary table.
1. **Exploration.** Serve a local web app with a catalog view (scatter plots
   of any catalog or summary column, lasso selection, a table of the
   selection) and a per-source page looked up by catalog source ID.

### Supported data

| Run kind          | Input rows                                                                    | harv model            |
| ----------------- | ----------------------------------------------------------------------------- | --------------------- |
| `rv`              | RV epochs from one instrument of any survey (APOGEE, SDSS-V, DESI, Gaia, ...) | `RVModel`             |
| `gaia_astrometry` | Gaia epoch astrometry (along-scan positions)                                  | `GaiaAstrometryModel` |

A run is one kind, read from one input table, and each source's data is a
single `RVData` or `GaiaAstrometryData`. Multi-instrument RV, SB2, and joint
RV + astrometry runs are planned (see "Planned features").

### Non-goals

- hq never re-implements orbit math, likelihoods, priors, sampling, or the
  structure of `Samples`. It calls harv's public API only. If hq needs
  something harv cannot do, the capability is added to harv (and its spec)
  first.
- hq does not do barycentric corrections. Input times must already be
  barycentric; hq only converts their time *scale* and format.
- hq does not choose priors. The prior and model are defined by the user in
  Python (see "The model file").
- hq does not manage numerical precision. Importing harv enables JAX's
  float64 mode (harv spec, "Core design principles"), and that applies in
  every hq process, including pool workers and MPI ranks.

### Lessons carried over

hq replaces the HQ pipeline built on The Joker (`hq-thejoker`) and generalizes
the `phobos` project. The design fixes the specific problems found in both:

| Problem in earlier pipelines                                     | hq rule                                                                    |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Failed sources only appear in logs and are silently retried      | Every source gets a recorded status (`ok` or `failed`) and error           |
| Stage completion means "a file exists"; stale results undetected | Every output records provenance hashes; mismatches refuse to resume        |
| One shared results file, merged by callbacks and `atexit` hooks  | Each process writes immutable result parts with an atomic rename           |
| Per-source scans of the full input table                         | One sort at prepare time; each process loads only its own sources          |
| Config keys silently ignored                                     | Unknown config keys are an error                                           |
| Survey-specific columns hard-coded (`APOGEE_ID`)                 | All columns, units and IDs come from the config                            |
| Prior constants and data paths in Python module constants        | Run settings in `hq.toml`, model and prior in `prior.py`, both snapshotted |

______________________________________________________________________

## Storage formats

hq stores everything it writes as Apache Parquet, read and written with
`pyarrow`. The one exception is the prior cache, which is harv's own HDF5
format (harv spec, "Building a prior cache") because harv streams it in
contiguous row slices.

Parquet suits hq's access patterns:

- **Population-scale reads are columnar.** Resume reads two columns of every
  per-source table; population inference reads a few parameter columns for
  every source. Neither touches anything else.
- **Results are written as immutable parts.** A process never appends to an
  existing file, so a crash can lose the results it has not yet written, but
  never damages results already on disk.
- **No per-source file structure.** Per-source lookups go through an index
  (source ID to file, row offset, row count), so there is no per-source group
  or file whose count grows with the sample.
- **Standard tools read the outputs directly.** `pyarrow.dataset`, polars,
  duckdb, and pandas can query a run's tables without hq.

Conventions shared by every Parquet file hq writes:

- Physical units are stored in each column's field metadata under the key
  `unit` (an astropy-parsable string, `""` for dimensionless).
- Run provenance (see "Provenance") is stored as JSON in the file's
  key-value metadata under the key `hq.provenance`.
- Files are written to `<name>.tmp` and renamed into place, so a file with its
  final name is always complete. Readers ignore `*.tmp`.

______________________________________________________________________

## Run directory

Everything about a run lives in one directory. Relative paths in the config
are resolved against it; absolute paths are used as given (survey data
usually lives on a scratch filesystem outside any repository).

```
run/
├── hq.toml                         # run configuration
├── prior.py                        # the model file: prior + model
├── data.parquet                    # one row per observation         (hq prepare)
├── data_index.parquet              # one row per source: row range   (hq prepare)
├── catalog.parquet                 # one row per prepared source     (hq prepare)
├── prior_cache.h5                  # shared prior library, harv HDF5 (hq prior-cache)
├── results/
│   ├── rejection/                  #                                 (hq run)
│   │   ├── 0003-of-0016-7f3a9c1e.sources.parquet
│   │   └── 0003-of-0016-7f3a9c1e.samples.parquet
│   └── mcmc/                       #                                 (hq mcmc)
├── summary.parquet                 # one row per source              (hq summarize)
└── logs/
    └── rejection-0003-of-0016.log  # one log per process slice
```

`hq init RUN_DIR` creates the directory with a template `hq.toml` and a
template `prior.py` for each run kind.

______________________________________________________________________

## Configuration (`hq.toml`)

The run configuration is a TOML file parsed with the standard-library
`tomllib`. It is loaded into a frozen `harv_hq.Config` object with
`Config.from_file(path)`. Validation is strict: unknown tables and keys, wrong
types, missing required tables and keys, and invalid values raise
`harv_hq.ConfigError` naming the offending key and table. Keys marked
"required" have no default.

Beyond types, these values are checked: `run.kind`, `mcmc.select`, and
`mcmc.chain_method` (`"sequential"`, `"parallel"`, or `"vectorized"`) against
their allowed values; `data.time_format` and `data.time_scale` against
astropy's `Time.FORMATS` and `Time.SCALES`; every count and size
(`min_n_obs`, `n_samples`, `top_k`, `batch_size`, `num_*`, `flush_*`,
`compact_n_sources`) must be positive; and `[data]` must hold the keys of
`run.kind` and none of the other kind's. An integer is accepted where a float
is expected, but a boolean is never accepted as a number.

Each table is an attribute of `Config` holding a frozen dataclass of its keys
(`config.run.seed`, `config.data.rv`, ...). The optional `[catalog]` and
`[mcmc]` tables are `None` when absent; the other optional tables take their
defaults. Path values (`run.model_file`, `data.file`, `catalog.file`) are
stored as absolute `pathlib.Path`s, and `config.run_dir` is the directory
holding `hq.toml`.

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

### `[results]`

How often a process writes a result part, and how large compacted parts are
(see "Result parts" and "Compaction"). A part is written when either flush
limit is reached, and when the process finishes.

| Key                 | Type    | Default   | Description                            |
| ------------------- | ------- | --------- | -------------------------------------- |
| `flush_n_sources`   | `int`   | `1000`    | Write a part after this many sources   |
| `flush_seconds`     | `float` | `600.0`   | Write a part after this much wall time |
| `compact_n_sources` | `int`   | `100_000` | Sources per part written by compaction |

The flush limits bound the work a crash can lose; smaller values mean more,
smaller files until compaction merges them.

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
- Every hq table stores the ID in a `source_id` column of that native type.
- IDs are never used as file or group names, so no characters are reserved.
  In the viewer, IDs are URL-encoded in paths (`+` in APOGEE IDs), and the
  source routes match the rest of the path (FastAPI's `{source_id:path}`),
  because a `/` in an ID is decoded before routing.

### Per-source randomness

Each source's PRNG key is derived from the run seed and the ID alone, so it
does not depend on shard layout, process count, or execution order:

```python
key = jax.random.fold_in(jax.random.key(seed), stable_hash(source_id))
```

`harv_hq.stable_hash(source_id)` is the first 4 bytes of the sha256 of
`str(source_id)` (UTF-8 encoded), read as a big-endian unsigned 32-bit
integer. Integer and string IDs that print the same hash the same. It is also
the way to select a reproducible subset of sources, e.g.
`stable_hash(id) % 100 == 0` in `select_rows`. Stage keys are split from it with
`jax.random.fold_in(key, stage)`, with `stage` 0 for rejection and 1 for MCMC.
The prior cache uses `jax.random.fold_in(jax.random.key(seed), 2**32 - 1)`.

______________________________________________________________________

## Stages

Each stage is a public method on `harv_hq.Run` (see "Public API"), and each
CLI subcommand is a thin wrapper around one method. Stages run in this order:

| CLI                       | Method                 | Reads                                                    | Writes                                                  |
| ------------------------- | ---------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| `hq init RUN_DIR --kind`  | `harv_hq.init_run`     | nothing                                                  | `hq.toml`, `prior.py`                                   |
| `hq prepare`              | `Run.prepare`          | `[data]`, `[catalog]` inputs                             | `data.parquet`, `data_index.parquet`, `catalog.parquet` |
| `hq prior-cache`          | `Run.make_prior_cache` | `prior.py`                                               | `prior_cache.h5`                                        |
| `hq run`                  | `Run.run_rejection`    | `data*.parquet`, `prior_cache.h5`                        | `results/rejection/*.parquet`                           |
| `hq compact`              | `Run.compact`          | `results/rejection/`                                     | `results/rejection/` (merged parts)                     |
| `hq summarize`            | `Run.summarize`        | `catalog.parquet`, `results/*/*.sources.parquet`         | `summary.parquet`                                       |
| `hq mcmc`                 | `Run.run_mcmc`         | `data*.parquet`, `summary.parquet`, `results/rejection/` | `results/mcmc/*.parquet`                                |
| `hq compact --stage mcmc` | `Run.compact`          | `results/mcmc/`                                          | `results/mcmc/` (merged parts)                          |
| `hq summarize`            | `Run.summarize`        | (again, to add MCMC columns)                             | `summary.parquet`                                       |
| `hq status`               | `Run.status`           | `results/*/*.sources.parquet`                            | nothing (prints counts)                                 |
| `hq serve`                | `harv_hq.create_app`   | everything above                                         | nothing                                                 |

Every CLI subcommand takes `--run-dir` (default: the current directory).

### `prepare`

1. Read the `[data]` table, keeping only the configured columns.
1. Apply the built-in cuts and then `select_rows`, if defined.
1. Convert times to TCB with astropy `Time`, and store them as MJD (TCB) in
   days.
1. Sort all rows by `(source_id, time)` once, and find source boundaries from
   the sorted IDs. No step scans the table per source.
1. Drop sources with fewer than `min_n_obs` observations, and record the
   number dropped in the log.
1. Write `data.parquet`, `data_index.parquet`, and `catalog.parquet`.

`prepare` refuses to overwrite existing prepared files unless
`overwrite=True` (`--overwrite`), because doing so invalidates every
downstream output.

#### `data.parquet`

One row per observation, sorted by `(source_id, time)`, written with row
groups of 65,536 rows. Key-value metadata holds the provenance, the run
`kind`, and a `data_id` (a uuid4 generated by this `prepare`, which every
downstream output records).

| Column                                                            | Type      | Field metadata                              |
| ----------------------------------------------------------------- | --------- | ------------------------------------------- |
| `source_id`                                                       | int64/str |                                             |
| `time`                                                            | float64   | `unit="day"`, `format="mjd"`, `scale="tcb"` |
| `rv`, `rv_err`                                                    | float64   | `unit` (`rv` runs)                          |
| `al_position`, `al_position_err`, `scan_angle`, `parallax_factor` | float64   | `unit` (`gaia_astrometry` runs)             |

#### `data_index.parquet`

One row per prepared source: `source_id`, `row_start` (int64, the source's
first row in `data.parquet`), and `n_obs` (int64). A source's observations are
rows `row_start` to `row_start + n_obs` of `data.parquet`.

`harv_hq.read_source(run_dir, source_id) -> RVData | GaiaAstrometryData`
reads that row range and rebuilds the harv data object. `time_ref` is left to
harv's default (the mean time). `harv_hq.read_sources(run_dir, source_ids)`
does the same for many sources with one read of the row groups they span and
returns a `dict`; the pipeline stages use it to load a process's slice (see
"Execution modes").

#### `catalog.parquet`

One row per prepared source: `source_id`, `n_obs`, `time_baseline` (days),
then the `[catalog]` columns joined on `source_id` (left join: prepared
sources with no catalog row get nulls). Catalog columns that collide with
these names raise `ConfigError`.

### `make_prior_cache`

Calls `make_setup()`, then
`harv.samplers.make_prior_cache(prior, model, n_samples, path, key=..., batch_size=...)`.
hq then writes the provenance JSON to the HDF5 file's root attribute
`hq.provenance`.

### `run_rejection`

Each process calls `make_setup()` once and builds one
`RejectionSampler(prior, model, batch_size=..., min_evidence_ess=...)`. It
loads the data for its slice (see "Execution modes") with `read_sources`.
Then, for each source in the slice that is not already done:

1. `samples = sampler.run_with_samples(data, prior_cache, key=..., top_k=..., ignore_non_finite=..., randomize_prior_order=...)`.
   `top_k` forces `return_logprobs` and `return_evidence_stats`, so every
   result carries `ln_likelihood`, `ln_prior`, and the evidence metadata.
1. Compute the source's summary statistics (see "Summary statistics") while
   the data and samples are in memory.
1. Add the source's samples and its sources-table row to the process's
   buffer, and write a part when a `[results]` limit is reached (see
   "Results").

harv's under-resolution `UserWarning` is captured per source with
`warnings.catch_warnings(record=True)` and stored in the source's `warnings`
column, not printed. Any exception is caught, its traceback is stored, and the
source is marked `failed`; the run continues.

Within a slice, sources are processed in ascending `n_obs`, so consecutive
sources share array shapes and reuse JIT compilations.

### MCMC follow-up (`run_mcmc`)

Reads `summary.parquet` (which must be current, see "Provenance") to select
sources. Each process calls `make_setup()` once, builds one
`NumpyroSampler(prior, model)`, and loads the data for its slice of the
selected sources. Then, for each selected source in its slice:

1. Load the source's rejection `Samples` (see `Run.load_source`) and pass its
   equal-weight resample (see "Weighted samples") as `init_samples`, so chains
   start at draws in proportion to their posterior weight.
1. `sampler.run(data, init_samples=..., key=..., num_chains=..., num_warmup=..., num_samples=..., chain_method=..., return_logprobs=True)`.
1. Compute `r_hat_max` and `ess_bulk_min` over all sampled parameters from
   `samples.to_arviz()` (`arviz.rhat`, `arviz.ess`), and the summary
   statistics.
1. Buffer and write the result as in `run_rejection`, with `mcmc_status`:

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
each slice stays sorted by `n_obs`. The slice depends only on
`data_index.parquet`, `i`, and `N`.

Each process loads the observations for its whole slice into memory once,
with `read_sources`. A slice holds `1/N` of the data, so memory per process
falls as `N` grows.

| Mode                     | Invocation                          | Slice               | Writes parts named        |
| ------------------------ | ----------------------------------- | ------------------- | ------------------------- |
| Single process (default) | `hq run`                            | `0/1`               | `0000-of-0001-<uuid>`     |
| Explicit shard           | `hq run --shard 3/16`               | `3/16`              | `0003-of-0016-<uuid>`     |
| Local pool               | `hq run --workers 8 [--shard 3/16]` | the process's slice | as above, by the parent   |
| MPI                      | `mpirun -n 64 hq run --mpi`         | `rank/size`         | `<rank>-of-<size>-<uuid>` |

- **Explicit shards** are for job arrays and task launchers (SLURM
  `--array=0-15` with `--shard $SLURM_ARRAY_TASK_ID/16`, disBatch). Shards are
  fully independent processes.
- **Local pool** runs a `ProcessPoolExecutor` with the `spawn` start method.
  Each worker is pinned to one CPU thread (BLAS and XLA environment variables
  set before JAX is imported), so `W` workers use `W` cores. The worker
  initializer imports the model file and builds the sampler once. The parent
  loads the slice's data and sends each task its source's arrays; workers
  return results as plain NumPy payloads, and only the parent buffers and
  writes parts.
- **MPI** uses `mpi4py` (the `harv-hq[mpi]` extra), imported lazily so a
  missing install raises `ImportError` naming `hq run --mpi`. Rank `r` of `n`
  processes slice `r/n` and writes its own parts, so ranks need no
  communication beyond a final barrier. Rank 0 additionally logs aggregate
  progress. `--mpi` is mutually exclusive with `--shard` and `--workers`.
- GPU runs use the single-process or explicit-shard modes, one process per
  device.

**Compaction at the end of a stage.** When one invocation has run the whole
stage, it compacts the stage's parts before exiting (see "Compaction"): a
single process or local pool on slice `0/1`, and MPI, where rank 0 compacts
after the final barrier. Explicit shards cannot know which shard finishes
last, so after a job array completes, `hq compact` (or `--stage mcmc`) is run
once, e.g. as a SLURM job with `--dependency=afterok:<array job id>`.
`--no-compact` skips the automatic step.

### Resume

Resume does not depend on shard layout. Before processing its slice, a
process reads the `source_id`, `status`, and `finished` columns of every
`results/<stage>/*.sources.parquet`. A source is done if its newest row (by
`finished`) has status `ok`. A source whose newest row is `failed` is retried
only with `--retry-failed`. Two rows with the same `finished` are copies of
one result (a part and its compacted copy, briefly, while compaction runs);
either may be used, and readers pick the one whose part stem sorts first.

So a run started as `--shard i/16` can be finished with `--mpi` on 64 ranks,
or with a single process, without recomputing finished sources. A crash loses
only the sources still in the crashed process's buffer.

`--overwrite` moves `results/<stage>/` to
`results/superseded-<timestamp>-<stage>/` (never deleting it) and starts
fresh.

______________________________________________________________________

## Results

### Result parts

A process writes its buffered results as one *part*: a pair of files in
`results/<stage>/` sharing a stem `<i>-of-<N>-<uuid8>`, where `uuid8` is the
first 8 hex digits of a fresh uuid4. The uuid makes part names unique, so
concurrent or repeated processes with the same slice never collide.

1. `<stem>.samples.parquet` is written first, then
1. `<stem>.sources.parquet`.

Each is written to `.tmp` and renamed. The sources file is the commit: a
source counts as written only when its row exists in a `.sources.parquet`. A
samples file without its sources file (from a crash between the two renames)
is ignored by every reader.

#### `*.sources.parquet`

One row per source in the part.

| Column                                     | Type            | Description                                                             |
| ------------------------------------------ | --------------- | ----------------------------------------------------------------------- |
| `source_id`                                | int64/str       |                                                                         |
| `status`                                   | str             | `"ok"` or `"failed"`                                                    |
| `error`                                    | str             | Full traceback for `failed`, else `""`                                  |
| `warnings`                                 | str             | Newline-joined captured warning messages, or `""`                       |
| `seed_hash`                                | int64           | `stable_hash(source_id)`                                                |
| `n_obs`                                    | int64           | Observations used                                                       |
| `started`, `finished`                      | timestamp (UTC) |                                                                         |
| `wall_time_s`                              | float64         | Wall time for this source                                               |
| `samples_row_start`, `n_samples`           | int64           | The source's row range in this part's samples file (0, 0 when `failed`) |
| one column per `Samples.metadata` key      |                 | e.g. `time_ref`, `time_ref_unit`, `ln_Z_int`, `n_prior_samples`         |
| summary statistics                         |                 | See "Summary statistics"                                                |
| `mcmc_status`, `r_hat_max`, `ess_bulk_min` |                 | `mcmc` stage only                                                       |

The `Samples.metadata` keys are listed in the key-value metadata entry
`hq.samples_metadata_keys` (JSON), so the `Samples` can be rebuilt.

#### `*.samples.parquet`

One row per posterior sample, grouped by source in the order of the sources
file, so each source's samples are one contiguous row range.

| Column                      | Type      | Description                                                                   |
| --------------------------- | --------- | ----------------------------------------------------------------------------- |
| `source_id`                 | int64/str |                                                                               |
| `sample_index`              | int32     | Index within the source's samples                                             |
| `chain`                     | int16     | `mcmc` stage only                                                             |
| one column per parameter    | float64   | Every nonlinear and linear parameter, unit in field metadata                  |
| `ln_likelihood`, `ln_prior` | float64   |                                                                               |
| `weight`                    | float64   | `Samples.weight` (`rejection` stage only), so readers need not reconstruct it |

Key-value metadata records the run-wide `Samples` structure:
`hq.model_type`, `hq.linear_extension_names`, `hq.nonlinear_names`, and
`hq.linear_names` (JSON). Every source in a run shares one model, so these are
identical across parts, and readers check that they are.

The mapping between a `Samples` and these columns belongs to harv: hq writes
`Samples.to_columns()` and rebuilds with `Samples.from_columns(...)` (harv
spec, "`to_columns` / `from_columns`"). **This requires a harv addition**,
which lands in harv, with its spec entry, before hq's results writer.

### Compaction

A large run leaves many small parts (one per process per flush), and some
sources appear more than once (a `failed` row superseded by a later `ok` one,
or duplicates from overlapping processes). `Run.compact(stage)` rewrites a
stage's parts into a few large ones holding only the newest row per source.

1. **Snapshot.** List the committed parts in `results/<stage>/` (sources files
   with their samples files) at the start. Only these are inputs; parts
   committed while compaction runs are left alone. If the snapshot is already
   compacted (every part is a compacted part and no source appears twice),
   return without writing anything.
1. **Check.** Every input's provenance must match the current one
   (`ProvenanceError` otherwise), and every input must carry the same
   run-wide `Samples` structure.
1. **Select.** Read the sources rows of all inputs, keep the newest row per
   source (the "Resume" rule), and sort them by `source_id`.
1. **Write.** For each block of `compact_n_sources` sources, gather their
   samples from the input samples files (row-range reads), and write a new
   part with stem `c-<uuid8>-<k:04d>` through the normal part protocol
   (samples file, then sources file, each renamed into place), with samples
   in the same `source_id` order and recomputed `samples_row_start`. Row
   groups hold 131,072 rows.
1. **Verify.** Re-read the new sources files and confirm they contain exactly
   the selected `(source_id, finished)` pairs.
1. **Delete inputs.** Remove the snapshot's input files, sources file first,
   then samples file. A file already gone (removed by a concurrent
   compaction) is skipped.

Readers stay correct at every point. Before step 6, each source is present in
an input part and possibly also in a compacted part with the same `finished`,
which the tie rule in "Resume" resolves; after it, only the compacted copy
remains. A crash at any step leaves a valid, possibly duplicated, results
directory, and rerunning compaction finishes the job. Compaction therefore
needs no lock, but it should not be run against a stage that is still being
written if the goal is a fully merged result: later parts survive it and need
another compaction.

Compaction drops superseded rows, including the tracebacks of `failed`
attempts that were later retried successfully; the `logs/` files keep them.
After compaction, every `*.samples.parquet` row belongs to a current result,
so population analyses can read the samples dataset directly with no join.

### Loading results

`harv_hq.Run.load_source(source_id) -> SourceResult` returns the source's
data, its rejection and MCMC `Samples` (or `None`), their statuses, and its
summary row. At first use, `Run` builds an in-memory index from the
`source_id`, `finished`, `samples_row_start`, and `n_samples` columns of every
sources file (newest row per source wins), then reads only the row groups of
the samples file that span the source's rows. If that file has since been
removed by compaction, the index is rebuilt and the read retried once.

For population analyses, the samples are directly readable as one dataset:
`pyarrow.dataset.dataset(run_dir / "results/rejection", format="parquet")`
filtered to `*.samples.parquet`. After compaction every row is current; before
it, superseded rows must be removed by joining on the newest sources rows.

### Provenance

Every output (`data.parquet`, `data_index.parquet`, `catalog.parquet`,
`prior_cache.h5`, every result part, `summary.parquet`) records a provenance
JSON object (Parquet key-value metadata, or the HDF5 root attribute for the
prior cache) with these keys:

| Key                                         | Description                                                                |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `hq_version`, `harv_version`, `jax_version` | Package versions                                                           |
| `config_sha256`                             | sha256 of `hq.toml` bytes                                                  |
| `model_file_sha256`                         | sha256 of the model file bytes (absent on the prepared data)               |
| `data_id`                                   | The `data_id` of the prepared data                                         |
| `prior_cache_id`                            | A uuid4 written into the prior cache when it was built (result parts only) |
| `created`                                   | ISO 8601 UTC timestamp                                                     |

A stage refuses to build on outputs whose `config_sha256`,
`model_file_sha256`, `data_id`, or `prior_cache_id` differ from the current
ones, raising `harv_hq.ProvenanceError` that names the mismatched field and
file. The fix is `--overwrite` on that stage. Version differences are logged
but not refused. `run_mcmc` also refuses a `summary.parquet` whose `created`
is older than the newest rejection part.

### Summary statistics

The worker computes these per source, while the data and samples are in
memory, and stores them as columns of the sources table:

| Column(s)                                            | Source                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `well_resolved`                                      | `Samples.acceptance_diagnostics(min_evidence_ess=...)`       |
| `period_unimodal`                                    | `Samples.period_unimodal(data)` on the equal-weight resample |
| `max_phase_gap`, `phase_coverage`, `periods_spanned` | the corresponding `Samples` methods, at the MAP sample       |
| `map_<param>`                                        | `Samples.map_sample()`, every parameter in `Samples.keys()`  |
| `<param>_p16`, `<param>_p50`, `<param>_p84`          | percentiles; weighted for rejection (see below)              |

`well_resolved` is rejection-only. Units are recorded in field metadata, in
the unit of the samples.

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

### Summary table (`summary.parquet`)

`Run.summarize()` reads the sources tables of both stages, keeps the newest
row per source, and joins them onto `catalog.parquet`. It reads no samples
and no data. One row per prepared source:

| Column(s)                                                                                                               | Source                                                        |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `source_id`, `n_obs`, `time_baseline`, `[catalog]` columns                                                              | `catalog.parquet`                                             |
| `rejection_status`, `rejection_error`                                                                                   | rejection sources table (`"pending"` / null when not yet run) |
| `n_prior_samples`, `ln_Z_int`, `ln_Z_int_mcse`, `ln_Z_int_ess`, `max_ln_likelihood`, `weight_captured`, `well_resolved` | rejection sources table                                       |
| `mcmc_status`, `r_hat_max`, `ess_bulk_min`                                                                              | mcmc sources table; `"not_selected"` / null when absent       |
| `final`                                                                                                                 | `"mcmc"` if `mcmc_status == "converged"`, else `"rejection"`  |
| `period_unimodal`, phase-coverage columns, `map_<param>`, `<param>_p*`                                                  | the sources table of the `final` stage                        |

Every source shares one model, so every `ok` source has the same parameter
columns; sources without results get nulls.

______________________________________________________________________

## Web viewer

`hq serve` starts a local web app for exploring a run. It is built with
FastAPI and Jinja2 templates, served by uvicorn, and plots in the browser with
Plotly.js loaded from a CDN. There is no frontend build step.

```python
harv_hq.create_app(run_dir: str | os.PathLike) -> fastapi.FastAPI
```

At startup the app loads `hq.toml`, the model file, and `catalog.parquet`
joined to `summary.parquet` (on `source_id`) into memory once, and builds the
results index used by `Run.load_source`. Per-source data and samples are read
on request. The summary and the results index are reloaded when the summary's
modification time changes. The server binds `127.0.0.1` by default; for a
run on a remote machine, use an SSH tunnel (`ssh -L 8000:localhost:8000 host`).

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

| Route                                                  | Returns                                                          |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| `GET /api/columns`                                     | Column names, dtypes, and units of the joined table              |
| `GET /api/catalog?columns=a,b,c&filter=...&status=...` | The requested columns as arrays, for all rows passing the filter |
| `POST /api/rows`                                       | Body `{source_ids, columns}`; the requested rows                 |
| `GET /api/source/{source_id}`                          | Data, samples, curves, summary row, statuses                     |

The `filter` parameter is a comma-separated list of `column:min:max` terms
(an empty bound is open, e.g. `bp_rp:0.5:` or `n_obs::20`), combined with
AND; a malformed term returns 400. `status` is a comma-separated list of
`rejection_status` values to keep (default: all). Non-finite values are encoded as JSON
`null`. `/api/catalog` returns whole columns, so the catalog view scales to
roughly 10^6 sources before payload size dominates; larger runs should filter
first.

______________________________________________________________________

## Public API

```python
import harv_hq

harv_hq.init_run(run_dir, *, kind)          # writes hq.toml + prior.py templates
config = harv_hq.Config.from_file(path)     # loads and validates an hq.toml
harv_hq.stable_hash(source_id) -> int       # see "Per-source randomness"
run = harv_hq.Run(run_dir)                  # loads and validates hq.toml
run.config                                  # harv_hq.Config (frozen)
run.prepare(*, overwrite=False)
run.make_prior_cache(*, overwrite=False)
run.run_rejection(*, shard=(0, 1), workers=1, mpi=False,
                  overwrite=False, retry_failed=False, compact=True)
run.compact(stage="rejection")              # or "mcmc"
run.summarize()
run.run_mcmc(*, shard=(0, 1), workers=1, mpi=False,
             overwrite=False, retry_failed=False, compact=True)
run.status() -> dict[str, dict[str, int]]   # stage -> status -> count
run.load_source(source_id) -> harv_hq.SourceResult

harv_hq.read_source(run_dir, source_id) -> RVData | GaiaAstrometryData
harv_hq.read_sources(run_dir, source_ids) -> dict[Any, RVData | GaiaAstrometryData]
harv_hq.create_app(run_dir) -> fastapi.FastAPI

# Exceptions
harv_hq.ConfigError, harv_hq.ProvenanceError
```

`ConfigError` subclasses `ValueError` (an invalid `hq.toml` or model file);
`ProvenanceError` subclasses `RuntimeError` (an output built from different
inputs). `init_run(run_dir, *, kind)` creates `run_dir` if needed, refuses a
non-empty one with `FileExistsError`, and returns it as a `Path`; `hq init`
takes `--kind rv|gaia_astrometry` (default `rv`).

`SourceResult` is a frozen dataclass with fields `source_id`, `data`
(`RVData | GaiaAstrometryData`), `rejection` and `mcmc` (`Samples | None`),
`rejection_status` and `mcmc_status` (`str`), and `summary`
(`dict[str, Any] | None`).

`Run`, `Config`, and `SourceResult` are `@final`, following the
abstract-final pattern; hq has no abstract bases in this version.

The CLI entry point is the `hq` console script, built with `argparse`.

______________________________________________________________________

## Dependencies

| Dependency                      | Used for                                     |
| ------------------------------- | -------------------------------------------- |
| `harv` (same version)           | everything model-related                     |
| `astropy`                       | reading input tables, time-scale conversion  |
| `pyarrow`                       | every Parquet file hq reads and writes       |
| `h5py`                          | provenance attribute on the harv prior cache |
| `arviz`                         | MCMC convergence diagnostics                 |
| `fastapi`, `uvicorn`, `jinja2`  | the web viewer                               |
| `mpi4py` (extra `harv-hq[mpi]`) | `--mpi`                                      |

`hq` pins `harv` to its own version (they are released in lockstep from one
tag).

______________________________________________________________________

## Testing

- Unit tests mirror `src/harv_hq/` under `packages/hq/tests/unit/`.
- An end-to-end test builds a small run from simulated data
  (`harv.simulate.simulate_rv_sb1_data` written out as a survey-style table,
  and `simulate_gaia_epoch_astrometry`), runs every stage with a small
  prior cache, and checks statuses, resume after a simulated crash (only
  buffered sources are lost), a change of shard count between invocations, an
  orphaned samples part being ignored, provenance refusals, and compaction
  (merged output equals the newest rows, superseded rows dropped, a crash
  injected after each compaction step still leaves a readable directory, and
  a second compaction of a compacted stage writes nothing).
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
  preference order. Per-source parameter sets differ (each source's offsets),
  so the samples schema would need a defined union of columns.
- **SB2 runs** (`kind = "sb2"`), needed for SDSS-V, where double-lined
  sources are identified in advance and routed to their own run. Each source
  becomes a `SystemData(primary=RVData, secondary=RVData)` fit with
  `JointModel.for_sb2` and `default_sb2_prior` (both already in harv; the
  shared prior cache works because `make_prior_cache` accepts a
  `JointModel`). Open questions: how the input table encodes the two
  components (a component column with one row per component-epoch, or paired
  `rv`/`rv2` columns per epoch), how component-qualified parameter names
  (`primary.rv_semiamp`) appear as columns, a derived mass-ratio column, and a
  two-component RV panel in the viewer.
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
