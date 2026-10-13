# The hq workflow

This page goes through the stages of an hq run in order: what each one reads and
writes, how to rerun a stage (from where it stopped, or from scratch), and which
changes to the inputs mean which stages have to be rerun. If you have not run hq
before, start with {doc}`getting-started`, which runs every stage on a small example.
Every `hq` subcommand's `--help` links back to this page.


## 1. The run directory

A run is a directory holding `hq.toml` and the model file (`prior.py`). Every stage
reads from and writes to this directory, and every subcommand except `hq init` takes
`--run-dir` (the default is the current directory; `hq init` takes the directory to
create as its argument). After a full run, it looks like this:

```
my-run/
├── hq.toml                     # the run configuration
├── prior.py                    # make_setup(), and optionally select_rows()
├── data.parquet                # per-source data            (hq prepare)
├── data_index.parquet          # where each source's rows are
├── catalog.parquet             # one row per source
├── prior_cache.h5              # the shared prior library   (hq prior-cache)
├── results/
│   ├── rejection/              # result parts               (hq run)
│   ├── mcmc/                   #                            (hq mcmc)
│   └── superseded-<time>-<stage>/   # results set aside by --overwrite
├── summary.parquet             # one row per source         (hq summarize)
└── logs/
```

Besides `hq.toml` and `prior.py`, the run directory may also hold your input table
and catalog, if `[data]` and `[catalog]` point there (as in {doc}`getting-started`).
Everything else in it is written by hq. The data, catalog, results, and summary are
Parquet, so they can be read with pyarrow, pandas, or polars without hq; the prior
library is HDF5, and `logs/` holds plain-text logs.


## 2. The stages, in order

| Command          | Reads                                       | Writes                                                  |
| ---------------- | ------------------------------------------- | ------------------------------------------------------- |
| `hq prepare`     | the `[data]` and `[catalog]` tables         | `data.parquet`, `data_index.parquet`, `catalog.parquet` |
| `hq prior-cache` | `prior.py`                                  | `prior_cache.h5`                                        |
| `hq run`         | the prepared data, `prior_cache.h5`         | `results/rejection/`                                    |
| `hq summarize`   | `catalog.parquet`, the results              | `summary.parquet`                                       |
| `hq mcmc`        | the prepared data, `summary.parquet`, the rejection results | `results/mcmc/`                         |
| `hq summarize`   | (again, to add the MCMC results)            | `summary.parquet`                                       |
| `hq compact`     | `results/<stage>/`                          | `results/<stage>/` (merged parts)                       |
| `hq status`      | the results                                 | nothing; prints counts per stage and status             |
| `hq serve`       | everything above                            | nothing; serves the viewer                              |

`hq prepare` reads the input table, applies the built-in cuts (a missing source ID,
non-finite times or measurements, non-positive uncertainties) and then the model
file's `select_rows`, if it defines one, converts the times to TCB, and drops sources
with fewer than `[prepare] min_n_obs` observations. It also joins the `[catalog]`
table, if there is one, onto the prepared sources.

`hq prior-cache` draws `[prior_cache] n_samples` samples from the prior that
`make_setup()` returns and writes them to one HDF5 file, which every source then
shares. `hq run` is the only stage that reads the cache, so it can be deleted once the
rejection results are final; the later stages then take its ID from the results.

`hq run` runs the rejection sampler on every prepared source. It can run as one
process, as a local pool (`--workers 8`), as one shard of a job array
(`--shard 3/16`), or across MPI ranks (`--mpi`, optionally with `--workers` on each
rank). Sources are split between processes the same way in every mode, and finished
sources are skipped however the earlier work was split, so a run started as a job
array can be finished with a single process. Each process writes its results in
batches ("parts"), so a crash loses only the sources it had not yet written.

`hq summarize` collects the newest result for every source into `summary.parquet`. It
reads only the small per-source tables, never the samples, so it is quick even for
large runs. Run it again whenever results change.

`hq mcmc` follows up a subset of sources with MCMC, started from their rejection
samples. It needs an `[mcmc]` table in `hq.toml`, and it reads `summary.parquet` to
choose the sources (by default, those the rejection sampler under-resolves but whose
period posterior is unimodal; a `select_for_mcmc(row)` function in the model file
replaces the rule). It refuses a summary that does not include the current rejection
results, so run `hq summarize` first. It takes the same execution options as `hq run`.

`hq compact` merges a stage's many small parts into a few large ones. A single
process, a local pool, or an MPI run compacts its stage automatically when it
finishes. A job array cannot know which of its shards finishes last, so after an
array, run `hq compact` (or `hq compact --stage mcmc`) once, e.g. as a job that
depends on the array.


## 3. Rerunning a stage

Every stage that loops over sources can simply be run again with the same command.
A source counts as done only when its newest result succeeded: sources that raised an
error (marked `failed`, with the traceback stored and shown in the viewer) are run
again, and finished sources are skipped. So after a timeout, a node failure, or a fix
for whatever made some sources fail, resubmitting the same command finishes the job.

To redo a stage from scratch instead, pass `--overwrite`. For `hq prepare` and `hq
prior-cache`, this replaces the files the stage writes. For `hq run` and `hq mcmc`,
the existing results are not deleted: `results/<stage>/` is moved aside to
`results/superseded-<time>-<stage>/`, and the stage starts over in an empty directory.
Nothing in hq reads the superseded directories, so they are only there in case you
want to compare against them, and you can delete them whenever you like.

`--no-compact` turns off the automatic compaction at the end of `hq run` and
`hq mcmc`. This is useful when you plan to run the stage again soon (for example, to
rerun failed sources), since compacting twice does the work twice, but there is no
harm in leaving it on.


## 4. What a change invalidates

Every file hq writes records what it was built from: a hash of each table of
`hq.toml` it depends on, a hash of the model file, the ID of the prepared data, and
for results, the ID of the prior cache. A stage refuses to build on outputs that
were made from different inputs, and stops with a `ProvenanceError` that names the
mismatched input and file. This means a run can never silently mix results from two
configurations, but it also means that after an edit, the stages that depend on it
have to be rerun with `--overwrite`, in order:

| What changed                                        | Rerun with `--overwrite`                                  |
| --------------------------------------------------- | --------------------------------------------------------- |
| `prior.py` (any edit, even to the prior alone)      | `prepare`, `prior-cache`, `run`, `mcmc`; then `summarize` |
| `[run]`                                             | `prepare`, `prior-cache`, `run`, `mcmc`; then `summarize` |
| `[data]` or `[prepare]`                             | `prepare`, `run`, `mcmc`; then `summarize`                |
| `[catalog]`, or the catalog file                    | `prepare`; then `summarize`                               |
| `[prior_cache]`                                     | `prior-cache`, `run`, `mcmc`; then `summarize`            |
| `[rejection]`                                       | `run`, `mcmc`; then `summarize`                           |
| `[mcmc]` (except `select` and `require_unimodal`)   | `mcmc`; then `summarize`                                  |
| `[mcmc]` `select` or `require_unimodal`             | nothing; run `hq summarize` and `hq mcmc` again           |
| `[results]` or `[serve]`                            | nothing                                                   |

The model file is hashed as a whole, because its `select_rows` decides which rows are
prepared. So even a prior-only change means preparing again. The prepared data are
identified by a hash of their contents, so preparing the same rows again keeps every
result valid. That is what makes a `[catalog]` change cheap: `hq prepare --overwrite`
rebuilds `catalog.parquet`, the prepared rows are unchanged, and `hq summarize` picks
up the new columns. A catalog edit is not refused anywhere, though, so it takes effect
only once you prepare again. Changing the `[mcmc]`
selection keys only changes which sources get MCMC, so existing MCMC results stay
valid and the next `hq mcmc` runs any newly selected sources. A new version of hq,
harv, or JAX is logged but not refused.

hq does not hash the input table itself, only the `[data]` settings that point to it,
and the same goes for the ID lists in `[prepare] include_ids` and `exclude_ids`. If you
replace or edit one of these files in place, run `hq prepare --overwrite`. If the
prepared rows change, the later stages then refuse the old results, and you rerun them
with `--overwrite` as for a `[data]` change.


## 5. Starting over

To start a run over completely while keeping its configuration, delete the outputs
hq wrote and run the stages again from `hq prepare`:

```bash
rm -r data.parquet data_index.parquet catalog.parquet prior_cache.h5 \
    summary.parquet results logs
```

This leaves `hq.toml`, `prior.py`, and any input files you keep in the run
directory. Running every stage with `--overwrite` in order does the same thing,
except that it keeps the old results in `results/superseded-*` directories.
