# Getting started with hq

In this guide, we will run a small simulated RV survey through every stage of an hq
run: we will prepare the data, build the prior samples library, run the rejection
sampler on every source, follow up the under-resolved sources with MCMC, and look at
the results in the viewer. Everything here runs on a laptop in a few minutes. See
{doc}`index` for how to install `harv-hq`.


## 1. Making a run directory

A run lives in its own directory. `hq init` creates one from a template for the kind
of data you have (`rv`, `gaia_astrometry`, or `sb2` for double-lined binaries):

```bash
hq init my-run --kind rv
```

This writes two files. `hq.toml` is the run's configuration: where the input table is
and which of its columns hold the source IDs, times, and RVs, how large the prior
samples library is, and the settings for each stage. `prior.py` is the model file,
which defines a `make_setup()` function that returns the prior and the model used for
every source. Both files are part of the run's provenance, so their contents are
recorded in every output.


## 2. A toy survey

hq reads one table with one row per observation, in any format that
`astropy.table.Table.read` can read. For this example, we will simulate 50 sources
with harv and write them out as a FITS table, with between 4 and 15 epochs per
source:

```python
import numpy as np
from astropy.table import Table, vstack
from unxt import ustrip

from harv.simulate import simulate_rv_sb1_data

tables = []
for i in range(50):
    data, _ = simulate_rv_sb1_data(seed=i, n_obs=4 + i % 12)
    tables.append(
        Table(
            {
                "source_id": [f"star-{i:03d}"] * data.n_obs,
                "time": np.asarray(ustrip("day", data.time)) + 2_460_000.0,
                "rv": np.asarray(ustrip("km/s", data.rv)),
                "rv_err": np.asarray(ustrip("km/s", data.rv_err)),
            }
        )
    )
vstack(tables).write("my-run/observations.fits")
```

The column names match the template's `[data]` table, and the times are Julian dates,
which is the template's default `time_format`. With a real survey, you would point
`[data]` at the survey's own column names instead (see {doc}`project-setup`).


## 3. Configuring the run

The template is set up for a large run, so we will shrink two settings in `hq.toml`
for this example: the number of prior samples in the shared library, and the number
of samples kept per source. We will also turn on MCMC follow-up, with shorter chains
than the defaults:

```toml
[prior_cache]
n_samples = 1_000_000

[rejection]
top_k = 256

[mcmc]
select = "under_resolved"
num_warmup = 500
num_samples = 500
```

The template's `prior.py` uses harv's default RV prior with periods between 2 and 4096
days. That range covers the simulated binaries, so we can leave it as it is. For a
real run, this is where you choose the period range and the amplitude scales for your
science case (see {doc}`../tutorials/rv/1-customize-prior`).


## 4. Running the stages

From inside the run directory, we run the stages in order:

```bash
cd my-run
hq prepare             # group the input table by source
hq prior-cache         # draw the shared prior samples library
hq run --workers 4     # the rejection sampler on every source
hq summarize           # one row per source: statuses and summary statistics
hq mcmc --workers 4    # MCMC on the sources the rejection sampler under-resolves
hq summarize           # again, to add the MCMC results
hq status              # how many sources are done, per stage
```

`hq prepare` writes the per-source data (`data.parquet`) and a catalog table with one
row per source. `hq prior-cache` draws the prior samples once and writes them to
`prior_cache.h5`, which every source then shares. `hq run` fits every source with the
rejection sampler, using four worker processes here, and writes the posterior samples
to `results/rejection/`. On a laptop, this takes about ten seconds for these 50
sources.

`hq summarize` writes `summary.parquet`, which has one row per source with its status,
evidence diagnostics, and percentiles of every parameter. `hq mcmc` reads it to pick
the sources to follow up: by default, those whose rejection samples do not resolve the
posterior well (`well_resolved` is false) but whose period posterior is unimodal. Each
of these gets MCMC chains started from its rejection samples. MCMC is much slower per
source than the rejection step, which is why only some sources get it: here, it takes
about two minutes for the 30 or so selected sources on four cores.

`hq run` and `hq mcmc` can be interrupted and run again with the same command. A
rerun skips the sources that are already done, so nothing finished is computed twice.


## 5. Looking at the results

`hq serve` starts a local web viewer:

```bash
hq serve               # then open http://127.0.0.1:8000
```

The front page plots any two columns of the summary table against each other (for
example, `n_obs` against `ln_Z_int`), and box or lasso selection on the plot lists the
selected sources in a table. Each source has its own page, with its data, posterior
orbit curves, and a scatter matrix of its samples. Selecting samples in the scatter
matrix limits the curves to those samples, and a toggle switches between the rejection
and MCMC samples when a source has both.

The results are also available from Python. Still inside `my-run`, `summary.parquet`
is an ordinary Parquet file (e.g., `pandas.read_parquet("summary.parquet")`), and a
`harv_hq.Run` loads everything hq holds for one source, including its data and its
samples as harv {py:class}`~harv.samplers.Samples` objects:

```python
import harv_hq

run = harv_hq.Run(".")  # the run directory
result = run.load_source("star-007")
result.rejection_status, result.mcmc_status
samples = result.mcmc if result.mcmc is not None else result.rejection
samples.plot_corner()
```


## Next steps

The {doc}`workflow` page goes through the stages in more detail: what each one writes,
how to rerun a stage from scratch, and which stages need to be rerun after a change to
`hq.toml`, `prior.py`, or the input data. To set up a run on a real survey and scale
it out to a cluster, see {doc}`project-setup`.
