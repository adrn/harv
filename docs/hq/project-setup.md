# Setting up a survey run

This page walks through setting up an hq run on a real survey and scaling it out to a
cluster. We use SDSS-V APOGEE exposure-level RVs as the example, but nothing here
depends on a particular file format: every survey-specific choice is a step where you
look at the data and decide. See {doc}`workflow` for what each stage does.


## 1. Create the project repository

A run is configuration plus code, and both belong in git. The data and the outputs do
not. A layout that has worked for us:

```
sdss5-binaries/
├── pyproject.toml          # depends on harv-hq
├── uv.lock
├── README.md               # what each run is, and where its data lives
├── scripts/
│   ├── make_input_table.py # raw SDSS-V products -> one flat exposure table
│   └── make_catalog.py     # raw SDSS-V products -> one row per star
├── runs/
│   ├── test-1pct/          # small subset, for tuning
│   │   ├── hq.toml
│   │   └── prior.py
│   └── full/
│       ├── hq.toml
│       └── prior.py
├── slurm/
│   ├── rejection.sbatch
│   └── mcmc.sbatch
└── .gitignore
```

```
# .gitignore
runs/*/*.parquet
runs/*/prior_cache.h5
runs/*/results/
runs/*/logs/
```

We use uv to manage the environment:

```bash
uv init --bare sdss5-binaries && cd sdss5-binaries
uv python pin 3.12            # or newer; harv requires >= 3.12
uv add harv-hq
```

If you are also developing harv or hq, depend on your checkout instead:

```toml
# pyproject.toml
[tool.uv.sources]
harv = { path = "../harv", editable = true }
harv-hq = { path = "../harv/packages/hq", editable = true }
# or, pinned to a commit:
# harv = { git = "https://github.com/adrn/harv", rev = "<sha>" }
# harv-hq = { git = "https://github.com/adrn/harv", rev = "<sha>", subdirectory = "packages/hq" }
```

harv currently depends on a fork of unxt, which its own `pyproject.toml` sets with a
uv `override-dependencies` entry. uv applies overrides only from the project being
locked, so when depending on a harv checkout, copy that entry into your project's
`[tool.uv]` table.

Add the MPI extra (`uv add "harv-hq[mpi]"`) only on machines where you will use
`--mpi`, and build `mpi4py` against the cluster's MPI library
(`MPICC=... uv pip install --no-binary mpi4py mpi4py`) rather than using a generic
wheel.


## 2. Understand the data before writing any config

Open the raw files in a notebook and answer the questions below. Write the answers in
the README, because they determine the config.

Which column identifies a star? SDSS-V carries several ID systems. Pick the one that
is stable across the data release and unique per star, and check that one star never
appears under two IDs. If the IDs you want to browse by in the viewer differ from the
ones you fit on, put the crosswalk in the catalog table (step 4).

What is the time column, exactly? Find its format (JD, MJD, ...), its time scale
(TDB, TAI, UTC, ...), and whether it is already barycentric. hq converts format and
scale but does not apply barycentric corrections. If the exposure times are
topocentric, compute barycentric times in `make_input_table.py` with astropy
(`Time.light_travel_time` with the star's coordinates and the observatory location).
Also check whether the time is the exposure start or its midpoint.

What are the RV and uncertainty columns, and in what units? Check whether the RV is
already barycentric-corrected. Look at the uncertainty distribution, too:
exposure-level errors are often underestimated, which matters for false-positive
binaries.

Which rows are bad? Find the quality flags, failed-fit sentinels (e.g., `-9999`), and
the S/N cuts the survey recommends. These become `select_rows` in the model file.

How many instruments are there? SDSS-V APOGEE observes from APO and LCO, and hq
currently fits one instrument per run (no zero-point offsets). So either make one run
per telescope, or one run on a single telescope. If the survey documents the two as
on a common zero point, combining them in one run is an explicit assumption, so write
it down. Fitting offsets between instruments is a planned hq feature.

Which sources are SB2s? Sources with RVs for both stars go to a separate `sb2` run
(see step 8), because fitting a single-lined model to two stars' RVs gives spurious
orbits. Find out how the survey flags them, and how its component labels are
assigned. The two-component fit assumes that component 1 is the same star at every
epoch. If the labels are assigned per exposure (e.g., "the brighter line"), they can
swap between epochs, and the labels need to be made consistent before hq sees them.

Should you fit exposures or visits? Exposures within one visit are taken minutes
apart and can share systematics, so treating them as independent epochs can overstate
the evidence for short-period variability. Decide whether to fit exposures as given,
add a `Jitter` extension in `prior.py`, or set `period_min` well above the
intra-visit spacing. Also note that harv's cost scales with the number of epochs, so
exposure-level runs cost several times more than visit-level ones.


## 3. Write the input-table script

`scripts/make_input_table.py` turns the raw products into one flat table, with one
row per exposure, written as Parquet or FITS. It does only what hq cannot do: gather
files, compute barycentric times if needed, and normalize column names. It applies no
quality cuts. Those go in `select_rows`, so that they are covered by the run's
provenance (the model file's hash is recorded in every output, but the script that
made the input table is not).

Write the output to scratch space, not into the repository, and record its path and
the git commit of the script in the README.


## 4. Write the catalog script

`scripts/make_catalog.py` writes one row per star with whatever you want to plot in
the viewer: Gaia photometry and parallax, derived colors and absolute magnitudes (the
viewer plots columns as given, so compute e.g. `bp_rp` and `abs_g` here), stellar
parameters, targeting flags, and any ID crosswalk.


## 5. Initialize a test run

```bash
uv run hq init runs/test-1pct --kind rv
```

In `runs/test-1pct/hq.toml`, point `[data]` at the input table (with an absolute
path), set `source_id`, `time`, `time_format`, `time_scale`, `rv`, `rv_err`, and
`rv_unit` from step 2, point `[catalog]` at the catalog, and start small:

```toml
[prior_cache]
n_samples = 1_000_000

[rejection]
top_k = 256
```

In `runs/test-1pct/prior.py`, set up `make_setup()` and `select_rows()`. For the
prior, choose `period_min` (above the intra-visit spacing, from step 2), `period_max`
(several times the longest baseline), `sigma_K0`, and `sigma_v0` (wide enough to
cover the survey's systemic velocities, since harv does no centering). Add `Jitter`
if you decided to in step 2. `select_rows(table)` gets the quality cuts from step 2
and, for the test run only, a deterministic subset of the sources:

```python
import numpy as np

import harv_hq


def select_rows(table):
    good = table["snr"] > 20  # your quality cuts
    # Hash each source once, not each exposure.
    ids = np.unique(table["source_id"])
    kept = ids[[harv_hq.stable_hash(sid) % 100 == 0 for sid in ids]]
    return good & np.isin(table["source_id"], kept)
```


## 6. Run the test locally and tune

```bash
cd runs/test-1pct
uv run hq prepare
uv run hq prior-cache
uv run hq run --workers 8
uv run hq status
uv run hq summarize
uv run hq serve          # http://127.0.0.1:8000
```

Here is what we look at before scaling up. `hq status` shows any `failed` sources;
read their tracebacks in the viewer and fix the cause (usually units, or a bad row
that slipped past `select_rows`). The distributions of `ln_Z_int_ess` and
`well_resolved` show whether the prior library is big enough: if most sources with
clear variability are under-resolved, the library is too small or the period range
too wide. Values of `weight_captured` well below 1 mean `top_k` is too small. We
also open a few known binaries and a few known constant stars in the viewer, to check
that the orbits and the null cases look right. Finally, the summary's per-source
`wall_time_s`, multiplied by the full source count, tells you how large the full run
will be.

After changing `hq.toml` or `prior.py`, rerun the affected stages with `--overwrite`
(see {doc}`workflow` for which ones). hq refuses to mix outputs built from old and new
inputs, so it will tell you if you miss one.


## 7. Full run on a cluster

Copy the tuned config to `runs/full/`, remove the subset from `select_rows`, and set
the full `n_samples`. The cache holds about `n_samples x n_params x 8` bytes (e.g.,
about 5 GB for $10^8$ samples with six stored parameters), so put it on fast scratch.
Run `prepare` and `prior-cache` once, then the rejection stage with either a job
array or MPI:

```bash
# SLURM job array: 64 independent shards
#SBATCH --array=0-63
uv run hq run --run-dir runs/full --shard ${SLURM_ARRAY_TASK_ID}/64

# or MPI across nodes (compacts automatically at the end)
srun uv run hq run --run-dir runs/full --mpi
```

If jobs time out or nodes fail, resubmit the same command: finished sources are
skipped, however the work was split before. A job array leaves many small result
parts, so merge them once the whole array has finished, for example with a dependent
job:

```bash
sbatch --dependency=afterok:<array job id> --wrap "uv run hq compact --run-dir runs/full"
```

Then summarize, run the MCMC follow-up the same way, and summarize again:

```bash
uv run hq summarize --run-dir runs/full
uv run hq mcmc --run-dir runs/full --shard ${SLURM_ARRAY_TASK_ID}/16   # array job
uv run hq compact --run-dir runs/full --stage mcmc                     # after the array
uv run hq summarize --run-dir runs/full
```


## 8. Several runs from one input table

A project often needs more than one run on the same exposures: one run per telescope,
or, for SDSS-V, a separate run for the double-lined (SB2) sources. Each run is its own
directory under `runs/`, with its own `hq.toml` and `prior.py`, and every run's
`[data] file` points at the same input table. Nothing else is shared: each run
prepares its own subset and builds its own prior cache.

To split the sources between runs, write their IDs to a file (one per line, or a
one-column table) and list it in `[prepare]`. One run includes the listed sources and
the other excludes them, so every source lands in exactly one run:

```toml
# runs/sb2/hq.toml
[prepare]
include_ids = "/scratch/sdss5/sb2-ids.txt"

# runs/full/hq.toml
[prepare]
exclude_ids = "/scratch/sdss5/sb2-ids.txt"
```

`hq prepare` logs how many listed IDs are not in the input table, which catches a list
built from a different data release. A split by a column of the table (e.g., by
telescope) needs no list: put it in each run's `select_rows`.

The SB2 run itself starts from its own template:

```bash
uv run hq init runs/sb2 --kind sb2
```

Its model file fits each source with `JointModel.for_sb2`, which shares the orbit and
the systemic velocity between the two stars and gives each its own semi-amplitude
(the secondary's is negative). The input table needs one row per star per
measurement, with a column saying which star the RV belongs to. `[data] component`
names that column, and `[data] components` maps its labels to the two stars:

```toml
[data]
component = "component"
components = { primary = 1, secondary = 2 }
```

If the survey delivers both stars' RVs on one row, reshape the table in
`make_input_table.py`. Give the SB1 rows in the same file the primary's label, and
both runs can read it:

```python
import numpy as np
from astropy.table import Table, vstack

paired = Table.read("sb2-exposures.fits")  # rv_1, rv_err_1, rv_2, rv_err_2 per row
stars = []
for label in (1, 2):
    star = paired["source_id", "time", f"rv_{label}", f"rv_err_{label}"]
    star.rename_columns([f"rv_{label}", f"rv_err_{label}"], ["rv", "rv_err"])
    star["component"] = label
    stars.append(star[np.isfinite(star["rv"])])  # an unmeasured star has no row
long = vstack(stars)
```

The summary of an `sb2` run has columns for each star's semi-amplitude
(`primary.rv_semiamp_p50`, `secondary.rv_semiamp_p50`, ...) and a derived
`mass_ratio` (`|K_primary| / |K_secondary|`, which is `M_secondary / M_primary`), and
the viewer draws both stars on one RV panel.


## 9. Explore and keep track

To browse a run on the cluster, serve it there and tunnel the port from your laptop:

```bash
uv run hq serve --run-dir runs/full                    # on the cluster
ssh -L 8000:localhost:8000 <cluster-login-node>        # on your laptop
```

When a run is final, commit `runs/full/hq.toml` and `prior.py` and tag the
repository. Every output records a sha256 of each `hq.toml` table it depends on, a
sha256 of the whole model file, and the versions of hq, harv, and JAX, so any result
file can be traced back to the configuration that produced it.
