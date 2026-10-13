# hq: running harv on surveys

:::{toctree}
:maxdepth: 1
:hidden:

getting-started
workflow
project-setup
:::

`hq` (the `harv-hq` package) runs harv over a whole survey: a table with one row per
observation for thousands to millions of sources, and optionally a catalog with one
row per source. It organizes a run as a directory holding a configuration file
(`hq.toml`) and a model file (`prior.py`), and it provides a command-line tool, `hq`,
whose subcommands are the stages of a run. The stages write files that the later
stages read (Parquet tables, and one HDF5 file for the shared prior library), the
per-source stages pick up where they stopped after a crash or a timed-out job, and
every output records what it was built from so that stale results are refused rather
than mixed in. A local web viewer plots the catalog and each source's data and
posterior samples.

hq does no modeling of its own. Every source is fit with harv's
{py:class}`~harv.samplers.RejectionSampler`, and the sources the rejection sampler
cannot resolve can be followed up with harv's {py:class}`~harv.samplers.NumpyroSampler`.
So if you have a handful of sources, you probably want to use harv directly (see
{doc}`../getting-started`). The guidance in {doc}`../at-scale` applies to hq runs too.

- {doc}`getting-started` runs a small simulated survey through every stage on a laptop.
- {doc}`workflow` describes every stage in order, how to rerun them, and which changes
  to the inputs invalidate which outputs.
- {doc}`project-setup` walks through setting up a real survey run, using SDSS-V
  exposure-level RVs as the example.

The full design is in the `harv-hq` specification
([`packages/hq/docs/spec.md`](https://github.com/adrn/harv/blob/main/packages/hq/docs/spec.md)).

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install harv-hq
```

:::

:::{tab-item} uv

```bash
uv add harv-hq
```

:::

::::

`harv-hq` is released together with harv, and each release requires the harv release
with the same version. To run stages over MPI, install the `mpi` extra
(`pip install "harv-hq[mpi]"`), building `mpi4py` against your cluster's MPI library.
