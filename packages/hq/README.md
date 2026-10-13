# harv-hq

Companion package to [harv](https://github.com/adrn/harv) for running it on whole
surveys: thousands to millions of sources, each with a handful of epochs. hq prepares
a survey's observation table, runs harv's rejection sampler on every source (on one
machine, a job array, or MPI), follows up under-resolved sources with MCMC, collects a
summary table, and serves a local web viewer for exploring the results. The
per-source stages (rejection and MCMC) pick up where they stopped after a crash or a
timed-out job, and every output records what it was built from, so stale results are
refused rather than mixed in.

```bash
pip install harv-hq
hq init my-run --kind rv
```

Documentation: [getting started](https://harvey.readthedocs.io/en/latest/hq/getting-started.html)
and [the workflow tutorial](https://harvey.readthedocs.io/en/latest/hq/workflow.html).

hq lives in the harv repository as a uv workspace member and is released in lockstep
with harv: each `harv-hq` release requires the `harv` release with the same version.
