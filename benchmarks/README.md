# benchmarks

This directory contains the code and results for scaling benchmarks of `harv` functionality.
Currently, this is mainly for `harv.samplers.RejectionSampler`, and we compare
benchmarks for different model parameterizations, epoch counts, prior library sizes, and
`batch_size`.

- `grid.py` defines the grid of benchmark cells and builds the data, priors, and
  models for each one.
- `test_rejection_scaling.py` is the benchmark itself, one `pytest-benchmark`
  test parametrized over that grid.
- `report.py` merges `results/*.json` into `docs/benchmarks.md` and its figures.
- `results/` holds the committed JSON from the runs that page is built from.

These are deliberately not part of the test suite: they live outside
`testpaths`, require `--bench` to run, and depend on the `bench` group that
is not installed in CI.
The results page is committed rather than rebuilt so we can compare CPU and GPU
performance.

For the measurements themselves see `docs/benchmarks.md`, for how to reproduce
them see `docs/running-benchmarks.md`, and for how to use the numbers when
running over a survey see `docs/at-scale.md`.

Most of the code in this directory was written by Claude Opus 5.

TODO: generalize the benchmark code so we can benchmark other functionality, like the
periodogram code.
