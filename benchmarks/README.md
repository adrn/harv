# benchmarks

This directory contains the code and results for scaling benchmarks of `harv` functionality.
Currently this covers `harv.samplers.RejectionSampler` (model parameterizations, epoch
counts, prior library sizes, and `batch_size`) and `harv.periodogram.periodogram`
(models, prior modes, epoch counts, grid sizes, and `n_terms`).

- `grid.py` defines the grid of benchmark cells and builds the data, priors, and
  models for each one.
- `test_rejection_scaling.py` is the benchmark itself, one `pytest-benchmark`
  test parametrized over that grid.
- `report.py` merges `results/rejection/*.json` into `docs/benchmarks/rejection.md` and its figures.
- `periodogram_grid.py`, `test_periodogram_scaling.py`, and `report_periodogram.py`
  are the same three pieces for `harv.periodogram.periodogram`, rendered into
  `docs/benchmarks/periodograms.md`.
- `results/rejection/` and `results/periodogram/` hold the committed JSON from the
  runs those pages are built from.

These are deliberately not part of the test suite: they live outside
`testpaths`, require `--bench` to run, and depend on the `bench` group that
is not installed in CI.
The results pages are committed rather than rebuilt so we can compare CPU and GPU
performance.

For the measurements themselves see `docs/benchmarks/`, for how to reproduce them
see `docs/benchmarks/running-the-benchmarks.md`, and for how to use the numbers when
running over a survey see `docs/at-scale.md`.

Most of the code in this directory was written by Claude Opus 5.

TODO: `report_periodogram.py` and `periodogram_grid.py` copy parts of `report.py` and
`grid.py` (result loading, deduplication, plotting, the compile table). Share that
code before adding a third benchmark.
