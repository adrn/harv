"""Fixtures and options for the rejection-sampler benchmarks.

Nothing here runs unless ``--bench`` is passed. See ``docs/running-benchmarks.md``.
"""

from __future__ import annotations

# x64 FIRST, before anything imports harv (and therefore JAX). harv deliberately
# does not enable it -- docs/sharp-bits.md makes it the user's job -- and every
# tutorial turns it on. float32 changes the sampler's *arithmetic*, not just its
# precision, so a float32 timing would not describe how anyone runs harv.
# Safe here: the root conftest.py only installs import hooks and creates no arrays.
import jax

jax.config.update("jax_enable_x64", True)

import functools  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import jax.random as jr  # noqa: E402
import pytest  # noqa: E402
from grid import Cell, build_prior_and_model, enumerate_cells  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("harv benchmarks")
    group.addoption(
        "--bench",
        action="store_true",
        default=False,
        help="Run the rejection-sampler benchmarks. Without this they all skip, "
        "so a stray `pytest benchmarks/` cannot start a multi-hour run.",
    )
    group.addoption(
        "--bench-full",
        action="store_true",
        default=False,
        help="Full cartesian product instead of the default star design (~10x longer).",
    )
    group.addoption(
        "--bench-smoke",
        action="store_true",
        default=False,
        help="Two tiny cells that exercise the whole pipeline in under a minute.",
    )
    group.addoption(
        "--bench-rounds",
        type=int,
        default=5,
        help="Timed rounds per cell, after one warmup round (default: 5).",
    )
    group.addoption(
        "--bench-cache-dir",
        default=None,
        help="Directory for HDF5 prior caches. Default: a pytest temp dir, "
        "rebuilt each run. Point it somewhere persistent to reuse them.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--bench"):
        return
    skip = pytest.mark.skip(
        reason="benchmarks require --bench (see docs/running-benchmarks.md)"
    )
    for item in items:
        if "benchmarks/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(skip)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize over the grid, chosen by the --bench-* flags."""
    if "cell" not in metafunc.fixturenames:
        return
    cells = enumerate_cells(
        full=metafunc.config.getoption("--bench-full"),
        smoke=metafunc.config.getoption("--bench-smoke"),
    )
    metafunc.parametrize("cell", cells, ids=[c.ident for c in cells])


@pytest.fixture(scope="session")
def rounds(request: pytest.FixtureRequest) -> int:
    n: int = request.config.getoption("--bench-rounds")
    return 1 if request.config.getoption("--bench-smoke") else n


@pytest.fixture(scope="session")
def grid_mode(request: pytest.FixtureRequest) -> str:
    """Which grid produced this run.

    Recorded into every result so report.py can reconstruct curve membership
    exactly. Inferring it from the curve names in the data does not work: curves
    share baseline cells, and a smoke run legitimately reuses a real curve name.
    """
    if request.config.getoption("--bench-smoke"):
        return "smoke"
    if request.config.getoption("--bench-full"):
        return "full"
    return "star"


@pytest.fixture(scope="session")
def device_info() -> dict[str, Any]:
    """Device and version metadata.

    pytest-benchmark's ``machine_info`` records the CPU and Python build but knows
    nothing about accelerators, so the GPU identity has to come from here or the
    CPU and GPU result files are indistinguishable in the report.
    """
    import harv

    device = jax.devices()[0]
    return {
        "device_platform": device.platform,
        "device_kind": device.device_kind,
        "device_count": jax.device_count(),
        "jax_version": jax.__version__,
        "harv_version": getattr(harv, "__version__", "unknown"),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "typecheck_hooks": not os.environ.get("HARV_NO_TYPECHECK"),
    }


@pytest.fixture(scope="session")
def cache_dir(request: pytest.FixtureRequest, tmp_path_factory: Any) -> Path:
    opt = request.config.getoption("--bench-cache-dir")
    if opt:
        path = Path(opt).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("prior-caches")


@pytest.fixture(scope="session")
def max_samples_by_parameterization(request: pytest.FixtureRequest) -> dict[str, int]:
    """Largest library each parameterization needs, so none is built oversized."""
    cells = enumerate_cells(
        full=request.config.getoption("--bench-full"),
        smoke=request.config.getoption("--bench-smoke"),
    )
    out: dict[str, int] = {}
    for cell in cells:
        out[cell.parameterization] = max(
            out.get(cell.parameterization, 0), cell.n_prior_samples
        )
    return out


# `maxsize=1` plus the parameterization-sorted cell list means exactly one prior
# library is resident at a time. At M = 1e7 a Gaia library is ~500 MB, so holding
# all six would be gigabytes for no reason.
@functools.lru_cache(maxsize=1)
def _build_memory_cache(parameterization: str, n_samples: int) -> Any:
    prior, model = build_prior_and_model(parameterization)
    return prior.sample(jr.key(0), n_samples, model=model)


@functools.cache
def _build_hdf5_cache(parameterization: str, n_samples: int, out_dir: str) -> str:
    from harv.samplers import make_prior_cache

    path = Path(out_dir) / f"{parameterization}-{n_samples}.h5"
    if path.exists():
        return str(path)
    prior, model = build_prior_and_model(parameterization)
    make_prior_cache(
        prior,
        model,
        n_samples,
        path,
        key=jr.key(0),
        batch_size=min(100_000, n_samples),
    )
    return str(path)


@pytest.fixture
def prior_cache(
    cell: Cell,
    cache_dir: Path,
    max_samples_by_parameterization: dict[str, int],
) -> Any:
    """The prior library for this cell, in whichever backend the cell names.

    The in-memory library is built once per parameterization at its largest
    required size and sliced down -- ``Samples`` slices all arrays along the
    leading axis (docs/spec.md, "Dict-style and index access"), so a slice is a
    view-shaped copy rather than a fresh round of prior sampling.
    """
    if cell.backend == "memory":
        n_max = max_samples_by_parameterization[cell.parameterization]
        full = _build_memory_cache(cell.parameterization, n_max)
        return full if cell.n_prior_samples == n_max else full[: cell.n_prior_samples]
    return _build_hdf5_cache(
        cell.parameterization, cell.n_prior_samples, str(cache_dir)
    )
