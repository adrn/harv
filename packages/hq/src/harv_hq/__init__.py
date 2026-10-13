"""harv-hq: pipelines, catalog preparation, and visualization for harv.

The public API is specified in ``packages/hq/docs/spec.md``.
"""

__all__ = (
    "CompactReport",
    "Config",
    "ConfigError",
    "ProvenanceError",
    "Run",
    "SourceResult",
    "create_app",
    "init_run",
    "read_source",
    "read_sources",
    "stable_hash",
)

from typing import Any

import os

import jax

# harv's likelihoods need float64. harv enables it on import with a warning
# meant for interactive users; hq always wants it, so it enables it first,
# before any module imports harv, and hq commands stay quiet. Like harv, it
# leaves an explicit JAX_ENABLE_X64 (e.g. =0) to the user.
if "JAX_ENABLE_X64" not in os.environ:
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from harv_hq._version import __version__
from harv_hq.compact import CompactReport
from harv_hq.config import Config, ConfigError
from harv_hq.ids import stable_hash
from harv_hq.init import init_run
from harv_hq.prepare import read_source, read_sources
from harv_hq.provenance import ProvenanceError
from harv_hq.run import Run, SourceResult


def __getattr__(name: str) -> Any:
    # The web stack (fastapi, starlette, pydantic, jinja2) is imported only
    # when the viewer is used, so pipeline processes and pool workers do not
    # pay for it.
    if name == "create_app":
        from harv_hq.web import create_app  # noqa: PLC0415

        return create_app
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
