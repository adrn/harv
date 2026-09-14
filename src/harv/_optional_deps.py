"""On-demand importers for optional dependencies.

harv's plotting and diagnostic layers need packages that are not required to
run a sampler: matplotlib and arviz (which itself imports matplotlib). Importing
either one is expensive, and on a cluster with node-local cache directories
every rank that imports matplotlib rebuilds its font list. Importing them here,
at the point of use, keeps ``import harv`` free of both so a sampling-only rank
never pays for them. See "Operational notes" in ``docs/at-scale.md``.

Each importer takes the name of the calling function, used only to write the
``ImportError`` message, and is private to the package.
"""

__all__ = ("get_arviz", "get_mpl")

from typing import Any


def get_mpl(func_name: str) -> tuple[Any, Any]:
    """Import matplotlib, returning ``(matplotlib, matplotlib.pyplot)``."""
    try:
        import matplotlib as mpl  # noqa: PLC0415  (optional dependency)
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as e:
        msg = f"matplotlib is required for {func_name}."
        raise ImportError(msg) from e
    return mpl, plt


def get_arviz(func_name: str) -> tuple[Any, Any]:
    """Import arviz, returning ``(arviz, arviz_base.labels.MapLabeller)``."""
    try:
        import arviz as az  # noqa: PLC0415  (optional dependency)
        from arviz_base.labels import MapLabeller  # noqa: PLC0415
    except ImportError as e:
        msg = f"arviz is required for {func_name}."
        raise ImportError(msg) from e
    return az, MapLabeller
