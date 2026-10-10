"""``hq init``: create a run directory from templates."""

__all__ = ("init_run",)

import os
from importlib import resources
from pathlib import Path

from harv_hq.config import KINDS


def init_run(run_dir: str | os.PathLike, *, kind: str) -> Path:
    """Create a run directory with a template ``hq.toml`` and ``prior.py``.

    Parameters
    ----------
    run_dir
        The directory to create. It may exist, but must be empty.
    kind
        The run kind, ``"rv"`` or ``"gaia_astrometry"``; chooses the
        templates.

    Returns
    -------
        The run directory.

    Raises
    ------
    ValueError
        If ``kind`` is not a known run kind.
    FileExistsError
        If ``run_dir`` exists and is not empty.

    Examples
    --------
    >>> import tempfile
    >>> run_dir = init_run(tempfile.mkdtemp(), kind="rv")
    >>> sorted(p.name for p in run_dir.iterdir())
    ['hq.toml', 'prior.py']
    """
    if kind not in KINDS:
        msg = f"kind must be one of {list(KINDS)}, got {kind!r}"
        raise ValueError(msg)
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        msg = f"{run_dir} is not empty; hq init needs a new or empty directory"
        raise FileExistsError(msg)
    run_dir.mkdir(parents=True, exist_ok=True)

    templates = resources.files("harv_hq") / "templates"
    for template, name in (
        (f"hq_{kind}.toml", "hq.toml"),
        (f"prior_{kind}.py", "prior.py"),
    ):
        (run_dir / name).write_text((templates / template).read_text())
    return run_dir
