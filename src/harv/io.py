"""Serialization helpers for harv sampler objects.

Provides :func:`save_sampler` and :func:`load_sampler` for persisting a
configured sampler (prior + extensions) to disk so that the same setup can be
reloaded without re-specifying all prior parameters.

The format is a standard Python pickle file.  Equinox modules (``eqx.Module``
subclasses) are picklable, so all prior distributions, extension objects, and
static fields round-trip exactly.

Examples
--------
Save and reload a :class:`~harv.samplers.RejectionSampler`:

>>> import harv
>>> sampler = harv.RejectionSampler(prior)  # doctest: +SKIP
>>> harv.save_sampler("sampler.pkl", sampler)  # doctest: +SKIP
>>> sampler2 = harv.load_sampler("sampler.pkl")  # doctest: +SKIP
"""

__all__ = ("load_sampler", "save_sampler")

import pickle
from pathlib import Path
from typing import Any


def save_sampler(path: str | Path, sampler: Any) -> None:
    """Save a configured sampler to a pickle file.

    All prior distributions, extensions, and static configuration are
    preserved.  The file can be reloaded with :func:`load_sampler` without
    needing to re-specify any prior parameters.

    Parameters
    ----------
    path : str or Path
        Output file path.  By convention use a ``.pkl`` extension.
    sampler : RejectionSampler or NumpyroSampler
        The sampler to save.  Any picklable object is accepted.

    Examples
    --------
    >>> import harv
    >>> save_sampler("sampler.pkl", sampler)  # doctest: +SKIP
    """
    with Path(path).open("wb") as f:
        pickle.dump(sampler, f)


def load_sampler(path: str | Path) -> Any:
    """Load a sampler from a pickle file written by :func:`save_sampler`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pkl`` checkpoint file.

    Returns
    -------
    sampler : RejectionSampler or NumpyroSampler
        Restored sampler with all prior distributions and extensions intact.

    Examples
    --------
    >>> import harv
    >>> sampler = load_sampler("sampler.pkl")  # doctest: +SKIP
    """
    with Path(path).open("rb") as f:
        return pickle.load(f)  # noqa: S301
