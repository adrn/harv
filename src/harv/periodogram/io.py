"""HDF5 persistence for interim period priors.

TODO: to review

The prior spec (knots + log-densities + unit) is written as a small HDF5
group, designed to coexist with ``Samples.to_hdf5`` output in the same file:
``Samples.from_hdf5`` reads only its own groups, so a per-source posterior
file can carry its interim period prior alongside the samples.
"""

__all__ = ("load_period_prior", "save_period_prior")

import os

import h5py
import numpy as np
import quaxed.numpy as jnp

from harv.distributions import QuantityDistribution
from harv.periodogram.distribution import LogGridDensity

_FORMAT_VERSION = 1
_DEFAULT_GROUP = "interim_period_prior"


def _write_group(
    parent: h5py.Group,
    density: LogGridDensity,
    unit: str,
    group: str,
    metadata: dict[str, int | float | str] | None,
) -> None:
    if group in parent:
        del parent[group]
    g = parent.create_group(group)
    g.create_dataset("ln_grid", data=np.asarray(density.ln_grid))
    g.create_dataset("log_density", data=np.asarray(density.log_density))
    g.attrs["unit"] = unit
    g.attrs["format_version"] = _FORMAT_VERSION
    for key, value in (metadata or {}).items():
        g.attrs[key] = value


def save_period_prior(
    file: str | os.PathLike | h5py.Group,
    prior: QuantityDistribution,
    *,
    group: str = _DEFAULT_GROUP,
    metadata: dict[str, int | float | str] | None = None,
) -> None:
    """Write an interim period prior spec to an HDF5 file or group.

    ``prior`` must be a `~harv.distributions.QuantityDistribution` wrapping a
    :class:`~harv.periodogram.LogGridDensity` (as returned by
    :func:`~harv.periodogram.tempered_period_prior` /
    :func:`~harv.periodogram.peak_period_prior`). An existing group of the
    same name is overwritten. Scalar ``metadata`` entries (e.g. builder
    provenance like ``{"builder": "tempered", "beta": 1.0, "floor": 0.1}``)
    are stored as group attributes.
    """
    density = prior.distribution if isinstance(prior, QuantityDistribution) else None
    if not isinstance(density, LogGridDensity):
        raise TypeError(
            "prior must be a QuantityDistribution wrapping a LogGridDensity; "
            f"got {type(prior).__name__}"
        )
    unit = prior.unit
    if not isinstance(unit, str):
        raise TypeError("prior must have a single scalar unit")

    if isinstance(file, h5py.Group):
        _write_group(file, density, unit, group, metadata)
    else:
        with h5py.File(file, "a") as f:
            _write_group(f, density, unit, group, metadata)


def _read_group(parent: h5py.Group, group: str) -> QuantityDistribution:
    if group not in parent:
        msg = f"No interim period prior group {group!r} in {parent.file.filename}"
        raise KeyError(msg)
    g = parent[group]
    return QuantityDistribution(
        LogGridDensity(
            jnp.asarray(np.asarray(g["ln_grid"])),
            jnp.asarray(np.asarray(g["log_density"])),
        ),
        str(g.attrs["unit"]),
    )


def load_period_prior(
    file: str | os.PathLike | h5py.Group,
    *,
    group: str = _DEFAULT_GROUP,
) -> QuantityDistribution:
    """Load an interim period prior previously written by `save_period_prior`.

    The round trip is exact: the stored arrays are the ``LogGridDensity``
    constructor arguments.
    """
    if isinstance(file, h5py.Group):
        return _read_group(file, group)
    with h5py.File(file, "r") as f:
        return _read_group(f, group)
