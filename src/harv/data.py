"""Data classes for representing time series data.

TODO: we could add support for metadata for the data type classes below.
"""

__all__ = [
    "AbstractAstrometryData",
    "GaiaAstrometryData",
    # "AbsoluteAstrometryData",
    "RadialVelocityData",
    "SourceData",
    "DatasetType",
    "InputData",
    "stack_datasets",
    "build_indicator_matrix",
]

from collections.abc import Iterator
from dataclasses import KW_ONLY, fields
from typing import ClassVar, TypeVar

import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import AbstractQuantity, Quantity, ustrip
from unxt.quantity import AllowValue

from .custom_types import NAngle, NFloatArray, NTime, NVelocity


class AbstractData(eqx.Module):
    """Abstract base class for observational data time series."""

    _obs_name: eqx.AbstractClassVar[str]
    _err_name: eqx.AbstractClassVar[str]

    # Note: time is defined in subclasses as a field, not as an abstract property
    # to avoid dataclass field ordering issues with equinox

    time: NTime
    """Barycentric TCB times."""

    _: KW_ONLY

    t_ref: NTime | None = None
    """Reference epoch. If None, uses mean observation time."""

    def __check_init__(self) -> None:
        """Compute t_ref from mean time if not provided."""
        if self.t_ref is None:
            object.__setattr__(
                self,
                "t_ref",
                jnp.mean(self.time),
            )

    @property
    def n_times(self) -> int:
        """Number of times / epochs / observations."""
        return len(self.time)

    def _get_obs(self) -> AbstractQuantity:
        """Get the observed values (e.g., positions, RVs)."""
        return getattr(self, self._obs_name)

    def _get_obs_err(self) -> AbstractQuantity:
        """Get the observed uncertainties."""
        return getattr(self, self._err_name)


class AbstractAstrometryData(AbstractData):
    """Abstract base class for astrometric data."""


class GaiaAstrometryData(AbstractAstrometryData):
    """Gaia epoch astrometry (along-scan measurements)."""

    _obs_name: ClassVar[str] = "al_position"
    _err_name: ClassVar[str] = "al_position_err"

    al_position: NAngle
    """Along-scan position."""

    al_position_err: NAngle
    """Along-scan uncertainty."""

    scan_angle: NAngle
    """Per-CCD scan angle."""

    parallax_factor: NFloatArray
    """AL parallax factors."""


# TODO: currently not supported, so commenting out
# class AbsoluteAstrometryData(AbstractAstrometryData):
#     """Traditional absolute astrometry (RA/Dec measurements)."""

#     time: NTime
#     """Observation times."""

#     ra: NAngle
#     """Right ascension."""

#     dec: NAngle
#     """Declination."""

#     ra_err: NAngle
#     """RA uncertainty."""

#     dec_err: NAngle
#     """Dec uncertainty."""


class RadialVelocityData(AbstractData):
    """Radial velocity measurements."""

    _obs_name: ClassVar[str] = "rv"
    _err_name: ClassVar[str] = "rv_err"

    rv: NVelocity
    """Radial velocities."""

    rv_err: NVelocity
    """Radial velocity uncertainties."""


# Type alias for all supported data types
DatasetType = AbstractAstrometryData | RadialVelocityData
_DT = TypeVar("_DT", bound=DatasetType)


class SourceData(eqx.Module):
    """Container for multiple named datasets for a single source.

    Accepts arbitrary named datasets via keyword arguments. Names are
    user-defined and can be anything (e.g., 'gaia', 'keck_rv', 'hst_imaging').
    """

    _datasets: dict[str, DatasetType]

    def __init__(self, **datasets: DatasetType) -> None:
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        for name, ds in datasets.items():
            if not isinstance(ds, AbstractData):
                raise TypeError(
                    f"Dataset '{name}' must be AbstractAstrometryData or "
                    f"RadialVelocityData, got {type(ds).__name__}"
                )
        object.__setattr__(self, "_datasets", datasets)

    # Dict-like interface:
    def __getitem__(self, name: str) -> DatasetType:
        return self._datasets[name]

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __len__(self) -> int:
        return len(self._datasets)

    def keys(self) -> Iterator[str]:
        """Get dataset names."""
        return iter(self._datasets.keys())

    def values(self) -> Iterator[DatasetType]:
        """Get dataset values."""
        return iter(self._datasets.values())

    def items(self) -> Iterator[tuple[str, DatasetType]]:
        """Get dataset (name, value) pairs."""
        return iter(self._datasets.items())

    # Other methods:
    def get_datasets_by_type(self, dtype: type[_DT]) -> dict[str, _DT]:
        """Get all datasets of a specific type."""
        return {k: v for k, v in self._datasets.items() if isinstance(v, dtype)}

    def _n_astrometry(self) -> int:
        """Number of astrometric datasets."""
        return len(self.get_datasets_by_type(AbstractAstrometryData))

    def _n_rv(self) -> int:
        """Number of radial velocity datasets."""
        return len(self.get_datasets_by_type(RadialVelocityData))


# Type alias for any top-level input accepted by the sampler and likelihoods.
# Use this instead of AbstractData in signatures that also accept SourceData.
InputData = AbstractData | SourceData


def stack_datasets(
    datasets: dict[str, AbstractData],
) -> AbstractData:
    """Concatenate multiple datasets in dict order into a single one.

    Parameters
    ----------
    datasets : dict[str, AbstractData]
        Ordered mapping of instrument name -> dataset.  Dict order determines
        the row order in the stacked output; it must match the order used when
        building the indicator matrix (see :func:`build_rv_indicator_matrix`).

    Returns
    -------
    data
        Single dataset containing all observations stacked in dict order.
    """
    # first make sure that all datasets have the same type:
    dset_types = {type(ds) for ds in datasets.values()}
    if len(dset_types) != 1:
        msg = f"All datasets must have the same type to stack (got: {dset_types})"
        raise ValueError(msg)

    # the reference dataset, which we use to get the field names and units for the
    # output dataset
    ref = next(iter(datasets.values()))

    # units for each field:
    all_units = {
        field.name: str(getattr(ref, field.name).unit)
        if hasattr(getattr(ref, field.name), "unit")
        else ""
        for field in fields(ref)
    }

    # NOTE: we assume that all datasets have the same fields and units, and we assume
    # that all fields are present in all datasets and are array-valued (so they can be
    # concatenated). That's true for current datasets, but we might want to relax these
    # assumptions in the future.
    all_data: dict[str, AbstractQuantity] = {
        name: Quantity(
            jnp.concatenate(
                [
                    ustrip(AllowValue, unit, getattr(ds, name))
                    for ds in datasets.values()
                ]
            ),
            unit,
        )
        for name, unit in all_units.items()
    }
    return type(ref)(**all_data)


def build_indicator_matrix(
    datasets: dict[str, AbstractData], reference: str
) -> tuple[AbstractData, jax.Array | None, tuple[str, ...] | None]:
    """Build indicator matrix for multi-survey data of the same type.

    Parameters
    ----------
    datasets : dict[str, AbstractData]
        Ordered mapping of instrument name -> dataset.  Dict order must match
        the order used when stacking (see :func:`stack_datasets`).
    reference : str
        Name of the reference instrument (its observations get no offset
        column).

    Returns
    -------
    indicator_matrix : jax.Array
        Shape ``(n_obs_total, n_non_ref)``.  ``indicator[i, j] = 1`` when
        observation ``i`` belongs to non-reference instrument ``j``.
    instrument_names : tuple[str, ...]
        Names of the non-reference instruments, in column order.

    """
    if reference not in datasets:
        msg = f"Reference instrument {reference!r} not in {list(datasets)}"
        raise ValueError(msg)

    non_ref_names = [k for k in datasets if k != reference]
    n_non_ref = len(non_ref_names)
    rows = []
    for name, ds in datasets.items():
        n_obs = len(ds.time)
        row = jnp.zeros((n_obs, n_non_ref))
        if name != reference:
            j = non_ref_names.index(name)
            row = row.at[:, j].set(1.0)
        rows.append(row)
    return (
        stack_datasets(datasets),
        jnp.concatenate(rows, axis=0),
        tuple(non_ref_names),
    )
