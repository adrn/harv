"""Dataset containers for multi-component and multi-instrument data."""

__all__ = (
    "DatasetType",
    "InputData",
    "SourceData",
    "SystemData",
)

from collections.abc import Iterator
from typing import TypeVar

import equinox as eqx
import quaxed.numpy as jnp
from unxt import AbstractQuantity

from harv.custom_types import NTime

from .datasets import (
    AbstractAstrometryData,
    AbstractData,
    DatasetType,
    RVData,
)

_DT = TypeVar("_DT", bound=DatasetType)


class AbstractDatasetContainer(eqx.Module):
    """Base class providing a dict-like interface over named datasets.

    Subclasses (``SystemData``, ``SourceData``) share this common interface
    but carry different semantic meaning.
    """

    _datasets: dict[str, DatasetType]

    def __getitem__(self, name: str) -> DatasetType:
        return self._datasets[name]

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __len__(self) -> int:
        return len(self._datasets)

    def keys(self) -> Iterator[str]:
        """Dataset/component names."""
        return iter(self._datasets.keys())

    def values(self) -> Iterator[DatasetType]:
        """Dataset/component values."""
        return iter(self._datasets.values())

    def items(self) -> Iterator[tuple[str, DatasetType]]:
        """(name, dataset) pairs."""
        return iter(self._datasets.items())

    def get_datasets_by_type(self, dtype: type[_DT]) -> dict[str, _DT]:
        """Get all datasets/components of a specific data type."""
        return {k: v for k, v in self._datasets.items() if isinstance(v, dtype)}


class SystemData(AbstractDatasetContainer):
    """Container for a multi-component system.

    Each named component holds a :class:`DatasetType` (e.g. :class:`RVData`,
    :class:`GaiaAstrometryData`) representing observations of a distinct
    physical body or photocenter in a gravitationally bound system.

    Unlike :class:`SourceData` (multiple instruments observing the *same*
    source), ``SystemData`` represents *resolved components* of a multi-body
    system (e.g. primary and secondary in an SB2).  The two containers may
    eventually be composed: ``SystemData`` for per-component spectroscopy,
    with a separate ``GaiaAstrometryData`` (or ``SourceData``) for the
    unresolved photocenter astrometry.

    Parameters are passed as keyword arguments where the key is the component
    name and the value is the dataset.

    Examples
    --------
    >>> data = SystemData(
    ...     primary=RVData(...),
    ...     secondary=RVData(...),
    ... )
    >>> data["primary"]
    RVData(...)
    """

    def __init__(self, **datasets: DatasetType) -> None:
        if not datasets:
            raise ValueError("At least one component must be provided")
        for name, ds in datasets.items():
            if not isinstance(ds, AbstractData):
                raise TypeError(
                    f"Component '{name}' must be an AbstractData subclass "
                    f"(RVData, GaiaAstrometryData, ...), got {type(ds).__name__}"
                )
        object.__setattr__(self, "_datasets", datasets)

    @property
    def t_ref(self) -> NTime:
        """Reference epoch from the first component."""
        return next(iter(self._datasets.values())).t_ref

    def _get_obs(self) -> AbstractQuantity:
        """Concatenated observations across all components (key order)."""
        return jnp.concatenate([ds._get_obs() for ds in self._datasets.values()])

    def _get_obs_err(self) -> AbstractQuantity:
        """Concatenated uncertainties across all components (key order)."""
        return jnp.concatenate([ds._get_obs_err() for ds in self._datasets.values()])


class SourceData(AbstractDatasetContainer):
    """Container for multiple named datasets for a single source.

    Accepts arbitrary named datasets via keyword arguments. Names are
    user-defined and can be anything (e.g., 'gaia', 'keck_rv', 'hst_imaging').
    """

    def __init__(self, **datasets: DatasetType) -> None:
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        for name, ds in datasets.items():
            if not isinstance(ds, AbstractData):
                raise TypeError(
                    f"Dataset '{name}' must be AbstractAstrometryData or "
                    f"RVData, got {type(ds).__name__}"
                )
        object.__setattr__(self, "_datasets", datasets)

    def _n_astrometry(self) -> int:
        """Number of astrometric datasets."""
        return len(self.get_datasets_by_type(AbstractAstrometryData))

    def _n_rv(self) -> int:
        """Number of radial velocity datasets."""
        return len(self.get_datasets_by_type(RVData))


# Type alias for any top-level input accepted by the sampler and likelihoods.
# Use this instead of AbstractData in signatures that also accept SourceData.
InputData = AbstractData | SourceData | SystemData
