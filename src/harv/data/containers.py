"""Dataset containers for multi-component and multi-instrument data."""

__all__ = (
    "DatasetType",
    "InputData",
    "SourceData",
    "SystemData",
)

from collections.abc import Iterator
from typing import TypeVar, cast

import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import AbstractQuantity

from harv.custom_types import NTime
from harv.data.datasets import (
    AbstractAstrometryData,
    AbstractData,
    DatasetType,
    RVData,
)
from harv.data.helpers import (
    _synchronize_t_refs,
    build_indicator_matrix,
    stack_datasets,
)

_DT = TypeVar("_DT", bound=DatasetType)


class AbstractDatasetContainer(eqx.Module):
    """Base class providing a dict-like interface over named datasets.

    Subclasses (SystemData, SourceData) share this common interface
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

    def get_datasets_by_type(self, data_type: type[_DT]) -> dict[str, _DT]:
        """Get all datasets/components of a specific data type."""
        return {k: v for k, v in self._datasets.items() if isinstance(v, data_type)}

    def _require_datasets_by_type(self, data_type: type[_DT]) -> dict[str, _DT]:
        datasets = self.get_datasets_by_type(data_type)
        if not datasets:
            msg = f"No datasets of type {data_type.__name__} found"
            raise ValueError(msg)
        return datasets

    def stacked_by_type(self, data_type: type[_DT]) -> _DT:
        """Stack all datasets of the requested type."""
        return cast("_DT", stack_datasets(self._require_datasets_by_type(data_type)))

    def indicator_data_by_type(
        self,
        data_type: type[_DT],
        reference: str,
    ) -> tuple[_DT, jax.Array | None, tuple[str, ...] | None]:
        """Return stacked data and indicator flags for one dataset type."""
        datasets = self._require_datasets_by_type(data_type)
        stacked, indicator, names = build_indicator_matrix(datasets, reference)
        return cast("_DT", stacked), indicator, names


class SystemData(AbstractDatasetContainer):
    """Container for a multi-component system.

    Each named component holds the same concrete data class representing
    observations of a distinct physical body or photocenter in a
    gravitationally bound system.
    """

    _dataset_type: type[AbstractData] = eqx.field(static=True)

    def __init__(self, **datasets: DatasetType) -> None:
        if not datasets:
            raise ValueError("At least one component must be provided")

        type_map: dict[str, str] = {}
        for name, ds in datasets.items():
            if not isinstance(ds, AbstractData):
                raise TypeError(
                    f"Component '{name}' must be an AbstractData subclass "
                    f"(RVData, GaiaAstrometryData, ...), got {type(ds).__name__}"
                )
            type_map[name] = type(ds).__name__

        dataset_types = {type(ds) for ds in datasets.values()}
        if len(dataset_types) != 1:
            msg = (
                "SystemData requires all component datasets to have the same "
                f"concrete data class; got {type_map}"
            )
            raise TypeError(msg)

        datasets = cast(
            "dict[str, DatasetType]",
            _synchronize_t_refs(cast("dict[str, AbstractData]", datasets)),
        )
        object.__setattr__(self, "_datasets", datasets)
        object.__setattr__(self, "_dataset_type", type(next(iter(datasets.values()))))

    @property
    def dataset_type(self) -> type[AbstractData]:
        """Concrete dataset class shared by all components."""
        return self._dataset_type

    @property
    def t_ref(self) -> NTime:
        """Reference epoch from the first component."""
        return next(iter(self._datasets.values())).t_ref

    def stacked(self) -> DatasetType:
        """Stack all component datasets."""
        return stack_datasets(self._datasets)

    def indicator_data(
        self,
        reference: str,
    ) -> tuple[DatasetType, jax.Array | None, tuple[str, ...] | None]:
        """Return stacked data and component-indicator flags."""
        return build_indicator_matrix(self._datasets, reference)

    def _get_obs(self) -> AbstractQuantity:
        """Concatenated observations across all components (key order)."""
        return jnp.concatenate([ds._get_obs() for ds in self._datasets.values()])

    def _get_obs_err(self) -> AbstractQuantity:
        """Concatenated uncertainties across all components (key order)."""
        return jnp.concatenate([ds._get_obs_err() for ds in self._datasets.values()])


class SourceData(AbstractDatasetContainer):
    """Container for multiple named datasets for a single source.

    Accepts arbitrary named datasets via keyword arguments. Names are
    user-defined and can be anything (e.g., gaia, keck_rv, hst_imaging).
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
        datasets = cast(
            "dict[str, DatasetType]",
            _synchronize_t_refs(cast("dict[str, AbstractData]", datasets)),
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
