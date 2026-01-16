"""Data classes for representing time series data."""

__all__ = [
    "AbstractAstrometry",
    "GaiaAstrometry",
    "AbsoluteAstrometry",
    "RadialVelocity",
    "SourceData",
    "DatasetType",
]

from abc import abstractmethod
from collections.abc import Iterator

import equinox as eqx

from .custom_types import NAngle, NFloatArray, NIntArray, NTime, NVelocity


class AbstractData(eqx.Module):  # type: ignore[misc]
    """Abstract base class for observational data time series."""

    @property
    @abstractmethod
    def time(self) -> NTime:
        """Observation times."""
        ...

    @property
    def n_times(self) -> int:
        """Number of times / epochs / observations."""
        return len(self.time)


class AbstractAstrometry(AbstractData):
    """Abstract base class for astrometric data."""


class GaiaAstrometry(AbstractAstrometry):
    """Gaia epoch astrometry (along-scan measurements)."""

    time: NTime  # Barycentric TCB times
    al_position: NAngle  # Along-scan position
    al_position_err: NAngle  # AL uncertainty
    scan_angle: NAngle  # Per-CCD scan angle θ
    parallax_factor: NFloatArray  # AL parallax factors
    t_ref: NTime  # Reference epoch for proper motion
    transit_index: NIntArray | None = None  # Optional transit grouping


class AbsoluteAstrometry(AbstractAstrometry):
    """Traditional absolute astrometry (RA/Dec measurements)."""

    time: NTime  # Observation times
    ra: NAngle  # Right ascension
    dec: NAngle  # Declination
    ra_err: NAngle  # RA uncertainty
    dec_err: NAngle  # Dec uncertainty
    t_ref: NTime  # Reference epoch for proper motion
    correlation: NFloatArray | None = None  # RA-Dec correlation coefficient
    parallax_factor: NFloatArray | None = None  # Optional parallax factors


# TODO: Future implementation - RelativeAstrometry for imaging data
# class RelativeAstrometry(AbstractAstrometry):
#     """Relative astrometry (companion position relative to star).
#
#     For direct imaging, interferometry, etc. where companion position
#     is measured relative to the host star.
#     """
#     time: NTime                           # Observation times
#     x: NAngle                             # x offset (e.g., RA direction)
#     y: NAngle                             # y offset (e.g., Dec direction)
#     x_error: NAngle                       # x uncertainty
#     y_error: NAngle                       # y uncertainty
#     correlation: NFloatArray | None = None  # x-y correlation coefficient
#


class AbstractRadialVelocity(AbstractData):
    """Abstract base class for radial velocity data."""


class RadialVelocity(AbstractRadialVelocity):
    """Radial velocity measurements."""

    time: NTime  # Observation times
    rv: NVelocity  # Radial velocities
    rv_err: NVelocity  # RV uncertainty


# Type alias for all supported data types
DatasetType = AbstractAstrometry | AbstractRadialVelocity


class SourceData(AbstractData):
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
                    f"Dataset '{name}' must be AbstractAstrometry or RadialVelocity, "
                    f"got {type(ds).__name__}"
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
    def get_datasets_by_type(self, dtype: type) -> dict[str, DatasetType]:
        """Get all datasets of a specific type."""
        return {k: v for k, v in self._datasets.items() if isinstance(v, dtype)}

    def n_astrometry(self) -> int:
        """Number of astrometric datasets."""
        return len(self.get_datasets_by_type(AbstractAstrometry))

    def n_rv(self) -> int:
        """Number of radial velocity datasets."""
        return len(self.get_datasets_by_type(RadialVelocity))
