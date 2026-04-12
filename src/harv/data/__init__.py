"""Data classes for representing time series data.

TODO: we could add support for metadata for the data type classes below.
"""

__all__ = (
    "AbstractAstrometryData",
    "AbstractData",
    "DatasetType",
    "GaiaAstrometryData",
    # "AbsoluteAstrometryData",
    "InputData",
    "RVData",
    "SourceData",
    "SystemData",
    "build_indicator_matrix",
    "stack_datasets",
)

from .containers import InputData, SourceData, SystemData
from .datasets import (
    AbstractAstrometryData,
    AbstractData,
    DatasetType,
    GaiaAstrometryData,
    RVData,
)
from .helpers import build_indicator_matrix, stack_datasets
