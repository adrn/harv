"""Data classes for representing time series data.

TODO: we could add support for metadata for the data type classes below.
"""

__all__ = (
    "AbstractAstrometryData",
    "GaiaAstrometryData",
    # "AbsoluteAstrometryData",
    "RVData",
    "SystemData",
    "SourceData",
    "DatasetType",
    "InputData",
    "stack_datasets",
    "build_indicator_matrix",
)

from collections.abc import Iterator
from dataclasses import KW_ONLY, fields
from typing import Any, ClassVar, TypeVar

import equinox as eqx
import jax
import numpy as np
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

    def plot(
        self,
        ax: Any = None,
        *,
        add_labels: bool = True,
        relative_to_t_ref: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot along-scan residuals vs time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  If ``None``, uses ``plt.gca()``.
        add_labels : bool, optional
            Add axis labels.
        relative_to_t_ref : bool, optional
            Plot time relative to ``t_ref``.
        **kwargs
            Passed to ``ax.errorbar()``.  Defaults can be overridden.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        al_unit = str(self.al_position.unit)
        time_unit = str(self.time.unit)

        t = np.asarray(ustrip(time_unit, self.time))
        if relative_to_t_ref and self.t_ref is not None:
            t = t - float(ustrip(time_unit, self.t_ref))

        style = kwargs.copy()
        style.setdefault("linestyle", "none")
        style.setdefault("marker", "o")
        style.setdefault("markersize", 4.0)
        style.setdefault("elinewidth", 1.0)
        style.setdefault("capsize", 0)
        style.setdefault("color", "k")
        style.setdefault("ecolor", "#666666")
        style.setdefault("zorder", 10)

        ax.errorbar(
            t,
            np.asarray(ustrip(al_unit, self.al_position)),
            yerr=np.asarray(ustrip(al_unit, self.al_position_err)),
            **style,
        )

        if add_labels:
            xlabel = (
                f"Time $-$ t_ref [{time_unit}]"
                if relative_to_t_ref
                else f"Time [{time_unit}]"
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"AL position [{al_unit}]")

        return ax


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


class RVData(AbstractData):
    """Radial velocity measurements."""

    _obs_name: ClassVar[str] = "rv"
    _err_name: ClassVar[str] = "rv_err"

    rv: NVelocity
    """Radial velocities."""

    rv_err: NVelocity
    """Radial velocity uncertainties."""

    def plot(
        self,
        ax: Any = None,
        *,
        rv_unit: str | None = None,
        add_labels: bool = True,
        relative_to_t_ref: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot RV data as error bars.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  If ``None``, uses ``plt.gca()``.
        rv_unit : str, optional
            Display unit for the RV axis.  Defaults to the data's own unit.
        add_labels : bool, optional
            Add axis labels.
        relative_to_t_ref : bool, optional
            Plot time relative to ``t_ref``.
        **kwargs
            Passed to ``ax.errorbar()``.  Defaults can be overridden.

        Returns
        -------
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from unxt import Quantity
        >>> data = RVData(
        ...     time=Quantity([0.0, 50.0, 100.0], "day"),
        ...     rv=Quantity([1.0, -2.0, 0.5], "km/s"),
        ...     rv_err=Quantity([0.5, 0.5, 0.5], "km/s"),
        ... )
        >>> ax = data.plot()  # uses errorbar() with sensible defaults
        >>> ax = data.plot(color="C1", markersize=6)  # override style
        >>> plt.close("all")
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if rv_unit is None:
            rv_unit = str(self.rv.unit)
        time_unit = str(self.time.unit)

        t = np.asarray(ustrip(time_unit, self.time))
        if relative_to_t_ref and self.t_ref is not None:
            t = t - float(ustrip(time_unit, self.t_ref))

        style = kwargs.copy()
        style.setdefault("linestyle", "none")
        style.setdefault("marker", "o")
        style.setdefault("markersize", 4.0)
        style.setdefault("elinewidth", 1.0)
        style.setdefault("capsize", 0)
        style.setdefault("color", "k")
        style.setdefault("ecolor", "#666666")
        style.setdefault("zorder", 10)

        ax.errorbar(
            t,
            np.asarray(ustrip(rv_unit, self.rv)),
            yerr=np.asarray(ustrip(rv_unit, self.rv_err)),
            **style,
        )

        if add_labels:
            xlabel = (
                f"Time $-$ t_ref [{time_unit}]"
                if relative_to_t_ref
                else f"Time [{time_unit}]"
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"RV [{rv_unit}]")

        return ax


# Type alias for all supported data types
DatasetType = AbstractAstrometryData | RVData
_DT = TypeVar("_DT", bound=DatasetType)


class SystemData(eqx.Module):
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

    _datasets: dict[str, DatasetType]

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

    # -- Dict-like interface (mirrors SourceData) --

    def __getitem__(self, name: str) -> DatasetType:
        return self._datasets[name]

    def __contains__(self, name: str) -> bool:
        return name in self._datasets

    def __len__(self) -> int:
        return len(self._datasets)

    def keys(self) -> Iterator[str]:
        """Component names."""
        return iter(self._datasets.keys())

    def values(self) -> Iterator[DatasetType]:
        """Component datasets."""
        return iter(self._datasets.values())

    def items(self) -> Iterator[tuple[str, DatasetType]]:
        """(name, dataset) pairs."""
        return iter(self._datasets.items())

    def get_datasets_by_type(self, dtype: type[_DT]) -> dict[str, _DT]:
        """Get all components of a specific data type."""
        return {k: v for k, v in self._datasets.items() if isinstance(v, dtype)}

    # -- Convenience properties --

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
                    f"RVData, got {type(ds).__name__}"
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
        return len(self.get_datasets_by_type(RVData))


# Type alias for any top-level input accepted by the sampler and likelihoods.
# Use this instead of AbstractData in signatures that also accept SourceData.
InputData = AbstractData | SourceData | SystemData


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
        if field.name != "t_ref"  # scalar, not array -- skip and recompute below
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
    # NOTE: t_ref is recomputed from the stacked time by __check_init__
    # TODO: we need to add a note somewhere (probably SourceData or all of the *Data
    # class docstrings) about how t_ref is handled when stacking datasets, since it's
    # not just concatenated but recomputed from the mean time. A potentially better
    # thing to do would be to check if one t_ref is set (use that), else throw an error.
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
