"""Observation data classes for time series data."""

__all__ = (
    "AbstractAstrometryData",
    "AbstractData",
    "DatasetType",
    "GaiaAstrometryData",
    # "AbsoluteAstrometryData",
    "RVData",
)

import dataclasses
from dataclasses import KW_ONLY
from typing import Any, ClassVar

import equinox as eqx
import numpy as np
from unxt import AbstractQuantity, Q
from unxt.quantity import ustrip

from harv.custom_types import NAngle, NFloatArray, NTime, NVelocity, ScalarQTime

# Optional dependency:
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class AbstractData(eqx.Module):
    """Abstract base class for observational data time series."""

    _obs_name: eqx.AbstractClassVar[str]
    _err_name: eqx.AbstractClassVar[str]

    # Note: time is defined in subclasses as a field, not as an abstract property
    # to avoid dataclass field ordering issues with equinox

    time: NTime
    """Barycentric TCB times."""

    _: KW_ONLY

    t_ref: ScalarQTime | None = None
    """Reference epoch. If None, uses mean observation time."""

    def __check_init__(self) -> None:
        """Compute t_ref from mean time if not provided."""
        if self.t_ref is None:
            # TODO: This is ugly - do we really need a concrete numpy mean here?
            # Use concrete NumPy mean so t_ref is a plain Python float wrapped in Q.
            # This avoids placing a JAX-traced array in a static metadata field
            # downstream.
            time_unit = str(self.time.unit)
            t_mean = float(np.mean(np.asarray(ustrip(time_unit, self.time))))
            object.__setattr__(self, "t_ref", Q(t_mean, time_unit))

    @property
    def n_times(self) -> int:
        """Number of times / epochs / observations."""
        return len(self.time)

    def __getitem__(self, key: Any) -> "AbstractData":
        """Return a new dataset with observations sliced along the time axis.

        Fields whose shape matches ``self.time.shape`` are sliced; scalar fields
        (e.g. ``t_ref``) are passed through unchanged.  Integer keys are converted
        to length-1 slices so that all arrays remain 1-d.

        Parameters
        ----------
        key : int, slice, or array-like
            Index or slice to apply to the observation axis.

        Returns
        -------
        AbstractData
            New instance of the same concrete class with sliced arrays.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv import RVData
        >>> data = RVData(
        ...     time=Q([0.0, 50.0, 100.0], "day"),
        ...     rv=Q([1.0, -2.0, 0.5], "km/s"),
        ...     rv_err=Q([0.5, 0.5, 0.5], "km/s"),
        ... )
        >>> data[:2].n_times
        2
        >>> data[0].n_times
        1
        """
        idx = slice(key, key + 1) if isinstance(key, int) else key
        obs_shape = self.time.shape
        fields_dict: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if hasattr(val, "shape") and val.shape == obs_shape:
                fields_dict[f.name] = val[idx]
            else:
                fields_dict[f.name] = val
        return type(self)(**fields_dict)

    def _get_obs(self) -> AbstractQuantity:
        """Get the observed values (e.g., positions, RVs)."""
        return getattr(self, self._obs_name)

    def _get_obs_err(self) -> AbstractQuantity:
        """Get the observed uncertainties."""
        return getattr(self, self._err_name)


class AbstractAstrometryData(AbstractData):
    """Abstract base class for astrometric data."""


class GaiaAstrometryData(AbstractAstrometryData):
    """Gaia epoch astrometry (along-scan measurements).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxt import Q
    >>> from harv import GaiaAstrometryData
    >>> data = GaiaAstrometryData(
    ...     time=Q([0.0, 100.0, 200.0], "day"),
    ...     al_position=Q([0.1, -0.2, 0.05], "mas"),
    ...     al_position_err=Q([0.01, 0.01, 0.01], "mas"),
    ...     scan_angle=Q([0.5, 1.2, 2.8], "rad"),
    ...     parallax_factor=jnp.array([0.3, -0.1, 0.4]),
    ... )
    >>> data.n_times
    3
    """

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
        al_unit: str | None = None,
        add_labels: bool = True,
        relative_to_t_ref: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Plot along-scan residuals vs time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  If ``None``, uses ``plt.gca()``.
        al_unit : str, optional
            Display unit for the along-scan position.  Defaults to the data's own unit.
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
        >>> import jax.numpy as jnp
        >>> import matplotlib.pyplot as plt
        >>> from unxt import Q
        >>> from harv import GaiaAstrometryData
        >>> data = GaiaAstrometryData(
        ...     time=Q([0.0, 100.0, 200.0], "day"),
        ...     al_position=Q([0.1, -0.2, 0.05], "mas"),
        ...     al_position_err=Q([0.01, 0.01, 0.01], "mas"),
        ...     scan_angle=Q([0.5, 1.2, 2.8], "rad"),
        ...     parallax_factor=jnp.array([0.3, -0.1, 0.4]),
        ... )
        >>> ax = data.plot()
        >>> plt.close("all")
        """
        from harv.plot import plot_timeseries_errorbar  # - circular imp.

        al_unit = al_unit or str(self.al_position.unit)
        return plot_timeseries_errorbar(
            self.time,
            self.al_position,
            self.al_position_err,
            ax=ax,
            time_unit=str(self.time.unit),
            # obs_unit=al_unit,
            t_ref=self.t_ref,
            relative_to_t_ref=relative_to_t_ref,
            ylabel=f"AL position [{al_unit}]",
            add_labels=add_labels,
            **kwargs,
        )


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
    """Radial velocity measurements.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv import RVData
    >>> data = RVData(
    ...     time=Q([0.0, 50.0, 100.0], "day"),
    ...     rv=Q([1.0, -2.0, 0.5], "km/s"),
    ...     rv_err=Q([0.5, 0.5, 0.5], "km/s"),
    ... )
    >>> data.n_times
    3
    """

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
        phase_fold: Any | None = None,
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
            Plot time relative to ``t_ref``.  Mutually exclusive with
            ``phase_fold``.
        phase_fold : Q["time"], optional
            If given, fold observations to orbital phase using this period:
            x = (time - t_ref) / phase_fold mod 1.  Mutually exclusive with
            ``relative_to_t_ref``.
        **kwargs
            Passed to ``ax.errorbar()``.  Defaults can be overridden.

        Returns
        -------
        ax : matplotlib.axes.Axes

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from unxt import Q
        >>> data = RVData(
        ...     time=Q([0.0, 50.0, 100.0], "day"),
        ...     rv=Q([1.0, -2.0, 0.5], "km/s"),
        ...     rv_err=Q([0.5, 0.5, 0.5], "km/s"),
        ... )
        >>> ax = data.plot()  # uses errorbar() with sensible defaults
        >>> ax = data.plot(color="C1", markersize=6)  # override style
        >>> ax = data.plot(phase_fold=Q(50.0, "day"))  # phase-folded
        >>> plt.close("all")
        """
        from harv.plot import plot_timeseries_errorbar  # - circular imp.

        if phase_fold is not None and relative_to_t_ref:
            msg = "phase_fold and relative_to_t_ref are mutually exclusive"
            raise ValueError(msg)

        rv_unit = rv_unit or str(self.rv.unit)
        return plot_timeseries_errorbar(
            self.time,
            self.rv,
            self.rv_err,
            ax=ax,
            time_unit=str(self.time.unit),
            obs_unit=rv_unit,
            t_ref=self.t_ref,
            relative_to_t_ref=relative_to_t_ref,
            phase_fold=phase_fold,
            ylabel=f"RV [{rv_unit}]",
            add_labels=add_labels,
            **kwargs,
        )


# Type alias for all supported data types
DatasetType = AbstractAstrometryData | RVData
