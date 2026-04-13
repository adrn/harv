"""Observation data classes for time series data."""

__all__ = (
    "AbstractAstrometryData",
    "AbstractData",
    "DatasetType",
    "GaiaAstrometryData",
    # "AbsoluteAstrometryData",
    "RVData",
)

from dataclasses import KW_ONLY
from typing import Any, ClassVar

import equinox as eqx
import quaxed.numpy as jnp
from unxt import AbstractQuantity

from harv.custom_types import NAngle, NFloatArray, NTime, NVelocity
from harv.plot import _plot_timeseries_errorbar

# Optional dependency:
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


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
        if ax is None:
            ax = plt.gca()

        al_unit = str(self.al_position.unit)
        return _plot_timeseries_errorbar(
            ax,
            self.time,
            self.al_position,
            self.al_position_err,
            time_unit=str(self.time.unit),
            obs_unit=al_unit,
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
        >>> from unxt import Q
        >>> data = RVData(
        ...     time=Q([0.0, 50.0, 100.0], "day"),
        ...     rv=Q([1.0, -2.0, 0.5], "km/s"),
        ...     rv_err=Q([0.5, 0.5, 0.5], "km/s"),
        ... )
        >>> ax = data.plot()  # uses errorbar() with sensible defaults
        >>> ax = data.plot(color="C1", markersize=6)  # override style
        >>> plt.close("all")
        """
        if ax is None:
            ax = plt.gca()

        if rv_unit is None:
            rv_unit = str(self.rv.unit)

        return _plot_timeseries_errorbar(
            ax,
            self.time,
            self.rv,
            self.rv_err,
            time_unit=str(self.time.unit),
            obs_unit=rv_unit,
            t_ref=self.t_ref,
            relative_to_t_ref=relative_to_t_ref,
            ylabel=f"RV [{rv_unit}]",
            add_labels=add_labels,
            **kwargs,
        )


# Type alias for all supported data types
DatasetType = AbstractAstrometryData | RVData
