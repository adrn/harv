"""Container for rejection sampler posterior samples.

This module provides the Samples class which stores posterior samples from
rejection sampling with dict-like access, unit handling, and analysis tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np
import quaxed.numpy as jnp
from unxt import Quantity, ustrip

from harv.likelihood._params import (
    CombinedOrbitParameters,
    GaiaAstrometryFullParameters,
    GaiaAstrometryOrbitParameters,
    RVFullParameters,
    RVOrbitParameters,
)

__all__ = ["Samples"]

# Maps stored class-name strings back to classes for HDF5 round-trips.
_ORBIT_CLS_BY_NAME: dict[str, type] = {
    "RVOrbitParameters": RVOrbitParameters,
    "GaiaAstrometryOrbitParameters": GaiaAstrometryOrbitParameters,
    "CombinedOrbitParameters": CombinedOrbitParameters,
}
_FULL_CLS_BY_NAME: dict[str, type] = {
    "RVFullParameters": RVFullParameters,
    "GaiaAstrometryFullParameters": GaiaAstrometryFullParameters,
}


class Samples(eqx.Module):
    """Container for rejection sampler posterior samples.

    Stores both nonlinear and linear parameter samples with metadata.
    Provides dict-like access with automatic unit conversion, statistical
    summaries, and visualization tools.

    Parameters
    ----------
    _nonlinear : dict[str, jnp.ndarray]
        Nonlinear parameter samples.
        Keys: "period" (days), "eccentricity", "phase_peri", and optionally
        "cos_i", "arg_peri", "lon_asc_node" depending on data type.
    _linear : jnp.ndarray
        Linear parameter samples, shape (n_samples, n_linear_params).
    _orbit_cls : type
        Orbit-only parameter class (e.g. RVOrbitParameters). Its
        ``data_type`` class variable gives the data type string.
    _full_cls : tuple[type, ...]
        Ordered tuple of full parameter classes (e.g. ``(RVFullParameters,)``
        or ``(GaiaAstrometryFullParameters, RVFullParameters)`` for combined).
        Linear parameter names are concatenated from each class's
        ``linear_param_names`` class variable.
    _linear_param_units : tuple[str, ...]
        Units of each linear parameter column, derived from the actual data
        at sampling time (e.g. ``rv_data.rv.unit`` for K and v0). Order
        matches the concatenated ``linear_param_names``.
    _metadata : dict[str, Any], optional
        Additional metadata (t_ref, acceptance rate, etc.).

    Examples
    --------
    >>> samples["period"]  # Returns Quantity in days
    >>> samples["eccentricity"]  # Returns dimensionless array
    >>> samples.n_samples  # Number of posterior samples
    """

    # Internal storage (dimensionless arrays)
    _nonlinear: dict[str, jnp.ndarray]
    _linear: jnp.ndarray

    # Class-driven metadata (static fields)
    _orbit_cls: type = eqx.field(static=True)
    _full_cls: tuple[type, ...] = eqx.field(static=True)
    _linear_param_units: tuple[str, ...] = eqx.field(static=True)
    # Time unit string for the period and t_peri derived quantities.
    # Must match the unit of the data used during sampling (e.g. "day", "yr").
    _time_unit: str = eqx.field(static=True)
    _metadata: dict[str, Any] = eqx.field(static=True)
    # Extra linear parameter names beyond those in _full_cls (e.g. per-instrument
    # RV offsets for multi-survey data).  Empty by default.
    _extra_linear_names: tuple[str, ...] = eqx.field(static=True, default=())

    @property
    def n_samples(self) -> int:
        """Number of posterior samples."""
        return len(next(iter(self._nonlinear.values())))

    @property
    def data_type(self) -> str:
        """Data type these samples correspond to."""
        return self._orbit_cls.data_type  # type: ignore[attr-defined]

    @property
    def _linear_param_names(self) -> tuple[str, ...]:
        """Linear parameter names, derived from _full_cls plus any extras."""
        base: tuple[str, ...] = sum(
            (cls.linear_param_names for cls in self._full_cls),  # type: ignore[attr-defined]
            (),
        )
        return base + self._extra_linear_names

    def keys(self) -> list[str]:
        """All available parameter names (nonlinear + linear + derived)."""
        base_keys = list(self._nonlinear.keys()) + list(self._linear_param_names)
        # Add derived quantities ("period" is now in _nonlinear; log_period is derived)
        derived_keys = ["log_period", "t_peri"]

        # Add inclination if cos_i is available
        if "cos_i" in self._nonlinear:
            derived_keys.append("inclination")

        return base_keys + derived_keys

    def __getitem__(self, key: str) -> Quantity[Any] | jnp.ndarray:
        """Get parameter samples with units restored.

        Parameters
        ----------
        key : str
            Parameter name.

        Returns
        -------
        values : Quantity | jnp.ndarray
            Parameter samples with appropriate units.

        Examples
        --------
        >>> samples["period"]  # Returns Quantity in days
        >>> samples["eccentricity"]  # Returns dimensionless array
        """
        # Check nonlinear parameters
        if key in self._nonlinear:
            return self._get_nonlinear_with_units(key)

        # Check linear parameters
        names = self._linear_param_names
        if key in names:
            idx = names.index(key)
            return self._get_linear_with_units(idx)

        # Check derived quantities
        if key == "log_period":
            return jnp.log10(self._nonlinear["period"])
        if key == "t_peri":
            period = self._nonlinear["period"]
            t_ref = self._metadata.get("t_ref", 0.0)
            t_ref_val = (
                ustrip(self._time_unit, t_ref) if isinstance(t_ref, Quantity) else t_ref
            )
            return Quantity(
                self._nonlinear["phase_peri"] * period + t_ref_val,
                self._time_unit,
            )
        if key == "inclination":
            if "cos_i" in self._nonlinear:
                return Quantity(jnp.arccos(self._nonlinear["cos_i"]), "rad")
            msg = "Inclination only available for astrometry/combined data"
            raise KeyError(msg)

        msg = f"Parameter '{key}' not found"
        raise KeyError(msg)

    def _get_nonlinear_with_units(self, key: str) -> Quantity[Any] | jnp.ndarray:
        """Get nonlinear parameter with appropriate units."""
        value = self._nonlinear[key]

        # Return dimensionless for most parameters
        if key in ["eccentricity", "phase_peri", "cos_i"]:
            return value

        if key == "period":
            return Quantity(value, self._time_unit)

        # Angles in radians
        if key in ["arg_peri", "lon_asc_node"]:
            return Quantity(value, "rad")

        return value

    def _get_linear_with_units(self, idx: int) -> Quantity[Any]:
        """Get linear parameter by column index with appropriate units."""
        value = self._linear[idx] if self._linear.ndim == 1 else self._linear[:, idx]
        unit = self._linear_param_units[idx]
        return Quantity(value, unit)

    def __len__(self) -> int:
        """Number of samples."""
        return self.n_samples

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Samples(n_samples={self.n_samples}, "
            f"data_type='{self.data_type}', "
            f"parameters={len(self.keys())})"
        )

    def median(
        self, key: str | None = None
    ) -> dict[str, Quantity[Any] | jnp.ndarray] | Quantity[Any] | jnp.ndarray:
        """Compute median values for parameters.

        Parameters
        ----------
        key : str, optional
            If provided, return median for this parameter only.
            If None, return dict of medians for all parameters.

        Returns
        -------
        median : dict or Quantity or Array
            Median value(s).

        Examples
        --------
        >>> samples.median("period")  # Median period
        >>> samples.median()  # Dict of all medians
        """
        if key is not None:
            return jnp.median(self[key])

        result: dict[str, Quantity[Any] | jnp.ndarray] = {}
        for param_key in self.keys():
            try:
                result[param_key] = jnp.median(self[param_key])
            except (KeyError, ValueError):
                continue
        return result

    def percentile(
        self, key: str, percentiles: list[float] | tuple[float, ...] = (16, 50, 84)
    ) -> list[Quantity[Any] | jnp.ndarray]:
        """Compute percentiles for a parameter.

        Parameters
        ----------
        key : str
            Parameter name.
        percentiles : list or tuple of float, optional
            Percentile values to compute (0-100). Default: (16, 50, 84)
            which corresponds to the 16th, 50th, 84th percentiles for Gaussian.

        Returns
        -------
        percentiles : list
            Percentile values with appropriate units.

        Examples
        --------
        >>> p16, p50, p84 = samples.percentile("eccentricity")
        >>> p5, p50, p95 = samples.percentile("period", [5, 50, 95])
        """
        values = self[key]
        return [jnp.percentile(values, p) for p in percentiles]

    def summary(self, params: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Compute summary statistics for parameters.

        For each parameter, computes:
        - median
        - mean
        - std (standard deviation)
        - percentiles (16th, 84th) for ±1-sigma equivalent

        Parameters
        ----------
        params : list of str, optional
            List of parameter names to summarize. If None, summarizes all.

        Returns
        -------
        summary : dict
            Dictionary mapping parameter names to their statistics.

        Examples
        --------
        >>> summary = samples.summary(["period", "eccentricity", "parallax"])
        >>> summary["period"]["median"]
        Quantity['time'](100.5, unit='d')
        >>> summary["eccentricity"]["std"]
        0.15
        """
        if params is None:
            params = self.keys()

        result: dict[str, dict[str, Any]] = {}
        for key in params:
            try:
                values = self[key]
                result[key] = {
                    "median": jnp.median(values),
                    "mean": jnp.mean(values),
                    "std": jnp.std(values),
                    "p16": jnp.percentile(values, 16),
                    "p84": jnp.percentile(values, 84),
                }
            except (KeyError, ValueError):
                continue

        return result

    def to_hdf5(self, filename: str | Path) -> None:
        """Save samples to HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Output HDF5 filename.

        Examples
        --------
        >>> samples.to_hdf5("posterior_samples.h5")
        """
        try:
            import h5py
        except ImportError as e:
            msg = (
                "h5py is required for HDF5 serialization. "
                "Install with: pip install h5py"
            )
            raise ImportError(msg) from e

        filename = Path(filename)

        with h5py.File(filename, "w") as f:
            # Store nonlinear parameters
            nl_group = f.create_group("nonlinear")
            for key, value in self._nonlinear.items():
                nl_group.create_dataset(key, data=np.asarray(value))

            # Store linear parameters
            f.create_dataset("linear", data=np.asarray(self._linear))

            # Store class references and metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["orbit_cls"] = self._orbit_cls.__name__
            meta_group.attrs["full_cls"] = ",".join(
                cls.__name__ for cls in self._full_cls
            )
            meta_group.attrs["linear_param_units"] = ",".join(self._linear_param_units)
            meta_group.attrs["time_unit"] = self._time_unit
            meta_group.attrs["extra_linear_names"] = ",".join(self._extra_linear_names)
            meta_group.attrs["n_samples"] = self.n_samples

            # Store custom metadata
            for key, value in self._metadata.items():
                if isinstance(value, int | float | str):
                    meta_group.attrs[key] = value
                elif hasattr(value, "value"):  # Quantity
                    meta_group.attrs[f"{key}_value"] = float(value.value)
                    meta_group.attrs[f"{key}_unit"] = str(value.unit)

    @classmethod
    def from_hdf5(cls, filename: str | Path) -> "Samples":
        """Load samples from HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Input HDF5 filename.

        Returns
        -------
        samples : Samples
            Loaded samples object.

        Examples
        --------
        >>> samples = Samples.from_hdf5("posterior_samples.h5")
        """
        try:
            import h5py
        except ImportError as e:
            msg = (
                "h5py is required for HDF5 serialization. "
                "Install with: pip install h5py"
            )
            raise ImportError(msg) from e

        filename = Path(filename)

        with h5py.File(filename, "r") as f:
            # Load nonlinear parameters
            nonlinear = {}
            for key in f["nonlinear"]:
                nonlinear[key] = jnp.array(f["nonlinear"][key][:])

            # Load linear parameters
            linear = jnp.array(f["linear"][:])

            # Load class references
            meta = f["metadata"]
            orbit_cls_name = meta.attrs["orbit_cls"]
            full_cls_names = meta.attrs["full_cls"].split(",")

            orbit_cls = _ORBIT_CLS_BY_NAME[orbit_cls_name]
            full_cls = tuple(_FULL_CLS_BY_NAME[n] for n in full_cls_names)
            linear_param_units = tuple(meta.attrs["linear_param_units"].split(","))
            # Fall back to "day" when loading files written before _time_unit was added.
            time_unit: str = meta.attrs.get("time_unit", "day")
            raw_extra = meta.attrs.get("extra_linear_names", "")
            extra_linear_names: tuple[str, ...] = (
                tuple(raw_extra.split(",")) if raw_extra else ()
            )

            # Load custom metadata
            metadata: dict[str, Any] = {}
            for key in meta.attrs:
                if key in [
                    "orbit_cls",
                    "full_cls",
                    "linear_param_units",
                    "time_unit",
                    "extra_linear_names",
                    "n_samples",
                ]:
                    continue
                if key.endswith("_value"):
                    # Skip, will be reconstructed with unit
                    continue
                if key.endswith("_unit"):
                    # Reconstruct Quantity
                    base_key = key[:-5]
                    value = meta.attrs[f"{base_key}_value"]
                    unit = meta.attrs[key]
                    metadata[base_key] = Quantity(value, unit)
                else:
                    metadata[key] = meta.attrs[key]

        return cls(
            _nonlinear=nonlinear,
            _linear=linear,
            _orbit_cls=orbit_cls,
            _full_cls=full_cls,
            _linear_param_units=linear_param_units,
            _time_unit=time_unit,
            _extra_linear_names=extra_linear_names,
            _metadata=metadata,
        )

    def plot_corner(  # noqa: C901
        self,
        params: list[str] | None = None,
        truths: dict[str, Any] | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Create corner plot of posterior samples using arviz.

        Parameters
        ----------
        params : list of str, optional
            Parameters to include in corner plot. If None, selects a default
            set based on data_type.
        truths : dict, optional
            Dictionary of true parameter values to overplot as reference values.
        **plot_kwargs
            Additional keyword arguments passed to arviz.plot_pair().

        Returns
        -------
        axes : np.ndarray
            Array of matplotlib axes from arviz.plot_pair().

        Examples
        --------
        >>> axes = samples.plot_corner()
        >>> axes = samples.plot_corner(params=["period", "eccentricity", "parallax"])
        >>> axes = samples.plot_corner(truths={"period": Quantity(100, "day")})
        """
        try:
            import arviz as az
        except ImportError as e:
            msg = (
                "arviz and matplotlib required for plotting. "
                "Install with: pip install arviz matplotlib"
            )
            raise ImportError(msg) from e

        # Select default parameters based on data type
        if params is None:
            if self.data_type == "astrometry":
                params = ["period", "eccentricity", "parallax", "semi_major_axis"]
            elif self.data_type == "rv":
                params = ["period", "eccentricity", "K", "v0"]
            elif self.data_type == "combined":
                params = ["period", "eccentricity", "parallax", "semi_major_axis", "K"]
            else:
                # Fallback: use first 4 available parameters
                params = self.keys()[:4]

        # Build data dictionary for arviz InferenceData
        data_dict = {}
        var_names = []
        reference_values = {}

        for param in params:
            try:
                values = self[param]
                if isinstance(values, Quantity):
                    # Store with unit in variable name
                    var_name = f"{param} [{values.unit}]"
                    data_dict[var_name] = np.asarray(values.value)[None, :]
                    var_names.append(var_name)
                    # Handle truths/reference values
                    if truths is not None and param in truths:
                        truth_val = truths[param]
                        if isinstance(truth_val, Quantity):
                            reference_values[var_name] = ustrip(values.unit, truth_val)
                        else:
                            reference_values[var_name] = float(truth_val)
                else:
                    data_dict[param] = np.asarray(values)[None, :]
                    var_names.append(param)
                    # Handle truths/reference values
                    if truths is not None and param in truths:
                        reference_values[param] = float(truths[param])
            except (KeyError, ValueError):
                continue

        if len(data_dict) == 0:
            msg = "No valid parameters found for plotting"
            raise ValueError(msg)

        # Create arviz InferenceData object
        idata = az.from_dict(posterior=data_dict)

        # Set default plot kwargs
        default_kwargs: dict[str, Any] = {
            "var_names": var_names,
            "kind": "kde",
            "marginals": True,
            "point_estimate": "median",
        }

        # Add reference values if provided
        if reference_values:
            default_kwargs["reference_values"] = reference_values
            default_kwargs["reference_values_kwargs"] = {"color": "C1", "lw": 2}

        # Merge with user kwargs (user kwargs take precedence)
        default_kwargs.update(plot_kwargs)

        # Create corner plot
        return az.plot_pair(idata, **default_kwargs)
