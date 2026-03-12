"""Container for rejection sampler posterior samples.

This module provides the Samples class which stores posterior samples from
rejection sampling with dict-like access, unit handling, and analysis tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from unxt import Quantity

__all__ = ["Samples"]


class Samples(eqx.Module):
    """Container for rejection sampler posterior samples.

    Stores both nonlinear and linear parameter samples with metadata.
    Provides dict-like access with automatic unit conversion, statistical
    summaries, and visualization tools.

    Parameters
    ----------
    _nonlinear : dict[str, jnp.ndarray]
        Nonlinear parameter samples (dimensionless arrays).
        Keys: "log_period", "eccentricity", "phase_peri", and optionally
        "cos_i", "arg_peri", "lon_asc_node" depending on data type.
    _linear : jnp.ndarray
        Linear parameter samples, shape (n_samples, n_linear_params).
    _linear_param_names : tuple[str, ...]
        Names of linear parameters in order.
    _data_type : {"astrometry", "rv", "combined", "sb2"}
        Type of data these samples correspond to.
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

    # Metadata (static fields for efficiency)
    _linear_param_names: tuple[str, ...] = eqx.field(static=True)
    _data_type: str = eqx.field(static=True)
    _metadata: dict[str, Any] = eqx.field(static=True)

    @property
    def n_samples(self) -> int:
        """Number of posterior samples."""
        return len(self._nonlinear["log_period"])

    @property
    def data_type(self) -> str:
        """Data type these samples correspond to."""
        return self._data_type

    def keys(self) -> list[str]:
        """All available parameter names (nonlinear + linear + derived)."""
        base_keys = list(self._nonlinear.keys()) + list(self._linear_param_names)
        # Add derived quantities
        derived_keys = ["period", "t_peri"]

        # Add inclination if cos_i is available
        if "cos_i" in self._nonlinear:
            derived_keys.append("inclination")

        return base_keys + derived_keys

    def __getitem__(self, key: str) -> Quantity | jnp.ndarray:
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
        if key in self._linear_param_names:
            idx = self._linear_param_names.index(key)
            return self._get_linear_with_units(key, idx)

        # Check derived quantities
        if key == "period":
            return Quantity(10.0 ** self._nonlinear["log_period"], "day")
        if key == "t_peri":
            period = 10.0 ** self._nonlinear["log_period"]
            return Quantity(
                self._nonlinear["phase_peri"] * period
                + self._metadata.get("t_ref", 0.0),
                "day",
            )
        if key == "inclination":
            if "cos_i" in self._nonlinear:
                return Quantity(jnp.arccos(self._nonlinear["cos_i"]), "rad")
            msg = "Inclination only available for astrometry/combined data"
            raise KeyError(msg)

        msg = f"Parameter '{key}' not found"
        raise KeyError(msg)

    def _get_nonlinear_with_units(self, key: str) -> Quantity | jnp.ndarray:
        """Get nonlinear parameter with appropriate units."""
        value = self._nonlinear[key]

        # Return dimensionless for most parameters
        if key in ["eccentricity", "phase_peri", "cos_i"]:
            return value

        # log_period is dimensionless but represents log10(period/day)
        if key == "log_period":
            return value

        # Angles in radians
        if key in ["arg_peri", "lon_asc_node"]:
            return Quantity(value, "rad")

        return value

    def _get_linear_with_units(self, key: str, idx: int) -> Quantity:
        """Get linear parameter with appropriate units."""
        # _linear should always be 2D (n_samples, n_linear_params)
        # but handle edge cases
        if self._linear.ndim == 1:
            # Edge case: single parameter vector (shouldn't normally happen)
            value = self._linear[idx]
        else:
            # Normal case: (n_samples, n_linear_params)
            value = self._linear[:, idx]

        # Astrometric parameters
        if key in ["alpha_0", "delta_0"]:
            return Quantity(value, "deg")
        if key in ["mu_alpha", "mu_delta"]:
            return Quantity(value, "mas/yr")
        if key in ["parallax", "semimajor_axis"]:
            return Quantity(value, "mas")

        # RV parameters
        if key in ["K", "K1", "K2", "v0"]:
            return Quantity(value, "km/s")

        # If unknown, return dimensionless
        return value

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
    ) -> dict[str, Quantity | float] | Quantity | float:
        """Compute median values for parameters.

        Parameters
        ----------
        key : str, optional
            If provided, return median for this parameter only.
            If None, return dict of medians for all parameters.

        Returns
        -------
        median : dict or Quantity or float
            Median value(s).

        Examples
        --------
        >>> samples.median("period")  # Median period
        >>> samples.median()  # Dict of all medians
        """
        if key is not None:
            values = self[key]
            if isinstance(values, Quantity):
                return Quantity(jnp.median(values.value), values.unit)
            return float(jnp.median(values))

        # Compute medians for all parameters
        result = {}
        for param_key in self.keys():
            try:
                values = self[param_key]
                if isinstance(values, Quantity):
                    result[param_key] = Quantity(jnp.median(values.value), values.unit)
                else:
                    result[param_key] = float(jnp.median(values))
            except (KeyError, ValueError):
                continue
        return result

    def percentile(
        self, key: str, percentiles: list[float] | tuple[float, ...] = (16, 50, 84)
    ) -> list[Quantity | float]:
        """Compute percentiles for a parameter.

        Parameters
        ----------
        key : str
            Parameter name.
        percentiles : list or tuple of float, optional
            Percentile values to compute (0-100). Default: (16, 50, 84)
            which corresponds to -1σ, median, +1σ for Gaussian.

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
        perc_array = jnp.array(percentiles)
        if isinstance(values, Quantity):
            percs = jnp.percentile(values.value, perc_array)
            return [Quantity(p, values.unit) for p in percs]
        percs = jnp.percentile(values, perc_array)
        return [float(p) for p in percs]

    def summary(self, params: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Compute summary statistics for parameters.

        For each parameter, computes:
        - median
        - mean
        - std (standard deviation)
        - percentiles (16th, 84th) for ±1σ equivalent

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

        summary = {}
        for key in params:
            try:
                values = self[key]

                # Extract numerical values
                if isinstance(values, Quantity):
                    vals = values.value
                    unit = values.unit
                else:
                    vals = values
                    unit = None

                # Compute statistics
                stats = {
                    "median": jnp.median(vals),
                    "mean": jnp.mean(vals),
                    "std": jnp.std(vals),
                    "p16": jnp.percentile(vals, 16),
                    "p84": jnp.percentile(vals, 84),
                }

                # Restore units if applicable
                if unit is not None:
                    stats = {k: Quantity(v, unit) for k, v in stats.items()}
                else:
                    stats = {k: float(v) for k, v in stats.items()}

                summary[key] = stats

            except (KeyError, ValueError):
                continue

        return summary

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
            msg = "h5py is required for HDF5 serialization. Install with: pip install h5py"
            raise ImportError(msg) from e

        filename = Path(filename)

        with h5py.File(filename, "w") as f:
            # Store nonlinear parameters
            nl_group = f.create_group("nonlinear")
            for key, value in self._nonlinear.items():
                nl_group.create_dataset(key, data=np.asarray(value))

            # Store linear parameters
            f.create_dataset("linear", data=np.asarray(self._linear))

            # Store metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["data_type"] = self._data_type
            meta_group.attrs["n_samples"] = self.n_samples
            meta_group.attrs["linear_param_names"] = ",".join(self._linear_param_names)

            # Store custom metadata
            for key, value in self._metadata.items():
                if isinstance(value, (int, float, str)):
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
            msg = "h5py is required for HDF5 serialization. Install with: pip install h5py"
            raise ImportError(msg) from e

        filename = Path(filename)

        with h5py.File(filename, "r") as f:
            # Load nonlinear parameters
            nonlinear = {}
            for key in f["nonlinear"].keys():
                nonlinear[key] = jnp.array(f["nonlinear"][key][:])

            # Load linear parameters
            linear = jnp.array(f["linear"][:])

            # Load metadata
            meta = f["metadata"]
            data_type = meta.attrs["data_type"]
            linear_param_names = tuple(meta.attrs["linear_param_names"].split(","))

            # Load custom metadata
            metadata = {}
            for key in meta.attrs.keys():
                if key in ["data_type", "n_samples", "linear_param_names"]:
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
            _linear_param_names=linear_param_names,
            _data_type=data_type,
            _metadata=metadata,
        )

    def plot_corner(
        self,
        params: list[str] | None = None,
        truths: dict[str, Any] | None = None,
        **plot_kwargs,
    ):
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
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = "arviz and matplotlib required for plotting. Install with: pip install arviz matplotlib"
            raise ImportError(msg) from e

        # Select default parameters based on data type
        if params is None:
            if self.data_type == "astrometry":
                params = ["period", "eccentricity", "parallax", "semimajor_axis"]
            elif self.data_type == "rv":
                params = ["period", "eccentricity", "K", "v0"]
            elif self.data_type == "combined":
                params = ["period", "eccentricity", "parallax", "semimajor_axis", "K"]
            elif self.data_type == "sb2":
                params = ["period", "eccentricity", "K1", "K2", "v0"]
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
                            reference_values[var_name] = float(
                                truth_val.to_value(values.unit)
                            )
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
        default_kwargs = {
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
        axes = az.plot_pair(idata, **default_kwargs)

        return axes
