"""Container for rejection sampler posterior samples.

This module provides the Samples class which stores posterior samples from
rejection sampling with dict-like access, unit handling, and analysis tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import h5py
import numpy as np
import quaxed.numpy as jnp
from numpyro import infer as _numpyro_infer
from unxt import AbstractQuantity, Q, ustrip

from harv.data import RVData, SourceData
from harv.kepler.orbits import astrometric_orbit_at_times, rv_at_times
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    RVParameters,
)
from harv.plot import get_t_grid

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

__all__ = ["Samples"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class _WarmStartMCMC:
    """Wrapper around ``numpyro.infer.MCMC`` with pre-set warm-start init params.

    Constructed by :meth:`RejectionSampler.init_mcmc`. Provides the full numpyro MCMC
    API via attribute delegation; only :meth:`run` is overridden to inject the
    rejection-sampler posterior positions as starting points unless the caller
    explicitly passes their own ``init_params``.

    Parameters
    ----------
    sampler :
        An instantiated numpyro MCMC kernel (e.g. ``NUTS(model)``).
    _init_params : dict[str, np.ndarray]
        Per-chain initial parameter values, shape ``(num_chains,)`` per key.
        Keys must match the numpyro site names used in the kernel's model.
    **mcmc_kwargs :
        Forwarded unchanged to ``numpyro.infer.MCMC.__init__``.
    """

    def __init__(
        self,
        sampler: Any,
        *,
        _init_params: dict[str, Any],
        **mcmc_kwargs: Any,
    ) -> None:
        self._mcmc = _numpyro_infer.MCMC(sampler, **mcmc_kwargs)
        self._init_params = _init_params

    def run(
        self,
        rng_key: Any,
        *args: Any,
        init_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run MCMC, using rejection-sampler positions as starting points.

        Parameters
        ----------
        rng_key :
            JAX key passed to the underlying ``numpyro.infer.MCMC.run``.
        *args :
            Positional arguments forwarded to ``MCMC.run``.
        init_params : dict, optional
            If provided, overrides the warm-start positions supplied at
            construction time.
        **kwargs :
            Keyword arguments forwarded to ``MCMC.run``.
        """
        self._mcmc.run(
            rng_key,
            *args,
            init_params=init_params if init_params is not None else self._init_params,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (get_samples, print_summary, ...) to the
        # underlying numpyro MCMC object.
        return getattr(self._mcmc, name)

    def __repr__(self) -> str:
        return (
            f"_WarmStartMCMC("
            f"num_chains={self._mcmc.num_chains}, "
            f"num_samples={self._mcmc.num_samples})"
        )


# Maps stored class-name strings back to classes for HDF5 round-trips.
_ORBIT_CLS_BY_NAME: dict[str, type] = {
    "RVParameters": RVParameters,
    "GaiaAstrometryParameters": GaiaAstrometryParameters,
}
_FULL_CLS_BY_NAME: dict[str, type] = {
    "RVParameters": RVParameters,
    "GaiaAstrometryParameters": GaiaAstrometryParameters,
}

# Nonlinear parameter units (fixed by physics).
_NONLINEAR_UNITS: dict[str, str] = {
    "period": "",  # filled from data at construction time
    "eccentricity": "",
    "phase_peri": "",
    "arg_peri": "rad",
    "cos_i": "",
    "lon_asc_node": "rad",
}


class Samples(eqx.Module):
    """Container for rejection sampler posterior samples.

    Stores both nonlinear and linear parameter samples as :class:`~unxt.Q`
    objects with units baked in. Provides dict-like access, statistical summaries,
    and visualization tools.

    Parameters
    ----------
    nonlinear : dict[str, Q]
        Nonlinear parameter samples, one Q per parameter.
        Keys: ``"period"``, ``"eccentricity"``, ``"phase_peri"``,
        and optionally ``"arg_peri"``, ``"cos_i"``, ``"lon_asc_node"``.
        Units: period has time units; angles have ``"rad"``; dimensionless
        parameters have unit ``""``.
    linear : dict[str, Q]
        Linear parameter samples, one Q per parameter.
        Keys: e.g. ``"rv_semiamp"``, ``"v_sys"`` for RV; ``"ra0"``, ``"dec0"``,
        ``"pmra"``, ``"pmdec"``, ``"parallax"``, ``"semi_major_axis"`` for
        astrometry.  Units are data-driven (e.g. ``"km/s"`` for RV).
    orbit_cls : type
        Nonlinear parameter class (e.g. ``RVParameters``).
    full_cls : tuple[type, ...]
        Ordered tuple of full parameter classes.
    metadata : dict[str, Any], optional
        Additional metadata (``t_ref``, acceptance rate, etc.).
    extra_linear_names : tuple[str, ...]
        Names of per-instrument RV offset parameters beyond the base linear
        set (e.g. ``("instr2_offset",)`` for multi-survey data).
    data_type : str
        One of ``"rv"``, ``"astrometry"``, or ``"combined"``.

    Examples
    --------
    >>> samples["period"]        # Q with time units
    >>> samples["eccentricity"]  # Q (dimensionless)
    >>> samples.n_samples        # number of posterior draws
    """

    # Pytree leaves -- Q arrays with units baked in
    nonlinear: dict[str, Q]
    linear: dict[str, Q]

    # Static fields -- not JAX leaves
    orbit_cls: type = eqx.field(static=True)
    full_cls: tuple[type, ...] = eqx.field(static=True)
    metadata: dict[str, Any] = eqx.field(static=True)
    # Names of per-instrument offsets stored in `linear` beyond the base set.
    extra_linear_names: tuple[str, ...] = eqx.field(static=True, default=())
    data_type: str = eqx.field(static=True, default="")

    @property
    def n_samples(self) -> int:
        """Number of posterior samples."""
        return int(next(iter(self.nonlinear.values())).shape[0])

    def keys(self) -> list[str]:
        """All available parameter names (nonlinear + linear + derived)."""
        base_keys = list(self.nonlinear.keys()) + list(self.linear.keys())
        derived_keys = ["log_period", "t_peri"]
        if "cos_i" in self.nonlinear:
            derived_keys.append("inclination")
        return base_keys + derived_keys

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def __getitem__(self, key: str) -> AbstractQuantity | jnp.ndarray:
        """Get parameter samples with units restored.

        Parameters
        ----------
        key : str
            Parameter name.

        Returns
        -------
        values : Q | jnp.ndarray
            Parameter samples with appropriate units.

        Examples
        --------
        >>> samples["period"]        # Q with time units
        >>> samples["eccentricity"]  # Q (dimensionless)
        """
        if key in self.nonlinear:
            return self.nonlinear[key]

        if key in self.linear:
            return self.linear[key]

        if key == "log_period":
            period = self.nonlinear["period"]
            return jnp.log10(ustrip(str(period.unit), period))

        if key == "t_peri":
            # Express t_peri in absolute time: t_ref + phase_peri * period.
            # phase_peri encodes the fractional orbital phase at t=0, so
            # phase_peri * period is the periastron time relative to t=0, and
            # adding t_ref converts it to the same absolute coordinate as data.time.
            period = self.nonlinear["period"]
            time_unit = str(period.unit)
            t_ref_raw = self.metadata.get("t_ref", 0.0)
            t_ref_val = (
                float(ustrip(time_unit, t_ref_raw))
                if isinstance(t_ref_raw, Q)
                else (float(t_ref_raw) if t_ref_raw is not None else 0.0)
            )
            phase_peri = ustrip("", self.nonlinear["phase_peri"])
            period_val = ustrip(time_unit, period)
            return Q(t_ref_val + phase_peri * period_val, time_unit)

        if key == "inclination":
            if "cos_i" in self.nonlinear:
                cos_i = ustrip("", self.nonlinear["cos_i"])
                return Q(jnp.arccos(cos_i), "rad")
            msg = "Inclination only available for astrometry/combined data"
            raise KeyError(msg)

        msg = f"Parameter '{key}' not found"
        raise KeyError(msg)

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
    ) -> dict[str, AbstractQuantity | jnp.ndarray] | AbstractQuantity | jnp.ndarray:
        """Compute median values for parameters.

        Parameters
        ----------
        key : str, optional
            If provided, return median for this parameter only.
            If None, return dict of medians for all parameters.

        Returns
        -------
        median : dict or Q or Array
            Median value(s).

        Examples
        --------
        >>> samples.median("period")  # Median period
        >>> samples.median()  # Dict of all medians
        """
        if key is not None:
            return jnp.median(self[key])

        result: dict[str, AbstractQuantity | jnp.ndarray] = {}
        for param_key in self.keys():
            try:
                result[param_key] = jnp.median(self[param_key])
            except (KeyError, ValueError):
                continue
        return result

    def percentile(
        self, key: str, percentiles: list[float] | tuple[float, ...] = (16, 50, 84)
    ) -> list[AbstractQuantity | jnp.ndarray]:
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
        - percentiles (16th, 84th) for +/-1-sigma equivalent

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
        Q['time'](100.5, unit='d')
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
        filename = Path(filename)

        with h5py.File(filename, "w") as f:
            # Store nonlinear parameters -- each as a dataset with a unit attr.
            nl_group = f.create_group("nonlinear")
            for key, qty in self.nonlinear.items():
                ds = nl_group.create_dataset(key, data=np.asarray(qty.value))
                ds.attrs["unit"] = str(qty.unit)

            # Store linear parameters -- each as a dataset with a unit attr.
            lin_group = f.create_group("linear")
            for key, qty in self.linear.items():
                ds = lin_group.create_dataset(key, data=np.asarray(qty.value))
                ds.attrs["unit"] = str(qty.unit)

            # Store class references and metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["orbit_cls"] = self.orbit_cls.__name__
            meta_group.attrs["data_type"] = self.data_type
            meta_group.attrs["full_cls"] = ",".join(
                cls.__name__ for cls in self.full_cls
            )
            meta_group.attrs["extra_linear_names"] = ",".join(self.extra_linear_names)
            meta_group.attrs["n_samples"] = self.n_samples

            # Store custom metadata
            for key, value in self.metadata.items():
                if isinstance(value, int | float | str):
                    meta_group.attrs[key] = value
                elif hasattr(value, "value"):  # Q
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
        filename = Path(filename)

        with h5py.File(filename, "r") as f:
            # Load class references
            meta = f["metadata"]
            orbit_cls_name = meta.attrs["orbit_cls"]
            full_cls_names = meta.attrs["full_cls"].split(",")

            orbit_cls = _ORBIT_CLS_BY_NAME[orbit_cls_name]
            full_cls = tuple(_FULL_CLS_BY_NAME[n] for n in full_cls_names)

            # data_type: read from file, or infer for old files
            _DATA_TYPE_BY_OLD_CLS = {
                "CombinedOrbitParameters": "combined",
                "RVMarginalizedParameters": "rv",
                "RVParameters": "rv",
                "GaiaAstrometryMarginalizedParameters": "astrometry",
                "GaiaAstrometryParameters": "astrometry",
            }
            data_type: str = meta.attrs.get(
                "data_type", _DATA_TYPE_BY_OLD_CLS.get(orbit_cls_name, "")
            )

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
                    "extra_linear_names",
                    "n_samples",
                    "data_type",
                    # old-format keys -- skip, handled separately
                    "linear_param_units",
                    "time_unit",
                ]:
                    continue
                if key.endswith("_value"):
                    continue
                if key.endswith("_unit"):
                    base_key = key[:-5]
                    value = meta.attrs[f"{base_key}_value"]
                    unit = meta.attrs[key]
                    metadata[base_key] = Q(value, unit)
                else:
                    metadata[key] = meta.attrs[key]

            nonlinear: dict[str, Q] = {}
            for key in f["nonlinear"]:
                ds = f["nonlinear"][key]
                unit = ds.attrs.get("unit", "")
                nonlinear[key] = Q(jnp.array(ds[:]), unit)

            linear: dict[str, Q] = {}
            for key in f["linear"]:
                ds = f["linear"][key]
                unit = ds.attrs.get("unit", "")
                linear[key] = Q(jnp.array(ds[:]), unit)

        return cls(
            nonlinear=nonlinear,
            linear=linear,
            orbit_cls=orbit_cls,
            full_cls=full_cls,
            extra_linear_names=extra_linear_names,
            data_type=data_type,
            metadata=metadata,
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
        >>> axes = samples.plot_corner(truths={"period": Q(100, "day")})
        """
        if not HAS_ARVIZ:
            msg = "arviz is required for corner plots."
            raise ImportError(msg)

        # Select default parameters based on data type
        if params is None:
            if self.data_type == "astrometry":
                params = ["period", "eccentricity", "parallax", "semi_major_axis"]
            elif self.data_type == "rv":
                params = ["period", "eccentricity", "rv_semiamp", "v_sys"]
            elif self.data_type == "combined":
                params = [
                    "period",
                    "eccentricity",
                    "parallax",
                    "semi_major_axis",
                    "rv_semiamp",
                ]
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
                if isinstance(values, Q):
                    # Store with unit in variable name
                    var_name = f"{param} [{values.unit}]"
                    data_dict[var_name] = np.asarray(values.value)[None, :]
                    var_names.append(var_name)
                    # Handle truths/reference values
                    if truths is not None and param in truths:
                        truth_val = truths[param]
                        if isinstance(truth_val, Q):
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

    def plot(
        self,
        data: Any = None,
        *,
        n_samples: int = 50,
        phase_fold: bool = False,
        apply_mean_offsets: bool = True,
        plot_kwargs: dict[str, Any] | None = None,
        data_plot_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """RV curve and/or astrometric orbit on sky.

        Selects panels automatically based on ``data_type``:

        - ``"rv"`` -- RV curve (time-domain or phase-folded); ``data`` must be
          a ``RVData`` or ``SourceData`` containing RV datasets.
        - ``"astrometry"`` -- on-sky orbital ellipses drawn from posterior
          samples; ``data`` is not required (orbit shape comes from samples).
        - ``"combined"`` -- both panels side by side; ``data`` must be a
          ``SourceData`` containing both ``GaiaAstrometryData`` and at least
          one ``RVData``.

        Parameters
        ----------
        data : RVData or SourceData, optional
            Observed data to overplot on the RV panel.  Required for ``"rv"``
            and ``"combined"`` data types; optional for ``"astrometry"``.
        n_samples : int, optional
            Number of posterior orbit curves to draw.  Default: 50.
        phase_fold : bool, optional
            If ``True``, fold the RV data and model curves to orbital phase
            using the median posterior period.  The mean v_sys is subtracted
            from the data so the y-axis shows the intrinsic RV variation.
            If ``False`` (default), plot RV vs time in the reference frame.
        apply_mean_offsets : bool, optional
            When ``True`` (default), shift each non-reference instrument's data
            points by the posterior mean offset so they land in the reference
            frame and can be compared directly to the model curves.  Has no
            effect when there are no multi-instrument offsets.
        plot_kwargs : dict, optional
            Style overrides for orbit model curves (passed to
            ``ax.plot()``).  Defaults: thin grey lines with no markers.
        data_plot_kwargs : dict, optional
            Style overrides for data points (passed to ``ax.errorbar()``).
            Defaults: filled circles with error bars.
        **kwargs :
            Additional keyword arguments forwarded to
            ``matplotlib.pyplot.subplots``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot(s).

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        ValueError
            If the data type is unknown.

        Examples
        --------
        >>> fig = samples.plot(data=rv_data)
        >>> fig = samples.plot(data=source_data, n_samples=100)
        >>> fig = samples.plot(data=source_data, phase_fold=True)
        >>> fig = samples.plot()  # astrometry only, no data points needed
        >>> fig = samples.plot(plot_kwargs={"color": "C3", "alpha": 0.3})
        """
        if not HAS_MPL:
            msg = "matplotlib is required for plotting. "
            raise ImportError(msg)

        if plot_kwargs is None:
            plot_kwargs = {}
        if data_plot_kwargs is None:
            data_plot_kwargs = {}

        dt = self.data_type
        if dt == "rv":
            fig, ax = plt.subplots(**kwargs)
            self._draw_rv(
                data,
                ax=ax,
                n_samples=n_samples,
                plt=plt,
                phase_fold=phase_fold,
                apply_mean_offsets=apply_mean_offsets,
                plot_kwargs=plot_kwargs,
                data_plot_kwargs=data_plot_kwargs,
            )
            fig.tight_layout()
            return fig
        if dt == "astrometry":
            fig, ax = plt.subplots(**kwargs)
            self._draw_astrometry(
                ax=ax,
                n_samples=n_samples,
                plot_kwargs=plot_kwargs,
            )
            fig.tight_layout()
            return fig
        if dt == "combined":
            figsize = kwargs.pop("figsize", (12, 5))
            fig, axes = plt.subplots(1, 2, figsize=figsize, **kwargs)
            self._draw_rv(
                data,
                ax=axes[0],
                n_samples=n_samples,
                plt=plt,
                phase_fold=phase_fold,
                apply_mean_offsets=apply_mean_offsets,
                plot_kwargs=plot_kwargs,
                data_plot_kwargs=data_plot_kwargs,
            )
            self._draw_astrometry(
                ax=axes[1],
                n_samples=n_samples,
                plot_kwargs=plot_kwargs,
            )
            fig.tight_layout()
            return fig
        msg = f"Unknown data_type '{dt}' for plot()."
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Private drawing helpers (draw into a caller-supplied Axes object)
    # ------------------------------------------------------------------

    def _draw_rv(  # noqa: C901
        self,
        data: Any,
        *,
        ax: Any,
        n_samples: int,
        plt: Any,
        phase_fold: bool = False,
        apply_mean_offsets: bool = True,
        plot_kwargs: dict[str, Any] | None = None,
        data_plot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """RV curve (time-domain or phase-folded) drawn into *ax*.

        Parameters
        ----------
        data : RVData or SourceData or None
            Observed RV data to overplot.
        ax : matplotlib.axes.Axes
            Axes to draw into.
        n_samples : int
            Number of posterior samples to draw as model curves.
        plt : module
            The matplotlib.pyplot module.
        phase_fold :
            When ``True`` fold data and model to orbital phase using the median
            period and subtract the mean v_sys so the y-axis shows only the
            intrinsic RV variation.  The mean anomaly is computed correctly
            even when individual samples have periods that differ from the
            median folding period.
        apply_mean_offsets :
            When ``True``, shift each non-reference instrument's data points
            by the posterior mean offset so they fall in the reference frame.
        plot_kwargs :
            Style overrides for orbit model curves (passed to ``ax.plot()``).
        data_plot_kwargs :
            Style overrides for data points (passed to ``ax.errorbar()``).
        """
        # Orbit curve style defaults (thin lines, no markers)
        orbit_style = (plot_kwargs or {}).copy()
        orbit_style.setdefault("linestyle", "-")
        orbit_style.setdefault("linewidth", 0.5)
        orbit_style.setdefault("alpha", 0.15)
        orbit_style.setdefault("marker", "")
        orbit_style.setdefault("color", "#555555")
        orbit_style.setdefault("rasterized", True)

        # Data style defaults (error bars with filled circles)
        data_style = (data_plot_kwargs or {}).copy()
        data_style.setdefault("linestyle", "none")
        data_style.setdefault("marker", "o")
        data_style.setdefault("markersize", 4.0)
        data_style.setdefault("elinewidth", 1.0)
        data_style.setdefault("capsize", 0)
        data_style.setdefault("zorder", 10)

        period_qty = self.nonlinear["period"]
        ecc_qty = self.nonlinear["eccentricity"]
        phase_peri_qty = self.nonlinear["phase_peri"]
        arg_peri_qty = self.nonlinear["arg_peri"]

        time_unit = str(period_qty.unit)
        period_vals = np.asarray(period_qty.value)  # plain (n,) for indexing

        K_qty = self.linear["rv_semiamp"]
        v0_qty = self.linear["v_sys"]
        rv_unit = str(K_qty.unit)
        K_vals = np.asarray(K_qty.value)
        v0_vals = np.asarray(v0_qty.value)

        median_period_val = float(np.median(period_vals))
        median_period = Q(median_period_val, time_unit)

        t_ref_raw = self.metadata.get("t_ref", 0.0)
        t_ref = t_ref_raw if isinstance(t_ref_raw, Q) else Q(t_ref_raw, time_unit)

        # Collect per-instrument datasets (multi-survey support).
        if isinstance(data, SourceData):
            rv_datasets: dict[str, RVData] = data.get_datasets_by_type(RVData)
        elif isinstance(data, RVData):
            rv_datasets = {"data": data}
        elif data is None:
            rv_datasets = {}
        else:
            msg = "data must be RVData or SourceData for data_type='rv'."
            raise ValueError(msg)

        # Per-instrument mean offsets (extra linear params beyond rv_semiamp and v_sys).
        mean_offsets: dict[str, Q] = {
            name: Q(float(np.mean(np.asarray(self.linear[name].value))), rv_unit)
            for name in self.extra_linear_names
        }

        # Mean v_sys used to centre phase-folded plots on zero.
        mean_v0 = Q(float(np.mean(v0_vals)), rv_unit)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # --- Data points ---
        for color_idx, (instr_name, rv_data) in enumerate(rv_datasets.items()):
            rv_obs = rv_data.rv
            rv_err = rv_data.rv_err

            # Shift non-reference instrument data into the reference frame.
            if apply_mean_offsets and instr_name in mean_offsets:
                rv_obs = rv_obs - mean_offsets[instr_name]

            if phase_fold:
                x_data = ustrip(
                    "",
                    ((rv_data.time - t_ref) / median_period) % Q(1.0, ""),
                )
                rv_obs = rv_obs - mean_v0
            else:
                x_data = ustrip(time_unit, rv_data.time)

            # Per-instrument style: start from data_style defaults, then
            # set color from the color cycle for multi-instrument plots.
            instr_style = data_style.copy()
            if "color" not in data_plot_kwargs:
                instr_style["color"] = colors[color_idx % len(colors)]
            label = instr_name if len(rv_datasets) > 1 else "data"
            instr_style.setdefault("label", label)
            ax.errorbar(
                x_data,
                ustrip(rv_unit, rv_obs),
                yerr=ustrip(rv_unit, rv_err),
                **instr_style,
            )

        # --- Posterior model curves ---
        n_draw = min(n_samples, self.n_samples)
        ecc_vals = np.asarray(ecc_qty.value)
        phase_peri_vals = np.asarray(phase_peri_qty.value)
        arg_peri_vals = np.asarray(arg_peri_qty.value)

        if phase_fold:
            # For each sample, evaluate the model in the sample's OWN orbital phase
            # frame (times = t_peri_i + phi * P_i) and then convert those times to
            # the DATA's display phase ((t - t_ref) / P_median mod 1).
            # This guarantees the model curve is x-aligned with every data point,
            # even when the sample's period differs from the reference period.
            phi_grid = np.linspace(0.0, 1.0, 500)
            for i in range(n_draw):
                period_i = Q(float(period_vals[i]), time_unit)
                ecc_i = float(ecc_vals[i])
                phase_peri_i = float(phase_peri_vals[i])
                arg_peri_i = Q(float(arg_peri_vals[i]), "rad")
                K_i = Q(float(K_vals[i]), rv_unit)
                # t_peri: phase_peri * period (no t_ref), matches _solve_kepler
                t_peri_i = Q(phase_peri_i * float(period_vals[i]), time_unit)
                # Times in sample i's own frame: one complete orbit from periastron.
                t_model = t_peri_i + Q(phi_grid, "") * period_i
                rv_model = rv_at_times(
                    t_model,
                    period_i,
                    ecc_i,
                    t_peri_i,
                    arg_peri_i,
                    K_i,
                    Q(0.0, rv_unit),
                )
                # Map model times to data display phase.
                x_model = np.asarray(
                    ustrip("", (t_model - t_ref) / median_period % Q(1.0, ""))
                )
                rv_vals = np.asarray(ustrip(rv_unit, rv_model))
                # Sort by display phase; insert NaN breaks at wrap-around jumps
                # so matplotlib does not draw a line across the plot.
                idx_sort = np.argsort(x_model)
                x_sorted = x_model[idx_sort]
                rv_sorted = rv_vals[idx_sort]
                gaps = np.where(np.diff(x_sorted) < -0.5)[0] + 1
                x_plot = np.insert(x_sorted.astype(float), gaps, np.nan)
                rv_plot = np.insert(rv_sorted.astype(float), gaps, np.nan)
                ax.plot(x_plot, rv_plot, **orbit_style)

            ax.set_xlabel("Orbital phase")
            ax.set_ylabel(f"RV $-$ $v_0$ [{rv_unit}]")
            ax.set_xlim(0.0, 1.0)
            ax.set_title(
                f"Phase-folded RV  (median P = {median_period_val:.1f} {time_unit})"
            )
        else:
            # Time-domain: dense grid spanning all observations.
            if rv_datasets:
                all_times = Q(
                    jnp.concatenate(
                        [rv_data.time.value for rv_data in rv_datasets.values()]
                    ),
                    time_unit,
                )
                t_grid = get_t_grid(all_times, median_period)
            else:
                t_grid = t_ref + Q(np.linspace(0.0, 1.0, 500), "") * median_period

            for i in range(n_draw):
                period_i = Q(float(period_vals[i]), time_unit)
                ecc_i = float(ecc_vals[i])
                phase_peri_i = float(phase_peri_vals[i])
                arg_peri_i = Q(float(arg_peri_vals[i]), "rad")
                K_i = Q(float(K_vals[i]), rv_unit)
                v0_i = Q(float(v0_vals[i]), rv_unit)
                # t_peri: phase_peri * period (no t_ref), matches _solve_kepler
                t_peri_i = Q(phase_peri_i * float(period_vals[i]), time_unit)
                rv_model = rv_at_times(
                    t_grid, period_i, ecc_i, t_peri_i, arg_peri_i, K_i, v0_i
                )
                ax.plot(
                    ustrip(time_unit, t_grid),
                    ustrip(rv_unit, rv_model),
                    **orbit_style,
                )

            ax.set_xlabel(f"Time [{time_unit}]")
            ax.set_ylabel(f"RV [{rv_unit}]")
            ax.set_title(f"RV (median P = {median_period_val:.1f} {time_unit})")

        if rv_datasets:
            ax.legend(loc="best")

    def _draw_astrometry(
        self,
        *,
        ax: Any,
        n_samples: int,
        plot_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """On-sky orbital ellipses drawn into *ax* for each posterior sample.

        Gaia along-scan measurements are 1-D projections and cannot be plotted
        directly as 2-D sky positions, so only the model orbit curves are shown.
        """
        orbit_style = (plot_kwargs or {}).copy()
        orbit_style.setdefault("linestyle", "-")
        orbit_style.setdefault("linewidth", 0.5)
        orbit_style.setdefault("alpha", 0.15)
        orbit_style.setdefault("marker", "")
        orbit_style.setdefault("color", "#555555")
        orbit_style.setdefault("rasterized", True)
        sma_qty = self.linear.get("semi_major_axis")
        sma_unit = str(sma_qty.unit) if sma_qty is not None else "mas"

        period_qty = self.nonlinear["period"]
        time_unit = str(period_qty.unit)
        period_vals = np.asarray(period_qty.value)
        ecc_vals = np.asarray(self.nonlinear["eccentricity"].value)
        phase_peri_vals = np.asarray(self.nonlinear["phase_peri"].value)
        arg_peri_vals = np.asarray(self.nonlinear["arg_peri"].value)
        cos_i_vals = np.asarray(self.nonlinear["cos_i"].value)
        lon_asc_vals = np.asarray(self.nonlinear["lon_asc_node"].value)
        sma_vals = np.asarray(sma_qty.value) if sma_qty is not None else None

        t_ref_raw = self.metadata.get("t_ref", 0.0)
        t_ref = t_ref_raw if isinstance(t_ref_raw, Q) else Q(t_ref_raw, time_unit)

        n_draw = min(n_samples, self.n_samples)
        for i in range(n_draw):
            period_i = Q(float(period_vals[i]), time_unit)
            ecc_i = float(ecc_vals[i])
            phase_peri_i = float(phase_peri_vals[i])
            arg_peri_i = Q(float(arg_peri_vals[i]), "rad")
            cos_i_i = float(cos_i_vals[i])
            lon_asc_i = Q(float(lon_asc_vals[i]), "rad")
            sma_val = float(sma_vals[i]) if sma_vals is not None else 1.0
            sma_i = Q(sma_val, sma_unit)
            # t_peri: phase_peri * period (no t_ref), matches _solve_kepler
            t_peri_i = Q(phase_peri_i * float(period_vals[i]), time_unit)

            # One full orbit: phi in [0, 1] -> times spanning exactly one period.
            # Use t_ref as origin so the ellipse is centered near the observations.
            phi_grid = np.linspace(0.0, 1.0, 500)
            times_grid = t_ref + Q(phi_grid, "") * period_i

            delta_ra, delta_dec = astrometric_orbit_at_times(
                times_grid,
                period_i,
                ecc_i,
                t_peri_i,
                arg_peri_i,
                cos_i_i,
                lon_asc_i,
                sma_i,
            )
            ax.plot(
                np.asarray(ustrip(sma_unit, delta_ra)),
                np.asarray(ustrip(sma_unit, delta_dec)),
                **orbit_style,
            )

        ax.set_xlabel(rf"$\Delta$RA [{sma_unit}]")
        ax.set_ylabel(rf"$\Delta$Dec [{sma_unit}]")
        ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_title("Orbit on sky")
