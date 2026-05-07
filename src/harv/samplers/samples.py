"""Container for rejection sampler posterior samples.

This module provides the Samples class which stores posterior samples from
rejection sampling with dict-like access, unit handling, and analysis tools.
"""

import warnings
from pathlib import Path
from typing import Any, overload

import equinox as eqx
import h5py
import numpy as np
import quaxed.numpy as jnp
from unxt import AbstractQuantity, Q, ustrip

try:
    import arviz as az
    from arviz_base.labels import MapLabeller

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False

__all__ = ["Samples"]


def _find_namespaced_keys(d: dict[str, Any], param_name: str) -> list[str]:
    """Return all keys in ``d`` that match ``param_name`` (bare or namespaced).

    Matches the bare name (e.g. ``"rv_semiamp"``) and any
    ``"component_name.param_name"`` form (e.g. ``"primary.rv_semiamp"``)
    used by :class:`~harv.models.JointModel` to namespace per-component
    parameters.  Keys are returned in dict insertion order so that
    callers picking the "first" match are deterministic.
    """
    return [k for k in d if k == param_name or k.endswith(f".{param_name}")]


class Samples(eqx.Module):
    """Container for posterior samples.

    Stores both nonlinear and linear parameter samples as :class:`~unxt.Q` objects with
    units baked in. Provides dict-like access, statistical summaries, and visualization
    tools.

    Parameters
    ----------
    nonlinear : dict[str, Q]
        Nonlinear parameter samples, one Q per parameter. Keys: ``"period"``,
        ``"eccentricity"``, ``"phase_peri"``, and optionally ``"arg_peri"``,
        ``"cos_i"``, ``"lon_asc_node"``. Units: period has time units; angles have
        ``"rad"``; dimensionless parameters have unit ``""``.
    linear : dict[str, Q]
        Linear parameter samples, one Q per parameter. Keys: e.g. ``"rv_semiamp"``,
        ``"v_sys"`` for RV; ``"ra0"``, ``"dec0"``, ``"pmra"``, ``"pmdec"``,
        ``"parallax"``, ``"semi_major_axis"`` for astrometry.  Units are data-driven
        (e.g. ``"km/s"`` for RV).
    data_type : str
        Informational label identifying the model that produced these samples (e.g.
        ``"RVModel"``, ``"GaiaAstrometryModel"``, ``"JointModel"``). Stored in HDF5 for
        round-tripping.
    metadata : dict[str, Any], optional
        Additional metadata (``t_ref``, ``num_chains``, acceptance rate, etc.).
    linear_extension_names : tuple[str, ...]
        Names of linear parameters introduced by extensions (instrument offsets,
        polynomial trends, etc.) beyond the base linear set.

    Examples
    --------
    ``Samples`` is normally produced by :meth:`RejectionSampler.run`, but can
    be constructed directly for testing or manual use:

    >>> from unxt import Q
    >>> from harv.samplers.samples import Samples
    >>> samples = Samples(
    ...     nonlinear={
    ...         "period": Q([100.0, 101.0, 99.5], "day"),
    ...         "eccentricity": Q([0.1, 0.15, 0.12], ""),
    ...         "phase_peri": Q([0.3, 0.31, 0.29], ""),
    ...         "arg_peri": Q([1.0, 1.1, 0.9], "rad"),
    ...     },
    ...     linear={
    ...         "rv_semiamp": Q([10.0, 11.0, 9.5], "km/s"),
    ...         "v_sys": Q([5.0, 5.1, 4.9], "km/s"),
    ...     },
    ...     data_type="rv",
    ...     metadata={"t_ref": Q(0.0, "day")},
    ... )
    >>> samples.n_samples
    3
    >>> "period" in samples
    True
    >>> samples.keys()[:4]
    ['period', 'eccentricity', 'phase_peri', 'arg_peri']
    """

    # Pytree leaves -- Q arrays with units baked in
    nonlinear: dict[str, Q]
    linear: dict[str, Q]

    # Static fields -- not JAX leaves
    data_type: str = eqx.field(static=True, default="")
    metadata: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    # Names of linear params introduced by extensions (offsets, trends, etc.).
    linear_extension_names: tuple[str, ...] = eqx.field(static=True, default=())

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

    @overload
    def __getitem__(self, key: str) -> "Q": ...

    @overload
    def __getitem__(self, key: int | slice | np.ndarray) -> "Samples": ...

    def __getitem__(self, key: str | int | slice | np.ndarray) -> "Q | Samples":
        """Get parameter samples by name, or return a sliced ``Samples``.

        Parameters
        ----------
        key : str or int or slice or array
            If ``str``, returns the named parameter array (with units).
            If ``int``, ``slice``, or boolean/integer array, returns a new
            ``Samples`` with all parameter arrays sliced along the sample axis.
            Integer keys are converted to length-1 slices to preserve 1-d shape.

        Returns
        -------
        values : Q or jnp.ndarray
            When ``key`` is a string.
        sliced : Samples
            When ``key`` is an int, slice, or array index.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 101.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 11.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.1], "km/s")},
        ...     data_type="rv",
        ...     metadata={"t_ref": Q(0.0, "day")},
        ... )
        >>> samples["period"].unit
        Unit("d")
        >>> samples["rv_semiamp"].shape
        (2,)
        >>> samples[:1].n_samples
        1
        >>> samples[0].n_samples
        1
        """
        if not isinstance(key, str):
            # int/slice/array index — return a sliced Samples preserving 1-d shape
            idx = slice(key, key + 1) if isinstance(key, int) else key
            sliced_nl = {k: v[idx] for k, v in self.nonlinear.items()}
            sliced_lin = {k: v[idx] for k, v in self.linear.items()}
            return Samples(
                nonlinear=sliced_nl,
                linear=sliced_lin,
                data_type=self.data_type,
                metadata=self.metadata,
                linear_extension_names=self.linear_extension_names,
            )

        if key in self.nonlinear:
            return self.nonlinear[key]

        if key in self.linear:
            return self.linear[key]

        if key == "log_period":
            period = self.nonlinear["period"]
            return jnp.log10(  # ty: ignore[invalid-return-type]
                ustrip(str(period.unit), period)
            )

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

    def wrap_angles(self) -> "Samples":
        """Wrap negative ``rv_semiamp`` / ``semi_major_axis`` to positive.

        For samples with negative ``rv_semiamp`` (or, in astrometry fits, negative
        ``semi_major_axis``) the orbit is physically equivalent to the positive case
        with ``arg_peri -> arg_peri + pi``. This method returns a new :class:`Samples`
        where:

        * negative ``rv_semiamp`` and ``semi_major_axis`` entries are flipped to
          positive,
        * the corresponding ``arg_peri`` values are shifted by ``pi`` and wrapped to
          ``[0, 2*pi)``.

        Joint models (e.g. SB2) namespace per-component linear parameters as
        ``"component.param_name"``; this method discovers every
        ``rv_semiamp``- and ``semi_major_axis``-suffixed key and flips them
        together, since they all share the same ``arg_peri``.  A single
        ω-shift flips an arbitrary number of K's and a's in lockstep.  The
        first ``rv_semiamp``-suffixed key (insertion order) determines the
        flip mask; ``semi_major_axis`` is used as the trigger only if no
        ``rv_semiamp`` is present (astrometry-only fits).

        No-op when ``arg_peri`` is absent from ``nonlinear``, when neither
        ``rv_semiamp`` nor ``semi_major_axis`` is in ``linear``, or when no entries are
        negative.

        Raises
        ------
        NotImplementedError
            If the model has multiple per-component ``arg_peri`` keys
            (e.g. ``"primary.arg_peri"`` and ``"secondary.arg_peri"``).
            All current harv joint factories share ``arg_peri``, so this is
            not expected to arise in practice.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 100.0], "day"),
        ...                "eccentricity": Q([0.1, 0.1], ""),
        ...                "phase_peri": Q([0.3, 0.3], ""),
        ...                "arg_peri": Q([1.0, 1.0], "rad")},
        ...     linear={"rv_semiamp": Q([-10.0, 10.0], "km/s"),
        ...             "v_sys": Q([0.0, 0.0], "km/s")},
        ...     data_type="rv",
        ...     metadata={"t_ref": Q(0.0, "day")},
        ... )
        >>> wrapped = samples.wrap_angles()
        >>> bool((wrapped["rv_semiamp"].value >= 0).all())
        True
        """
        # `arg_peri` must exist; otherwise there's no shift to apply.  We
        # support the typical case of a single shared `arg_peri` (either bare
        # or under a single component's namespace).  Per-component arg_peri is
        # not the default for any joint factory, so out-of-scope here.
        omega_keys = _find_namespaced_keys(self.nonlinear, "arg_peri")
        if not omega_keys:
            return self
        if len(omega_keys) > 1:
            msg = (
                "wrap_angles does not yet support multiple per-component "
                f"arg_peri keys; got {omega_keys!r}.  All current harv joint "
                "models share arg_peri, so this should not arise in practice."
            )
            raise NotImplementedError(msg)
        omega_key = omega_keys[0]

        # Discover every rv_semiamp- and semi_major_axis-suffixed key.  This
        # matches both bare names (single-component) and dot-namespaced names
        # like "primary.rv_semiamp" (joint / SB2).
        K_keys = _find_namespaced_keys(self.linear, "rv_semiamp")
        a_keys = _find_namespaced_keys(self.linear, "semi_major_axis")

        # Trigger: prefer rv_semiamp (RV is more discriminating for the sign
        # convention).  In a joint model with shared `arg_peri`, every K
        # flips together with the same ω-shift, so the choice of trigger
        # doesn't affect the math — we just need *one* deterministic pick.
        if K_keys:
            trigger = self.linear[K_keys[0]]
        elif a_keys:
            trigger = self.linear[a_keys[0]]
        else:
            return self

        flip = ustrip(str(trigger.unit), trigger) < 0
        if not jnp.any(flip):
            return self

        # Flip every K and every a together: in a joint model they all share
        # `arg_peri`, so a single ω-shift flips their signs in lockstep.
        new_lin = dict(self.linear)
        for k in (*K_keys, *a_keys):
            v = new_lin[k]
            v_val = ustrip(str(v.unit), v)
            new_lin[k] = Q(jnp.where(flip, -v_val, v_val), v.unit)

        arg_peri = self.nonlinear[omega_key]
        arg_val = ustrip(str(arg_peri.unit), arg_peri)

        new_nl = dict(self.nonlinear)
        new_nl[omega_key] = Q(
            jnp.where(flip, jnp.mod(arg_val + jnp.pi, 2.0 * jnp.pi), arg_val),
            arg_peri.unit,
        )

        return Samples(
            nonlinear=new_nl,
            linear=new_lin,
            data_type=self.data_type,
            metadata=self.metadata,
            linear_extension_names=self.linear_extension_names,
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
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 102.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 12.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.2], "km/s")},
        ...     data_type="rv",
        ...     metadata={"t_ref": Q(0.0, "day")},
        ... )
        >>> med = samples.median("period")
        >>> med.unit
        Unit("d")
        >>> all_medians = samples.median()
        >>> "period" in all_medians
        True
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
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 102.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 12.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.2], "km/s")},
        ...     data_type="rv",
        ...     metadata={"t_ref": Q(0.0, "day")},
        ... )
        >>> p16, p50, p84 = samples.percentile("eccentricity")
        >>> len(samples.percentile("period", [5, 50, 95]))
        3
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
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 102.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 12.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.2], "km/s")},
        ...     data_type="rv",
        ...     metadata={"t_ref": Q(0.0, "day")},
        ... )
        >>> summary = samples.summary(["period", "eccentricity"])
        >>> sorted(summary.keys())
        ['eccentricity', 'period']
        >>> sorted(summary["period"].keys())
        ['mean', 'median', 'p16', 'p84', 'std']
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
        Save posterior samples and reload them:

        >>> samples.to_hdf5("posterior_samples.h5")  # doctest: +SKIP
        >>> reloaded = Samples.from_hdf5("posterior_samples.h5")  # doctest: +SKIP
        >>> reloaded.n_samples == samples.n_samples  # doctest: +SKIP
        True
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

            # Store metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["data_type"] = self.data_type
            meta_group.attrs["linear_extension_names"] = ",".join(
                self.linear_extension_names
            )
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
        >>> samples = Samples.from_hdf5("posterior_samples.h5")  # doctest: +SKIP
        >>> samples.n_samples  # doctest: +SKIP
        42
        >>> samples.data_type  # doctest: +SKIP
        'rv'
        """
        filename = Path(filename)

        with h5py.File(filename, "r") as f:
            meta = f["metadata"]

            data_type: str = meta.attrs.get("data_type", "")

            raw_extra = meta.attrs.get("linear_extension_names", "") or meta.attrs.get(
                "offset_names", ""
            )
            linear_extension_names: tuple[str, ...] = (
                tuple(raw_extra.split(",")) if raw_extra else ()
            )

            # Load custom metadata
            metadata: dict[str, Any] = {}
            for key in meta.attrs:
                if key in [
                    "linear_extension_names",
                    "offset_names",
                    "n_samples",
                    "data_type",
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
            data_type=data_type,
            linear_extension_names=linear_extension_names,
            metadata=metadata,
        )

    def to_arviz(
        self, params: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> Any:
        """Export samples to an ``arviz.InferenceData`` object.

        Parameters
        ----------
        params : list of str, optional
            Parameters to include.  If ``None``, all parameters returned by
            :meth:`keys` are included.
        labels : dict[str, str], optional
            Override display names for specific parameters, e.g. ``{"period": "period
            [day]", "rv_semiamp": "K [km/s]"}``. Parameters not listed use their plain
            parameter name as the label.

        Returns
        -------
        idata : arviz.InferenceData
            Inference data suitable for ``arviz.plot_pair``, ``arviz.summary``, etc.

        Raises
        ------
        ImportError
            If ``arviz`` is not installed.

        Examples
        --------
        >>> idata = samples.to_arviz(["period", "eccentricity"])  # doctest: +SKIP
        """
        if not HAS_ARVIZ:
            msg = "arviz is required for to_arviz()."
            raise ImportError(msg)

        if params is None:
            params = self.keys()

        num_chains: int = int(self.metadata.get("num_chains", 1))
        n_per_chain, remainder = divmod(self.n_samples, num_chains)
        if remainder != 0:
            # Fall back to treating all samples as a single chain.
            num_chains = 1
            n_per_chain = self.n_samples
            warnings.warn(
                "Number of samples is not divisible by num_chains. "
                "Falling back to a single chain.",
                category=UserWarning,
                stacklevel=1,
            )

        data_dict: dict[str, Any] = {}
        for param in params:
            try:
                values = self[param]
                var_name = (labels or {}).get(param, param)
                if isinstance(values, Q):
                    arr = np.asarray(values.value).reshape(num_chains, n_per_chain)
                else:
                    arr = np.asarray(values).reshape(num_chains, n_per_chain)
                data_dict[var_name] = arr
            except (KeyError, ValueError):
                continue

        if len(data_dict) == 0:
            msg = "No valid parameters found"
            raise ValueError(msg)

        return az.from_dict({"posterior": data_dict})

    def plot_corner(  # noqa: C901
        self,
        params: list[str] | None = None,
        truths: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
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
        labels : dict[str, str], optional
            Override display names for specific parameters, e.g. ``{"period": "period
            [day]", "rv_semiamp": "K [km/s]"}``. Parameters not listed use their plain
            parameter name as the label.
        **plot_kwargs
            Additional keyword arguments passed to arviz.plot_pair().

        Returns
        -------
        axes : np.ndarray
            Array of matplotlib axes from arviz.plot_pair().

        Examples
        --------
        Default corner plot (selects parameters based on data type):

        >>> axes = samples.plot_corner()  # doctest: +SKIP

        Specify parameters and overplot true values:

        >>> from unxt import Q  # doctest: +SKIP
        >>> axes = samples.plot_corner(  # doctest: +SKIP
        ...     params=["period", "eccentricity"],
        ...     truths={"period": Q(100, "day"), "eccentricity": 0.3},
        ... )
        """
        if not HAS_ARVIZ:
            msg = "arviz is required for corner plots."
            raise ImportError(msg)

        # Select default parameters based on available params
        if params is None:
            params = self.keys()

        # Build data dictionary for arviz InferenceData
        data_dict = {}
        var_names = []
        reference_values = {}

        param_units = {}
        for param in params:
            try:
                values = self[param]
                if isinstance(values, Q):
                    param_units[param] = str(values.unit)
                    values = values.value
                data_dict[param] = np.asarray(values)[None, :]
                var_names.append(param)
                if truths is not None and param in truths:
                    truth_val = truths[param]
                    # Strip a Q truth into the plotted unit so the reference
                    # marker lines up with the (already-unitless) sample axis.
                    # ``values`` has already been unwrapped above, so the
                    # recorded ``param_units[param]`` is the source of truth.
                    if isinstance(truth_val, Q):
                        target_unit = param_units.get(param, "")
                        reference_values[param] = float(ustrip(target_unit, truth_val))
                    else:
                        reference_values[param] = float(truth_val)
            except (KeyError, ValueError):
                continue

        if len(data_dict) == 0:
            msg = "No valid parameters found for plotting"
            raise ValueError(msg)

        # Create arviz InferenceData object
        idata = az.from_dict({"posterior": data_dict})

        # Set default plot kwargs
        default_kwargs: dict[str, Any] = {
            "var_names": var_names,
            "marginal": True,
            "triangle": "lower",
        }

        user_labels = labels or {}
        resolved_labels = {
            k: user_labels[k]
            if k in user_labels
            else f"{k} [{param_units[k]}]"
            if param_units.get(k)
            else k
            for k in params
        }
        if any(resolved_labels[k] != k for k in params):
            default_kwargs["labeller"] = MapLabeller(
                var_name_map=resolved_labels  # ty: ignore[invalid-argument-type]
            )

        # Add reference values if provided
        if reference_values:
            default_kwargs.setdefault("visuals", {})
            default_kwargs["visuals"]["point_estimate"] = {"color": "C1"}
            default_kwargs.setdefault("stats", {})
            default_kwargs["stats"]["point_estimate"] = reference_values

        # Merge with user kwargs (user kwargs take precedence)
        default_kwargs.update(plot_kwargs)

        return az.plot_pair(idata, **default_kwargs)
