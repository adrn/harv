"""Plotting utilities."""

__all__ = (
    "get_t_grid",
    "plot_gaia_astrometry",
    "plot_gaia_sky_orbit",
    "plot_rv",
)

import warnings
from typing import Any

import numpy as np
import quaxed.numpy as jnp
from unxt import Q, ustrip

from harv.custom_types import BatchQTime, NQAny, NTime, QTime, ScalarQTime
from harv.data import GaiaAstrometryData, RVData, SourceData, SystemData
from harv.extensions.multi_survey import MultiSurveyOffset
from harv.kepler.orbits import astrometric_orbit_at_times, rv_at_times
from harv.samplers import Samples

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    import tinygp
except ImportError:
    tinygp = None  # type: ignore[assignment]


# Default styles:
_DEFAULT_ERRORBAR_STYLE: dict[str, Any] = {
    "linestyle": "none",
    "marker": "o",
    "markersize": 3.0,
    "elinewidth": 1.0,
    "capsize": 0,
    "color": "k",
    "ecolor": "#888888",
    "zorder": 10,
}

_DEFAULT_LINE_STYLE: dict[str, Any] = {
    "linestyle": "-",
    "linewidth": 0.5,
    "marker": "",
    "color": "tab:blue",
}


# Helper functions:


def plot_timeseries_errorbar(
    time: NTime,
    obs: NQAny,
    obs_err: NQAny,
    *,
    time_unit: str | None = None,
    obs_unit: str | None = None,
    t_ref: Any | None = None,
    relative_to_t_ref: bool = False,
    phase_fold: ScalarQTime | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_labels: bool = True,
    ax: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Plot observation vs time as error bars (internal helper).

    Parameters
    ----------
    time, obs, obs_err
        Quantity arrays for time, observation, and observation uncertainty.
    time_unit, obs_unit
        Unit strings for axes.
    t_ref
        Reference epoch (Quantity or None).
    relative_to_t_ref
        Whether to subtract ``t_ref`` from times before plotting.
    phase_fold
        If provided, a period Quantity. Plot ``(time - t_ref) / phase_fold mod 1``
        on the x-axis instead of absolute time.
    xlabel, ylabel
        Axis label overrides.
    add_labels
        Whether to set axis labels.
    ax
        Matplotlib axes to draw on.
    **kwargs
        Forwarded to ``ax.errorbar()``, overriding defaults.
    """
    if ax is None:
        _, ax = plt.subplots()

    if time_unit is None:
        time_unit: str = str(time.unit)

    if obs_unit is None:
        obs_unit: str = str(obs.unit)

    if phase_fold is not None:
        t_ref_val = ustrip(time_unit, t_ref) if t_ref is not None else 0.0
        x = (
            (ustrip(time_unit, time) - t_ref_val) / ustrip(time_unit, phase_fold)
        ) % 1.0
        _xlabel = "orbital phase"

    else:
        x = ustrip(time_unit, time)

        if relative_to_t_ref and t_ref is not None:
            x = x - ustrip(time_unit, t_ref)

        _xlabel = (
            f"time $-$ t_ref [{time_unit}]"
            if relative_to_t_ref
            else f"time [{time_unit}]"
        )

    # Merge default and user styles, with user overrides (kwargs) taking precedence:
    style = {**_DEFAULT_ERRORBAR_STYLE, **kwargs}

    ax.errorbar(
        np.asarray(x),
        np.asarray(ustrip(obs_unit, obs)),
        yerr=np.asarray(ustrip(obs_unit, obs_err)),
        **style,
    )

    if add_labels:
        if xlabel is None:
            xlabel = _xlabel
        ax.set(xlabel=xlabel, ylabel=ylabel)

    return ax


def get_t_grid(
    times: BatchQTime,
    period: ScalarQTime,
    *,
    span_buffer_factor: float = 0.1,
    n_points_per_period: int = 256,
    max_t_grid: int | None = int(1e6),
) -> NTime:
    """Dense time grid spanning the observation baseline with a small buffer.

    Generates a regular grid of times for plotting model orbits over data. The grid
    resolution adapts to the period so that fast orbits are well-resolved while
    long-period orbits don't create excessive grids.

    Parameters
    ----------
    times : Q["time"]
        Observation times.
    period : Q["time"]
        Orbital period (scalar).  Used to set the grid spacing as ``period /
        n_points_per_period``.
    span_buffer_factor : float, optional
        Fractional buffer added to each side of the observation baseline. Default: 0.1
        (10% on each side).
    n_points_per_period : int, optional
        Number of grid points per orbital period.  Default: 256.
    max_t_grid : int or None, optional
        Maximum number of grid points. Default: 1e6. Set to None to disable.

    Returns
    -------
    t_grid : Q["time"]
        Regular time grid spanning the buffered observation range.

    Examples
    --------
    >>> from unxt import Q
    >>> times = Q([0.0, 50.0, 100.0], "day")
    >>> t_grid = get_t_grid(times, Q(30.0, "day"))
    >>> len(t_grid) > 128
    True
    """
    time_unit = str(times.unit)
    t_vals = np.asarray(times.value)
    t_min, t_max = t_vals.min(), t_vals.max()
    span = t_max - t_min

    p_val = float(ustrip(time_unit, period))
    dt = p_val / n_points_per_period

    n_grid = span / dt if dt > 0 else 1
    if max_t_grid is not None and n_grid > max_t_grid:
        dt = span / max_t_grid

    grid = np.arange(
        t_min - span * span_buffer_factor,
        t_max + span * span_buffer_factor + dt,
        dt,
    )
    return Q(grid, time_unit)


def get_alpha(n: int) -> float:
    """Get alpha (transparency) for plotting many samples, to avoid overplotting."""
    return max(0.08, min(0.8, 8.0 / n))


# --- Hacky, extension-specific plotting helpers ---


def _get_sample_scalar_value(samples: Any, name: str, sample_index: int) -> float:
    """Extract one scalar sample value by bare or qualified parameter name."""
    candidate_names: list[str] = []
    for mapping in (samples.nonlinear, samples.linear):
        if name in mapping:
            candidate_names.append(name)  # noqa: PERF401

    suffix = f".{name}"
    for mapping in (samples.nonlinear, samples.linear):
        for key in mapping:
            if key.endswith(suffix):
                candidate_names.append(key)  # noqa: PERF401

    candidate_names = list(dict.fromkeys(candidate_names))
    if not candidate_names:
        msg = f"Could not find sample values for extension parameter {name!r}"
        raise KeyError(msg)
    if len(candidate_names) > 1:
        msg = (
            f"Ambiguous extension parameter {name!r}; matched sample keys "
            f"{tuple(candidate_names)}"
        )
        raise ValueError(msg)

    key = candidate_names[0]
    qty = samples.nonlinear[key] if key in samples.nonlinear else samples.linear[key]
    return float(np.asarray(qty.value)[sample_index])


def _get_extension_sample_values(
    samples: Any,
    extensions: tuple[Any, ...],
    sample_index: int,
) -> dict[str, float]:
    """Collect scalar extension parameter values for one posterior sample."""
    param_names = tuple(
        dict.fromkeys(p.name for ext in extensions for p in ext.extra_params())
    )
    return {
        name: _get_sample_scalar_value(samples, name, sample_index)
        for name in param_names
    }


def _gp_plot_signal(
    ext: Any,
    hp: dict[str, float],
    residuals: Any,
    data_err: Any,
    t_grid: Any,
    data_times: Any,
) -> Any:
    """Return the GP conditional mean used to overlay time-domain RV curves."""
    if tinygp is None:
        msg = "tinygp is required for GP plotting support"
        raise ImportError(msg)

    kernel = ext.kernel_builder(hp)
    gp = tinygp.GaussianProcess(kernel, data_times, diag=data_err**2)
    _, cond = gp.condition(residuals, t_grid)
    return cond.loc


def _plot_extension_extra_noise(
    ext: Any,
    hp: dict[str, float],
    data_err: Any,
) -> Any | None:
    """Private plotting adapter for extension-driven error-bar widening.

    Keep plot-specific behavior out of ``AbstractExtension``. If more
    extensions need custom plotting behavior later, replace this type-dispatch
    block with an optional plotting capability/protocol instead of adding plot
    hooks back onto the base extension API.
    """
    from .extensions.jitter import Jitter  # noqa: PLC0415

    if isinstance(ext, Jitter):
        return jnp.asarray(hp["jitter"])
    return None


def _plot_extension_rv_signal(
    ext: Any,
    hp: dict[str, float],
    residuals: Any,
    data_err: Any,
    t_grid: Any,
    data_times: Any,
) -> Any | None:
    """Private plotting adapter for extension-driven RV curve adjustments."""
    from .extensions.gp import GP  # noqa: PLC0415

    if isinstance(ext, GP):
        return _gp_plot_signal(ext, hp, residuals, data_err, t_grid, data_times)
    return None


def _component_linear_param(samples: Any, comp_name: str, param: str, i: int) -> Any:
    """Return sample *i* of a linear parameter, preferring the namespaced key.

    For SB2 JointModel samples the linear params are stored as
    ``"primary.rv_semiamp"`` / ``"secondary.rv_semiamp"`` etc. to avoid
    collisions. For plain RVModel samples the bare key ``"rv_semiamp"`` is used.
    This helper tries the qualified key first and falls back to the bare key.
    """
    qualified = f"{comp_name}.{param}"
    key = qualified if qualified in samples.linear else param
    return samples[key][i]


# --- end of hacky extension-specific plotting helpers ---


# These are the main public API plotting functions:


def plot_rv(  # noqa: C901 -- plotting code is inherently complex
    samples: Samples,
    data: RVData | SourceData | SystemData | None = None,
    extensions: tuple[Any, ...] = (),
    *,
    n_samples: int | None = 128,
    relative_to_t_ref: bool = False,
    relative_to_median_v_sys: bool = False,
    phase_fold_median: bool = False,
    apply_median_offsets: bool = True,
    plot_kwargs: dict[str, Any] | None = None,
    data_plot_kwargs: dict[str, Any] | None = None,
    extra_err_plot_kwargs: dict[str, Any] | None = None,
    color_cycler: Any | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot RV curves computed from (posterior) samples over data.

    Draws ``n_samples`` posterior Keplerian RV curves sampled from *samples*. When
    *extensions* are provided, each curve is augmented by the extension contributions
    (e.g. GP conditional mean) and error bars are widened by any extra noise terms (e.g.
    jitter).

    Parameters
    ----------
    samples : Samples
        Posterior samples from :class:`~harv.samplers.RejectionSampler` or
        :class:`~harv.samplers.NumpyroSampler`.
    data : RVData or SourceData or SystemData, optional
        Observed RV data to overplot.  When ``None``, only orbit model curves are
        drawn (no data points, no instrument-colour cycling).
    extensions : tuple of AbstractExtension, optional
        Extensions used during sampling. Plotting has private built-in support
        for GP conditional-mean overlays and jitter-driven error-bar widening.
        Default: no extensions.
    n_samples : int | None, optional
        Number of posterior curves to draw.  Set to None to draw all samples.  Default:
        128.
    relative_to_t_ref : bool, optional
        Whether to plot time relative to the reference epoch (t_ref) of the data.
    relative_to_median_v_sys : bool, optional
        Whether to shift all curves by the median systemic velocity (v_sys) of the
        samples, so that the curves show only the relative RV variations. Only applies
        when a "v_sys" parameter is present in the samples. Default: False.
    phase_fold_median : bool, optional
        If ``True``, fold data and model to orbital phase using the sample closest to
        the median period. Phase zero is set to that sample's ``t_peri`` value. Only
        that single reference orbit curve is drawn — plotting multiple samples on a
        phase axis defined by one period is misleading when the posterior has period
        spread. Any GP extension contributions are also suppressed when phase folding.
        Default: ``False``.
    apply_median_offsets : bool, optional
        Shift non-reference instrument data by the posterior median offset so
        all instruments land in the reference frame.  Only applies when a
        :class:`~harv.extensions.MultiSurveyOffset` extension is present.
        Default: ``True``.
    plot_kwargs : dict, optional
        Style overrides for orbit model curves.
    data_plot_kwargs : dict, optional
        Style overrides for data points.
    extra_err_plot_kwargs : dict, optional
        Style overrides for the widened error bars drawn when a jitter extension
        is present.  Keys override the defaults (marker="", ecolor="#6b2828",
        alpha=0.5).
    color_cycler : matplotlib cycler, optional
        Colour cycler used to assign distinct colours to each instrument's data
        points.  When ``None`` (default) the current ``axes.prop_cycle`` from
        ``matplotlib.rcParams`` is used.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  If ``None`` (default), a new figure and axes are
        created and the axes object is returned.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots`` when *ax* is ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes plotted to.

    Raises
    ------
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> ax = plot_rv(samples, rv_data)  # doctest: +SKIP
    >>> ax = plot_rv(samples, rv_data, extensions=(jitter, gp))  # doctest: +SKIP
    >>> ax = plot_rv(samples, rv_data, phase_fold_median=True)  # doctest: +SKIP
    """
    if plt is None:
        msg = "matplotlib is required for plot_rv."
        raise ImportError(msg)

    if phase_fold_median and extensions:
        from .extensions.gp import GP  # noqa: PLC0415

        if any(isinstance(ext, GP) for ext in extensions):
            warnings.warn(
                "phase_fold_median=True suppresses GP contributions from model curves "
                "but the data still contains GP variance, so the orbit will not "
                "overlay the data. Consider plotting without phase_fold_median "
                "when GP extensions are present.",
                UserWarning,
                stacklevel=2,
            )

    if ax is None:
        _, ax = plt.subplots(**kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    if data_plot_kwargs is None:
        data_plot_kwargs = {}
    if extra_err_plot_kwargs is None:
        extra_err_plot_kwargs = {}

    n_draw = min(len(samples), n_samples) if n_samples is not None else len(samples)

    # plotting style defaults - first for orbits:
    plot_kwargs = mpl.cbook.normalize_kwargs(plot_kwargs, mpl.lines.Line2D)
    orbit_style = {**_DEFAULT_LINE_STYLE, **plot_kwargs}
    # alpha is set after draw_indices is known (see below)

    # for data, this is mainly passed through to plot_timeseries_errorbar, which handles
    # merging user config with defaults
    data_style = data_plot_kwargs.copy()

    # Collect per-instrument datasets
    if isinstance(data, SourceData | SystemData):
        rv_datasets: dict[str, RVData] = data.get_datasets_by_type(RVData)
        if len(rv_datasets) == 0:
            msg = "No RVData found in provided data."
            raise ValueError(msg)
    elif isinstance(data, RVData):
        rv_datasets = {"data": data}
    elif data is None:
        rv_datasets = {}
    else:
        msg = "data must be RVData, SourceData, or SystemData."
        raise ValueError(msg)

    # we pull the time unit off of the data - we could make this configurable?
    _data = next(iter(rv_datasets.values())) if rv_datasets else None
    time_unit = (
        str(_data.time.unit) if _data is not None else str(samples["period"].unit)
    )
    rv_unit = (
        str(_data.rv.unit) if _data is not None else str(samples["rv_semiamp"].unit)
    )

    # Extract median period and t_ref for phase-folding and plotting
    median_period = QTime.from_(jnp.median(samples["period"]))
    ref_idx = int(jnp.argmin(jnp.abs(samples["period"] - median_period)))
    ref_period = samples["period"][ref_idx]
    ref_t_peri = samples["t_peri"][ref_idx]

    # When phase-folding, only the reference sample defines the phase axis so
    # plotting other samples (at different periods) would be misleading.
    draw_indices = [ref_idx] if phase_fold_median else range(n_draw)
    orbit_style.setdefault("alpha", get_alpha(len(draw_indices)))

    t_ref = Q(
        samples.metadata.get("t_ref", 0.0),
        samples.metadata.get("t_ref_unit", time_unit),
    )

    # Per-instrument median offsets sourced directly from MultiSurveyOffset extensions
    # so that only genuine instrument offsets are applied, not trend or GP params.
    offset_names: set[str] = set()
    for ext in extensions:
        if isinstance(ext, MultiSurveyOffset):
            offset_names.update(ext.instrument_names)

    median_offsets: dict[str, Q] = {
        name: Q(float(np.median(np.asarray(samples.linear[name].value))), rv_unit)
        for name in offset_names
        if name in samples.linear
    }

    # Per-dataset median extra noise (e.g. jitter). Computed separately for each
    # dataset so that instruments with different n_times are handled correctly.
    extra_err_per_dataset: dict[str, Any] = {}
    if extensions and rv_datasets:
        for ds_name, ds in rv_datasets.items():
            ds_err = ds.rv_err
            per_sample = []
            for i in draw_indices:
                hp_i = _get_extension_sample_values(samples, extensions, i)
                extra_var_i = Q(jnp.zeros(ds.n_times), rv_unit) ** 2
                for ext in extensions:
                    val = _plot_extension_extra_noise(ext, hp_i, ds_err)
                    if val is not None:
                        extra_q = Q(jnp.broadcast_to(val, ds.n_times), rv_unit)
                        extra_var_i = extra_var_i + extra_q**2
                per_sample.append(jnp.sqrt(extra_var_i))
            if per_sample:
                extra_err_per_dataset[ds_name] = jnp.median(
                    jnp.stack(per_sample), axis=0
                )

    # If a user specified, a global shift:
    median_v0 = (
        jnp.median(samples["v_sys"])
        if relative_to_median_v_sys and "v_sys" in samples
        else Q(0.0, rv_unit)
    )

    # Color cycler for instruments
    if color_cycler is not None:
        colors = list(color_cycler.by_key()["color"])
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Plot the data:
    for color_idx, (instr_name, rv_data) in enumerate(rv_datasets.items()):
        rv_obs = rv_data.rv
        rv_err = rv_data.rv_err

        if apply_median_offsets and instr_name in median_offsets:
            rv_obs = rv_obs - median_offsets[instr_name]

        instr_style = data_style.copy()
        if "color" not in instr_style:
            instr_style["color"] = colors[color_idx % len(colors)]
        label = instr_name if len(rv_datasets) > 1 else "data"
        instr_style.setdefault("label", label)

        phase_fold = ref_period if phase_fold_median else None
        phase_zero = ref_t_peri if phase_fold_median else t_ref

        plot_timeseries_errorbar(
            rv_data.time,
            rv_obs - median_v0,
            rv_err,
            time_unit=time_unit,
            obs_unit=rv_unit,
            t_ref=phase_zero,
            relative_to_t_ref=relative_to_t_ref,
            phase_fold=phase_fold,
            ax=ax,
            **instr_style,
        )

        # Widened error bars showing median extra noise (e.g. jitter)
        median_extra_err = extra_err_per_dataset.get(instr_name)
        if median_extra_err is not None:
            wide_style: dict[str, Any] = {
                "marker": "",
                "ecolor": "#C72222",  # red to stand out
                "alpha": 0.65,
                "zorder": data_style.get("zorder", 1) - 1,
                **extra_err_plot_kwargs,
            }
            rv_err_wide = jnp.sqrt(rv_err**2 + median_extra_err**2)
            plot_timeseries_errorbar(
                rv_data.time,
                rv_obs - median_v0,
                rv_err_wide,
                time_unit=time_unit,
                obs_unit=rv_unit,
                t_ref=phase_zero,
                relative_to_t_ref=relative_to_t_ref,
                phase_fold=phase_fold,
                ax=ax,
                **wide_style,
            )

    # Plot the orbit curves:
    phase_grid = jnp.linspace(0.0, 1.0, 1024)  # should this be customizable?
    if phase_fold_median:
        # Common reference-time grid defined by the chosen reference sample.
        t_grid = ref_t_peri + Q(phase_grid, "") * ref_period

        # the x values to plot below
        x_plot = phase_grid

        ax_set_info = {
            "xlabel": "phase",
            "ylabel": f"RV [{rv_unit}]",
            "xlim": (0.0, 1.0),
            "title": "Phase-folded radial velocity data and posterior orbits",
        }

    else:
        # don't phase fold:
        if rv_datasets:
            all_times = QTime.from_(
                jnp.concatenate([rv_data.time for rv_data in rv_datasets.values()])
            )
            t_grid = get_t_grid(all_times, median_period)
        else:
            t_grid = ref_t_peri + Q(phase_grid, "") * median_period

        x_plot = ustrip(time_unit, t_grid)

        if relative_to_t_ref:
            x_plot = x_plot - ustrip(time_unit, t_ref)

        ax_set_info = {
            "xlabel": f"time [{time_unit}]",
            "ylabel": f"RV [{rv_unit}]",
            "title": "Radial velocity data and posterior orbits",
        }

    # When data is present, draw one set of orbit curves per instrument/component so
    # that SB2 secondaries (with their own rv_semiamp) are rendered correctly and GP
    # residuals are computed against the matching dataset.
    if rv_datasets:
        for instr_name, _rv_data in rv_datasets.items():
            for i in draw_indices:
                sample_data: dict[str, Any] = {
                    "period": samples["period"][i],
                    "eccentricity": samples["eccentricity"][i],
                    "t_peri": samples["t_peri"][i],
                    "arg_peri": samples["arg_peri"][i],
                    "rv_semiamp": _component_linear_param(
                        samples, instr_name, "rv_semiamp", i
                    ),
                    "v_sys": _component_linear_param(samples, instr_name, "v_sys", i),
                }
                rv_model = rv_at_times(t_grid, **sample_data)

                # Extension contributions (e.g. GP conditional mean) computed
                # against this component's data, not the first dataset.
                if extensions and not phase_fold_median:
                    rv_at_data = rv_at_times(
                        _rv_data.time,
                        sample_data["period"],
                        sample_data["eccentricity"],
                        sample_data["t_peri"],
                        sample_data["arg_peri"],
                        sample_data["rv_semiamp"],
                        sample_data["v_sys"],
                    )
                    residuals = jnp.asarray(
                        ustrip(rv_unit, _rv_data.rv) - ustrip(rv_unit, rv_at_data)
                    )
                    err_data_raw = ustrip(rv_unit, _rv_data.rv_err)
                    t_data_raw = ustrip(time_unit, _rv_data.time)
                    hp_i = _get_extension_sample_values(samples, extensions, i)
                    for ext in extensions:
                        contrib = _plot_extension_rv_signal(
                            ext,
                            hp_i,
                            residuals,
                            err_data_raw,
                            ustrip(time_unit, t_grid),
                            t_data_raw,
                        )
                        if contrib is not None:
                            rv_model = rv_model + Q(contrib, rv_unit)

                ax.plot(x_plot, ustrip(rv_unit, rv_model - median_v0), **orbit_style)
    else:
        # data=None: draw orbit curves using bare parameter keys (no component context)
        for i in draw_indices:
            sample_data: dict[str, Any] = {
                "period": samples["period"][i],
                "eccentricity": samples["eccentricity"][i],
                "t_peri": samples["t_peri"][i],
                "arg_peri": samples["arg_peri"][i],
                "rv_semiamp": samples["rv_semiamp"][i],
                "v_sys": samples["v_sys"][i],
            }
            rv_model = rv_at_times(t_grid, **sample_data)
            ax.plot(x_plot, ustrip(rv_unit, rv_model - median_v0), **orbit_style)

    ax.legend(loc="best")
    ax.set(**ax_set_info)

    return ax


def plot_gaia_sky_orbit(
    orbit_params: dict[str, Any],
    data: GaiaAstrometryData | None = None,
    *,
    n_grid: int = 500,
    errorbar_scale: float = 1.0,
    plot_kwargs: dict[str, Any] | None = None,
    data_plot_kwargs: dict[str, Any] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a single astrometric orbit ellipse on the sky.

    Draws the photocentre orbit projected onto the sky-plane (``ΔRA`` vs ``ΔDec``)
    for one set of orbital parameters.  When *data* is provided, each Gaia epoch
    is rendered as a short line segment in the scan direction at the model-predicted
    photocentre offset, with half-length equal to the along-scan measurement
    uncertainty (scaled by *errorbar_scale*).  This shows the 1-D constraint each
    epoch contributes.

    Parameters
    ----------
    orbit_params : dict
        Orbital parameters for a single sample.  Required keys: ``"period"``,
        ``"eccentricity"``, ``"t_peri"``, ``"arg_peri"``, ``"cos_i"``,
        ``"lon_asc_node"``, ``"semi_major_axis"``.  ``"t_peri"`` should be the
        absolute periastron time (i.e. ``t_ref + phase_peri * period``).
    data : GaiaAstrometryData, optional
        Gaia epoch astrometry data.  When ``None``, only the model ellipse is
        drawn.
    n_grid : int, optional
        Number of phase points used to draw the smooth orbit curve.  Default: 500.
    errorbar_scale : float, optional
        Scale factor applied to the half-length of each scan-direction line
        segment (default 1.0 = 1-sigma).
    plot_kwargs : dict, optional
        Style overrides for the orbit curve (forwarded to ``ax.plot``).
    data_plot_kwargs : dict, optional
        Style overrides for the per-epoch scan-direction segments (forwarded to
        ``ax.plot``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  If ``None`` (default), a new figure is created and
        returned.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots`` when *ax* is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure if *ax* was ``None``, else ``None``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> fig = plot_gaia_sky_orbit(orbit_params, data=gaia_data)  # doctest: +SKIP
    """
    if plt is None:
        msg = "matplotlib is required for plot_gaia_sky_orbit."
        raise ImportError(msg)

    return_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    if data_plot_kwargs is None:
        data_plot_kwargs = {}

    period = orbit_params["period"]
    eccentricity = orbit_params["eccentricity"]
    t_peri = orbit_params["t_peri"]
    arg_peri = orbit_params["arg_peri"]
    cos_i = orbit_params["cos_i"]
    lon_asc_node = orbit_params["lon_asc_node"]
    sma = orbit_params["semi_major_axis"]

    sma_unit = str(sma.unit)

    # Smooth orbit curve over one full period, anchored at periastron.
    phi_grid = np.linspace(0.0, 1.0, n_grid)
    times_grid = t_peri + Q(phi_grid, "") * period
    delta_ra_grid, delta_dec_grid = astrometric_orbit_at_times(
        times_grid,
        period,
        eccentricity,
        t_peri,
        arg_peri,
        cos_i,
        lon_asc_node,
        sma,
    )

    orbit_style = {**_DEFAULT_LINE_STYLE, "color": "#555555", **plot_kwargs}
    orbit_style.setdefault("rasterized", True)
    ax.plot(
        np.asarray(ustrip(sma_unit, delta_ra_grid)),
        np.asarray(ustrip(sma_unit, delta_dec_grid)),
        **orbit_style,
    )

    if data is not None:
        # Model-predicted photocentre offsets at each observation epoch
        delta_ra_e, delta_dec_e = astrometric_orbit_at_times(
            data.time,
            period,
            eccentricity,
            t_peri,
            arg_peri,
            cos_i,
            lon_asc_node,
            sma,
        )
        delta_ra_e_v = np.asarray(ustrip(sma_unit, delta_ra_e))
        delta_dec_e_v = np.asarray(ustrip(sma_unit, delta_dec_e))

        # Scan direction unit vector in (ΔRA, ΔDec) plane: (sin ψ, cos ψ).
        # This matches the LPC convention used in the model design matrix.
        psi = np.asarray(ustrip("rad", data.scan_angle))
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        al_err = np.asarray(ustrip(sma_unit, data.al_position_err))
        half_len = errorbar_scale * al_err

        # Per-epoch scan-direction segments centred on the model prediction.
        seg_x = np.stack(
            [delta_ra_e_v - half_len * sin_psi, delta_ra_e_v + half_len * sin_psi],
            axis=1,
        )
        seg_y = np.stack(
            [delta_dec_e_v - half_len * cos_psi, delta_dec_e_v + half_len * cos_psi],
            axis=1,
        )

        seg_style = {
            "color": "k",
            "linewidth": 1.0,
            "alpha": 0.7,
            **data_plot_kwargs,
        }
        for k in range(seg_x.shape[0]):
            ax.plot(seg_x[k], seg_y[k], **seg_style)

    # Central star marker
    ax.plot(0, 0, marker="*", color="goldenrod", markersize=10, zorder=20)

    ax.set_xlabel(rf"$\Delta\alpha^*$ [{sma_unit}]")
    ax.set_ylabel(rf"$\Delta\delta$ [{sma_unit}]")
    ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.3)

    if return_fig:
        fig.tight_layout()
        return fig
    return None


def plot_gaia_astrometry(  # noqa: C901 -- plotting code is inherently complex
    samples: Samples,
    data: GaiaAstrometryData,
    extensions: tuple[Any, ...] = (),  # reserved for parity with plot_rv
    *,
    n_samples: int | None = 128,
    phase_fold_median: bool = False,
    plot_kwargs: dict[str, Any] | None = None,
    data_plot_kwargs: dict[str, Any] | None = None,
    sky_orbit_kwargs: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (10, 5),
    axes: tuple[Any, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot Gaia epoch-astrometry data with posterior orbit overlays.

    Produces a two-panel figure:

    - **Panel 1**: along-scan position vs time (or orbital phase if
      *phase_fold_median* is true) with multi-sample posterior orbit overlays.
      Median proper motion and zero-point offsets are subtracted from the data
      so that the parallax + orbital signal is visible.  When phase folding,
      the median parallax contribution is also subtracted (parallax has annual
      period and would smear when folded at the orbital period); only the
      single reference (median-period) sample is drawn.
    - **Panel 2**: on-sky photocentre orbital ellipse (delegated to
      :func:`plot_gaia_sky_orbit`) for the median-period sample, with each
      Gaia epoch shown as a scan-direction segment at the model-predicted
      photocentre offset.

    Parameters
    ----------
    samples : Samples
        Posterior samples from a Gaia astrometry or joint model.
    data : GaiaAstrometryData
        The data conditioned on by the model.  Required (panel 1 needs the
        scan angles and parallax factors).
    extensions : tuple of AbstractExtension, optional
        Currently unused; reserved for parity with :func:`plot_rv`.
    n_samples : int or None, optional
        Number of posterior orbit overlays to draw on panel 1.  Set to ``None``
        to draw every sample.  Default: 128.
    phase_fold_median : bool, optional
        If ``True``, fold panel 1 to orbital phase using the sample closest to
        the median period.  Phase zero is set to that sample's ``t_peri``.
        Only the reference orbit curve is drawn (multiple samples on a phase
        axis defined by one period would be misleading), and the median
        parallax contribution is auto-subtracted from the data.
        Default: ``False``.
    plot_kwargs : dict, optional
        Style overrides for the panel-1 orbit-model lines.
    data_plot_kwargs : dict, optional
        Style overrides for the panel-1 data error bars.
    sky_orbit_kwargs : dict, optional
        Forwarded to :func:`plot_gaia_sky_orbit` for panel 2.
    figsize : tuple, optional
        Figure size when *axes* is ``None``.  Default: ``(10, 5)``.
    axes : (matplotlib.axes.Axes, matplotlib.axes.Axes), optional
        Two axes to draw into.  If ``None`` (default), a new 1x2 figure is
        created and returned.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots`` when *axes* is ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The figure if *axes* was ``None``, else ``None``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> fig = plot_gaia_astrometry(samples, data=gaia_data)  # doctest: +SKIP
    >>> fig = plot_gaia_astrometry(  # doctest: +SKIP
    ...     samples, data=gaia_data, phase_fold_median=True
    ... )
    """
    if plt is None:
        msg = "matplotlib is required for plot_gaia_astrometry."
        raise ImportError(msg)

    return_fig = axes is None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, **kwargs)
    ax_t, ax_sky = axes

    if plot_kwargs is None:
        plot_kwargs = {}
    if data_plot_kwargs is None:
        data_plot_kwargs = {}
    if sky_orbit_kwargs is None:
        sky_orbit_kwargs = {}

    n_draw = min(len(samples), n_samples) if n_samples is not None else len(samples)

    plot_kwargs = mpl.cbook.normalize_kwargs(plot_kwargs, mpl.lines.Line2D)
    orbit_style = {**_DEFAULT_LINE_STYLE, "color": "#555555", **plot_kwargs}

    time_unit = str(data.time.unit)
    obs_unit = str(data.al_position.unit)

    # Reference sample (closest to median period) — defines phase folding +
    # which sample to draw on panel 2.
    median_period = jnp.median(samples["period"])
    ref_idx = int(jnp.argmin(jnp.abs(samples["period"] - median_period)))

    # When phase-folding, only the reference sample is drawn (matches plot_rv).
    draw_indices = [ref_idx] if phase_fold_median else range(n_draw)
    orbit_style.setdefault("alpha", get_alpha(len(draw_indices)))

    # Median linear params for the data subtraction, returned in the requested unit.
    def _median_in(name: str, unit: str) -> float:
        if name not in samples.linear:
            msg = f"Required linear parameter {name!r} is missing from samples"
            raise KeyError(msg)
        qty = samples.linear[name]
        med_val = float(np.median(np.asarray(qty.value)))
        med_q = Q(med_val, str(qty.unit))
        return float(ustrip(unit, med_q))

    ra0_v = _median_in("ra0", obs_unit)
    dec0_v = _median_in("dec0", obs_unit)
    pmra_v = _median_in("pmra", f"{obs_unit}/yr")
    pmdec_v = _median_in("pmdec", f"{obs_unit}/yr")
    parallax_v = _median_in("parallax", obs_unit)

    # Per-epoch geometric quantities
    psi = ustrip("rad", data.scan_angle)
    sin_psi = np.asarray(jnp.sin(psi))
    cos_psi = np.asarray(jnp.cos(psi))
    parallax_factor = np.asarray(data.parallax_factor)
    dt_yr = np.asarray(ustrip("yr", data.time - data.t_ref))

    linear_subtract = (
        ra0_v * sin_psi
        + dec0_v * cos_psi
        + (pmra_v * dt_yr) * sin_psi
        + (pmdec_v * dt_yr) * cos_psi
    )
    if phase_fold_median:
        # parallax wobble has annual period; fold at orbital period would smear it.
        linear_subtract = linear_subtract + parallax_v * parallax_factor

    al_data = np.asarray(ustrip(obs_unit, data.al_position)) - linear_subtract
    al_err = np.asarray(ustrip(obs_unit, data.al_position_err))

    # Data x-axis: time (or phase if folding)
    ref_t_peri = samples["t_peri"][ref_idx]
    ref_period = samples["period"][ref_idx]
    if phase_fold_median:
        x_data = (
            (
                np.asarray(ustrip(time_unit, data.time))
                - float(ustrip(time_unit, ref_t_peri))
            )
            / float(ustrip(time_unit, ref_period))
        ) % 1.0
        xlabel = "orbital phase"
    else:
        x_data = np.asarray(ustrip(time_unit, data.time))
        xlabel = f"time [{time_unit}]"

    # Plot data (sorted by x for a clean ordering, but errorbar doesn't need that)
    data_style = {**_DEFAULT_ERRORBAR_STYLE, **data_plot_kwargs}
    ax_t.errorbar(x_data, al_data, yerr=al_err, **data_style)

    # Sort epoch order for the per-sample model lines (so polylines look sensible)
    # TODO: don't plot the orbit only at the data epochs, but use a finer grid of times
    # like in plot_rv. Oh but maybe we can't do that easily because the parallax factor
    # is only defined at the data epochs? Hmm. Maybe show residuals instead then?
    order = np.argsort(x_data)
    x_sorted = x_data[order]
    sin_psi_s = sin_psi[order]
    cos_psi_s = cos_psi[order]
    parallax_factor_s = parallax_factor[order]
    times_sorted = data.time[order]

    # Per-sample model lines through the data epochs
    for i in draw_indices:
        period_i = samples["period"][i]
        ecc_i = samples["eccentricity"][i]
        t_peri_i = samples["t_peri"][i]
        arg_peri_i = samples["arg_peri"][i]
        cos_i_i = samples["cos_i"][i]
        lon_asc_i = samples["lon_asc_node"][i]
        sma_i = samples["semi_major_axis"][i]
        parallax_i = samples["parallax"][i]

        delta_ra_i, delta_dec_i = astrometric_orbit_at_times(
            times_sorted,
            period_i,
            ecc_i,
            t_peri_i,
            arg_peri_i,
            cos_i_i,
            lon_asc_i,
            sma_i,
        )
        al_orbit_i = (
            np.asarray(ustrip(obs_unit, delta_ra_i)) * sin_psi_s
            + np.asarray(ustrip(obs_unit, delta_dec_i)) * cos_psi_s
        )
        if phase_fold_median:
            al_model_i = al_orbit_i
        else:
            parallax_i_v = float(ustrip(obs_unit, parallax_i))
            al_model_i = al_orbit_i + parallax_i_v * parallax_factor_s

        ax_t.plot(x_sorted, al_model_i, **orbit_style)

    ax_t.set_xlabel(xlabel)
    ax_t.set_ylabel(f"AL position $-$ linear model [{obs_unit}]")
    ax_t.set_title("Along-scan vs " + ("phase" if phase_fold_median else "time"))

    # Panel 2: delegate to plot_gaia_sky_orbit using the reference sample.
    ref_orbit_params = {
        "period": samples["period"][ref_idx],
        "eccentricity": samples["eccentricity"][ref_idx],
        "t_peri": samples["t_peri"][ref_idx],
        "arg_peri": samples["arg_peri"][ref_idx],
        "cos_i": samples["cos_i"][ref_idx],
        "lon_asc_node": samples["lon_asc_node"][ref_idx],
        "semi_major_axis": samples["semi_major_axis"][ref_idx],
    }
    plot_gaia_sky_orbit(ref_orbit_params, data=data, ax=ax_sky, **sky_orbit_kwargs)
    ax_sky.set_title("Sky-projected orbit (median sample)")

    if return_fig:
        fig.tight_layout()
        return fig
    return None
