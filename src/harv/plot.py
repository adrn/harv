"""Plotting utilities."""

__all__ = (
    "get_t_grid",
    "plot_gaia_astrometry",
    "plot_gaia_sky_orbit",
    "plot_rv",
)

import warnings
from typing import Any

import equinox as eqx
import jax
import numpy as np
import quaxed.numpy as jnp
from unxt import Q, ustrip
from unxt.quantity import AllowValue

from harv.custom_types import BatchQTime, NQAny, NTime, ScalarQTime
from harv.data import GaiaAstrometryData, RVData, SourceData, SystemData
from harv.models.extensions.multi_survey import MultiSurveyOffset
from harv.samplers import Samples

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt: Any = None


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
    min_t_grid: int | None = None,
) -> NTime:
    """Dense time grid spanning the observation baseline with a small buffer.

    Generates a regular grid of times for plotting model orbits over data. The grid
    resolution adapts to the period so that fast orbits are well-resolved while
    long-period orbits don't create excessive grids.

    Parameters
    ----------
    times
        Observation times.
    period
        Orbital period (scalar).  Used to set the grid spacing as ``period /
        n_points_per_period``.
    span_buffer_factor
        Fractional buffer added to each side of the observation baseline. Default: 0.1
        (10% on each side).
    n_points_per_period
        Number of grid points per orbital period.  Default: 256.
    max_t_grid
        Maximum number of grid points. Default: 1e6. Set to None to disable.
    min_t_grid
        Minimum number of grid points. Default: None. Set to None to disable.

    Returns
    -------
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

    buffer = max(span * span_buffer_factor, 0.5 * p_val)
    full = span + 2 * buffer  # buffered baseline actually plotted

    # Points to resolve the period across the buffered baseline, then clamp
    n_grid = int(np.ceil(full / dt)) + 1
    if max_t_grid is not None:
        n_grid = min(n_grid, max_t_grid)
    if min_t_grid is not None:
        n_grid = max(n_grid, min_t_grid)

    grid = np.linspace(t_min - buffer, t_max + buffer, n_grid)
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


def _component_sample_params(
    samples: "Samples",
    comp_model: Any,
    data: Any,
    namespace: str,
    i: int,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    """Like _assemble_sample_params, but resolves namespaced keys for joint models.

    For each param ``X`` declared by ``comp_model``, looks up
    ``f"{namespace}.{X}"`` in ``samples`` first, then falls back to the bare
    name ``X``.  Used to pull per-component values from a joint posterior
    (e.g. SB2 ``"primary.rv_semiamp"`` → ``"rv_semiamp"`` for the primary
    component).  Returns sample ``i`` (scalars), in the form ``comp_model.predict``
    / ``comp_model.predict_at_times`` expects.
    """
    base_nl_units = {
        p.name: p.unit for p in comp_model.parameterization.params() if not p.linear
    }
    linear_units = comp_model._linear_param_units(data)

    def _lookup(samples_dict: dict[str, Any], name: str) -> Any:
        qualified = f"{namespace}.{name}"
        return (
            samples_dict[qualified] if qualified in samples_dict else samples_dict[name]
        )

    nl_for_model: dict[str, Any] = {}
    for name in comp_model._all_nonlinear_names():
        if (
            name not in samples.nonlinear
            and f"{namespace}.{name}" not in samples.nonlinear
        ):
            continue
        value = _lookup(samples.nonlinear, name)[i]
        nl_for_model[name] = (
            value if base_nl_units.get(name, "") else ustrip(str(value.unit), value)
        )

    linear_stripped: dict[str, jax.Array] = {}
    for name in linear_units:
        if name not in samples.linear and f"{namespace}.{name}" not in samples.linear:
            continue
        value = _lookup(samples.linear, name)[i]
        linear_stripped[name] = jnp.asarray(ustrip(linear_units[name], value))

    return nl_for_model, linear_stripped


def _strip_multisurvey_offsets(model: Any) -> Any:
    """Return a copy of *model* with any MultiSurveyOffset extension removed.

    The offset extension's ``indicator_matrix`` is fixed to the original
    data's row count, so it cannot be evaluated on an arbitrary plotting time
    grid.  The smooth-curve overlay uses this stripped model;
    per-instrument median offsets are applied to data points separately.
    """
    new_exts = tuple(
        e for e in model.extensions if not isinstance(e, MultiSurveyOffset)
    )
    if len(new_exts) == len(model.extensions):
        return model
    return eqx.tree_at(lambda m: m.extensions, model, new_exts)


# These are the main public API plotting functions:


def plot_rv(  # noqa: C901 -- plotting code is inherently complex
    samples: Samples,
    data: RVData | SourceData | SystemData | None = None,
    model: Any = None,
    *,
    n_samples: int | None = 128,
    time_grid: BatchQTime | None = None,
    show_signal_components: bool = False,
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

    Draws ``n_samples`` posterior RV curves sampled from *samples*.  The
    predicted curve and error model are obtained by delegating to the *model*:

    - The smooth curve at the plotting grid uses
      :meth:`~harv.models.RVModel.predict_at_times`, which folds in every
      design-matrix extension (e.g.
      :class:`~harv.models.extensions.MonomialTrend`).
    - Error bars are widened using
      :meth:`~harv.models.component.AbstractComponentModel._full_obs_err`
      (jitter, GP diagonal).
    - GP structured noise is overlaid via
      :meth:`~harv.models.extensions.GP.conditional_mean`.

    :class:`~harv.models.extensions.MultiSurveyOffset` is subsetted out for
    the smooth-curve overlay (its ``indicator_matrix`` is bound to the
    original data's row count and cannot be evaluated on a plotting grid);
    per-instrument median offsets are still applied to data points so that
    instruments land in the reference frame.

    Parameters
    ----------
    samples
        Posterior samples from :class:`~harv.samplers.RejectionSampler` or
        :class:`~harv.samplers.NumpyroSampler`.
    data
        Observed RV data to overplot.  When ``None``, only orbit model curves are
        drawn (no data points, no instrument-colour cycling).
    model
        The RV component model (or :class:`~harv.models.JointModel` for SB2)
        whose extensions and parameterization define the prediction.  When
        ``None`` (default), a bare ``RVModel()`` is used — the previous
        no-extension behavior.
    n_samples
        Number of posterior curves to draw.  Set to None to draw all samples.  Default:
        128.
    time_grid
        Explicit time grid used to evaluate and plot the posterior orbit curves.
        When provided, this is used instead of the default phase grid or
        :func:`get_t_grid`. If ``phase_fold_median=True``, the supplied time grid
        is converted to phase using the reference sample's period and periastron time.
    show_signal_components
        Whether to plot the Keplerian signal and the combined extension-driven
        contribution as separate curves instead of plotting their sum. This
        decomposition view is only supported for time-domain RV plots with
        observed data. Default: ``False``.
    relative_to_t_ref
        Whether to plot time relative to the reference epoch (t_ref) of the data.
    relative_to_median_v_sys
        Whether to shift all curves by the median systemic velocity (v_sys) of the
        samples, so that the curves show only the relative RV variations. Only applies
        when a "v_sys" parameter is present in the samples. Default: False.
    phase_fold_median
        If ``True``, fold data and model to orbital phase using the sample closest to
        the median period. Phase zero is set to that sample's ``t_peri`` value. Only
        that single reference orbit curve is drawn — plotting multiple samples on a
        phase axis defined by one period is misleading when the posterior has period
        spread. When plot-aware extensions are present, the reference sample's
        extension contribution is subtracted from the data before folding so the
        Keplerian orbit overlays the phase-folded points. Default: ``False``.
    apply_median_offsets
        Shift non-reference instrument data by the posterior median offset so
        all instruments land in the reference frame.  Only applies when a
        :class:`~harv.models.extensions.MultiSurveyOffset` extension is present.
        Default: ``True``.
    plot_kwargs
        Style overrides for orbit model curves.
    data_plot_kwargs
        Style overrides for data points.
    extra_err_plot_kwargs
        Style overrides for the widened error bars drawn when a jitter extension
        is present.  Keys override the defaults (marker="", ecolor="#6b2828",
        alpha=0.5).
    color_cycler
        Colour cycler used to assign distinct colours to each instrument's data
        points.  When ``None`` (default) the current ``axes.prop_cycle`` from
        ``matplotlib.rcParams`` is used.
    ax
        Axes to draw into.  If ``None`` (default), a new figure and axes are
        created and the axes object is returned.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots`` when *ax* is ``None``.

    Returns
    -------
        The axes plotted to.

    Raises
    ------
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    >>> ax = plot_rv(samples, rv_data)  # doctest: +SKIP
    >>> ax = plot_rv(samples, rv_data, model=sampler.model)  # doctest: +SKIP
    >>> ax = plot_rv(samples, rv_data, phase_fold_median=True)  # doctest: +SKIP
    """
    if plt is None:
        msg = "matplotlib is required for plot_rv."
        raise ImportError(msg)

    if model is None:
        from harv.models.rv import RVModel as _RVModel  # noqa: PLC0415

        model = _RVModel()

    # JointModel?  Build a per-instrument component-model resolver.
    from harv.models.joint import JointModel as _JointModel  # noqa: PLC0415

    def _component_model_for(instr_name: str) -> Any:
        if isinstance(model, _JointModel) and instr_name in model.components:
            return model.components[instr_name]
        return model

    if show_signal_components and phase_fold_median:
        warnings.warn(
            "show_signal_components=True is only supported for time-domain RV "
            "plots and will be ignored when phase_fold_median=True.",
            UserWarning,
            stacklevel=2,
        )
        show_signal_components = False

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

    if show_signal_components and not rv_datasets:
        warnings.warn(
            "show_signal_components=True requires observed RV data so extension "
            "contributions can be decomposed and will be ignored when data is None.",
            UserWarning,
            stacklevel=2,
        )
        show_signal_components = False

    # we pull the time unit off of the data - we could make this configurable?
    _data = next(iter(rv_datasets.values())) if rv_datasets else None
    time_unit = (
        str(_data.time.unit) if _data is not None else str(samples["period"].unit)
    )
    rv_unit = (
        str(_data.rv.unit) if _data is not None else str(samples["rv_semiamp"].unit)
    )

    # Extract median period and t_ref for phase-folding and plotting
    median_period = Q["time"].from_(  # ty: ignore[unresolved-reference]
        jnp.median(samples["period"])
    )
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

    # Per-instrument median offsets sourced directly from any MultiSurveyOffset
    # extension on the model so that only genuine instrument offsets are
    # applied to data points, not trend or GP params.
    offset_names: set[str] = set()
    for ds_name in rv_datasets:
        for ext in _component_model_for(ds_name).extensions:
            if isinstance(ext, MultiSurveyOffset):
                offset_names.update(ext.instrument_names)

    median_offsets: dict[str, Q] = {
        name: Q(float(np.median(np.asarray(samples.linear[name].value))), rv_unit)
        for name in offset_names
        if name in samples.linear
    }

    # Per-dataset median effective error: delegate to the component model's
    # extension-modified covariance (jitter adds quadrature, GP adds full
    # off-diagonal structure — we take its diagonal for display).  Computed
    # across draw_indices and the median sigma per data point is used to draw
    # the widened error bars.
    extra_err_per_dataset: dict[str, Any] = {}
    if rv_datasets:
        for ds_name, ds in rv_datasets.items():
            comp_model = _component_model_for(ds_name)
            if not comp_model.extensions:
                continue
            ds_err = jnp.asarray(ustrip(rv_unit, ds.rv_err))
            per_sample_eff_err: list[Any] = []
            for i in draw_indices:
                nl_i, _lin_i = _component_sample_params(
                    samples, comp_model, ds, ds_name, i
                )
                cov = comp_model._full_obs_err(ds_err, nl_i, ds)
                eff = jnp.sqrt(cov) if cov.ndim == 1 else jnp.sqrt(jnp.diag(cov))
                per_sample_eff_err.append(eff)
            if per_sample_eff_err:
                # Only show the *extra* over the raw obs error (in quadrature)
                # so the widened bars are visually distinct from the originals.
                median_eff = jnp.median(jnp.stack(per_sample_eff_err), axis=0)
                extra_var = jnp.clip(median_eff**2 - ds_err**2, 0.0)
                extra_err_per_dataset[ds_name] = Q(jnp.sqrt(extra_var), rv_unit)

    # If a user specified, a global shift:
    median_v0 = (
        jnp.median(samples["v_sys"])
        if relative_to_median_v_sys and "v_sys" in samples
        else Q(0.0, rv_unit)
    )

    # Color cycler for instruments
    # TODO: will need to be smarter if number of instruments > number of colors
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

        comp_model_for_instr = _component_model_for(instr_name)
        if phase_fold_median and comp_model_for_instr.extensions:
            # Subtract the reference sample's structured extension contribution
            # (trend at data times + GP conditional mean at data times) so the
            # observed RV folds cleanly onto the Keplerian orbit overlay.
            nl_ref, lin_ref = _component_sample_params(
                samples, comp_model_for_instr, rv_data, instr_name, ref_idx
            )
            curve_model = _strip_multisurvey_offsets(comp_model_for_instr)
            # Keplerian + trend contribution at data times (no offsets,
            # no zero-point — those are part of the *model* prediction, not the
            # noise we want to subtract).
            from harv.models.extensions.gp import GP  # noqa: PLC0415

            # Build a "trend-only" model (drop GP and offsets, keep e.g.
            # MonomialTrend) so its predict_at_times yields the deterministic
            # extension contribution.
            trend_only_exts = tuple(
                e for e in curve_model.extensions if not isinstance(e, GP)
            )
            trend_only_model = eqx.tree_at(
                lambda m: m.extensions, curve_model, trend_only_exts
            )
            # Baseline Keplerian-only prediction (no extensions at all).
            kepler_only_model = eqx.tree_at(lambda m: m.extensions, curve_model, ())
            y_full = trend_only_model.predict_at_times(
                rv_data.time,
                nl_ref,
                lin_ref,
                t_ref=rv_data.t_ref,
            )
            y_kepler = kepler_only_model.predict_at_times(
                rv_data.time,
                nl_ref,
                lin_ref,
                t_ref=rv_data.t_ref,
            )
            trend_contrib = y_full - y_kepler  # bare jax array in rv_unit

            # GP conditional mean (if a GP extension is present), conditioned
            # on the full Keplerian + trend residuals.
            gp_contrib = jnp.zeros_like(trend_contrib)
            residuals_full = jnp.asarray(ustrip(rv_unit, rv_obs) - jnp.asarray(y_full))
            err_data_arr = jnp.asarray(ustrip(rv_unit, rv_err))
            for ext in comp_model_for_instr.extensions:
                if isinstance(ext, GP):
                    hp = _get_extension_sample_values(samples, (ext,), ref_idx)
                    t_unit_ext = ext.time_unit or time_unit
                    gp_contrib = gp_contrib + jnp.asarray(
                        ext.conditional_mean(
                            residuals_full,
                            jnp.asarray(ustrip(t_unit_ext, rv_data.time)),
                            jnp.asarray(ustrip(t_unit_ext, rv_data.time)),
                            err_data_arr,
                            hp,
                        )
                    )
            phase_fold_signal = Q(trend_contrib + gp_contrib, rv_unit)
            rv_obs = rv_obs - phase_fold_signal

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
        # Common reference-time grid defined by the chosen reference sample,
        # unless the user explicitly provides a plotting grid.
        if time_grid is None:
            t_grid = ref_t_peri + Q(phase_grid, "") * ref_period
            x_plot = phase_grid
        else:
            t_grid = time_grid
            x_plot = (
                (ustrip(time_unit, t_grid) - float(ustrip(time_unit, ref_t_peri)))
                / float(ustrip(time_unit, ref_period))
            ) % 1.0

        ax_set_info = {
            "xlabel": "phase",
            "ylabel": f"RV [{rv_unit}]",
            "xlim": (0.0, 1.0),
            "title": "Phase-folded radial velocity data and posterior orbits",
        }

    else:
        # don't phase fold:
        if time_grid is not None:
            t_grid = time_grid
        elif rv_datasets:
            all_times = Q["time"].from_(  # ty: ignore[unresolved-reference]
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

    # Build the orbit-curve overlays via model.predict_at_times so trend
    # contributions and any other design-matrix extension fold in automatically.
    # MultiSurveyOffset is subsetted out: its indicator_matrix is bound to the
    # original data's row count, and the per-instrument offsets are already
    # being applied to data points via `median_offsets` above.
    from harv.models.extensions.gp import GP  # noqa: PLC0415

    def _curve_for_sample(
        comp_model: Any,
        rv_data_ref: RVData,
        instr_name: str,
        i: int,
    ) -> tuple[jax.Array, jax.Array, bool]:
        """Return (kepler_rv_array, extension_rv_array, has_extension_signal).

        Both arrays are in rv_unit, bare jax arrays of shape len(t_grid).
        """
        nl_i, lin_i = _component_sample_params(
            samples, comp_model, rv_data_ref, instr_name, i
        )
        curve_model = _strip_multisurvey_offsets(comp_model)
        # Keplerian-only baseline (no design-matrix extensions).
        kepler_only = eqx.tree_at(lambda m: m.extensions, curve_model, ())
        y_kepler = kepler_only.predict_at_times(
            t_grid,
            nl_i,
            lin_i,
            t_ref=rv_data_ref.t_ref,
            obs_unit=rv_unit,
        )
        # Full design-matrix prediction (Keplerian + trend + any other
        # design-matrix extension).
        y_full = curve_model.predict_at_times(
            t_grid,
            nl_i,
            lin_i,
            t_ref=rv_data_ref.t_ref,
            obs_unit=rv_unit,
        )
        ext_curve = y_full - y_kepler

        # GP conditional-mean overlay (covariance extension, not design-matrix
        # — predicted on the grid by conditioning on data residuals).
        has_signal = bool(ext_curve.any())
        if not phase_fold_median:
            err_data_arr = jnp.asarray(ustrip(rv_unit, rv_data_ref.rv_err))
            for ext in comp_model.extensions:
                if isinstance(ext, GP):
                    # Residuals against the *full* deterministic prediction at
                    # the data times.
                    y_at_data = curve_model.predict(nl_i, lin_i, rv_data_ref)
                    residuals = jnp.asarray(
                        ustrip(rv_unit, rv_data_ref.rv) - jnp.asarray(y_at_data)
                    )
                    t_unit_ext = ext.time_unit or time_unit
                    hp = _get_extension_sample_values(samples, (ext,), i)
                    gp_grid = ext.conditional_mean(
                        residuals,
                        jnp.asarray(ustrip(t_unit_ext, rv_data_ref.time)),
                        jnp.asarray(ustrip(t_unit_ext, t_grid)),
                        err_data_arr,
                        hp,
                    )
                    ext_curve = ext_curve + jnp.asarray(gp_grid)
                    has_signal = True
        return jnp.asarray(y_kepler), jnp.asarray(ext_curve), has_signal

    if rv_datasets:
        # When data is present, draw one set of curves per instrument/component
        # so SB2 secondaries (with their own rv_semiamp) and per-component GP
        # extensions are rendered correctly.  Orbit color is offset by 1 from
        # the data color (so C0 data pairs with a C1 orbit overlay).
        for color_idx, (instr_name, _rv_data) in enumerate(rv_datasets.items()):
            comp_model_curve = _component_model_for(instr_name)
            total_style = orbit_style.copy()
            total_style["color"] = colors[(color_idx + 1) % len(colors)]
            kepler_style = orbit_style.copy()
            kepler_style["color"] = colors[(color_idx + 1) % len(colors)]
            extension_style = orbit_style.copy()
            extension_style["color"] = colors[(color_idx + 2) % len(colors)]
            extension_style.setdefault("linestyle", "--")

            for draw_idx, i in enumerate(draw_indices):
                y_kepler, y_ext, has_signal = _curve_for_sample(
                    comp_model_curve,
                    _rv_data,
                    instr_name,
                    i,
                )
                if show_signal_components:
                    kepler_plot_style = kepler_style.copy()
                    extension_plot_style = extension_style.copy()
                    if color_idx == 0 and draw_idx == 0:
                        kepler_plot_style.setdefault("label", "Keplerian")
                        extension_plot_style.setdefault("label", "Extensions")

                    ax.plot(
                        x_plot,
                        np.asarray(y_kepler)
                        - float(ustrip(AllowValue, rv_unit, median_v0)),
                        **kepler_plot_style,
                    )
                    if has_signal:
                        ax.plot(
                            x_plot,
                            np.asarray(y_ext),
                            **extension_plot_style,
                        )
                else:
                    y_model = y_kepler + y_ext
                    ax.plot(
                        x_plot,
                        np.asarray(y_model)
                        - float(ustrip(AllowValue, rv_unit, median_v0)),
                        **total_style,
                    )
    else:
        # data=None: no per-instrument context — use the model directly with
        # bare keys from samples, via a synthetic single-instrument RVData
        # required by predict_at_times.  No GP overlay (no data residuals to
        # condition on).
        from harv.models.rv import RVModel as _RVModel  # noqa: PLC0415

        # Build a synthetic single-instrument data shim to satisfy
        # _linear_param_units(data) when assembling lin values.
        dummy_data = RVData(
            time=Q(jnp.zeros(1), time_unit),
            rv=Q(jnp.zeros(1), rv_unit),
            rv_err=Q(jnp.ones(1), rv_unit),
            t_ref=t_ref,
        )
        comp_model_for_curve = model if isinstance(model, _RVModel) else _RVModel()
        curve_model = _strip_multisurvey_offsets(comp_model_for_curve)
        for i in draw_indices:
            nl_i, lin_i = _component_sample_params(
                samples, curve_model, dummy_data, "data", i
            )
            y_model = curve_model.predict_at_times(
                t_grid,
                nl_i,
                lin_i,
                t_ref=t_ref,
                obs_unit=rv_unit,
            )
            ax.plot(
                x_plot,
                np.asarray(y_model) - float(ustrip(AllowValue, rv_unit, median_v0)),
                **orbit_style,
            )

    ax.legend(loc="best")
    ax.set(**ax_set_info)

    return ax


def plot_gaia_sky_orbit(  # noqa: C901 -- plotting code is inherently complex
    model: Any,
    samples: Samples,
    *,
    data: GaiaAstrometryData | None = None,
    n_grid: int = 500,
    errorbar_scale: float = 1.0,
    plot_kwargs: dict[str, Any] | None = None,
    data_plot_kwargs: dict[str, Any] | None = None,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Plot a single astrometric orbit ellipse on the sky.

    Draws the photocenter orbit projected onto the sky-plane (``ΔRA`` vs ``ΔDec``)
    for one posterior sample.  When *data* is provided, each Gaia epoch is
    rendered as a short line segment in the scan direction at the model-
    predicted photocenter offset, with half-length equal to the along-scan
    measurement uncertainty (scaled by *errorbar_scale*).

    The orbit-only sky path (no PM, no parallax) is constructed by delegating
    to :meth:`~harv.models.GaiaAstrometryModel.predict_orbit_sky`, so this
    function automatically supports both Standard and Thiele-Innes
    parameterizations.

    Parameters
    ----------
    model
        The :class:`~harv.models.GaiaAstrometryModel` whose parameterization
        defines the orbit.
    samples
        Posterior samples containing exactly one sample.  Select beforehand
        with ``samples[i]`` or :meth:`Samples.map_sample`.
    data
        Gaia epoch astrometry data.  When ``None``, only the orbit ellipse is
        drawn (no per-epoch markers).
    n_grid
        Number of phase points used to draw the smooth orbit curve.  Default 500.
    errorbar_scale
        Scale factor on the half-length of each scan-direction line segment
        (default ``1.0`` = 1-sigma).
    plot_kwargs
        Style overrides for the orbit curve (forwarded to ``ax.plot``).
    data_plot_kwargs
        Style overrides for the per-epoch scan-direction segments.
    ax
        Axes to draw into.  ``None`` (default) creates a new figure.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots`` when *ax* is ``None``.

    Returns
    -------
        The figure if *ax* was ``None``, else ``None``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If *samples* does not contain exactly one posterior sample.
    """
    if plt is None:
        msg = "matplotlib is required for plot_gaia_sky_orbit."
        raise ImportError(msg)

    if len(samples) != 1:
        msg = (
            f"plot_gaia_sky_orbit expects exactly one posterior sample, but "
            f"got {len(samples)}. Select a single sample first, e.g. "
            "`samples[i]` or `samples.map_sample()`."
        )
        raise ValueError(msg)

    return_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    if plot_kwargs is None:
        plot_kwargs = {}
    if data_plot_kwargs is None:
        data_plot_kwargs = {}

    # Assemble (nl, lin) from the single sample.  Strip dimensioned nl to the
    # form the parameterization expects (period stays as Q[time]).  Linear
    # values: strip each to its own stored unit so the resulting (dRA, dDec)
    # land in that same unit.
    base_nl_units = {
        p.name: p.unit for p in model.parameterization.params() if not p.linear
    }
    nl = {
        name: samples[name][0]
        if base_nl_units.get(name, "")
        else jnp.asarray(ustrip(str(samples[name][0].unit), samples[name][0]))
        for name in samples.nonlinear
    }
    linear_unit_map: dict[str, str] = {
        name: str(samples[name][0].unit) for name in samples.linear
    }
    lin = {
        name: jnp.asarray(ustrip(linear_unit_map[name], samples[name][0]))
        for name in samples.linear
    }

    # Display unit: take from the orbit-amplitude linear param if present.
    if "semi_major_axis" in linear_unit_map:
        sma_unit = linear_unit_map["semi_major_axis"]
    elif "ti_A" in linear_unit_map:
        sma_unit = linear_unit_map["ti_A"]
    else:
        msg = (
            "plot_gaia_sky_orbit: samples must contain either 'semi_major_axis' "
            "(Standard) or 'ti_A' (Thiele-Innes) linear params."
        )
        raise ValueError(msg)

    period_q = nl["period"]  # Q[time]
    phase_peri_v = float(ustrip(AllowValue, "", samples["phase_peri"][0]))
    t_peri_q = phase_peri_v * period_q

    # Smooth orbit curve over one full period, anchored at periastron.
    phi_grid = np.linspace(0.0, 1.0, n_grid)
    times_grid = t_peri_q + Q(phi_grid, "") * period_q
    delta_ra_grid, delta_dec_grid = model.predict_orbit_sky(nl, lin, times_grid)

    orbit_style = {**_DEFAULT_LINE_STYLE, "color": "#555555", **plot_kwargs}
    orbit_style.setdefault("rasterized", True)
    ax.plot(
        np.asarray(delta_ra_grid),
        np.asarray(delta_dec_grid),
        **orbit_style,
    )

    if data is not None:
        # Model-predicted photocentre offsets at each observation epoch
        delta_ra_e, delta_dec_e = model.predict_orbit_sky(nl, lin, data.time)
        delta_ra_e_v = np.asarray(delta_ra_e)
        delta_dec_e_v = np.asarray(delta_dec_e)

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


def plot_gaia_astrometry(
    samples: Samples,
    data: GaiaAstrometryData,
    model: Any = None,
    *,
    data_plot_kwargs: dict[str, Any] | None = None,
    sky_orbit_kwargs: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (10, 5),
    axes: tuple[Any, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot the best-fit Gaia astrometry model for a single posterior sample.

    Produces a two-panel goodness-of-fit figure for one posterior sample:

    - **Panel 1 (left)**: on-sky photocentre orbital ellipse (delegated to
      :func:`plot_gaia_sky_orbit`), with each Gaia epoch shown as a
      scan-direction segment at the model-predicted photocentre offset.
    - **Panel 2 (right)**: along-scan position residual vs time — the observed
      ``al_position`` minus the *full* predicted model for the sample.  The
      prediction is built by delegating to the model
      (:meth:`~harv.models.GaiaAstrometryModel.predict`), which folds in every
      design-matrix extension (e.g.
      :class:`~harv.models.extensions.MonomialTrend` (astrometry=True),
      :class:`~harv.models.extensions.MultiSurveyOffset`).  Covariance-only
      extensions (jitter, GP) widen the residual error bars via
      :meth:`~harv.models.component.AbstractComponentModel._full_obs_err`;
      GP conditional-mean structured noise is additionally subtracted from the
      residual so only true noise remains.

    *samples* must contain exactly one posterior sample.  Select one beforehand
    with ``samples[i]`` or :meth:`~harv.samplers.Samples.map_sample`.

    Parameters
    ----------
    samples
        Posterior samples from a Gaia astrometry or joint model.  Must contain
        exactly one sample; otherwise a :class:`ValueError` is raised.
    data
        The data conditioned on by the model.  Required.
    model
        The :class:`~harv.models.GaiaAstrometryModel` whose extensions and
        parameterization define the prediction.  When ``None`` (default), a
        bare ``GaiaAstrometryModel()`` is used — equivalent to the previous
        no-extension behavior.
    data_plot_kwargs
        Style overrides for the panel-2 residual error bars.
    sky_orbit_kwargs
        Forwarded to :func:`plot_gaia_sky_orbit` for panel 1.
    figsize
        Figure size when *axes* is ``None``.  Default ``(10, 5)``.
    axes
        Two axes to draw into, ``(sky, residual)``.  ``None`` (default) creates
        a new figure.
    **kwargs
        Forwarded to ``matplotlib.pyplot.subplots`` when *axes* is ``None``.

    Returns
    -------
        The figure if *axes* was ``None``, else ``None``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If *samples* does not contain exactly one posterior sample.

    Examples
    --------
    >>> fig = plot_gaia_astrometry(samples[0], data=gaia_data)  # doctest: +SKIP
    >>> fig = plot_gaia_astrometry(  # doctest: +SKIP
    ...     samples.map_sample(), data=gaia_data, model=sampler.model
    ... )
    """
    if plt is None:
        msg = "matplotlib is required for plot_gaia_astrometry."
        raise ImportError(msg)

    if len(samples) != 1:
        msg = (
            f"plot_gaia_astrometry expects exactly one posterior sample, but "
            f"got {len(samples)}. Select a single sample first, e.g. "
            "`samples[i]` to pick one by index or `samples.map_sample()` for "
            "the maximum a posteriori sample."
        )
        raise ValueError(msg)

    if model is None:
        from harv.models.astrometry import (  # noqa: PLC0415
            GaiaAstrometryModel as _GaiaAstrometryModel,
        )

        model = _GaiaAstrometryModel()

    return_fig = axes is None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, **kwargs)
    ax_sky, ax_resid = axes

    if data_plot_kwargs is None:
        data_plot_kwargs = {}
    if sky_orbit_kwargs is None:
        sky_orbit_kwargs = {}

    obs_unit = str(data.al_position.unit)

    # --- Panel 2: along-scan residual vs time ---
    # Full prediction (orbital + PM + parallax + zero-point + every
    # design-matrix extension) is computed by the model's likelihood-path
    # primitives — single source of truth shared with `log_prob` / `chi_squared`.
    from harv.samplers.samples import _assemble_sample_params  # noqa: PLC0415

    nl, lin = _assemble_sample_params(samples, model, data, i=0)
    y_pred = np.asarray(model.predict(nl, lin, data))
    residual = np.asarray(ustrip(obs_unit, data.al_position)) - y_pred

    # GP conditional-mean overlay: structured noise from any GP extension is
    # predicted at the data times (conditioned on the current residual) and
    # subtracted, so the displayed residual reflects only the genuine
    # uncorrelated noise.
    from harv.models.extensions.gp import GP  # noqa: PLC0415

    err_obs = np.asarray(ustrip(obs_unit, data.al_position_err))
    for ext in model.extensions:
        if isinstance(ext, GP):
            t_unit = ext.time_unit or str(data.time.unit)
            t_data = jnp.asarray(ustrip(t_unit, data.time))
            hp = _get_extension_sample_values(samples, (ext,), 0)
            gp_mean = np.asarray(
                ext.conditional_mean(
                    jnp.asarray(residual),
                    t_data,
                    t_data,
                    jnp.asarray(err_obs),
                    hp,
                )
            )
            residual = residual - gp_mean

    # Extension-modified error bars (jitter + GP diagonal) via the same
    # covariance path the likelihood uses.
    cov = model._full_obs_err(jnp.asarray(err_obs), nl, data)
    err_eff = np.asarray(jnp.sqrt(cov) if cov.ndim == 1 else jnp.sqrt(jnp.diag(cov)))

    plot_timeseries_errorbar(
        data.time,
        Q(residual, obs_unit),
        Q(err_eff, obs_unit),
        obs_unit=obs_unit,
        ylabel=f"AL position residual [{obs_unit}]",
        ax=ax_resid,
        **data_plot_kwargs,
    )
    ax_resid.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_resid.set_title("Along-scan residual vs time")

    # --- Panel 1: sky-projected orbit (delegated) ---
    plot_gaia_sky_orbit(model, samples, data=data, ax=ax_sky, **sky_orbit_kwargs)
    ax_sky.set_title("Sky-projected orbit")

    if return_fig:
        fig.tight_layout()
        return fig
    return None
