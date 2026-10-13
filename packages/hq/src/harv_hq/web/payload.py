"""JSON payloads for the web viewer: catalog columns, filters, and source pages.

See ``packages/hq/docs/spec.md``, "Web viewer". Non-finite floats become
``None`` (JSON ``null``) everywhere, since JSON has no NaN.
"""

__all__ = (
    "MAX_CURVE_SAMPLES",
    "MAX_TIME_GRID",
    "column_info",
    "filter_mask",
    "jsonable",
    "source_payload",
)

import logging
import math
from typing import Any, cast

import jax
import numpy as np
import pyarrow as pa

# pyarrow.compute generates its functions at import, so ty cannot see them.
import pyarrow.compute as pc
from unxt import Q, ustrip

from harv.data import GaiaAstrometryData, RVData, SystemData
from harv.plot import _strip_multi_survey_offsets, get_time_grid
from harv.samplers import Samples
from harv.samplers.samples import _assemble_sample_params
from harv_hq.config import SB2_COMPONENTS
from harv_hq.prepare import PreparedSource
from harv_hq.run import STAGES, SourceResult
from harv_hq.summary_stats import _value_and_unit

logger = logging.getLogger(__name__)

MAX_CURVE_SAMPLES = 256
MAX_TIME_GRID = 4096
# Points along one period of the Gaia sky-plane orbit.
_SKY_GRID = 256
# Significant digits kept in curve values, the bulk of a source payload.
_CURVE_DIGITS = 6


def jsonable(values: Any) -> Any:
    """Values as JSON-ready Python objects, with non-finite floats as ``None``.

    Parameters
    ----------
    values
        A pyarrow array, a NumPy array (any shape), a dict, or a scalar.

    Returns
    -------
        Lists, dicts, and scalars.

    Examples
    --------
    >>> import numpy as np
    >>> jsonable(np.array([[1.0, np.nan], [np.inf, 2.0]]))
    [[1.0, None], [None, 2.0]]
    >>> jsonable({"a": float("nan"), "b": "x"})
    {'a': None, 'b': 'x'}
    """
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        if pa.types.is_binary(values.type) or pa.types.is_temporal(values.type):
            values = values.cast(pa.string())
        if pa.types.is_floating(values.type):
            values = pc.if_else(  # ty: ignore[unresolved-attribute]
                pc.is_finite(values),  # ty: ignore[unresolved-attribute]
                values,
                pa.scalar(None, values.type),
            )
        return values.to_pylist()
    if isinstance(values, dict):
        return {k: jsonable(v) for k, v in values.items()}
    if isinstance(values, float):
        return values if math.isfinite(values) else None
    if isinstance(values, bytes):
        return values.decode()
    if hasattr(values, "isoformat"):
        return values.isoformat()
    if isinstance(values, (np.ndarray, jax.Array)):
        array = np.asarray(values, dtype=float)
        out = array.astype(object)
        out[~np.isfinite(array)] = None
        return out.tolist()
    return values


def column_info(table: pa.Table, units: dict[str, str]) -> list[dict[str, Any]]:
    """Each column's name, dtype, unit, and whether it can be plotted."""
    return [
        {
            "name": field.name,
            "dtype": str(field.type),
            "unit": units.get(field.name, ""),
            "numeric": _numeric(field.type),
        }
        for field in table.schema
    ]


def _numeric(dtype: pa.DataType) -> bool:
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype)


def filter_mask(table: pa.Table, filter_text: str, status_text: str) -> pa.Array:
    """Rows passing the catalog filters (spec, "JSON API").

    Parameters
    ----------
    table
        The joined catalog and summary table.
    filter_text
        Comma-separated ``column:min:max`` terms, ANDed; an empty bound is
        open. Rows with a null in a filtered column are dropped.
    status_text
        Comma-separated ``rejection_status`` values to keep; empty keeps all.

    Returns
    -------
        A boolean mask over the rows.

    Raises
    ------
    ValueError
        For a malformed term, an unknown or non-numeric column, a bound that
        is not a number, or a status filter before ``hq summarize``.

    Examples
    --------
    >>> import pyarrow as pa
    >>> table = pa.table({"g": [10.0, 12.0, 14.0], "rejection_status": ["ok"] * 3})
    >>> filter_mask(table, "g:11:", "ok").to_pylist()
    [False, True, True]
    """
    mask = pa.array(np.ones(table.num_rows, dtype=bool))
    for term in filter(None, filter_text.split(",")):
        parts = term.rsplit(":", 2)
        if len(parts) != 3:
            msg = f"filter term {term!r} is not column:min:max"
            raise ValueError(msg)
        name, low, high = parts
        if name not in table.column_names or not _numeric(table[name].type):
            msg = f"filter column {name!r} is not a numeric column"
            raise ValueError(msg)
        for bound, compare in ((low, pc.greater_equal), (high, pc.less_equal)):  # ty: ignore[unresolved-attribute]
            if not bound:
                continue
            try:
                value = float(bound)
            except ValueError:
                msg = f"filter bound {bound!r} in {term!r} is not a number"
                raise ValueError(msg) from None
            mask = pc.and_(mask, compare(table[name], value))  # ty: ignore[unresolved-attribute]
    if statuses := [s for s in status_text.split(",") if s]:
        if "rejection_status" not in table.column_names:
            msg = "no rejection_status column yet; run hq summarize first"
            raise ValueError(msg)
        keep = pc.is_in(table["rejection_status"], value_set=pa.array(statuses))  # ty: ignore[unresolved-attribute]
        mask = pc.and_(mask, keep)  # ty: ignore[unresolved-attribute]
    return pc.fill_null(mask, fill_value=False)


def source_payload(result: SourceResult, model: Any) -> dict[str, Any]:
    """Everything the source page shows, as JSON-ready data.

    Parameters
    ----------
    result
        The source's :meth:`~harv_hq.Run.load_source` result.
    model
        The run's model (from ``make_setup``), for the orbit curves.

    Returns
    -------
        ``source_id``, ``kind``, ``data`` (arrays and units, per star for
        ``sb2``), ``summary``, and per stage its ``status``, ``error``,
        ``warnings``, and, when it has samples, up to
        :data:`MAX_CURVE_SAMPLES` of them with their curves.
    """
    data = result.data
    stages = {}
    for stage in STAGES:
        row = getattr(result, f"{stage}_row") or {}
        samples: Samples | None = getattr(result, stage)
        stages[stage] = {
            "status": getattr(result, f"{stage}_status"),
            "mcmc_status": row.get("mcmc_status"),
            "error": row.get("error", ""),
            "warnings": row.get("warnings", ""),
        }
        if samples is not None:
            weighted = stage == "rejection"
            shown = samples[_shown(samples, weighted=weighted)]
            stages[stage] |= _samples_payload(shown, data, model)
            if weighted:
                stages[stage]["weight"] = jsonable(np.asarray(shown.weight))
    summary = jsonable(result.summary)
    if summary is not None:
        summary["source_id"] = str(result.source_id)
    return {
        # 19-digit IDs overflow JS numbers, so IDs travel as strings.
        "source_id": str(result.source_id),
        "kind": _kind(data),
        "data": _data_payload(data),
        "summary": summary,
        "stages": stages,
    }


def _shown(samples: Samples, *, weighted: bool) -> np.ndarray:
    """Indices of the samples sent to the browser.

    Weighted (top-K) samples: drawn without replacement in proportion to
    weight, so the shown set follows the posterior. Equal-weight samples:
    evenly spaced (the chains are thinned).
    """
    n = samples.n_samples
    if not weighted:
        return np.unique(np.linspace(0, n - 1, min(n, MAX_CURVE_SAMPLES)).astype(int))
    weights = np.asarray(samples.weight, dtype=float)
    nonzero = np.flatnonzero(weights > 0)
    if len(nonzero) <= MAX_CURVE_SAMPLES:
        return nonzero
    rng = np.random.default_rng(0)
    p = weights[nonzero] / weights[nonzero].sum()
    return np.sort(rng.choice(nonzero, MAX_CURVE_SAMPLES, replace=False, p=p))


def _kind(data: PreparedSource) -> str:
    if isinstance(data, SystemData):
        return "sb2"
    return "rv" if isinstance(data, RVData) else "gaia_astrometry"


def _data_payload(data: PreparedSource) -> dict[str, Any]:
    """The data's arrays and units; per component (``primary``, ...) for SB2."""
    if isinstance(data, SystemData):
        return {
            name: _data_payload(cast("RVData", data[name])) for name in SB2_COMPONENTS
        }
    if isinstance(data, RVData):
        columns = {"time": data.time, "rv": data.rv, "rv_err": data.rv_err}
    else:
        columns = {
            "time": data.time,
            "al_position": data.al_position,
            "al_position_err": data.al_position_err,
        }
    return {
        **{name: jsonable(np.asarray(value.value)) for name, value in columns.items()},
        "units": {name: str(value.unit) for name, value in columns.items()},
    }


def _samples_payload(
    samples: Samples, data: PreparedSource, model: Any
) -> dict[str, Any]:
    # Some derived keys are plain arrays, not Quantities.
    params = {name: _value_and_unit(samples[name]) for name in samples.keys()}  # noqa: SIM118
    payload: dict[str, Any] = {
        "params": {name: jsonable(value) for name, (value, _) in params.items()},
        "units": {name: unit for name, (_, unit) in params.items()},
    }
    if not samples.n_samples:
        return payload
    # A bad sample set (NaN draws, an unusual parameterization) loses its
    # curves, not the whole page.
    try:
        if isinstance(data, SystemData):
            payload["curves"] = _sb2_curves(samples, data, model)
        elif isinstance(data, RVData):
            payload["curves"] = _rv_curves(samples, data, model)
        else:
            payload["curves"] = _gaia_curves(samples, data, model)
    except Exception as err:
        logger.exception("curves failed")
        payload["curves_error"] = f"{type(err).__name__}: {err}"
    return payload


def _curve_json(values: Any) -> Any:
    """Curve values rounded to ``_CURVE_DIGITS`` significant digits of their scale.

    Full float64 precision would make a source payload several times larger
    for no visible difference.
    """
    array = np.asarray(values, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    if finite.size and finite.max() > 0:
        scale = math.floor(math.log10(finite.max()))
        array = np.round(array, max(0, _CURVE_DIGITS - 1 - scale))
    return jsonable(array)


def _rv_grid(samples: Samples, time: Any) -> Any:
    """A time grid over ``time`` resolving the median sampled period."""
    periods = np.asarray(samples["period"].value)
    periods = periods[np.isfinite(periods)]
    if not periods.size:
        msg = "no sample has a finite period"
        raise ValueError(msg)
    period = Q(np.median(periods), samples["period"].unit)
    return get_time_grid(time, period, max_time_grid=MAX_TIME_GRID)


def _rv_on_grid(samples: Samples, data: RVData, model: Any, grid: Any) -> Any:
    """RV on ``grid``, one row per sample, rounded for JSON."""
    # A MultiSurveyOffset is bound to the data's rows, so it cannot be
    # evaluated on a grid; the curves are drawn without the offsets.
    model = _strip_multi_survey_offsets(model)
    unit = str(data.rv.unit)
    nonlinear, linear = _assemble_sample_params(samples, model, data)

    def curve(nl: dict[str, Any], lin: dict[str, Any]) -> jax.Array:
        return model.predict_at_times(
            nl, lin, grid, time_ref=data.time_ref, obs_unit=unit
        )

    return _curve_json(jax.vmap(curve)(nonlinear, linear))


def _rv_curves(samples: Samples, data: RVData, model: Any) -> dict[str, Any]:
    """RV on a grid over the baseline, one row per sample."""
    grid = _rv_grid(samples, data.time)
    return {
        "time": jsonable(np.asarray(grid.value)),
        "rv": _rv_on_grid(samples, data, model, grid),
    }


def _sb2_curves(samples: Samples, data: SystemData, model: Any) -> dict[str, Any]:
    """Each star's RV on one grid over both stars' baseline, one row per sample.

    Each star is the RV case on its own component of the samples and model.
    """
    time = Q(
        np.concatenate([ustrip("day", data[name].time) for name in SB2_COMPONENTS]),
        "day",
    )
    grid = _rv_grid(samples, time)
    return {
        "time": jsonable(np.asarray(grid.value)),
        **{
            name: _rv_on_grid(
                samples.component(name),
                cast("RVData", data[name]),
                model.components[name],
                grid,
            )
            for name in SB2_COMPONENTS
        },
    }


def _gaia_curves(
    samples: Samples, data: GaiaAstrometryData, model: Any
) -> dict[str, Any]:
    """Sky-plane orbits over one period, and along-scan residuals at the data.

    A parameterization without Campbell elements (Fourier) has no sky orbit,
    so only the residuals are returned.
    """
    nonlinear, linear = _assemble_sample_params(samples, model, data)
    phase = np.linspace(0.0, 1.0, _SKY_GRID)
    al = ustrip(str(data.al_position.unit), data.al_position)

    def sky(nl: dict[str, Any], lin: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        times = nl["period"] * (nl["phase_peri"] + phase)
        return model.predict_orbit_sky(nl, lin, times)

    def residual(nl: dict[str, Any], lin: dict[str, Any]) -> jax.Array:
        return al - model.predict(nl, lin, data)

    curves = {"residual": _curve_json(jax.vmap(residual)(nonlinear, linear))}
    if "phase_peri" in nonlinear and hasattr(model.parameterization, "sky_orbit"):
        ra, dec = jax.vmap(sky)(nonlinear, linear)
        curves |= {"ra": _curve_json(ra), "dec": _curve_json(dec)}
    return curves
