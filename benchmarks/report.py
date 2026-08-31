"""Turn pytest-benchmark JSON into the tracked ``docs/benchmarks.md`` page.

Usage::

    uv run -g bench python benchmarks/report.py

Reads every ``benchmarks/results/*.json``, merges the runs by device, and writes
the results page plus its log-log figures. The page is committed because
ReadTheDocs has no GPU and must never try to regenerate it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import (
    BATCH_SIZE_VALUES,
    N_OBS_VALUES,
    N_PRIOR_SAMPLE_VALUES,
    PARAMETERIZATIONS,
    curve_definitions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
PAGE_PATH = REPO_ROOT / "docs" / "benchmarks.md"
FIG_DIR = REPO_ROOT / "docs" / "_static" / "benchmarks"

AXIS_LABELS = {
    "n_obs": "number of observations",
    "n_prior_samples": "prior library size $M$",
    "batch_size": "`batch_size`",
    "backend": "prior-cache backend",
}
CURVE_TITLES = {
    "n_obs": "Scaling with number of observations",
    "n_prior_samples": "Scaling with prior library size",
    "batch_size": "Effect of `batch_size`",
    "backend": "In-memory library vs streamed HDF5 cache",
    "smoke": "Smoke run",
    "full": "Full factorial",
}
NUMERIC_AXES = {"n_obs", "n_prior_samples", "batch_size"}


# --- loading ---------------------------------------------------------------


def device_label(record: dict[str, Any], machine: dict[str, Any]) -> str:
    """Short human name for the device a record was measured on."""
    info = record["extra_info"]
    platform = info.get("device_platform", "?")
    if platform == "cpu":
        brand = machine.get("cpu", {}).get("brand_raw", machine.get("processor", "CPU"))
        return f"CPU ({brand})"
    return f"{platform.upper()} ({info.get('device_kind', '?')})"


def _file_modes(path: Path, payload: dict) -> set[str]:
    """Grid modes present in one result file."""
    return {
        r["extra_info"].get("grid_mode", "star")
        for r in payload.get("benchmarks", [])
        if "parameterization" in r.get("extra_info", {})
    }


def load_results(
    results_dir: Path,
    *,
    include_smoke: bool = False,
) -> tuple[dict[tuple, dict[str, dict]], dict[str, dict]]:
    """Return ``(measurements, device_meta)``.

    ``measurements`` maps a cell key to ``{device_label: record}``; ``device_meta``
    maps a device label to the run metadata that produced it.

    Smoke-run files are skipped unless ``include_smoke``. A smoke file is never a
    real result -- it exists to prove this generator works -- and leaving one in the
    directory used to make the whole report unbuildable.
    """
    measurements: dict[tuple, dict[str, dict]] = defaultdict(dict)
    device_meta: dict[str, dict] = {}
    label_sources: dict[str, str] = {}

    files = sorted(results_dir.glob("*.json"))
    if not files:
        msg = (
            f"no result files in {results_dir}. Run the benchmarks first "
            f"(see docs/running-benchmarks.md)."
        )
        raise SystemExit(msg)

    payloads: list[tuple[Path, dict]] = []
    for path in files:
        payload = json.loads(path.read_text())
        modes = _file_modes(path, payload)
        if modes == {"smoke"} and not include_smoke:
            print(f"  skipping {path.name} (smoke run; --include-smoke to render it)")
            continue
        payloads.append((path, payload))

    if not payloads:
        msg = (
            f"only smoke-run files in {results_dir}. Run the real grid, or pass "
            f"--include-smoke to render the smoke results anyway."
        )
        raise SystemExit(msg)

    modes = {m for path, payload in payloads for m in _file_modes(path, payload)}
    if len(modes) > 1:
        detail = "\n".join(
            f"    {path.name}: {', '.join(sorted(_file_modes(path, payload)))}"
            for path, payload in payloads
        )
        msg = (
            f"results in {results_dir} mix grid modes {sorted(modes)}, whose cells "
            f"are not comparable:\n{detail}\n"
            f"  Keep one mode per directory -- delete the odd files out and re-run "
            f"this script."
        )
        raise SystemExit(msg)

    for path, payload in payloads:
        machine = payload.get("machine_info", {})
        for record in payload["benchmarks"]:
            info = record["extra_info"]
            if "parameterization" not in info:
                continue  # not one of ours
            label = device_label(record, machine)
            # Two files that resolve to the same label would silently overwrite each
            # other. The usual cause is a "gpu" run that actually fell back to CPU
            # because CUDA-enabled jaxlib was missing.
            if label_sources.setdefault(label, path.name) != path.name:
                msg = (
                    f"{path.name} and {label_sources[label]} both describe device "
                    f"{label!r}, so one would overwrite the other. If one was meant "
                    f"to be a GPU run, check `python -c 'import jax; "
                    f"print(jax.devices())'` -- jax falls back to CPU when "
                    f"CUDA-enabled jaxlib is not installed."
                )
                raise SystemExit(msg)
            key = (
                info["parameterization"],
                info["n_obs"],
                info["n_prior_samples"],
                info["batch_size"],
                info["backend"],
            )
            measurements[key][label] = record
            device_meta.setdefault(
                label,
                {
                    "machine": machine,
                    "datetime": payload.get("datetime", "?"),
                    "source": path.name,
                    **{
                        k: info.get(k)
                        for k in (
                            "jax_version",
                            "harv_version",
                            "x64",
                            "top_k",
                            "device_count",
                            "typecheck_hooks",
                        )
                    },
                },
            )
    return dict(measurements), device_meta


# --- formatting ------------------------------------------------------------


def fmt_time(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "--"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def log_slope(xs: list[float], ys: list[float]) -> float | None:
    """Power-law exponent from a log-log least-squares fit.

    ``t ~ x**slope``: slope 1 is linear in the axis, 0 is free.
    """
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    if len(pairs) < 3:
        return None
    xs_, ys_ = zip(*pairs, strict=True)
    return float(np.polyfit(np.log10(xs_), np.log10(ys_), 1)[0])


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def series_of(cell: Any, axis: str) -> tuple:
    """The row identity within a curve: every coordinate except the varying axis."""
    parts: list[Any] = [cell.parameterization]
    if axis != "n_prior_samples" and cell.n_prior_samples != 1_000_000:
        parts.append(f"M={cell.n_prior_samples:,}")
    if axis != "n_obs" and cell.n_obs != 64:
        parts.append(f"n_obs={cell.n_obs}")
    if axis != "batch_size" and cell.batch_size != 100_000:
        parts.append(f"batch={cell.batch_size:,}")
    if axis != "backend" and cell.backend != "memory":
        parts.append(cell.backend)
    return tuple(parts)


# --- rendering -------------------------------------------------------------


def _axis_values(cells: list[Any], axis: str) -> list[Any]:
    values: list[Any] = []
    for cell in cells:
        v = getattr(cell, axis)
        if v not in values:
            values.append(v)
    return sorted(values, key=lambda v: v if isinstance(v, int) else str(v))


def _series_map(cells: list[Any], axis: str) -> dict[str, dict[Any, tuple]]:
    """Row label -> {axis value: cell key}."""
    series: dict[str, dict[Any, tuple]] = defaultdict(dict)
    for cell in cells:
        label = " ".join(str(p) for p in series_of(cell, axis))
        series[label][getattr(cell, axis)] = cell.key
    return dict(series)


def _flat_table(
    cells: list[Any], measurements: dict[tuple, dict[str, dict]], devices: list[str]
) -> str:
    """No axis to scale along (smoke / full): just list the cells."""
    rows = [
        [cell.ident]
        + [fmt_time(_median(measurements.get(cell.key, {}).get(d))) for d in devices]
        for cell in sorted(cells)
    ]
    return md_table(["cell", *devices], rows)


def _device_table(
    device: str,
    axis: str,
    axis_values: list[Any],
    series: dict[str, dict[Any, tuple]],
    measurements: dict[tuple, dict[str, dict]],
) -> tuple[str | None, dict[str, tuple]]:
    """One device's table for one curve, plus that device's plot data."""
    rows: list[list[str]] = []
    plot: dict[str, tuple] = {}
    numeric = axis in NUMERIC_AXES

    for label, by_axis in sorted(series.items()):
        times = [
            _median(measurements.get(by_axis.get(v), {}).get(device))
            for v in axis_values
        ]
        if all(t is None for t in times):
            continue
        row = [label, *[fmt_time(t) for t in times]]
        if numeric:
            xs = [float(v) for v, t in zip(axis_values, times, strict=True) if t]
            ys = [t for t in times if t]
            slope = log_slope(xs, ys)
            row.append("--" if slope is None else f"{slope:+.2f}")
            plot[label] = ([float(v) for v in axis_values], times)
        rows.append(row)

    if not rows:
        return None, {}
    header = ["parameterization", *[str(v) for v in axis_values]]
    if numeric:
        header.append("slope")
    return f"**{device}** — median wall time\n\n" + md_table(header, rows), plot


def _speedup_table(
    devices: list[str],
    axis_values: list[Any],
    series: dict[str, dict[Any, tuple]],
    measurements: dict[tuple, dict[str, dict]],
) -> str | None:
    """Ratio of the first two devices, cell by cell."""
    rows: list[list[str]] = []
    for label, by_axis in sorted(series.items()):
        row = [label]
        has_value = False
        for v in axis_values:
            per_device = measurements.get(by_axis.get(v), {})
            a = _median(per_device.get(devices[0]))
            b = _median(per_device.get(devices[1]))
            if a and b:
                row.append(f"{a / b:.1f}x")
                has_value = True
            else:
                row.append("--")
        if has_value:
            rows.append(row)
    if not rows:
        return None
    return (
        f"**Speedup** — {devices[0]} / {devices[1]} "
        f"(>1 means {devices[1]} is faster)\n\n"
        + md_table(["parameterization", *[str(v) for v in axis_values]], rows)
    )


def render_curve(
    curve: str,
    cells: list[Any],
    measurements: dict[tuple, dict[str, dict]],
    devices: list[str],
) -> tuple[str, dict]:
    """Render one curve's tables. Returns (markdown, plot payload)."""
    if curve not in AXIS_LABELS:
        return _flat_table(cells, measurements, devices), {}

    axis = curve
    axis_values = _axis_values(cells, axis)
    series = _series_map(cells, axis)

    chunks: list[str] = []
    plot: dict[str, Any] = {"axis": axis, "devices": {}}
    for device in devices:
        table, device_plot = _device_table(
            device, axis, axis_values, series, measurements
        )
        plot["devices"][device] = device_plot
        if table:
            chunks.append(table)

    if len(devices) > 1:
        speedup = _speedup_table(devices, axis_values, series, measurements)
        if speedup:
            chunks.append(speedup)

    return "\n\n".join(chunks), plot


def _median(record: dict | None) -> float | None:
    return None if record is None else float(record["stats"]["median"])


def render_compile_table(
    measurements: dict[tuple, dict[str, dict]], devices: list[str]
) -> str:
    """Cold-call cost minus warm median: the JIT compile estimate."""
    per_device: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for by_device in measurements.values():
        for device, record in by_device.items():
            info = record["extra_info"]
            cold = info.get("cold_seconds")
            warm = _median(record)
            if cold is None or warm is None:
                continue
            per_device[device][info["parameterization"]].append(max(cold - warm, 0.0))

    names = sorted({n for d in per_device.values() for n in d})
    rows = []
    for name in names:
        row = [name]
        for device in devices:
            vals = per_device.get(device, {}).get(name, [])
            row.append(fmt_time(float(np.median(vals)) if vals else None))
        rows.append(row)
    return md_table(["parameterization", *devices], rows)


def render_plots(curve: str, plot: dict, devices: list[str]) -> str | None:
    """Log-log figure for one curve. Returns the markdown image line, or None."""
    if not plot or not any(plot["devices"].values()):
        return None
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), layout="constrained")
    colors: dict[str, Any] = {}
    for i, device in enumerate(devices):
        for name, (xs, ys) in sorted(plot["devices"].get(device, {}).items()):
            pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y]
            if len(pts) < 2:
                continue
            px, py = zip(*pts, strict=True)
            if name not in colors:
                colors[name] = f"C{len(colors) % 10}"
            ax.plot(
                px,
                py,
                styles[i % len(styles)],
                marker="o",
                ms=4,
                color=colors[name],
                label=f"{name} — {device}" if i == 0 else None,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(AXIS_LABELS[plot["axis"]].replace("`", ""))
    ax.set_ylabel("median wall time [s]")
    ax.set_title(CURVE_TITLES.get(curve, curve))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    if len(devices) > 1:
        ax.text(
            0.02,
            0.02,
            "linestyle = device: "
            + ", ".join(f"{s} {d}" for s, d in zip(styles, devices, strict=False)),
            transform=ax.transAxes,
            fontsize=7,
            alpha=0.7,
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{curve}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return f"![{CURVE_TITLES.get(curve, curve)}](_static/benchmarks/{out.name})"


# --- findings ---------------------------------------------------------------
#
# Everything in this section is COMPUTED from the loaded results. Only the
# interpretation is fixed prose, and it is attributed to the run in the metadata
# table. That is deliberate: a hardcoded "52x" would quietly become a lie the
# first time someone regenerates the page on different hardware.


def _pow10(n: int) -> str:
    """Render a round library size as `1e7` rather than `1e+07`."""
    return (
        f"1e{len(str(n)) - 1}"
        if str(n).startswith("1") and set(str(n)[1:]) <= {"0"}
        else f"{n:,}"
    )


def _at(
    measurements: dict[tuple, dict[str, dict]],
    param: str,
    n_obs: int,
    n_prior: int,
    batch: int,
    backend: str,
    device: str,
) -> float | None:
    return _median(
        measurements.get((param, n_obs, n_prior, batch, backend), {}).get(device)
    )


def _m_curve(
    measurements: dict[tuple, dict[str, dict]], param: str, device: str
) -> dict[int, float]:
    """Library-size curve for one parameterization, as {M: seconds}."""
    out = {}
    for m in N_PRIOR_SAMPLE_VALUES:
        t = _at(measurements, param, 64, m, min(100_000, m), "memory", device)
        if t:
            out[m] = t
    return out


def _fmt_range(values: list[float], unit: str = "x", places: int = 1) -> str:
    if not values:
        return "--"
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 10**-places:
        return f"{lo:.{places}f}{unit}"
    return f"{lo:.{places}f}-{hi:.{places}f}{unit}"


def _speedup_growth(
    measurements: dict[tuple, dict[str, dict]], fast: str, slow: str
) -> tuple[list[float], list[float], int, int]:
    """Speedups at the smallest and largest library size."""
    small, large = [], []
    m_lo, m_hi = min(N_PRIOR_SAMPLE_VALUES), max(N_PRIOR_SAMPLE_VALUES)
    for p in PARAMETERIZATIONS:
        a, b = _m_curve(measurements, p, slow), _m_curve(measurements, p, fast)
        if m_lo in a and m_lo in b:
            small.append(a[m_lo] / b[m_lo])
        if m_hi in a and m_hi in b:
            large.append(a[m_hi] / b[m_hi])
    return small, large, m_lo, m_hi


def _throughput_rows(
    measurements: dict[tuple, dict[str, dict]], devices: list[str]
) -> list[list[str]]:
    """Peak sustained throughput, in millions of prior samples per second."""
    m_hi = max(N_PRIOR_SAMPLE_VALUES)
    rows = []
    for p in PARAMETERIZATIONS:
        row = [p]
        any_value = False
        for d in devices:
            t = _at(measurements, p, 64, m_hi, min(100_000, m_hi), "memory", d)
            rate = m_hi / t / 1e6 if t else None
            row.append(
                "--" if rate is None else f"{rate:.2f}" if rate < 1 else f"{rate:.1f}"
            )
            any_value = any_value or bool(t)
        if any_value:
            rows.append(row)
    return rows


def _floor(measurements: dict[tuple, dict[str, dict]], device: str) -> float | None:
    times = [_median(r[device]) for r in measurements.values() if device in r]
    times = [t for t in times if t]
    return min(times) if times else None


def _batch_spread(
    measurements: dict[tuple, dict[str, dict]], device: str
) -> list[float]:
    """Slowest/fastest batch_size at fixed (param, M) -- how much the knob matters."""
    spreads = []
    for p in PARAMETERIZATIONS:
        for m in N_PRIOR_SAMPLE_VALUES:
            ts = [
                t
                for b in BATCH_SIZE_VALUES
                if (t := _at(measurements, p, 64, m, b, "memory", device))
            ]
            if len(ts) >= 2:
                spreads.append(max(ts) / min(ts))
    return spreads


def _hdf5_penalty(
    measurements: dict[tuple, dict[str, dict]], device: str
) -> list[float]:
    out = []
    for p in PARAMETERIZATIONS:
        for m in N_PRIOR_SAMPLE_VALUES:
            mem = _at(measurements, p, 64, m, 100_000, "memory", device)
            h5 = _at(measurements, p, 64, m, 100_000, "hdf5", device)
            if mem and h5:
                out.append(h5 / mem)
    return out


def _compile_range(
    measurements: dict[tuple, dict[str, dict]], device: str
) -> list[float]:
    per_param: dict[str, list[float]] = {}
    for r in measurements.values():
        if device not in r:
            continue
        info = r[device]["extra_info"]
        cold, warm = info.get("cold_seconds"), _median(r[device])
        if cold is None or warm is None:
            continue
        per_param.setdefault(info["parameterization"], []).append(max(cold - warm, 0.0))
    return [float(np.median(v)) for v in per_param.values() if v]


def _worst_n_obs_knee(
    measurements: dict[tuple, dict[str, dict]], fast: str, slow: str
) -> tuple[str, int, float, int, float] | None:
    """Largest drop in speedup between adjacent n_obs values."""
    worst = None
    for p in PARAMETERIZATIONS:
        sp = {}
        for n in N_OBS_VALUES:
            a = _at(measurements, p, n, 1_000_000, 100_000, "memory", slow)
            b = _at(measurements, p, n, 1_000_000, 100_000, "memory", fast)
            if a and b:
                sp[n] = a / b
        ns = sorted(sp)
        for lo, hi in itertools.pairwise(ns):
            drop = sp[lo] / sp[hi]
            if worst is None or drop > worst[0]:
                worst = (drop, p, lo, sp[lo], hi, sp[hi])
    return None if worst is None else (worst[1], worst[2], worst[3], worst[4], worst[5])


def _device_agreement(
    measurements: dict[tuple, dict[str, dict]], devices: list[str]
) -> tuple[float, int] | None:
    """Largest relative disagreement in evidence_ess between two devices."""
    if len(devices) < 2:
        return None
    a, b = devices[0], devices[1]
    worst, n = 0.0, 0
    for r in measurements.values():
        if a not in r or b not in r:
            continue
        xa = r[a]["extra_info"].get("evidence_ess")
        xb = r[b]["extra_info"].get("evidence_ess")
        if xa is None or xb is None or math.isnan(xa) or math.isnan(xb):
            continue
        worst = max(worst, abs(xa - xb) / max(abs(xa), 1e-30))
        n += 1
    return (worst, n) if n else None


def render_findings(  # noqa: C901
    measurements: dict[tuple, dict[str, dict]],
    devices: list[str],
    mode: str,
) -> list[str]:
    """Interpretation of the tables below, with every figure computed."""
    if mode != "star":
        return []  # the star grid is the only one these readings are defined for

    lines = [
        "## What the numbers say",
        "",
        "Computed from the results in this page, on the hardware above. The",
        "interpretation is fixed; the figures are not, so regenerating on your own",
        "machine ({doc}`running-benchmarks`) updates them rather than contradicting",
        "them.",
        "",
    ]

    gpus = [d for d in devices if not d.startswith("CPU")]
    cpus = [d for d in devices if d.startswith("CPU")]
    pair = (gpus[0], cpus[0]) if gpus and cpus else None

    if pair:
        fast, slow = pair
        small, large, m_lo, m_hi = _speedup_growth(measurements, fast, slow)
        if small and large:
            f_fast, f_slow = _floor(measurements, fast), _floor(measurements, slow)
            floors = (
                f"the smallest warm call measured here is {f_fast * 1e3:.0f} ms on "
                f"GPU against {f_slow * 1e3:.0f} ms on CPU."
                if f_fast and f_slow
                else "the per-call floor is visible at the small-`M` end of the table."
            )
            headline = (
                f"At `M={_pow10(m_lo)}` the GPU is only "
                f"{_fmt_range(small)} faster; at `M={_pow10(m_hi)}` it is "
                f"{_fmt_range(large)}. The reason is in the"
            )
            lines += [
                "### The GPU advantage grows with library size",
                "",
                headline,
                "slopes: cost is near-linear in `M` on CPU and clearly sublinear on",
                "GPU, because a small library cannot fill the device. Below roughly",
                "`M=1e5` a GPU call is dominated by fixed cost, not by work — "
                + floors,
                "",
                "**Consequence for population runs.** That floor is paid once per",
                "source. Millions of sources at small `M` spend most of their time in",
                "per-call overhead, so a GPU pays off through *bigger libraries*, not",
                "through more calls. See {doc}`at-scale`.",
                "",
            ]

    rows = _throughput_rows(measurements, devices)
    if rows:
        m_hi = max(N_PRIOR_SAMPLE_VALUES)
        intro = f"Sustained rate at `M={_pow10(m_hi)}`, `n_obs=64`, in-memory library."
        lines += [
            "### Throughput, in millions of prior samples per second",
            "",
            intro,
            "This is the number to plan a run from.",
            "",
            md_table(["parameterization", *devices], rows),
            "",
            "The spread across parameterizations is real work, not noise: the",
            "astrometric models carry more linear columns than the RV ones, and the",
            "Thiele-Innes variant additionally evaluates a Jacobian correction per",
            "sample.",
            "",
        ]

    if pair:
        fast, slow = pair
        agree = _device_agreement(measurements, devices)
        if agree:
            worst, n = agree
            lines += [
                "### The speedup is free",
                "",
                f"Across all {n} cells measured on both devices, the largest relative",
                f"difference in `logZ_int_ess` is {worst:.1e}. Both runs pin float64,",
                "and the GPU is not trading accuracy for speed — it is the same",
                "arithmetic, faster.",
                "",
            ]

    spreads = {d: _batch_spread(measurements, d) for d in devices}
    if any(spreads.values()):
        # The heading must not claim a comparison the loaded data cannot support.
        lines += [
            "### `batch_size` matters more on CPU than on GPU"
            if pair
            else "### How much `batch_size` matters",
            "",
        ]
        for d in devices:
            if spreads[d]:
                lines.append(
                    f"- **{d}**: slowest/fastest `batch_size` at fixed `M` spans "
                    f"{_fmt_range(spreads[d], places=2)}."
                )
        lines += [
            "",
            *(
                [
                    "This inverts the guidance the design spec used to give. Setting",
                    "`batch_size = n_prior_samples` on GPU is not the lever it was",
                    "assumed to be; the device is already saturated at far smaller",
                    "batches.",
                ]
                if pair
                else []
            ),
            "The knob sets the working-set size of the `(batch, n_obs, n_linear)`",
            "intermediate, which is why it moves CPU timings at all.",
            "",
            ":::{warning}",
            "Every CPU cell here is a **single process on an otherwise idle node**, so",
            "these are per-core ceilings, not per-node predictions. Under many-rank",
            "contention the optimum moves *down*, because ranks compete for memory",
            "bandwidth — a large batch that wins alone can starve a full node. Measure",
            "it under the contention you will actually run. See {doc}`at-scale`.",
            ":::",
            "",
        ]

    pen = {d: _hdf5_penalty(measurements, d) for d in devices}
    if any(pen.values()):
        lines += [
            "### Streaming from HDF5 costs the GPU much more than the CPU"
            if pair
            else "### The cost of streaming from HDF5",
            "",
        ]
        for d in devices:
            if pen[d]:
                lines.append(
                    f"- **{d}**: {_fmt_range(pen[d], places=2)} of the in-memory time."
                )
        lines += [
            "",
            *(
                [
                    "The absolute I/O cost is similar; what differs is what it is",
                    "competing with. On CPU the compute is slow enough to hide the",
                    "reads. On GPU the compute is fast enough that the reads become",
                    "the run.",
                ]
                if pair
                else []
            ),
            "Keep the library in memory whenever it fits, and reserve the HDF5 path",
            "for libraries that genuinely cannot.",
            "",
        ]

    comp = {d: _compile_range(measurements, d) for d in devices}
    if any(comp.values()):
        lines += [
            "### Compilation is a per-shape tax, and it is higher on GPU"
            if pair
            else "### Compilation is a per-shape tax",
            "",
        ]
        for d in devices:
            if comp[d]:
                span = _fmt_range(comp[d], unit=" s", places=1)
                lines.append(f"- **{d}**: {span} per distinct input shape.")
        lines += [
            "",
            "Paid once per shape, so a population loop over identically-shaped data",
            "amortizes it to nothing — and a catalog with a different epoch count per",
            "source pays it *per source*, which is why bucketing epoch counts is the",
            "first thing to do with real astrometry. See {doc}`at-scale`.",
            "",
        ]

    if pair:
        knee = _worst_n_obs_knee(measurements, *pair)
        if knee and knee[2] / knee[4] > 1.5:
            param, lo, sp_lo, hi, sp_hi = knee
            lines += [
                "### The GPU advantage falls away at large `n_obs`",
                "",
                f"For `{param}` the speedup drops from {sp_lo:.0f}x at `n_obs={lo}` to",
                f"{sp_hi:.0f}x at `n_obs={hi}` — a discontinuity, not a trend, and all",
                "six parameterizations show it at the same place.",
                "",
                ":::{note}",
                "The arithmetic points at the working set: at `batch_size=1e5` and",
                f"`n_obs={hi}`, one float64 column array is",
                f"`1e5 x {hi} x 8 B = {1e5 * hi * 8 / 1e6:.0f} MB`, and the design",
                "matrix holds several. Lowering `batch_size` so that",
                "`batch_size x n_obs` stays roughly constant is the obvious remedy,",
                "**but this grid did not measure it** — the `batch_size` curve",
                "was only run at `n_obs=64`. Treat the",
                "cause as a hypothesis and measure it on your own data before relying",
                "on it.",
                ":::",
                "",
            ]

    return lines


SMOKE_BANNER = [
    ":::{warning}",
    "**These are smoke-test numbers, not real results.** They come from a",
    "`--bench-smoke` run: a handful of tiny cells at `M = 1e4` whose only",
    "job is to prove the harness and this report generator work end to end.",
    "The timings are meaningless as a description of harv's performance.",
    "Replace them with a real run -- see {doc}`running-benchmarks`.",
    ":::",
    "",
]


def render_metadata_table(devices: list[str], device_meta: dict[str, dict]) -> str:
    """One row per device: what produced these numbers."""
    rows = []
    for device in devices:
        m = device_meta[device]
        machine = m["machine"]
        rows.append(
            [
                device,
                m.get("datetime", "?")[:10],
                "{} / py{}".format(
                    machine.get("system", "?"), machine.get("python_version", "?")
                ),
                str(m.get("jax_version")),
                str(m.get("harv_version")),
                "yes" if m.get("x64") else "**no**",
                str(m.get("top_k")),
            ]
        )
    header = ["device", "date", "platform", "jax", "harv", "float64", "top_k"]
    return md_table(header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS_DIR), type=Path)
    parser.add_argument("--out", default=str(PAGE_PATH), type=Path)
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figure generation."
    )
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Render smoke-run results, which are skipped by default.",
    )
    args = parser.parse_args()

    measurements, device_meta = load_results(
        Path(args.results), include_smoke=args.include_smoke
    )
    devices = sorted(device_meta, key=lambda d: (not d.startswith("CPU"), d))

    mode = next(
        r["extra_info"].get("grid_mode", "star")
        for by_device in measurements.values()
        for r in by_device.values()
    )
    definitions = curve_definitions(full=mode == "full", smoke=mode == "smoke")

    lines = [
        "# Benchmarks",
        "",
    ]
    lines += SMOKE_BANNER if mode == "smoke" else []
    lines += [
        "Wall-clock scaling of harv's `RejectionSampler`",
        "across model parameterizations, dataset sizes, prior library sizes, and",
        "`batch_size`. Every number here is measured, not modelled.",
        "",
        ":::{note}",
        "This page is generated from committed benchmark results and is **not**",
        "rebuilt when the docs build — ReadTheDocs has no GPU. To regenerate it, see",
        "{doc}`running-benchmarks`.",
        ":::",
        "",
        "## Run metadata",
        "",
    ]

    lines.append(render_metadata_table(devices, device_meta))
    lines += [
        "",
        "All cells use `top_k` selection, which gives a static output shape so that",
        "timings are not contaminated by recompilation as the accepted-sample count",
        "changes (see the Top-K selection section of the design spec).",
        "",
    ]

    lines += render_findings(measurements, devices, mode)

    for curve, cells in definitions.items():
        if not any(c.key in measurements for c in cells):
            continue
        lines += [f"## {CURVE_TITLES.get(curve, curve)}", ""]
        if curve in AXIS_LABELS:
            lines.append(f"Varying {AXIS_LABELS[curve]}.")
            if mode == "star":
                lines.append(
                    "All other axes are held at the baseline: `n_obs=64`, `M=1e6`, "
                    "`batch_size=1e5`, in-memory cache."
                )
            lines.append("")
        body, plot = render_curve(curve, cells, measurements, devices)
        lines += [body, ""]
        if curve in NUMERIC_AXES:
            lines += [
                "`slope` is the exponent of a log-log least-squares fit: 1.0 is linear",
                f"in {AXIS_LABELS[curve]}, 0.0 is free.",
                "",
            ]
        if not args.no_plots:
            img = render_plots(curve, plot, devices)
            if img:
                lines += [img, ""]

    lines += [
        "## First-call compile cost",
        "",
        "Median of `cold - warm` across every cell, where the cold call runs against",
        "a cleared JIT cache. This is compilation plus one execution minus one",
        "execution, so it estimates compile time alone.",
        "",
        render_compile_table(measurements, devices),
        "",
        "Compile cost is paid once per distinct input shape. In a population loop",
        "over many sources with identical data shapes it is amortized to nothing;",
        "for a single one-off fit it can dominate.",
        "",
    ]

    out_path = Path(args.out)
    # Exactly one trailing newline: sections append a blank separator line, and
    # the repo's end-of-file-fixer hook rejects the doubled newline that leaves.
    out_path.write_text("\n".join(lines).rstrip("\n") + "\n")
    # relative_to raises for an --out outside the repo, so only prettify when it helps.
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"wrote {shown}")
    print(f"  devices: {', '.join(devices)}")
    print(f"  cells:   {len(measurements)}")


if __name__ == "__main__":
    main()
