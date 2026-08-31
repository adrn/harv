"""Turn pytest-benchmark JSON into the tracked ``docs/benchmarks.md`` page.

Usage::

    uv run -g bench python benchmarks/report.py

Reads every ``benchmarks/results/*.json``, merges the runs by device, and writes
the results page plus its log-log figures. The page is committed because
ReadTheDocs has no GPU and must never try to regenerate it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grid import curve_definitions

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


def load_results(
    results_dir: Path,
) -> tuple[dict[tuple, dict[str, dict]], dict[str, dict]]:
    """Return ``(measurements, device_meta)``.

    ``measurements`` maps a cell key to ``{device_label: record}``; ``device_meta``
    maps a device label to the run metadata that produced it.
    """
    measurements: dict[tuple, dict[str, dict]] = defaultdict(dict)
    device_meta: dict[str, dict] = {}

    files = sorted(results_dir.glob("*.json"))
    if not files:
        msg = (
            f"no result files in {results_dir}. Run the benchmarks first "
            f"(see docs/running-benchmarks.md)."
        )
        raise SystemExit(msg)

    for path in files:
        payload = json.loads(path.read_text())
        machine = payload.get("machine_info", {})
        for record in payload["benchmarks"]:
            info = record["extra_info"]
            if "parameterization" not in info:
                continue  # not one of ours
            label = device_label(record, machine)
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
    args = parser.parse_args()

    measurements, device_meta = load_results(Path(args.results))
    devices = sorted(device_meta, key=lambda d: (not d.startswith("CPU"), d))

    modes = {
        r["extra_info"].get("grid_mode", "star")
        for by_device in measurements.values()
        for r in by_device.values()
    }
    if len(modes) > 1:
        msg = (
            f"results mix grid modes {sorted(modes)}; their cells are not "
            f"comparable. Keep one mode per results directory."
        )
        raise SystemExit(msg)
    mode = modes.pop()
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
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"  devices: {', '.join(devices)}")
    print(f"  cells:   {len(measurements)}")


if __name__ == "__main__":
    main()
