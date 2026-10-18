"""Turn periodogram pytest-benchmark JSON into ``docs/benchmarks-periodogram.md``.

Usage::

    uv run --group bench python benchmarks/report_periodogram.py

Reads the periodogram records from every ``benchmarks/results/*.json`` (records
from the rejection benchmarks are ignored, as ``report.py`` ignores these) and
writes the results page plus one log-log figure per curve.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from periodogram_grid import BASELINE, curve_definitions
from report import REPO_ROOT, RESULTS_DIR, device_label, fmt_time, log_slope, md_table

PAGE_PATH = REPO_ROOT / "docs" / "benchmarks-periodogram.md"

AXIS_LABELS = {
    "n_obs": "number of observations",
    "n_grid": "number of trial frequencies",
    "n_terms": "number of Fourier harmonics `n_terms`",
}


def load_results(
    results_dir: Path, *, include_smoke: bool = False
) -> tuple[dict[tuple, dict[str, dict]], dict[str, dict], str]:
    """Return ``(measurements, device_meta, grid_mode)``; keys are cell keys."""
    measurements: dict[tuple, dict[str, dict]] = defaultdict(dict)
    device_meta: dict[str, dict] = {}
    modes: set[str] = set()
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        machine = payload.get("machine_info", {})
        for record in payload.get("benchmarks", []):
            info = record["extra_info"]
            if info.get("benchmark") != "periodogram":
                continue
            if info["grid_mode"] == "smoke" and not include_smoke:
                continue
            modes.add(info["grid_mode"])
            label = device_label(record, machine)
            key = tuple(
                info[k] for k in ("model", "prior", "n_terms", "n_obs", "n_grid")
            )
            if label in measurements[key]:
                msg = f"{path.name}: duplicate {label!r} measurement for {key}"
                raise SystemExit(msg)
            measurements[key][label] = record
            device_meta.setdefault(
                label,
                {
                    "date": payload.get("datetime", "?")[:10],
                    "platform": "{} / py{}".format(
                        machine.get("system", "?"), machine.get("python_version", "?")
                    ),
                    "jax": info.get("jax_version"),
                    "harv": info.get("harv_version"),
                    "x64": info.get("x64"),
                },
            )
    if not measurements:
        msg = f"no periodogram results in {results_dir}; see docs/running-benchmarks.md"
        raise SystemExit(msg)
    if len(modes) > 1:
        msg = f"results mix grid modes {sorted(modes)}; keep one mode per directory"
        raise SystemExit(msg)
    return dict(measurements), device_meta, modes.pop()


def _stat(record: dict | None, which: str) -> float | None:
    if record is None:
        return None
    if which == "warm":
        return float(record["stats"]["median"])
    return record["extra_info"].get(f"{which}_seconds")


def render_curve(
    curve: str,
    cells: list[Any],
    measurements: dict[tuple, dict[str, dict]],
    devices: list[str],
) -> tuple[list[str], dict[str, dict[str, tuple[list, list]]]]:
    """Tables (one per device) and the plot payload ``{device: {series: (x, y)}}``."""
    xs = sorted({getattr(c, curve) for c in cells})
    series: dict[str, dict[int, tuple]] = defaultdict(dict)
    for c in cells:
        series[c.series][getattr(c, curve)] = c.key

    lines: list[str] = []
    plot: dict[str, dict[str, tuple[list, list]]] = {}
    for device in devices:
        rows = []
        plot[device] = {}
        for name, by_x in series.items():
            warm = [
                _stat(measurements.get(by_x.get(x), {}).get(device), "warm") for x in xs
            ]
            if not any(warm):
                continue
            eager = _stat(
                measurements.get(by_x.get(getattr(BASELINE, curve)), {}).get(device),
                "eager",
            )
            slope = log_slope(
                [x for x, t in zip(xs, warm, strict=True) if t],
                [t for t in warm if t],
            )
            rows.append(
                [
                    name,
                    *(fmt_time(t) for t in warm),
                    "--" if slope is None else f"{slope:.2f}",
                    fmt_time(eager),
                ]
            )
            plot[device][name] = (xs, warm)
        if rows:
            header = [
                "series",
                *(f"{x:,}" for x in xs),
                "slope",
                f"eager @ {getattr(BASELINE, curve):,}",
            ]
            lines += [f"**{device}**", "", md_table(header, rows), ""]
    return lines, plot


def render_plot(
    curve: str, plot: dict, devices: list[str], fig_dir: Path
) -> str | None:
    """Log-log figure for one curve, written under ``fig_dir``."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), layout="constrained")
    colors: dict[str, str] = {}
    for i, device in enumerate(devices):
        for name, (xs, ys) in sorted(plot.get(device, {}).items()):
            pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y]
            if len(pts) < 2:
                continue
            colors.setdefault(name, f"C{len(colors) % 10}")
            ax.plot(
                *zip(*pts, strict=True),
                styles[i % len(styles)],
                marker="o",
                ms=4,
                color=colors[name],
                label=f"{name} — {device}",
            )
    if not colors:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(AXIS_LABELS[curve].replace("`", ""))
    ax.set_ylabel("median wall time, jitted [s]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"periodogram-{curve}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return f"![{AXIS_LABELS[curve]}](_static/benchmarks/{out.name})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS_DIR), type=Path)
    parser.add_argument("--out", default=str(PAGE_PATH), type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    measurements, device_meta, mode = load_results(
        args.results, include_smoke=args.include_smoke
    )
    devices = sorted(device_meta, key=lambda d: (not d.startswith("CPU"), d))

    lines = ["# Periodogram benchmarks", ""]
    if mode == "smoke":
        lines += [
            ":::{warning}",
            "Smoke-test numbers from a `--bench-smoke` run. They only prove the",
            "harness works; see {doc}`running-benchmarks`.",
            ":::",
            "",
        ]
    lines += [
        "Wall-clock cost of one `harv.periodogram.periodogram` call for a single",
        "source, under `jax.jit`, by model, prior mode, number of observations,",
        "number of trial frequencies, and number of harmonics.",
        "",
        ":::{note}",
        "Generated from committed results; not rebuilt with the docs. See",
        "{doc}`running-benchmarks`.",
        ":::",
        "",
        "## Run metadata",
        "",
        md_table(
            ["device", "date", "platform", "jax", "harv", "float64"],
            [
                [
                    d,
                    m["date"],
                    m["platform"],
                    str(m["jax"]),
                    str(m["harv"]),
                    "yes" if m["x64"] else "**no**",
                ]
                for d, m in device_meta.items()
            ],
        ),
        "",
        "`flat` is a fixed Gaussian amplitude prior, `period_dependent` a",
        "`LinearPriorCallable` resolved at every trial period, and `profile` is",
        "`prior=False` (amplitudes maximized). `eager` is the median of calling",
        "`periodogram` directly, without `jax.jit`, at the baseline value of the",
        "axis: that is what a one-off call costs, and it is dominated by op-by-op",
        "dispatch rather than arithmetic. `slope` is the log-log exponent of the",
        "jitted times.",
        "",
    ]

    for curve, cells in curve_definitions(
        full=mode == "full", smoke=mode == "smoke"
    ).items():
        if curve not in AXIS_LABELS or not any(c.key in measurements for c in cells):
            continue
        lines += [f"## Scaling with {AXIS_LABELS[curve]}", ""]
        if mode == "star":
            held = sorted({"n_obs", "n_grid", "n_terms"} - {curve})
            lines += [
                "Held at "
                + ", ".join(f"`{a}={getattr(BASELINE, a):,}`" for a in held)
                + ".",
                "",
            ]
        body, plot = render_curve(curve, cells, measurements, devices)
        lines += body
        # Figures live beside the page, so a smoke report never touches docs/.
        fig_dir = args.out.parent / "_static" / "benchmarks"
        if not args.no_plots and (img := render_plot(curve, plot, devices, fig_dir)):
            lines += [img, ""]

    compile_rows = []
    for device in devices:
        gaps = [
            max(_stat(r, "cold") - _stat(r, "warm"), 0.0)
            for by_device in measurements.values()
            if (r := by_device.get(device)) is not None
        ]
        compile_rows.append([device, fmt_time(float(np.median(gaps)))])
    lines += [
        "## First-call compile cost",
        "",
        "Median of `cold - warm` across every cell, the cold call against a cleared",
        "JIT cache. Paid once per distinct data shape.",
        "",
        md_table(["device", "compile"], compile_rows),
    ]

    args.out.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {args.out}  ({len(measurements)} cells, {', '.join(devices)})")


if __name__ == "__main__":
    main()
