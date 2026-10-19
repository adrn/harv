"""Turn periodogram pytest-benchmark JSON into ``docs/benchmarks/periodograms.md``.

Usage::

    uv run --group bench python benchmarks/report_periodogram.py

Reads every ``benchmarks/results/periodogram/*.json`` and writes the results page
plus one log-log figure per curve.
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
from report import REPO_ROOT, device_label, fmt_time, log_slope, md_table

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "periodogram"
PAGE_PATH = REPO_ROOT / "docs" / "benchmarks" / "periodograms.md"

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
    n_smoke = 0
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        machine = payload.get("machine_info", {})
        for record in payload.get("benchmarks", []):
            info = record["extra_info"]
            if info.get("benchmark") != "periodogram":
                continue
            if info["grid_mode"] == "smoke" and not include_smoke:
                n_smoke += 1
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
        msg = (
            f"only smoke-run results in {results_dir}; pass --include-smoke to "
            "render them"
            if n_smoke
            else f"no periodogram results in {results_dir}; "
            "see docs/benchmarks/running-the-benchmarks.md"
        )
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


def _capped(record: dict | None) -> bool:
    """Whether profile mode timed fewer harmonics than the cell asked for."""
    if record is None:
        return False
    info = record["extra_info"]
    return info.get("n_terms_effective", info["n_terms"]) != info["n_terms"]


def _fmt(record: dict | None) -> str:
    return fmt_time(_stat(record, "warm")) + (" †" if _capped(record) else "")


def render_flat(
    cells: list[Any], measurements: dict[tuple, dict[str, dict]], devices: list[str]
) -> list[str]:
    """One row per cell, for the full factorial, which has no single axis."""
    rows = [
        [
            c.model,
            c.prior,
            str(c.n_terms),
            str(c.n_obs),
            f"{c.n_grid:,}",
            *(_fmt(measurements[c.key].get(d)) for d in devices),
        ]
        for c in cells
        if c.key in measurements
    ]
    header = ["model", "prior", "n_terms", "n_obs", "n_grid", *devices]
    return [md_table(header, rows), ""]


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
            records = [measurements.get(by_x.get(x), {}).get(device) for x in xs]
            if not any(records):
                continue
            # Capped cells timed a smaller model, so they stay out of fits and plots.
            warm = [None if _capped(r) else _stat(r, "warm") for r in records]
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
                    *(_fmt(r) for r in records),
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
    return f"![{AXIS_LABELS[curve]}](figures/{out.name})"


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

    lines = ["# Periodograms", ""]
    if mode == "smoke":
        lines += [
            ":::{warning}",
            "Smoke-test numbers from a `--bench-smoke` run. They only prove the",
            "harness works; see {doc}`running-the-benchmarks`.",
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
        "{doc}`running-the-benchmarks`.",
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
        "axis, after a first call has compiled the scan: that is what each call",
        "in a plain Python loop over same-shape sources costs. The difference",
        "from the jitted time is per-op dispatch of the code around the scan.",
        "`slope` is the log-log exponent of the",
        "jitted times. † marks a profile-mode cell where `periodogram` lowered",
        "`n_terms` to keep the trial model overdetermined, so it timed fewer",
        "harmonics than the cell asked for; those cells are left out of `slope`",
        "and the figures.",
        "",
    ]

    for curve, cells in curve_definitions(
        full=mode == "full", smoke=mode == "smoke"
    ).items():
        if not any(c.key in measurements for c in cells):
            continue
        if curve not in AXIS_LABELS:
            lines += ["## Full grid", "", *render_flat(cells, measurements, devices)]
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
        fig_dir = args.out.parent / "figures"
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
        "JIT cache. Paid once per distinct data shape, whether or not the caller",
        "wraps `periodogram` in `jax.jit`.",
        "",
        md_table(["device", "compile"], compile_rows),
    ]

    args.out.write_text("\n".join(lines).rstrip("\n") + "\n")
    print(f"wrote {args.out}  ({len(measurements)} cells, {', '.join(devices)})")


if __name__ == "__main__":
    main()
