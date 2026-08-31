"""Grid cell definition and enumeration for the rejection-sampler benchmarks.

The design is a **star** (one-axis-at-a-time) rather than a full factorial: each
curve varies a single axis with the others pinned to :data:`BASELINE`, which is
what a fitted scaling exponent actually needs. ``--bench-full`` swaps in the full
cartesian product when an interaction is suspected.

Axes and baselines are documented in ``docs/running-benchmarks.md``.
"""

from __future__ import annotations

__all__ = (
    "BASELINE",
    "BATCH_SIZE_VALUES",
    "N_OBS_VALUES",
    "N_PRIOR_SAMPLE_VALUES",
    "PARAMETERIZATIONS",
    "Cell",
    "build_data",
    "build_prior_and_model",
    "curve_definitions",
    "enumerate_cells",
)

import itertools
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from unxt import Q

# --- fixed constants -------------------------------------------------------

# Selection policy for every cell.
#
# ``top_k`` gives a *static* output shape, so the ``jax.vmap`` over the
# conditional-linear solve is traced once instead of once per distinct acceptance
# count (docs/spec.md, "Top-K selection"). Without it, timings would be dominated
# by recompilation noise that varies with the data rather than the axis under test.
TOP_K = 256

# Simulated-system truth. Fixed across every cell so that only the axis under
# test varies -- in particular the SNR per observation is constant, so curve A
# measures the cost of more data, not the cost of a better-constrained posterior.
RV_TRUTH: dict[str, Any] = {
    "period": Q(200.0, "day"),
    "eccentricity": 0.3,
    "rv_semiamp": Q(10.0, "km/s"),
    "v_sys": Q(0.0, "km/s"),
    "rv_err": Q(0.5, "km/s"),
    "baseline": Q(3.0, "yr"),
}
GAIA_TRUTH: dict[str, Any] = {
    "period": Q(400.0, "day"),
    "eccentricity": 0.2,
    "semi_major_axis": Q(3.0, "mas"),
    "al_error": Q(0.1, "mas"),
    "baseline": Q(5.0, "yr"),
}

# --- axes ------------------------------------------------------------------

# Number of observations.
#
# For RV this is the number of spectra; for Gaia it is the number of along-scan
# epochs (one row per field-of-view transit, per ``GaiaAstrometryData``). Gaia DR4
# sources land in the 60-250 range, so the top of this ladder is the realistic
# regime, not an extrapolation.
N_OBS_VALUES: tuple[int, ...] = (8, 16, 32, 64, 128, 256)

N_PRIOR_SAMPLE_VALUES: tuple[int, ...] = (10_000, 100_000, 1_000_000, 10_000_000)
BATCH_SIZE_VALUES: tuple[int, ...] = (10_000, 100_000, 1_000_000)

PARAMETERIZATIONS: tuple[str, ...] = (
    "StandardRV",
    "EcoswEsinwRV",
    "FourierRV",
    "StandardGaiaAstrometry",
    "ThieleInnesGaiaAstrometry",
    "FourierGaiaAstrometry",
)

RV_PARAMETERIZATIONS = PARAMETERIZATIONS[:3]
GAIA_PARAMETERIZATIONS = PARAMETERIZATIONS[3:]

# The two parameterizations carried through curves C and D.
#
# Curves A and B already establish that the six behave alike up to a constant, so
# the batch_size and backend axes -- the expensive ones -- only need one RV and one
# astrometry representative.
REPRESENTATIVE: tuple[str, ...] = ("StandardRV", "StandardGaiaAstrometry")


@dataclass(frozen=True, order=True)
class Cell:
    """One benchmark measurement: a point in the grid."""

    parameterization: str
    n_obs: int
    n_prior_samples: int
    batch_size: int
    backend: Literal["memory", "hdf5"]
    curve: str

    @property
    def model_kind(self) -> Literal["rv", "gaia"]:
        return "rv" if self.parameterization in RV_PARAMETERIZATIONS else "gaia"

    @property
    def ident(self) -> str:
        """Stable pytest id."""
        return (
            f"{self.parameterization}-n{self.n_obs}"
            f"-M{self.n_prior_samples}-b{self.batch_size}-{self.backend}"
        )

    @property
    def key(self) -> tuple[Any, ...]:
        """Measurement identity: everything except which curve asked for it."""
        return (
            self.parameterization,
            self.n_obs,
            self.n_prior_samples,
            self.batch_size,
            self.backend,
        )

    def asdict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_kind"] = self.model_kind
        d["top_k"] = TOP_K
        return d


BASELINE = Cell(
    parameterization="StandardRV",
    n_obs=64,
    n_prior_samples=1_000_000,
    batch_size=100_000,
    backend="memory",
    curve="baseline",
)
# Centre of the star. Every curve varies exactly one field of this.


def curve_definitions(
    *, full: bool = False, smoke: bool = False
) -> dict[str, list[Cell]]:
    """Map each curve name to the cells that belong to it, *before* deduplication.

    Curves overlap: the baseline point sits on three of them. Execution wants each
    measurement once (:func:`enumerate_cells`), but the report wants full rows, so
    membership has to survive somewhere. It survives here.
    """
    if smoke:
        # A real (if tiny) n_obs curve, not two isolated points: three axis values
        # are the minimum that exercises slope fitting and plotting, so the smoke
        # run checks the whole report pipeline rather than half of it.
        return {
            "n_obs": [
                replace(
                    BASELINE,
                    parameterization=name,
                    n_obs=n_obs,
                    n_prior_samples=10_000,
                    batch_size=10_000,
                    curve="n_obs",
                )
                for name in REPRESENTATIVE
                for n_obs in (8, 16, 32)
            ]
        }

    if full:
        return {
            "full": [
                Cell(
                    parameterization=name,
                    n_obs=n_obs,
                    n_prior_samples=n_prior,
                    batch_size=batch,
                    backend=backend,
                    curve="full",
                )
                for name, n_obs, n_prior, batch, backend in itertools.product(
                    PARAMETERIZATIONS,
                    N_OBS_VALUES,
                    N_PRIOR_SAMPLE_VALUES,
                    BATCH_SIZE_VALUES,
                    ("memory", "hdf5"),
                )
                # A batch larger than the library is clamped by the sampler's
                # padding, so those cells would duplicate batch == M.
                if batch <= n_prior
            ]
        }

    return {
        # Curve A -- cost vs number of observations, all six parameterizations.
        "n_obs": [
            replace(BASELINE, parameterization=name, n_obs=n_obs, curve="n_obs")
            for name in PARAMETERIZATIONS
            for n_obs in N_OBS_VALUES
        ],
        # Curve B -- cost vs prior library size, all six parameterizations.
        "n_prior_samples": [
            replace(
                BASELINE,
                parameterization=name,
                n_prior_samples=n_prior,
                batch_size=min(BASELINE.batch_size, n_prior),
                curve="n_prior_samples",
            )
            for name in PARAMETERIZATIONS
            for n_prior in N_PRIOR_SAMPLE_VALUES
        ],
        # Curve C -- the GPU knob. docs/spec.md asserts batch_size = n_prior_samples
        # on GPU and 100_000 on CPU; this is the axis that checks it.
        "batch_size": [
            replace(
                BASELINE,
                parameterization=name,
                n_prior_samples=n_prior,
                batch_size=batch,
                curve="batch_size",
            )
            for name in REPRESENTATIVE
            for n_prior in (1_000_000, 10_000_000)
            for batch in BATCH_SIZE_VALUES
        ],
        # Curve D -- in-memory library vs streamed HDF5 cache.
        "backend": [
            replace(
                BASELINE,
                parameterization=name,
                n_prior_samples=n_prior,
                backend=backend,
                curve="backend",
            )
            for name in REPRESENTATIVE
            for n_prior in (1_000_000, 10_000_000)
            for backend in ("memory", "hdf5")
        ],
    }


def enumerate_cells(*, full: bool = False, smoke: bool = False) -> list[Cell]:
    """Flatten :func:`curve_definitions` into the list of measurements to take.

    Deduplicated (curves share baseline points) and sorted by parameterization,
    which is what lets the prior-cache fixture hold exactly one sample library in
    memory at a time.
    """
    cells: list[Cell] = []
    for curve_cells in curve_definitions(full=full, smoke=smoke).values():
        cells += curve_cells
    return _dedupe(cells)


def _dedupe(cells: list[Cell]) -> list[Cell]:
    """Drop duplicate measurement points, keeping the first curve that claimed one.

    Two cells are the same *measurement* when everything but ``curve`` matches, so
    a baseline point shared by three curves is timed once and reported by all
    three -- report.py re-derives membership from :func:`curve_definitions`.
    """
    seen: dict[tuple[Any, ...], Cell] = {}
    for cell in cells:
        seen.setdefault(cell.key, cell)
    return sorted(seen.values())


# --- model / prior / data construction -------------------------------------


def build_prior_and_model(parameterization: str) -> tuple[Any, Any]:
    """Build the ``(HarvPrior, model)`` pair for a parameterization.

    Imported lazily so that ``benchmarks/conftest.py`` can pin x64 before harv
    (and therefore JAX) is imported.
    """
    import harv.models as hm
    from harv.models.astrometry import GaiaAstrometryModel
    from harv.models.parameterizations import ThieleInnesGaiaAstrometry
    from harv.models.rv import RVModel

    rv_period_bounds = {"period_min": Q(10.0, "day"), "period_max": Q(3000.0, "day")}
    gaia_period_bounds = {"period_min": Q(50.0, "day"), "period_max": Q(3000.0, "day")}
    gaia_scales = {
        "sigma_a0": Q(1e3, "AU"),
        "sigma_parallax": Q(100.0, "mas"),
        "sigma_pos": Q(1e3, "mas"),
        "sigma_vtan": Q(200.0, "km/s"),
    }

    match parameterization:
        case "StandardRV" | "EcoswEsinwRV":
            param = getattr(hm, parameterization)()
            prior = param.default_prior(
                **rv_period_bounds,
                sigma_K0=Q(30.0, "km/s"),
                sigma_v0=Q(50.0, "km/s"),
            )
            return prior, RVModel(parameterization=param)

        case "FourierRV":
            # Fourier parameterizations have no data-driven defaults by design;
            # every scale must be explicit (docs/spec.md, "Priors are explicit").
            param = hm.FourierRV(n_terms=2)
            prior = param.default_prior(
                **rv_period_bounds,
                sigma_amp=Q(30.0, "km/s"),
                sigma_v0=Q(50.0, "km/s"),
            )
            return prior, RVModel(parameterization=param)

        case "StandardGaiaAstrometry":
            param = hm.StandardGaiaAstrometry()
            prior = param.default_prior(**gaia_period_bounds, **gaia_scales)
            return prior, GaiaAstrometryModel(parameterization=param)

        case "ThieleInnesGaiaAstrometry":
            # `from_data` is the documented constructor, but it derives
            # `a_floor = med(sigma_AL)/sqrt(N)` from the data -- which would make
            # the prior library depend on n_obs and defeat the build-once-slice-
            # many cache. A fixed floor is data-independent and still exercises
            # the Jacobian-correction path, which is the cost being measured.
            param = ThieleInnesGaiaAstrometry(
                a_floor=0.01, apply_jacobian_correction=True
            )
            prior = param.default_prior(**gaia_period_bounds, **gaia_scales)
            return prior, GaiaAstrometryModel(parameterization=param)

        case "FourierGaiaAstrometry":
            param = hm.FourierGaiaAstrometry(n_terms=2)
            prior = param.default_prior(
                **gaia_period_bounds,
                sigma_amp=Q(10.0, "mas"),
                sigma_pos=Q(1e3, "mas"),
                sigma_pm=Q(100.0, "mas/yr"),
                sigma_parallax=Q(100.0, "mas"),
            )
            return prior, GaiaAstrometryModel(parameterization=param)

        case _:  # pragma: no cover - guarded by PARAMETERIZATIONS
            msg = f"unknown parameterization {parameterization!r}"
            raise ValueError(msg)


def build_data(model_kind: str, n_obs: int) -> Any:
    """Simulate a dataset with ``n_obs`` observations."""
    from harv.simulate.astrometry import simulate_gaia_epoch_astrometry
    from harv.simulate.rv import simulate_rv_sb1_data

    if model_kind == "rv":
        data, _ = simulate_rv_sb1_data(seed=42, n_obs=n_obs, **RV_TRUTH)
    else:
        data, _ = simulate_gaia_epoch_astrometry(seed=42, n_obs=n_obs, **GAIA_TRUTH)
    return data
