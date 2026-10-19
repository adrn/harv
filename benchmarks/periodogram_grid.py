"""Grid cell definition and enumeration for the periodogram benchmarks.

Same **star** design as :mod:`grid`: each curve varies one axis with the others
pinned to :data:`BASELINE`. Every curve is drawn for every ``(model, prior)``
series, because the three prior modes take different code paths (a fixed Gaussian,
a ``LinearPriorCallable`` resolved per trial period, and the unregularized
least-squares solve) and that difference is the thing worth seeing.

Axes and baselines are documented in ``docs/benchmarks/running-the-benchmarks.md``.
"""

__all__ = (
    "BASELINE",
    "MODELS",
    "N_GRID_VALUES",
    "N_TERMS_VALUES",
    "PRIORS",
    "PeriodogramCell",
    "curve_definitions",
    "enumerate_cells",
)

import itertools
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from grid import N_OBS_VALUES
from unxt import Q

# --- axes ------------------------------------------------------------------

MODELS: tuple[str, ...] = ("FourierRV", "FourierGaiaAstrometry")

# "flat": sigma_amp. "period_dependent": sigma_0 (P/P_ref)^p, a LinearPriorCallable
# resolved at every trial period. "profile": prior=False, amplitudes maximized.
PRIORS: tuple[str, ...] = ("flat", "period_dependent", "profile")

# Trial frequencies. A Gaia DR4 baseline (~5.5 yr) at the default
# samples_per_peak=8 gives ~1.5e3 down to P = 10 d and ~1.6e4 down to P = 1 d, so
# the baseline sits in the realistic range and 1e5 is a deliberately fine grid.
# The scan is a single jax.vmap over the grid with no chunking, so peak memory
# grows with n_grid * n_obs; the top of this ladder is also a memory probe.
N_GRID_VALUES: tuple[int, ...] = (1_000, 10_000, 100_000)

N_TERMS_VALUES: tuple[int, ...] = (1, 2, 3, 4)

PERIOD_MIN = Q(1.0, "day")
PERIOD_MAX = Q(3000.0, "day")
# The scan needs no data to size a grid of explicit n_grid; pinning time_span
# keeps the grid identical across n_obs.
TIME_SPAN = Q(2000.0, "day")


@dataclass(frozen=True, order=True)
class PeriodogramCell:
    """One periodogram benchmark measurement: a point in the grid."""

    model: str
    prior: Literal["flat", "period_dependent", "profile"]
    n_terms: int
    n_obs: int
    n_grid: int
    curve: str

    @property
    def model_kind(self) -> Literal["rv", "gaia"]:
        return "rv" if self.model == "FourierRV" else "gaia"

    @property
    def series(self) -> str:
        return f"{self.model} {self.prior}"

    @property
    def ident(self) -> str:
        """Stable pytest id."""
        return f"{self.model}-{self.prior}-H{self.n_terms}-n{self.n_obs}-g{self.n_grid}"

    @property
    def key(self) -> tuple[Any, ...]:
        """Measurement identity: everything except which curve asked for it."""
        return (self.model, self.prior, self.n_terms, self.n_obs, self.n_grid)

    def asdict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_kind"] = self.model_kind
        d["benchmark"] = "periodogram"
        return d


BASELINE = PeriodogramCell(
    model="FourierGaiaAstrometry",
    prior="flat",
    n_terms=2,
    n_obs=64,
    n_grid=10_000,
    curve="baseline",
)


def curve_definitions(
    *, full: bool = False, smoke: bool = False
) -> dict[str, list[PeriodogramCell]]:
    """Map each curve name to its cells, *before* deduplication."""
    series = list(itertools.product(MODELS, PRIORS))

    if smoke:
        # Every model and prior mode, at n_obs low enough that profile mode caps
        # n_terms, so the run reaches every build_prior branch and the report's
        # capped-cell marking.
        return {
            "n_obs": [
                replace(
                    BASELINE, model=m, prior=p, n_obs=n, n_grid=1_000, curve="n_obs"
                )
                for m, p in series
                for n in (8, 16, 32)
            ]
        }

    if full:
        return {
            "full": [
                PeriodogramCell(m, p, h, n, g, "full")
                for (m, p), h, n, g in itertools.product(
                    series, N_TERMS_VALUES, N_OBS_VALUES, N_GRID_VALUES
                )
            ]
        }

    return {
        "n_obs": [
            replace(BASELINE, model=m, prior=p, n_obs=n, curve="n_obs")
            for m, p in series
            for n in N_OBS_VALUES
        ],
        "n_grid": [
            replace(BASELINE, model=m, prior=p, n_grid=g, curve="n_grid")
            for m, p in series
            for g in N_GRID_VALUES
        ],
        "n_terms": [
            replace(BASELINE, model=m, prior=p, n_terms=h, curve="n_terms")
            for m, p in series
            for h in N_TERMS_VALUES
        ],
    }


def enumerate_cells(
    *, full: bool = False, smoke: bool = False
) -> list[PeriodogramCell]:
    """Flatten :func:`curve_definitions`, timing each shared point once."""
    seen: dict[tuple[Any, ...], PeriodogramCell] = {}
    for cells in curve_definitions(full=full, smoke=smoke).values():
        for cell in cells:
            seen.setdefault(cell.key, cell)
    return sorted(seen.values())
