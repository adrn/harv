"""``summary.parquet``: one row per prepared source, from the sources tables.

See ``packages/hq/docs/spec.md``, "Summary table". Reads the catalog and the
newest sources row per source of each stage; never the samples or the data.
"""

__all__ = ("SUMMARY_FILE", "summarize")

from pathlib import Path
from typing import Any

import pyarrow as pa

# pyarrow.compute generates its functions at import, so ty cannot see them.
import pyarrow.compute as pc

from harv_hq._parquet_io import read_parquet, write_parquet
from harv_hq.prepare import CATALOG_FILE
from harv_hq.results import ResultsIndex, read_source_rows

SUMMARY_FILE = "summary.parquet"
_EVIDENCE = (
    "n_prior_samples",
    "ln_Z_int",
    "ln_Z_int_mcse",
    "ln_Z_int_ess",
    "max_ln_likelihood",
    "weight_captured",
    "well_resolved",
)
_MCMC = ("r_hat_max", "ess_bulk_min")
_NAMED_STATS = ("period_unimodal", "max_phase_gap", "phase_coverage", "periods_spanned")


def summarize(run_dir: Path, *, expected: dict[str, dict[str, Any] | None]) -> Path:
    """Write ``summary.parquet`` and return its path.

    Parameters
    ----------
    run_dir
        The run directory.
    expected
        Each stage's current provenance; its parts are checked against it
        (``None``: not checked). The rejection entry is also recorded as the
        summary's provenance.

    Returns
    -------
        The path written.

    Raises
    ------
    ProvenanceError
        If a result part was built from different inputs.
    """
    catalog = read_parquet(run_dir / CATALOG_FILE)
    ids = catalog.table["source_id"].to_pylist()
    units = dict(catalog.units)
    rows = {}
    for stage, provenance in expected.items():
        index = ResultsIndex.build(
            run_dir / "results" / stage, expected_provenance=provenance
        )
        rows[stage], stage_units = _aligned(index, ids)
        units.update(stage_units)
    rejection, mcmc = rows["rejection"], rows["mcmc"]
    n = len(ids)

    def column(stage_rows: dict[str, pa.Array], name: str) -> pa.Array:
        return stage_rows.get(name, pa.nulls(n))

    columns: dict[str, Any] = dict(
        zip(catalog.table.column_names, catalog.table.columns, strict=True)
    )
    columns["rejection_status"] = pc.fill_null(
        column(rejection, "status").cast(pa.string()), "pending"
    )
    columns["rejection_error"] = column(rejection, "error")
    columns.update({name: column(rejection, name) for name in _EVIDENCE})
    columns["mcmc_status"] = pc.fill_null(
        column(mcmc, "mcmc_status").cast(pa.string()), "not_selected"
    )
    columns.update({name: column(mcmc, name) for name in _MCMC})
    use_mcmc = pc.equal(columns["mcmc_status"], "converged")  # ty: ignore[unresolved-attribute]
    columns["final"] = pc.if_else(use_mcmc, "mcmc", "rejection")  # ty: ignore[unresolved-attribute]
    for name in filter(_is_statistic, rejection):
        columns[name] = pc.if_else(  # ty: ignore[unresolved-attribute]
            use_mcmc, column(mcmc, name), rejection[name]
        )

    path = run_dir / SUMMARY_FILE
    table = pa.table(columns)
    write_parquet(
        path,
        table,
        units={k: v for k, v in units.items() if k in table.column_names},
        metadata={"provenance": expected["rejection"]},
    )
    return path


def _aligned(
    index: ResultsIndex, ids: list[Any]
) -> tuple[dict[str, pa.Array], dict[str, str]]:
    """The index's newest rows, one per ``ids`` entry (null where absent)."""
    records = list(index.records.values())
    if not records:
        return {}, {}
    table, units = read_source_rows(records)
    position = {sid: i for i, sid in enumerate(index.records)}
    take = pa.array([position.get(sid) for sid in ids], type=pa.int64())
    aligned = table.take(take)
    return dict(zip(aligned.column_names, aligned.columns, strict=True)), units


def _is_statistic(name: str) -> bool:
    """Whether a sources column is a per-source summary statistic."""
    return (
        name in _NAMED_STATS
        or name.startswith("map_")
        or name.endswith(("_p16", "_p50", "_p84"))
    )
