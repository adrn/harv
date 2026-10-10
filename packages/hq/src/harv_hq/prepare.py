"""``hq prepare``: turn a survey table into per-source data, and read it back.

See ``packages/hq/docs/spec.md``, "``prepare``". The input table is sorted
once by ``(source_id, time)``; source boundaries come from the sorted IDs, so
no step scans the table per source. Readers find a source's rows through
``data_index.parquet`` and read only the row groups that hold them.
"""

__all__ = ("PreparedData", "prepare", "read_source", "read_sources")

import logging
import os
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, final

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table
from astropy.time import Time
from unxt import Q

from harv.data import GaiaAstrometryData, RVData
from harv_hq._parquet_io import (
    groups_spanning,
    read_metadata,
    read_parquet,
    row_group_starts,
    write_parquet,
)
from harv_hq.config import Config, ConfigError
from harv_hq.provenance import ProvenanceError, make_provenance

logger = logging.getLogger(__name__)

DATA_FILE = "data.parquet"
INDEX_FILE = "data_index.parquet"
CATALOG_FILE = "catalog.parquet"
ROW_GROUP_SIZE = 65_536

# Per kind: (observation column, uncertainty column) and every value column in
# data.parquet order. parallax_factor is dimensionless and has no unit key.
_OBS_ERR = {
    "rv": ("rv", "rv_err"),
    "gaia_astrometry": ("al_position", "al_position_err"),
}
_VALUE_COLUMNS = {
    "rv": ("rv", "rv_err"),
    "gaia_astrometry": (
        "al_position",
        "al_position_err",
        "scan_angle",
        "parallax_factor",
    ),
}
_UNIT_KEYS = {
    "rv": "rv_unit",
    "rv_err": "rv_unit",
    "al_position": "al_position_unit",
    "al_position_err": "al_position_unit",
    "scan_angle": "scan_angle_unit",
}
_RESERVED_CATALOG_COLUMNS = ("source_id", "n_obs", "time_baseline")


def prepare(
    config: Config,
    *,
    select_rows: Callable[[Table], Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Write ``data.parquet``, ``data_index.parquet`` and ``catalog.parquet``.

    Parameters
    ----------
    config
        The run configuration.
    select_rows
        The model file's optional ``select_rows(table)`` hook. It receives the
        full input table, every column included, after the built-in cuts.
    overwrite
        Replace existing prepared files. Doing so invalidates every
        downstream output, which then refuses to resume (``data_id``
        changes).

    Raises
    ------
    FileExistsError
        If prepared files exist and ``overwrite`` is false.
    """
    run_dir = config.run_dir
    outputs = [run_dir / name for name in (DATA_FILE, INDEX_FILE, CATALOG_FILE)]
    existing = [p.name for p in outputs if p.exists()]
    if existing and not overwrite:
        msg = (
            f"{run_dir} already has prepared data ({', '.join(existing)}); "
            "pass overwrite=True (hq prepare --overwrite) to replace it"
        )
        raise FileExistsError(msg)

    kind = config.run.kind
    data = config.data
    table = _read_table(data.file, data.format, data.hdu)
    n_input = len(table)

    keep = _builtin_cuts(table, config)
    if select_rows is not None:
        keep &= np.asarray(select_rows(table), dtype=bool)
    table = table[keep]

    source_id = _id_array(table[data.source_id])
    time = Time(
        np.asarray(table[data.time]), format=data.time_format, scale=data.time_scale
    ).tcb.mjd
    values, units = _value_arrays(table, config)

    order = np.lexsort((time, source_id))
    source_id, time = source_id[order], time[order]
    values = {name: arr[order] for name, arr in values.items()}

    ids, starts, counts = np.unique(source_id, return_index=True, return_counts=True)
    enough = counts >= config.prepare.min_n_obs
    rows = np.repeat(enough, counts)
    source_id, time = source_id[rows], time[rows]
    values = {name: arr[rows] for name, arr in values.items()}
    ids, counts = ids[enough], counts[enough]
    if len(ids) == 0:
        msg = (
            f"no sources left in {data.file} after the cuts and min_n_obs = "
            f"{config.prepare.min_n_obs} ({n_input} input rows)"
        )
        raise ValueError(msg)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    logger.info(
        "prepare: %d input rows, %d kept after cuts, %d sources "
        "(%d dropped with fewer than %d observations)",
        n_input,
        len(order),
        len(ids),
        int((~enough).sum()),
        config.prepare.min_n_obs,
    )

    data_id = uuid.uuid4().hex
    metadata = {
        "provenance": make_provenance(config, data_id=data_id),
        "kind": kind,
        "data_id": data_id,
    }
    write_parquet(
        run_dir / DATA_FILE,
        {"source_id": source_id, "time": time, **values},
        units={"time": "day", **units},
        field_metadata={"time": {"format": "mjd", "scale": "tcb"}},
        metadata=metadata,
        row_group_size=ROW_GROUP_SIZE,
    )

    baseline = np.maximum.reduceat(time, starts) - np.minimum.reduceat(time, starts)
    _write_catalog(
        config, ids=ids, n_obs=counts, time_baseline=baseline, metadata=metadata
    )

    # Written last: until it exists, there is no usable prepared data.
    write_parquet(
        run_dir / INDEX_FILE,
        {"source_id": ids, "row_start": starts, "n_obs": counts.astype(np.int64)},
        metadata=metadata,
    )


def _read_table(path: Path, fmt: str | None, hdu: int | str | None) -> Table:
    kwargs: dict[str, Any] = {}
    if fmt is not None:
        kwargs["format"] = fmt
    if hdu is not None:
        kwargs["hdu"] = hdu
    return Table.read(path, **kwargs)


def _missing(column: Any) -> np.ndarray:
    return (
        np.ma.getmaskarray(column)
        if hasattr(column, "mask")
        else np.zeros(len(column), dtype=bool)
    )


def _builtin_cuts(table: Table, config: Config) -> np.ndarray:
    """Rows with an ID, finite time/observation/uncertainty, positive uncertainty."""
    data = config.data
    obs_key, err_key = _OBS_ERR[config.run.kind]
    obs_col, err_col = getattr(data, obs_key), getattr(data, err_key)
    keep = ~_missing(table[data.source_id])
    for name in (data.time, obs_col, err_col):
        column = table[name]
        keep &= ~_missing(column)
        keep &= np.isfinite(np.asarray(np.ma.getdata(column), dtype=float))
    keep &= np.asarray(np.ma.getdata(table[err_col]), dtype=float) > 0
    return keep


def _id_array(column: Any) -> np.ndarray:
    """Source IDs as int64 for integer columns, str otherwise."""
    values = np.asarray(np.ma.getdata(column))
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    if values.dtype.kind == "S":
        return np.char.decode(values).astype(str)
    return values.astype(str)


def _value_arrays(
    table: Table, config: Config
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """The kind's value columns as float64, renamed to their data.parquet names."""
    values: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    for name in _VALUE_COLUMNS[config.run.kind]:
        column = table[getattr(config.data, name)]
        values[name] = np.asarray(np.ma.getdata(column), dtype=np.float64)
        if name in _UNIT_KEYS:
            unit = (
                column.unit
                if column.unit is not None
                else getattr(config.data, _UNIT_KEYS[name])
            )
            units[name] = str(Q(1.0, str(unit)).unit)
    return values, units


def _write_catalog(
    config: Config,
    *,
    ids: np.ndarray,
    n_obs: np.ndarray,
    time_baseline: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    columns: dict[str, pa.Array] = {
        "source_id": pa.array(ids),
        "n_obs": pa.array(n_obs.astype(np.int64)),
        "time_baseline": pa.array(time_baseline, type=pa.float64()),
    }
    units = {"time_baseline": "day"}

    if config.catalog is not None:
        catalog = config.catalog
        table = _read_table(catalog.file, catalog.format, catalog.hdu)
        if catalog.source_id not in table.colnames:
            msg = (
                f"[catalog] source_id column {catalog.source_id!r} "
                f"is not in {catalog.file}"
            )
            raise ConfigError(msg)
        names = catalog.columns
        if names is None:
            names = [
                n
                for n in table.colnames
                if n != catalog.source_id and table[n].ndim == 1
            ]
        clash = sorted(set(names) & set(_RESERVED_CATALOG_COLUMNS))
        if clash:
            msg = f"[catalog] column(s) {clash} collide with hq's own catalog columns"
            raise ConfigError(msg)
        missing = sorted(set(names) - set(table.colnames))
        if missing:
            msg = f"[catalog] column(s) {missing} are not in {catalog.file}"
            raise ConfigError(msg)

        catalog_ids = _cast_catalog_ids(table[catalog.source_id], ids.dtype)
        position = {cid: i for i, cid in enumerate(catalog_ids.tolist())}
        if len(position) != len(catalog_ids):
            msg = f"[catalog] source IDs in {catalog.file} are not unique"
            raise ConfigError(msg)
        rows = np.array([position.get(i, -1) for i in ids.tolist()], dtype=np.int64)
        found = rows >= 0
        take = pa.array(np.where(found, rows, 0), mask=~found)
        for name in names:
            column = table[name]
            if column.ndim != 1:
                msg = f"[catalog] column {name!r} is not one value per source"
                raise ConfigError(msg)
            source = pa.array(np.asarray(np.ma.getdata(column)), mask=_missing(column))
            columns[name] = source.take(take)
            if getattr(column, "unit", None) is not None:
                units[name] = str(column.unit)

    write_parquet(
        config.run_dir / CATALOG_FILE, pa.table(columns), units=units, metadata=metadata
    )


def _cast_catalog_ids(column: Any, dtype: np.dtype) -> np.ndarray:
    values = _id_array(column)
    if dtype.kind != "i":
        return values.astype(str)
    try:
        return values.astype(np.int64)
    except ValueError as err:
        msg = (
            "[catalog] source IDs cannot be cast to the data table's integer "
            f"IDs: {err}"
        )
        raise ConfigError(msg) from err


@final
class PreparedData:
    """An open handle on a run's prepared data, for repeated reads.

    Reads ``data_index.parquet`` once and keeps ``data.parquet`` open, so a
    process that reads many sources (a pipeline slice, the viewer) pays the
    open cost once.

    Parameters
    ----------
    run_dir
        The run directory.

    Raises
    ------
    FileNotFoundError
        If the run has no prepared data (``hq prepare`` has not run).
    ProvenanceError
        If ``data.parquet`` and ``data_index.parquet`` come from different
        ``prepare`` runs.
    """

    def __init__(self, run_dir: str | os.PathLike) -> None:
        run_dir = Path(run_dir)
        index_path, data_path = run_dir / INDEX_FILE, run_dir / DATA_FILE
        if not index_path.exists():
            msg = f"{run_dir} has no prepared data; run hq prepare first"
            raise FileNotFoundError(msg)
        index = read_parquet(index_path)
        data_meta = read_metadata(data_path)
        if index.metadata["data_id"] != data_meta["data_id"]:
            msg = (
                f"{index_path} and {data_path} come from different prepare runs; "
                "rerun hq prepare --overwrite"
            )
            raise ProvenanceError(msg)

        self.kind: str = data_meta["kind"]
        self.data_id: str = data_meta["data_id"]
        self.provenance: dict[str, Any] = data_meta["provenance"]
        self.path = data_path
        ids = index.table.column("source_id").to_pylist()
        self._is_int = pa.types.is_integer(index.table.schema.field("source_id").type)
        self._ranges = dict(
            zip(
                ids,
                zip(
                    index.table.column("row_start").to_pylist(),
                    index.table.column("n_obs").to_pylist(),
                    strict=True,
                ),
                strict=True,
            )
        )
        self.source_ids: list[Any] = ids

        self._file = pq.ParquetFile(data_path)
        self._units = {
            field.name: field.metadata[b"unit"].decode()
            for field in self._file.schema_arrow
            if field.metadata and b"unit" in field.metadata
        }
        self._group_starts = row_group_starts(self._file)

    def n_obs(self, source_id: Any) -> int:
        """The number of prepared observations of one source."""
        return self._ranges[self._key(source_id)][1]

    def read(self, source_ids: Iterable[Any]) -> dict[Any, RVData | GaiaAstrometryData]:
        """Read many sources, touching only the row groups that hold them.

        Parameters
        ----------
        source_ids
            IDs to read, in any order.

        Returns
        -------
            A dict from source ID (as stored: ``int`` or ``str``) to its harv
            data object, in the order given.

        Raises
        ------
        KeyError
            Naming an ID that is not in the prepared data.
        """
        keys = [self._key(sid) for sid in source_ids]
        ranges = [self._ranges[k] for k in keys]
        if not ranges:
            return {}

        groups = sorted(
            {
                g
                for start, n in ranges
                for g in groups_spanning(self._group_starts, start, n)
            }
        )
        table = self._file.read_row_groups(groups)
        # Map a global row number to its position in the concatenated groups.
        offsets = {}
        position = 0
        for g in groups:
            offsets[g] = position - int(self._group_starts[g])
            position += int(self._group_starts[g + 1] - self._group_starts[g])

        columns = {name: table.column(name).to_numpy() for name in table.column_names}
        out: dict[Any, RVData | GaiaAstrometryData] = {}
        for key, (start, n) in zip(keys, ranges, strict=True):
            lo = start + offsets[groups_spanning(self._group_starts, start, n).start]
            out[key] = self._build({k: v[lo : lo + n] for k, v in columns.items()})
        return out

    def _key(self, source_id: Any) -> Any:
        key = int(source_id) if self._is_int else str(source_id)
        if key not in self._ranges:
            msg = f"source {source_id!r} is not in the prepared data"
            raise KeyError(msg)
        return key

    def _build(self, rows: dict[str, np.ndarray]) -> RVData | GaiaAstrometryData:
        time = Q(rows["time"], "day")
        if self.kind == "rv":
            return RVData(
                time=time,
                rv=Q(rows["rv"], self._units["rv"]),
                rv_err=Q(rows["rv_err"], self._units["rv_err"]),
            )
        return GaiaAstrometryData(
            time=time,
            al_position=Q(rows["al_position"], self._units["al_position"]),
            al_position_err=Q(rows["al_position_err"], self._units["al_position_err"]),
            scan_angle=Q(rows["scan_angle"], self._units["scan_angle"]),
            parallax_factor=jnp.asarray(rows["parallax_factor"]),
        )


def read_sources(
    run_dir: str | os.PathLike, source_ids: Iterable[Any]
) -> dict[Any, RVData | GaiaAstrometryData]:
    """Read many sources' prepared data, with one read of the row groups they span.

    Parameters
    ----------
    run_dir
        The run directory.
    source_ids
        IDs to read.

    Returns
    -------
        A dict from source ID to its ``RVData`` or ``GaiaAstrometryData``.
    """
    return PreparedData(run_dir).read(source_ids)


def read_source(
    run_dir: str | os.PathLike, source_id: Any
) -> RVData | GaiaAstrometryData:
    """Read one source's prepared data.

    Parameters
    ----------
    run_dir
        The run directory.
    source_id
        The source ID.

    Returns
    -------
        Its ``RVData`` or ``GaiaAstrometryData``; ``time_ref`` is harv's
        default (the mean time).
    """
    return next(iter(PreparedData(run_dir).read([source_id]).values()))
