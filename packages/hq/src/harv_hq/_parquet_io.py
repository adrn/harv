"""The one place hq writes and reads Parquet.

Owns the conventions in ``packages/hq/docs/spec.md``, "Storage formats":
units in each column's field metadata (key ``unit``), hq's key-value metadata
as JSON under ``hq.<name>`` keys, and an atomic write (to ``<name>.tmp``, then
renamed) so a file with its final name is always complete. No other module
calls ``pyarrow.parquet.write_table`` directly.
"""

__all__ = (
    "ParquetContents",
    "groups_spanning",
    "read_metadata",
    "read_parquet",
    "read_rows",
    "row_group_starts",
    "write_parquet",
)

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_PREFIX = "hq."


@final
@dataclass(frozen=True)
class ParquetContents:
    """A table read by :func:`read_parquet`, with its decoded metadata."""

    table: pa.Table
    units: dict[str, str]
    metadata: dict[str, Any]


def write_parquet(
    path: str | os.PathLike,
    data: pa.Table | Mapping[str, Any],
    *,
    units: Mapping[str, str] | None = None,
    field_metadata: Mapping[str, Mapping[str, str]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    row_group_size: int | None = None,
) -> None:
    """Write a table atomically, with hq's unit and metadata conventions.

    Parameters
    ----------
    path
        Destination file. Written to ``<path>.tmp`` and renamed into place; on
        failure the temporary file is removed and ``path`` is untouched.
    data
        A ``pyarrow.Table`` or a mapping of column name to array.
    units
        Unit string per column, stored as field metadata ``unit``.
    field_metadata
        Any further per-column field metadata (e.g. a time column's format).
    metadata
        Key-value metadata; each value is JSON-encoded under ``hq.<key>``.
    row_group_size
        Rows per Parquet row group (pyarrow's default when ``None``).
    """
    path = Path(path)
    table = data if isinstance(data, pa.Table) else pa.table(dict(data))

    per_field: dict[str, dict[str, str]] = {}
    for name, unit in (units or {}).items():
        per_field.setdefault(name, {})["unit"] = unit
    for name, extra in (field_metadata or {}).items():
        per_field.setdefault(name, {}).update(extra)
    unknown = sorted(set(per_field) - set(table.column_names))
    if unknown:
        msg = f"field metadata given for columns {unknown} not in the table"
        raise ValueError(msg)

    schema = pa.schema(
        [
            field.with_metadata(per_field[field.name])
            if field.name in per_field
            else field
            for field in table.schema
        ],
        metadata={
            f"{_PREFIX}{key}": json.dumps(value)
            for key, value in (metadata or {}).items()
        },
    )
    table = table.cast(schema)

    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp, row_group_size=row_group_size)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_parquet(
    path: str | os.PathLike, columns: list[str] | None = None
) -> ParquetContents:
    """Read a table written by :func:`write_parquet`.

    Parameters
    ----------
    path
        The file to read.
    columns
        Columns to read; all of them when ``None``.

    Returns
    -------
        The table, its per-column units (columns without a ``unit`` are
        omitted), and its decoded ``hq.*`` key-value metadata.
    """
    table = pq.read_table(path, columns=columns)
    units = {
        field.name: field.metadata[b"unit"].decode()
        for field in table.schema
        if field.metadata and b"unit" in field.metadata
    }
    return ParquetContents(
        table=table, units=units, metadata=_decode(table.schema.metadata)
    )


def read_metadata(path: str | os.PathLike) -> dict[str, Any]:
    """Read only a file's ``hq.*`` key-value metadata, from its footer.

    Parameters
    ----------
    path
        The file to read.

    Returns
    -------
        The decoded metadata, keyed without the ``hq.`` prefix.
    """
    return _decode(pq.read_schema(path).metadata)


def row_group_starts(parquet_file: pq.ParquetFile) -> np.ndarray:
    """First row of each row group, plus the total row count at the end."""
    sizes = [
        parquet_file.metadata.row_group(i).num_rows
        for i in range(parquet_file.num_row_groups)
    ]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


def groups_spanning(starts: np.ndarray, first: int, n_rows: int) -> range:
    """The row groups holding rows ``first`` to ``first + n_rows``."""
    lo = int(np.searchsorted(starts, first, side="right")) - 1
    hi = int(np.searchsorted(starts, first + n_rows, side="left"))
    return range(lo, max(hi, lo + 1))


def read_rows(
    parquet_file: pq.ParquetFile,
    first: int,
    n_rows: int,
    *,
    starts: np.ndarray | None = None,
) -> pa.Table:
    """Read rows ``first`` to ``first + n_rows``, decoding only their row groups.

    Parameters
    ----------
    parquet_file
        An open file.
    first, n_rows
        The row range.
    starts
        Cached :func:`row_group_starts` of the file, to skip recomputing it.

    Returns
    -------
        The rows, with the file's schema.
    """
    if starts is None:
        starts = row_group_starts(parquet_file)
    groups = groups_spanning(starts, first, n_rows)
    table = parquet_file.read_row_groups(list(groups))
    return table.slice(first - int(starts[groups.start]), n_rows)


def _decode(raw: Mapping[bytes, bytes] | None) -> dict[str, Any]:
    """Decode ``hq.*`` keys; other writers' metadata (e.g. pandas) is ignored."""
    return {
        key.decode().removeprefix(_PREFIX): json.loads(value)
        for key, value in (raw or {}).items()
        if key.decode().startswith(_PREFIX)
    }
