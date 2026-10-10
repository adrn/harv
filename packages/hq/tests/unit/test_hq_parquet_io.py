"""Tests for ``harv_hq._parquet_io``: units, metadata, and atomic writes."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from harv_hq import _parquet_io
from harv_hq._parquet_io import read_metadata, read_parquet, write_parquet

PROVENANCE = {"config_sha256": "abc", "versions": {"harv": "1.0"}, "n": 3}


def test_round_trip(tmp_path):
    path = tmp_path / "t.parquet"
    write_parquet(
        path,
        {"source_id": ["a", "b"], "time": [1.0, 2.0], "rv": [3.0, 4.0]},
        units={"time": "day", "rv": "km / s"},
        field_metadata={"time": {"format": "mjd", "scale": "tcb"}},
        metadata={"provenance": PROVENANCE, "kind": "rv"},
    )
    contents = read_parquet(path)
    assert contents.table.column("rv").to_pylist() == [3.0, 4.0]
    assert contents.units == {"time": "day", "rv": "km / s"}
    assert contents.metadata == {"provenance": PROVENANCE, "kind": "rv"}
    time_field = contents.table.schema.field("time")
    assert time_field.metadata[b"format"] == b"mjd"
    assert time_field.metadata[b"unit"] == b"day"


def test_column_subset_and_footer_only(tmp_path):
    path = tmp_path / "t.parquet"
    write_parquet(path, {"a": [1], "b": [2]}, metadata={"provenance": PROVENANCE})
    assert read_parquet(path, columns=["b"]).table.column_names == ["b"]
    assert read_metadata(path) == {"provenance": PROVENANCE}


def test_accepts_a_pyarrow_table_and_row_group_size(tmp_path):
    path = tmp_path / "t.parquet"
    write_parquet(path, pa.table({"x": np.arange(10)}), row_group_size=3)
    assert pq.ParquetFile(path).num_row_groups == 4


def test_other_metadata_is_ignored(tmp_path):
    path = tmp_path / "t.parquet"
    table = pa.table({"x": [1]}).replace_schema_metadata({"pandas": "{}"})
    write_parquet(path, table, metadata={"kind": "rv"})
    assert read_metadata(path) == {"kind": "rv"}


def test_unknown_field_metadata_column_raises(tmp_path):
    with pytest.raises(ValueError, match="not in the table"):
        write_parquet(tmp_path / "t.parquet", {"x": [1]}, units={"y": "m"})


def test_failed_write_leaves_nothing_and_keeps_the_old_file(tmp_path, monkeypatch):
    path = tmp_path / "t.parquet"
    write_parquet(path, {"x": [1]})

    def broken_write(table, where, **kwargs: object):
        where.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(_parquet_io.pq, "write_table", broken_write)
    with pytest.raises(OSError, match="disk full"):
        write_parquet(path, {"x": [2]})

    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.parquet"]
    assert pq.read_table(path).column("x").to_pylist() == [1]
