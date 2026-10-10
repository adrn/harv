"""Tests for ``hq prepare`` and ``read_source`` / ``read_sources``."""

import shutil
import time

import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.table import Table
from astropy.time import Time
from unxt import ustrip

from harv.data import GaiaAstrometryData, RVData
from harv_hq import ConfigError, ProvenanceError, Run, read_source, read_sources
from harv_hq import prepare as prepare_module
from harv_hq._parquet_io import read_metadata, read_parquet
from harv_hq.cli import main
from harv_hq.prepare import PreparedData
from harv_hq.provenance import check_provenance, make_provenance, sha256_file


def kept_rows(table, sid):
    """The input rows of one source that survive the built-in cuts, time-sorted."""
    rows = table[table["source_id"] == sid]
    rows = rows[np.isfinite(rows["rv"]) & (rows["rv_err"] > 0)]
    rows.sort("time")
    return rows


class TestRV:
    def test_round_trip(self, rv_run):
        Run(rv_run.run_dir).prepare()
        kept = rv_run.kept
        data = read_sources(rv_run.run_dir, kept)

        assert list(data) == list(kept)
        for sid, source in data.items():
            assert isinstance(source, RVData)
            rows = kept_rows(rv_run.table, sid)
            expected_time = Time(rows["time"], format="jd", scale="tdb").tcb.mjd
            np.testing.assert_allclose(
                ustrip("day", source.time), expected_time, rtol=0, atol=1e-9
            )
            np.testing.assert_array_equal(ustrip("km/s", source.rv), rows["rv"])
            np.testing.assert_array_equal(ustrip("km/s", source.rv_err), rows["rv_err"])

    def test_index_and_dropped_sources(self, rv_run):
        Run(rv_run.run_dir).prepare()
        index = read_parquet(rv_run.run_dir / "data_index.parquet").table
        stored = dict(
            zip(index["source_id"].to_pylist(), index["n_obs"].to_pylist(), strict=True)
        )
        assert stored == rv_run.kept
        # Row ranges tile data.parquet with no gaps.
        starts = index["row_start"].to_numpy()
        np.testing.assert_array_equal(
            starts[1:], starts[:-1] + index["n_obs"].to_numpy()[:-1]
        )

    def test_ids_with_plus_and_slash(self, rv_run):
        Run(rv_run.run_dir).prepare()
        assert (
            read_source(rv_run.run_dir, "J1234/5678").n_obs
            == rv_run.n_obs["J1234/5678"]
        )
        plus = "2M00000001+0000007"
        assert read_source(rv_run.run_dir, plus).n_obs == rv_run.n_obs[plus]

    def test_select_rows_sees_every_column(self, rv_run):
        """The hook can cut on columns the config does not map (here, snr)."""
        prior = rv_run.run_dir / "prior.py"
        prior.write_text(
            prior.read_text()
            + "\ndef select_rows(table):\n    return table['snr'] > 10\n"
        )
        Run(rv_run.run_dir).prepare()
        bad = "2M00000003+0000021"
        assert read_source(rv_run.run_dir, bad).n_obs == rv_run.n_obs[bad] - 1

    def test_column_unit_wins_else_config_unit(self, rv_run):
        Run(rv_run.run_dir).prepare()
        units = read_parquet(rv_run.run_dir / "data.parquet").units
        assert units == {"time": "day", "rv": "km / s", "rv_err": "km / s"}

        table = Table(rv_run.table, copy=True)
        table["rv"].unit = table["rv_err"].unit = None
        table.write(rv_run.run_dir / "observations.ecsv", overwrite=True)
        config = rv_run.run_dir / "hq.toml"
        config.write_text(
            config.read_text().replace('rv_unit = "km/s"', 'rv_unit = "m/s"')
        )
        Run(rv_run.run_dir).prepare(overwrite=True)
        source = read_source(rv_run.run_dir, "J1234/5678")
        assert str(source.rv.unit) == "m / s"

    def test_metadata_and_provenance(self, rv_run):
        Run(rv_run.run_dir).prepare()
        meta = read_metadata(rv_run.run_dir / "data.parquet")
        assert meta["kind"] == "rv"
        assert meta["provenance"]["data_id"] == meta["data_id"]
        assert meta["provenance"]["model_file_sha256"] == sha256_file(
            rv_run.run_dir / "prior.py"
        )
        for name in ("data_index.parquet", "catalog.parquet"):
            assert read_metadata(rv_run.run_dir / name)["data_id"] == meta["data_id"]
        time_field = pq.read_schema(rv_run.run_dir / "data.parquet").field("time")
        assert time_field.metadata[b"scale"] == b"tcb"

    def test_editing_the_model_file_makes_prepared_data_stale(self, rv_run):
        """select_rows lives in prior.py, so an edit must force a re-prepare."""
        run = Run(rv_run.run_dir)
        run.prepare()
        meta = read_metadata(rv_run.run_dir / "data.parquet")
        prior = rv_run.run_dir / "prior.py"
        prior.write_text(prior.read_text() + "\n# tightened cuts\n")
        current = make_provenance(run.config, data_id=meta["data_id"])
        with pytest.raises(ProvenanceError, match="model_file_sha256"):
            check_provenance(meta["provenance"], current, path="data.parquet")

    def test_overwrite(self, rv_run):
        run = Run(rv_run.run_dir)
        run.prepare()
        first = read_metadata(rv_run.run_dir / "data.parquet")["data_id"]
        with pytest.raises(FileExistsError, match="overwrite"):
            run.prepare()
        run.prepare(overwrite=True)
        assert read_metadata(rv_run.run_dir / "data.parquet")["data_id"] != first

    def test_no_sources_left(self, rv_run):
        config = rv_run.run_dir / "hq.toml"
        config.write_text(
            config.read_text().replace("min_n_obs = 3", "min_n_obs = 1000")
        )
        with pytest.raises(ValueError, match="no sources left"):
            Run(rv_run.run_dir).prepare()


class TestCatalog:
    def test_join(self, rv_run):
        Run(rv_run.run_dir).prepare()
        contents = read_parquet(rv_run.run_dir / "catalog.parquet")
        table = contents.table
        index = read_parquet(rv_run.run_dir / "data_index.parquet").table
        assert table["source_id"].to_pylist() == index["source_id"].to_pylist()
        assert table.column_names == [
            "source_id",
            "n_obs",
            "time_baseline",
            "bp_rp",
            "abs_g",
        ]
        assert contents.units == {"time_baseline": "day", "abs_g": "mag"}

        rows = {r["source_id"]: r for r in table.to_pylist()}
        assert "2M99999999+9999999" not in rows  # catalog-only sources are dropped
        assert rows["2M00000001+0000007"]["bp_rp"] is None  # no catalog row: null
        source = rows["J1234/5678"]
        assert source["bp_rp"] is not None
        t = kept_rows(rv_run.table, "J1234/5678")["time"]
        assert source["time_baseline"] == pytest.approx(float(t.max() - t.min()))

    def test_reserved_column_name(self, rv_run):
        catalog = rv_run.catalog.copy()
        catalog.rename_column("bp_rp", "n_obs")
        catalog.write(rv_run.run_dir / "catalog.ecsv", overwrite=True)
        with pytest.raises(ConfigError, match="collide"):
            Run(rv_run.run_dir).prepare()

    def test_duplicate_ids(self, rv_run):
        catalog = rv_run.catalog.copy()
        catalog.add_row(catalog[0])
        catalog.write(rv_run.run_dir / "catalog.ecsv", overwrite=True)
        with pytest.raises(ConfigError, match="not unique"):
            Run(rv_run.run_dir).prepare()

    def test_selected_columns(self, rv_run):
        config = rv_run.run_dir / "hq.toml"
        config.write_text(config.read_text() + 'columns = ["abs_g"]\n')
        Run(rv_run.run_dir).prepare()
        names = read_parquet(rv_run.run_dir / "catalog.parquet").table.column_names
        assert names == ["source_id", "n_obs", "time_baseline", "abs_g"]


class TestGaia:
    def test_round_trip_with_integer_ids(self, gaia_run):
        Run(gaia_run.run_dir).prepare()
        sid = next(iter(gaia_run.n_obs))
        source = read_source(gaia_run.run_dir, sid)
        assert isinstance(source, GaiaAstrometryData)
        assert read_source(gaia_run.run_dir, str(sid)).n_obs == source.n_obs

        rows = gaia_run.table[gaia_run.table["source_id"] == sid]
        rows.sort("time")
        np.testing.assert_array_equal(ustrip("mas", source.al_position), rows["al"])
        # The column's own unit (rad) wins over the config's scan_angle_unit (deg).
        np.testing.assert_allclose(ustrip("rad", source.scan_angle), rows["psi"])
        np.testing.assert_array_equal(
            np.asarray(source.parallax_factor), rows["plx_factor"]
        )

        index = read_parquet(gaia_run.run_dir / "data_index.parquet").table
        assert index.schema.field("source_id").type == "int64"


class TestReading:
    def test_reads_only_the_needed_row_groups(self, rv_run, monkeypatch):
        monkeypatch.setattr(prepare_module, "ROW_GROUP_SIZE", 8)
        Run(rv_run.run_dir).prepare()
        prepared = PreparedData(rv_run.run_dir)
        assert prepared._file.num_row_groups > 5

        calls = []
        original = prepared._file.read_row_groups
        monkeypatch.setattr(
            prepared._file,
            "read_row_groups",
            lambda groups: calls.append(groups) or original(groups),
        )
        sid = "2M00000010+0000070"
        source = prepared.read([sid])[sid]
        assert len(calls) == 1
        assert len(calls[0]) <= 2  # a source spans at most two 8-row groups here
        assert source.n_obs == rv_run.n_obs[sid]

        # Sources whose rows straddle a row-group boundary still come back whole.
        everything = prepared.read(prepared.source_ids)
        assert {s: d.n_obs for s, d in everything.items()} == rv_run.kept

    def test_unknown_source(self, rv_run):
        Run(rv_run.run_dir).prepare()
        with pytest.raises(KeyError, match="not in the prepared data"):
            read_source(rv_run.run_dir, "nope")

    def test_not_prepared(self, rv_run):
        with pytest.raises(FileNotFoundError, match="hq prepare"):
            read_source(rv_run.run_dir, "J1234/5678")

    def test_mixed_prepare_runs_are_refused(self, rv_run, tmp_path):
        Run(rv_run.run_dir).prepare()
        shutil.copy(
            rv_run.run_dir / "data_index.parquet", tmp_path / "old_index.parquet"
        )
        Run(rv_run.run_dir).prepare(overwrite=True)
        shutil.copy(
            tmp_path / "old_index.parquet", rv_run.run_dir / "data_index.parquet"
        )
        with pytest.raises(ProvenanceError, match="different prepare runs"):
            PreparedData(rv_run.run_dir)


def test_cli_prepare(rv_run):
    assert main(["prepare", "--run-dir", str(rv_run.run_dir)]) == 0
    with pytest.raises(SystemExit) as excinfo:
        main(["prepare", "--run-dir", str(rv_run.run_dir)])
    assert "already has prepared data" in str(excinfo.value.code)


def test_large_table_prepares_quickly(tmp_path, rv_run):
    """10^5 rows over 10^4 sources: guards against per-source scans (loose bound)."""
    rng = np.random.default_rng(0)
    n_rows = 100_000
    table = Table(
        {
            "source_id": rng.integers(0, 10_000, n_rows),
            "time": 2_460_000.0 + rng.uniform(0, 1000, n_rows),
            "rv": rng.normal(0, 10, n_rows),
            "rv_err": np.full(n_rows, 1.0),
            "snr": np.full(n_rows, 50.0),
        }
    )
    table.write(rv_run.run_dir / "observations.ecsv", overwrite=True)
    config = rv_run.run_dir / "hq.toml"
    text = config.read_text()
    config.write_text(text[: text.index("\n[catalog]")] + "\n")

    start = time.perf_counter()
    Run(rv_run.run_dir).prepare()
    assert time.perf_counter() - start < 60
    assert len(PreparedData(rv_run.run_dir).source_ids) > 9_000
