"""Tests for the web viewer: routes, filters, IDs, and source payloads."""

import shutil
import types
from datetime import UTC, datetime
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import uvicorn
from fastapi.testclient import TestClient
from unxt import Q

from harv.samplers import Samples
from harv_hq import Run, create_app
from harv_hq._model_file import ModelFile
from harv_hq.cli import main
from harv_hq.results import (
    PartWriter,
    ResultsIndex,
    process_source,
    read_source_row,
    read_source_rows,
)
from harv_hq.summarize import SUMMARY_FILE
from harv_hq.web import payload as payload_module
from harv_hq.web.app import _Viewer
from harv_hq.web.payload import (
    MAX_CURVE_SAMPLES,
    _curve_json,
    _gaia_curves,
    _rv_curves,
    _samples_payload,
    _shown,
    jsonable,
)

PLUS, SLASH = "2M00000001+0000007", "J1234/5678"  # IDs that need URL encoding
MCMC_ID = "2M00000003+0000021"
MCMC_CHAINS = 2


def write_mcmc_result(run, source_id):
    """An MCMC result made from the rejection samples, without running NUTS.

    The viewer only needs a result to show; a real run would compile NUTS
    here too, and this module's memory is what CI runners run short of.
    """
    rejection = run.load_source(source_id)
    samples = rejection.rejection
    chained = Samples(
        nonlinear=samples.nonlinear,
        linear=samples.linear,
        model_type=samples.model_type,
        metadata={**samples.metadata, "num_chains": MCMC_CHAINS},
        ln_likelihood=samples.ln_likelihood,
        ln_prior=samples.ln_prior,
    )
    columns = {"mcmc_status": "converged", "r_hat_max": 1.0, "ess_bulk_min": 100.0}
    payload = process_source(
        source_id,
        rejection.data,
        stage="mcmc",
        compute=lambda: (chained, columns, {}),
    )
    writer = PartWriter(
        run.run_dir / "results" / "mcmc",
        stage="mcmc",
        shard=(0, 1),
        provenance=run._stage_provenance("mcmc"),
        flush_n_sources=1,
        flush_seconds=60.0,
    )
    writer.add(payload)
    return samples.n_samples


@pytest.fixture(scope="module")
def rv_dir(tmp_path_factory, rv_run_factory):
    """An RV run with rejection results, one MCMC result, and a summary."""
    fixture = rv_run_factory(tmp_path_factory.mktemp("web"))
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection()
    write_mcmc_result(run, MCMC_ID)
    run.summarize()
    return fixture.run_dir


@pytest.fixture(scope="module")
def client(rv_dir):
    return TestClient(create_app(rv_dir))


@pytest.fixture(scope="module")
def gaia_client(tmp_path_factory, gaia_run_factory):
    fixture = gaia_run_factory(tmp_path_factory.mktemp("web-gaia"))
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection()
    run.summarize()
    return TestClient(create_app(fixture.run_dir)), fixture


@pytest.fixture(scope="module")
def sb2_client(tmp_path_factory, sb2_run_factory):
    fixture = sb2_run_factory(tmp_path_factory.mktemp("web-sb2"))
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection()
    run.summarize()
    return TestClient(create_app(fixture.run_dir)), fixture


def summary(rv_dir):
    return pq.read_table(rv_dir / SUMMARY_FILE)


class TestPages:
    def test_index(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "catalog.js" in response.text

    @pytest.mark.parametrize("sid", [PLUS, SLASH])
    def test_source_page(self, client, sid):
        response = client.get(f"/source/{quote(sid, safe='')}")
        assert response.status_code == 200
        assert "source.js" in response.text

    def test_unknown_source(self, client):
        assert client.get("/source/nope").status_code == 404
        assert client.get("/api/source/nope").status_code == 404

    def test_static(self, client):
        for name in ("catalog.js", "source.js", "style.css"):
            assert client.get(f"/static/{name}").status_code == 200


class TestCatalog:
    def test_columns(self, client, rv_dir):
        columns = {c["name"]: c for c in client.get("/api/columns").json()}
        assert set(columns) == set(summary(rv_dir).column_names)
        assert columns["abs_g"]["unit"] == "mag"
        assert columns["abs_g"]["numeric"]
        assert not columns["rejection_status"]["numeric"]

    def test_whole_columns(self, client, rv_dir):
        body = client.get("/api/catalog?columns=source_id,abs_g,ln_Z_int").json()
        table = summary(rv_dir)
        assert body["source_id"] == table["source_id"].to_pylist()
        assert body["abs_g"] == table["abs_g"].to_pylist()

    def test_filters(self, client, rv_dir):
        body = client.get(
            "/api/catalog?columns=source_id,abs_g,n_obs&filter=abs_g:1:,n_obs::6"
        ).json()
        table = summary(rv_dir).to_pylist()
        # Sources missing from the catalog have a null abs_g: filtered out.
        expected = [
            r["source_id"]
            for r in table
            if r["abs_g"] is not None and r["abs_g"] >= 1 and r["n_obs"] <= 6
        ]
        assert body["source_id"] == expected
        assert 0 < len(expected) < len(table)
        ok = client.get("/api/catalog?status=ok").json()["source_id"]
        assert len(ok) == len(table)
        assert client.get("/api/catalog?status=failed").json() == {"source_id": []}

    @pytest.mark.parametrize(
        "query",
        [
            "filter=abs_g:1",
            "filter=nope:1:2",
            "filter=rejection_status:1:2",
            "filter=abs_g:x:",
            "columns=source_id,nope",
        ],
    )
    def test_malformed_requests(self, client, query):
        response = client.get(f"/api/catalog?{query}")
        assert response.status_code == 400
        assert response.json()["detail"]

    def test_rows(self, client, rv_dir):
        response = client.post(
            "/api/rows",
            json={"source_ids": [SLASH, PLUS], "columns": ["source_id", "abs_g"]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["source_id"] == [SLASH, PLUS]
        rows = {r["source_id"]: r for r in summary(rv_dir).to_pylist()}
        assert body["abs_g"] == [rows[SLASH]["abs_g"], rows[PLUS]["abs_g"]]
        missing = {"source_ids": ["nope"], "columns": ["source_id"]}
        assert client.post("/api/rows", json=missing).status_code == 404

    def test_reloads_when_the_summary_changes(self, rv_dir, tmp_path):
        run_dir = shutil.copytree(rv_dir, tmp_path / "run")
        client = TestClient(create_app(run_dir))
        assert client.get("/api/catalog?status=ok").status_code == 200
        (run_dir / SUMMARY_FILE).unlink()
        # Back to catalog.parquet alone: no statuses to filter on yet.
        response = client.get("/api/catalog?status=ok")
        assert response.status_code == 400
        assert "hq summarize" in response.json()["detail"]
        assert "rejection_status" not in {
            c["name"] for c in client.get("/api/columns").json()
        }


class TestSourcePayload:
    @pytest.mark.parametrize("sid", [PLUS, SLASH])
    def test_rv(self, client, sid):
        response = client.get(f"/api/source/{quote(sid, safe='')}")
        assert response.status_code == 200
        body = response.json()
        assert body["source_id"] == sid
        assert body["kind"] == "rv"
        assert body["summary"]["source_id"] == sid
        n_obs = len(body["data"]["time"])
        assert len(body["data"]["rv"]) == len(body["data"]["rv_err"]) == n_obs
        assert body["data"]["units"]["rv"] == "km / s"

        rejection = body["stages"]["rejection"]
        assert rejection["status"] == "ok"
        assert rejection["error"] == ""
        n = len(rejection["params"]["period"])
        assert 0 < n <= MAX_CURVE_SAMPLES
        assert all(len(v) == n for v in rejection["params"].values())
        curves = rejection["curves"]
        assert len(curves["rv"]) == n
        assert {len(c) for c in curves["rv"]} == {len(curves["time"])}
        assert len(curves["time"]) <= 4096
        assert body["stages"]["mcmc"]["status"] == "pending"
        assert "params" not in body["stages"]["mcmc"]

    def test_both_stages(self, client, rv_dir):
        body = client.get(f"/api/source/{quote(MCMC_ID, safe='')}").json()
        mcmc = body["stages"]["mcmc"]
        assert mcmc["status"] == "ok"
        assert mcmc["mcmc_status"] == "converged"
        n = Run(rv_dir).load_source(MCMC_ID).mcmc.n_samples
        assert len(mcmc["params"]["period"]) == n  # every draw, under the cap
        assert len(mcmc["curves"]["rv"]) == n
        assert "params" in body["stages"]["rejection"]

    def test_gaia(self, gaia_client):
        client, fixture = gaia_client
        sid = next(iter(fixture.kept))
        # 19-digit IDs travel as strings: JS numbers cannot hold them exactly.
        ids = client.get("/api/catalog").json()["source_id"]
        assert str(sid) in ids
        body = client.get(f"/api/source/{sid}").json()
        assert body["source_id"] == str(sid)
        assert body["summary"]["source_id"] == str(sid)
        assert body["kind"] == "gaia_astrometry"
        n_obs = fixture.kept[sid]
        assert len(body["data"]["al_position"]) == n_obs
        curves = body["stages"]["rejection"]["curves"]
        n = len(body["stages"]["rejection"]["params"]["period"])
        assert np.shape(curves["ra"]) == np.shape(curves["dec"]) == (n, 256)
        assert np.shape(curves["residual"]) == (n, n_obs)
        assert client.get("/api/source/12345").status_code == 404
        assert client.get("/api/source/not-an-int").status_code == 404

    def test_sb2(self, sb2_client):
        client, fixture = sb2_client
        body = client.get("/api/source/103").json()
        assert body["kind"] == "sb2"
        table = fixture.table[fixture.table["source_id"] == 103]
        for name, label in (("primary", 1), ("secondary", 2)):
            data = body["data"][name]
            assert len(data["rv"]) == np.sum(table["component"] == label)
            assert data["units"]["rv"] == "km / s"

        rejection = body["stages"]["rejection"]
        assert "primary.rv_semiamp" in rejection["params"]
        assert "mass_ratio_p50" in body["summary"]
        curves = rejection["curves"]
        n = len(rejection["params"]["period"])
        for name in ("primary", "secondary"):
            assert np.shape(curves[name]) == (n, len(curves["time"]))
        # The stars move in antiphase about the shared systemic velocity.
        primary, secondary = (
            np.asarray(curves[k][0]) for k in ("primary", "secondary")
        )
        assert np.corrcoef(primary, secondary)[0, 1] < -0.99
        assert client.get("/source/103").status_code == 200


class TestReload:
    def test_a_rewritten_summary_is_picked_up_without_reimporting(
        self, rv_dir, tmp_path, monkeypatch
    ):
        run_dir = shutil.copytree(rv_dir, tmp_path / "run")
        viewer = _Viewer(run_dir)
        first = viewer.current()
        assert viewer.current() is first  # unchanged: the same snapshot
        loads = []
        load = ModelFile.load
        monkeypatch.setattr(
            ModelFile, "load", lambda path: loads.append(path) or load(path)
        )
        # Rewritten at once, likely within one mtime tick: a new inode.
        Run(run_dir).summarize()
        second = viewer.current()
        assert second is not first
        assert second.run is not first.run  # a fresh results index
        assert loads == []  # the model file is not imported again
        assert second.run.config is viewer.run.config

    def test_a_snapshot_is_one_version(self, rv_dir, tmp_path):
        """Its results index and summary row do not follow later changes."""
        run_dir = shutil.copytree(rv_dir, tmp_path / "run")
        snapshot = _Viewer(run_dir).current()
        assert set(snapshot.run._indexes) == {"rejection", "mcmc"}  # built now
        before = snapshot.summary_row(SLASH)
        assert before["source_id"] == SLASH
        (run_dir / SUMMARY_FILE).unlink()
        assert snapshot.summary_row(SLASH) == before

    def test_a_summary_deleted_while_loading_falls_back_to_the_catalog(
        self, rv_dir, tmp_path, monkeypatch
    ):
        run_dir = shutil.copytree(rv_dir, tmp_path / "run")
        viewer = _Viewer(run_dir)
        (run_dir / SUMMARY_FILE).unlink()
        snapshot = viewer.current()
        assert snapshot.stamp is None
        assert "rejection_status" not in snapshot.table.column_names


class TestRobustPayloads:
    @pytest.fixture
    def rv_source(self, rv_dir):
        run = Run(rv_dir)
        result = run.load_source(SLASH)
        model = run.model_file.setup("rv")[1]
        return result, model

    @staticmethod
    def with_periods(samples, periods) -> Samples:
        return Samples(
            nonlinear={**samples.nonlinear, "period": Q(periods, "day")},
            linear=samples.linear,
            model_type=samples.model_type,
            metadata=samples.metadata,
            ln_likelihood=samples.ln_likelihood,
            ln_prior=samples.ln_prior,
        )

    def test_nan_periods_are_skipped(self, rv_source):
        result, model = rv_source
        samples = result.rejection
        periods = np.asarray(samples["period"].value).copy()
        periods[0] = np.nan
        curves = _rv_curves(self.with_periods(samples, periods), result.data, model)
        assert curves["rv"][0] == [None] * len(curves["time"])
        assert curves["rv"][1][0] is not None

    def test_curves_that_fail_leave_the_rest_of_the_page(self, rv_source):
        result, model = rv_source
        samples = result.rejection
        nan = self.with_periods(samples, np.full(samples.n_samples, np.nan))
        payload = _samples_payload(nan, result.data, model)
        assert "no sample has a finite period" in payload["curves_error"]
        assert "curves" not in payload
        assert len(payload["params"]["period"]) == samples.n_samples

    def test_a_page_survives_failing_curves(self, client, monkeypatch):
        def broken(*args: object, **kwargs: object):
            raise RuntimeError("boom")

        monkeypatch.setattr(payload_module, "_rv_curves", broken)
        response = client.get(f"/api/source/{quote(SLASH, safe='')}")
        assert response.status_code == 200
        assert "boom" in response.json()["stages"]["rejection"]["curves_error"]

    def test_no_samples_shown(self, rv_source):
        result, model = rv_source
        empty = result.rejection[np.array([], dtype=int)]
        payload = _samples_payload(empty, result.data, model)
        assert payload["params"]["period"] == []
        assert "curves" not in payload
        assert "curves_error" not in payload

    def test_gaia_without_a_sky_orbit(self, gaia_client):
        _, fixture = gaia_client
        run = Run(fixture.run_dir)
        result = run.load_source(next(iter(fixture.kept)))
        model = run.model_file.setup("gaia_astrometry")[1]
        # A parameterization with no sky orbit (as a Fourier fit has none).
        stub = types.SimpleNamespace(
            parameterization=types.SimpleNamespace(
                params=model.parameterization.params
            ),
            predict=model.predict,
            _all_nonlinear_names=model._all_nonlinear_names,
            _linear_param_units=model._linear_param_units,
        )
        curves = _gaia_curves(result.rejection, result.data, stub)
        assert set(curves) == {"residual"}


def test_curve_values_keep_six_significant_digits():
    assert _curve_json(np.array([123.456789, np.nan, -1.0])) == [123.457, None, -1.0]
    assert _curve_json(np.array([0.00123456789])) == [0.00123457]
    assert _curve_json(np.array([[0.0, np.inf]])) == [[0.0, None]]


def test_read_source_row(rv_dir):
    records = ResultsIndex.build(rv_dir / "results" / "rejection").records
    record = records[SLASH]
    assert read_source_row(record) == read_source_rows([record])[0].to_pylist()[0]
    assert pq.ParquetFile(record.sources_path).metadata.row_group(0).num_rows <= 4096


def test_load_source_returns_the_stage_rows(rv_dir):
    result = Run(rv_dir).load_source(MCMC_ID)
    assert result.rejection_row["status"] == "ok"
    assert result.rejection_row["source_id"] == MCMC_ID
    assert result.mcmc_row["mcmc_status"] == "converged"
    assert Run(rv_dir).load_source(PLUS).mcmc_row is None


def _weighted(n, ln_likelihood):
    return Samples(
        nonlinear={"period": Q(np.arange(1.0, n + 1), "day")},
        linear={},
        model_type="RVModel",
        metadata={"ln_Z_int": 0.0, "n_prior_samples": n},
        ln_likelihood=np.asarray(ln_likelihood, dtype=float),
        ln_prior=np.zeros(n),
    )


def test_shown_samples():
    # Weighted: at most the cap, never a zero-weight sample, no repeats.
    ln_l = np.zeros(1000)
    ln_l[::2] = -np.inf
    shown = _shown(_weighted(1000, ln_l), weighted=True)
    assert len(shown) == len(set(shown)) == MAX_CURVE_SAMPLES
    assert np.all(shown % 2 == 1)
    # Fewer nonzero weights than the cap: all of them.
    ten = np.where(np.arange(1000) < 10, 0.0, -np.inf)
    assert len(_shown(_weighted(1000, ten), weighted=True)) == 10
    # Equal weight: evenly spaced, first and last included.
    even = _shown(_weighted(1000, np.zeros(1000)), weighted=False)
    assert len(even) == MAX_CURVE_SAMPLES
    assert even[0] == 0
    assert even[-1] == 999


def test_jsonable_turns_non_finite_into_null():
    column = pa.chunked_array([[1.0, float("nan"), None, float("inf")]])
    assert jsonable(column) == [1.0, None, None, None]
    assert jsonable(pa.array([1, 2])) == [1, 2]
    assert jsonable(np.array([np.nan, 2.0])) == [None, 2.0]
    # FITS strings and timestamps, which JSON cannot hold as they are.
    assert jsonable(pa.array([b"HARPS"])) == ["HARPS"]
    when = datetime(2026, 1, 2, tzinfo=UTC)
    assert jsonable(pa.array([when]))[0].startswith("2026-01-02")
    assert jsonable({"telescope": b"HARPS", "at": when}) == {
        "telescope": "HARPS",
        "at": when.isoformat(),
    }


def test_cli_serve(rv_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs))
    )
    config = rv_dir / "hq.toml"
    text = config.read_text()
    try:
        config.write_text(text + '\n[serve]\nhost = "0.0.0.0"\nport = 9001\n')
        assert main(["serve", "--run-dir", str(rv_dir)]) == 0
    finally:
        config.write_text(text)
    ((app, kwargs),) = calls
    assert kwargs == {"host": "0.0.0.0", "port": 9001}  # noqa: S104
    assert app.title.startswith("hq:")
