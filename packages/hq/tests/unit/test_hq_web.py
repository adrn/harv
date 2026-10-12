"""Tests for the web viewer: routes, filters, IDs, and source payloads."""

import shutil
from urllib.parse import quote

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from unxt import Q

from harv.samplers import Samples
from harv_hq import Run, create_app
from harv_hq import cli as cli_module
from harv_hq.cli import main
from harv_hq.summarize import SUMMARY_FILE
from harv_hq.web.payload import MAX_CURVE_SAMPLES, _shown, jsonable

PLUS, SLASH = "2M00000001+0000007", "J1234/5678"  # IDs that need URL encoding
MCMC_ID = "2M00000003+0000021"
HOOK = f"""

def select_for_mcmc(row):
    return row["source_id"] == {MCMC_ID!r}
"""
MCMC = """
[mcmc]
num_chains = 2
num_warmup = 20
num_samples = 20
max_r_hat = 1e6
min_ess_bulk = 1
"""


@pytest.fixture(scope="module")
def rv_dir(tmp_path_factory, rv_run_factory):
    """An RV run with rejection results, one MCMC result, and a summary."""
    fixture = rv_run_factory(tmp_path_factory.mktemp("web"))
    model_file = fixture.run_dir / "prior.py"
    model_file.write_text(model_file.read_text() + HOOK)
    config = fixture.run_dir / "hq.toml"
    config.write_text(config.read_text() + MCMC)
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection()
    run.summarize()
    run.run_mcmc()
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

    def test_both_stages(self, client):
        body = client.get(f"/api/source/{quote(MCMC_ID, safe='')}").json()
        mcmc = body["stages"]["mcmc"]
        assert mcmc["status"] == "ok"
        assert mcmc["mcmc_status"] == "converged"
        assert len(mcmc["params"]["period"]) == 2 * 20  # every draw, under the cap
        assert len(mcmc["curves"]["rv"]) == 2 * 20
        assert "params" in body["stages"]["rejection"]

    def test_gaia(self, gaia_client):
        client, fixture = gaia_client
        sid = next(iter(fixture.kept))
        # 19-digit IDs travel as strings: JS numbers cannot hold them exactly.
        ids = client.get("/api/catalog").json()["source_id"]
        assert str(sid) in ids
        body = client.get(f"/api/source/{sid}").json()
        assert body["source_id"] == str(sid)
        assert body["kind"] == "gaia_astrometry"
        n_obs = fixture.kept[sid]
        assert len(body["data"]["al_position"]) == n_obs
        curves = body["stages"]["rejection"]["curves"]
        n = len(body["stages"]["rejection"]["params"]["period"])
        assert np.shape(curves["ra"]) == np.shape(curves["dec"]) == (n, 256)
        assert np.shape(curves["residual"]) == (n, n_obs)
        assert client.get("/api/source/12345").status_code == 404
        assert client.get("/api/source/not-an-int").status_code == 404


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


def test_cli_serve(rv_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_module.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs))
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
