"""Tests for ``harv_hq.provenance``."""

import hashlib
import logging
from datetime import datetime

import pytest

from harv_hq import Config, ProvenanceError, init_run
from harv_hq.provenance import check_provenance, make_provenance, sha256_file


@pytest.fixture
def config(tmp_path):
    run_dir = init_run(tmp_path / "run", kind="rv")
    return Config.from_file(run_dir / "hq.toml")


def test_sha256_file(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"hello" * 1_000_000)
    assert sha256_file(path) == hashlib.sha256(b"hello" * 1_000_000).hexdigest()


def test_make_provenance_keys(config):
    record = make_provenance(config, data_id="d1", prior_cache_id="p1")
    assert set(record) == {
        "hq_version",
        "harv_version",
        "jax_version",
        "config_sha256",
        "model_file_sha256",
        "data_id",
        "prior_cache_id",
        "created",
    }
    assert record["config_sha256"] == sha256_file(config.config_path)
    assert record["model_file_sha256"] == sha256_file(config.run.model_file)
    assert datetime.fromisoformat(record["created"]).utcoffset().total_seconds() == 0


def test_make_provenance_optional_keys(config):
    record = make_provenance(config)
    assert "data_id" not in record
    assert "prior_cache_id" not in record


def test_check_passes_on_match(config):
    record = make_provenance(config, data_id="d1")
    check_provenance(record, record, path="x.parquet")


@pytest.mark.parametrize(
    "field", ["config_sha256", "model_file_sha256", "data_id", "prior_cache_id"]
)
def test_check_refuses_a_mismatch(config, field):
    expected = make_provenance(config, data_id="d1", prior_cache_id="p1")
    found = {**expected, field: "something else"}
    with pytest.raises(ProvenanceError, match=rf"x\.parquet.*{field}.*--overwrite"):
        check_provenance(found, expected, path="x.parquet")


def test_check_refuses_a_missing_field(config):
    expected = make_provenance(config, data_id="d1")
    found = {k: v for k, v in expected.items() if k != "data_id"}
    with pytest.raises(ProvenanceError, match="data_id = None"):
        check_provenance(found, expected, path="x.parquet")


def test_check_ignores_fields_not_expected(config):
    expected = make_provenance(config, data_id="d1")
    found = {**expected, "prior_cache_id": "anything"}
    check_provenance(found, expected, path="data.parquet")


def test_check_logs_version_differences(config, caplog):
    expected = make_provenance(config)
    found = {**expected, "harv_version": "0.0.1"}
    with caplog.at_level(logging.WARNING, logger="harv_hq.provenance"):
        check_provenance(found, expected, path="x.parquet")
    assert "harv 0.0.1" in caplog.text


def test_editing_the_model_file_changes_its_hash(config):
    before = make_provenance(config)
    config.run.model_file.write_text(config.run.model_file.read_text() + "\n# edit\n")
    after = make_provenance(config)
    with pytest.raises(ProvenanceError, match="model_file_sha256"):
        check_provenance(before, after, path="part.parquet")
