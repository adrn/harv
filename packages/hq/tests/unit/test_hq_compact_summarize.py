"""Tests for compaction, ``summarize``, and ``status``."""

import itertools
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from harv.samplers import RejectionSampler
from harv_hq import Run
from harv_hq import compact as compact_module
from harv_hq import results as results_module
from harv_hq.cli import main
from harv_hq.prepare import PreparedData
from harv_hq.results import Payload, ResultsIndex, write_part

FLUSH = 4
COMPACT = 5


def rejected_run(fixture, *, compact=False):
    """A run with rejection results in several 4-source parts, uncompacted."""
    config = fixture.run_dir / "hq.toml"
    config.write_text(
        config.read_text()
        + f"\n[results]\nflush_n_sources = {FLUSH}\ncompact_n_sources = {COMPACT}\n"
    )
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection(compact=compact)
    return run


def stage_dir(run):
    return run.run_dir / "results" / "rejection"


def parts(run):
    return sorted(p.name for p in stage_dir(run).glob("*.sources.parquet"))


def snapshot(run):
    """Every source's status and samples, as a reader sees them."""
    index = ResultsIndex.build(stage_dir(run))
    fresh = Run(run.run_dir)
    return {
        sid: (
            record.status,
            None
            if record.status != "ok"
            else {
                k: np.asarray(v)
                for k, v in fresh.load_source(sid)
                .rejection.to_columns()
                .columns.items()
            },
        )
        for sid, record in index.records.items()
    }


def assert_same(a, b):
    assert a.keys() == b.keys()
    for sid in a:
        assert a[sid][0] == b[sid][0]
        if a[sid][1] is not None:
            assert a[sid][1].keys() == b[sid][1].keys()
            for k in a[sid][1]:
                np.testing.assert_array_equal(a[sid][1][k], b[sid][1][k])


def assert_fully_compacted(run, n_sources):
    names = parts(run)
    assert names
    assert all(name.startswith("c-") for name in names)
    assert len(names) == math.ceil(n_sources / COMPACT)
    rows = sum(pq.read_metadata(stage_dir(run) / n).num_rows for n in names)
    assert rows == n_sources  # one row per source


class TestCompact:
    def test_merges_parts_and_keeps_every_result(self, rv_run):
        run = rejected_run(rv_run)
        before = snapshot(run)
        assert len(parts(run)) == math.ceil(len(rv_run.kept) / FLUSH)

        report = run.compact()
        assert report.n_sources == len(rv_run.kept)
        assert report.n_dropped_rows == 0
        assert report.n_output_parts == math.ceil(len(rv_run.kept) / COMPACT)
        assert_fully_compacted(run, len(rv_run.kept))
        assert_same(snapshot(run), before)

        # A population read needs no join: one contiguous range per source.
        samples = ds.dataset(
            sorted(str(p) for p in stage_dir(run).glob("*.samples.parquet"))
        ).to_table(columns=["source_id", "sample_index"])
        ids = samples["source_id"].to_pylist()
        assert set(ids) == set(rv_run.kept)
        changes = sum(a != b for a, b in itertools.pairwise(ids))
        assert changes == len(rv_run.kept) - 1

    def test_a_compacted_stage_is_left_alone(self, rv_run):
        run = rejected_run(rv_run)
        run.compact()
        names = parts(run)
        assert run.compact().n_output_parts == 0
        assert parts(run) == names

    def test_a_retried_failure_keeps_only_the_ok_row(self, rv_run, monkeypatch):
        config = rv_run.run_dir / "hq.toml"
        config.write_text(
            config.read_text() + f"\n[results]\ncompact_n_sources = {COMPACT}\n"
        )
        run = Run(rv_run.run_dir)
        run.prepare()
        run.make_prior_cache()
        bad = "J1234/5678"
        original = RejectionSampler.run_with_samples

        def flaky(self, data, *args: object, **kwargs: object):
            if data.n_obs == rv_run.n_obs[bad]:
                raise ValueError("boom")
            return original(self, data, *args, **kwargs)

        monkeypatch.setattr(RejectionSampler, "run_with_samples", flaky)
        run.run_rejection(compact=False)
        n_failed = ResultsIndex.build(stage_dir(run)).status_counts()["failed"]
        monkeypatch.setattr(RejectionSampler, "run_with_samples", original)
        run.run_rejection(compact=False)

        report = run.compact()
        assert report.n_dropped_rows == n_failed
        rows = ds.dataset([str(stage_dir(run) / n) for n in parts(run)]).to_table(
            columns=["source_id", "status"]
        )
        assert sorted(rows["source_id"].to_pylist()) == sorted(rv_run.kept)
        assert set(rows["status"].to_pylist()) == {"ok"}

    def test_a_part_committed_during_compaction_survives(self, rv_run, monkeypatch):
        run = rejected_run(rv_run)
        provenance = run._stage_provenance("rejection")
        original = compact_module._verify
        late = {}

        def verify_after_a_late_part(outputs, wanted):
            # Another process commits a part after the snapshot was taken.
            row = {
                "source_id": "late",
                "status": "failed",
                "error": "x",
                "finished": datetime.now(UTC),
            }
            late["path"] = write_part(
                stage_dir(run),
                "0000-of-0001-late0000",
                [Payload(row=row)],
                {"provenance": provenance, "stage": "rejection"},
            )
            original(outputs, wanted)

        monkeypatch.setattr(compact_module, "_verify", verify_after_a_late_part)
        run.compact()
        assert late["path"].exists()
        assert "late" in ResultsIndex.build(stage_dir(run)).records

    @pytest.mark.parametrize("step", ["after write", "after verify", "mid delete"])
    def test_a_crash_at_any_step_leaves_a_readable_stage(
        self, rv_run, monkeypatch, step
    ):
        run = rejected_run(rv_run)
        before = snapshot(run)
        with monkeypatch.context() as patch:
            if step == "after write":
                patch.setattr(compact_module, "_verify", _crash)
            elif step == "after verify":
                patch.setattr(compact_module, "_delete_inputs", _crash)
            else:
                deleted = []
                unlink = Path.unlink

                def unlink_then_crash(self, *args: object, **kwargs: object):
                    if len(deleted) == 3:
                        _crash()
                    deleted.append(self)
                    unlink(self, *args, **kwargs)

                patch.setattr(Path, "unlink", unlink_then_crash)
            with pytest.raises(RuntimeError, match="crash"):
                run.compact()

        assert_same(snapshot(run), before)
        assert Run(run.run_dir).status()["rejection"] == {"ok": len(rv_run.kept)} | {
            "pending": 0
        }
        Run(run.run_dir).summarize()
        run.compact()  # finishes the job
        assert_fully_compacted(run, len(rv_run.kept))
        assert_same(snapshot(run), before)

    def test_mismatched_output_deletes_nothing(self, rv_run, monkeypatch):
        run = rejected_run(rv_run)
        names = parts(run)
        original = compact_module._payloads
        # A writer bug: one source of each block goes missing.
        monkeypatch.setattr(
            compact_module, "_payloads", lambda block: original(block)[:-1]
        )
        with pytest.raises(RuntimeError, match="inputs were left in place"):
            run.compact()
        assert set(names) <= set(parts(run))


def _crash(*args: object, **kwargs: object):
    raise RuntimeError("crash")


class TestAutomaticCompaction:
    def test_a_whole_stage_run_compacts(self, rv_run):
        run = rejected_run(rv_run, compact=True)
        assert_fully_compacted(run, len(rv_run.kept))

    def test_explicit_shards_and_no_compact_do_not(self, rv_run):
        run = rejected_run(rv_run)  # compact=False
        assert not any(name.startswith("c-") for name in parts(run))
        run.run_rejection(shard=(1, 4), overwrite=True)
        assert not any(name.startswith("c-") for name in parts(run))

    def test_mpi_rank_0_compacts_after_the_barrier(self, rv_run, monkeypatch):
        config = rv_run.run_dir / "hq.toml"
        config.write_text(
            config.read_text()
            + f"\n[results]\nflush_n_sources = {FLUSH}\ncompact_n_sources = {COMPACT}\n"
        )
        run = Run(rv_run.run_dir)
        run.prepare()
        run.make_prior_cache()

        class FakeComm:
            def __init__(self, rank):
                self.rank = rank

            def Get_rank(self):
                return self.rank

            def Get_size(self):
                return 2

            def Barrier(self):
                pass

        # Rank 1 finishes first; rank 0 compacts once both are done.
        for rank in (1, 0):
            monkeypatch.setattr("harv_hq.run.mpi_comm", lambda r=rank: FakeComm(r))
            run.run_rejection(mpi=True)
            if rank == 1:
                assert not any(n.startswith("c-") for n in parts(run))
        assert_fully_compacted(run, len(rv_run.kept))


class TestSummarize:
    def test_columns_and_statuses(self, rv_run, monkeypatch):
        config = rv_run.run_dir / "hq.toml"
        config.write_text(config.read_text() + "\n[results]\nflush_n_sources = 4\n")
        run = Run(rv_run.run_dir)
        run.prepare()
        run.make_prior_cache()
        bad = "J1234/5678"
        original = RejectionSampler.run_with_samples

        def flaky(self, data, *args: object, **kwargs: object):
            if data.n_obs == rv_run.n_obs[bad]:
                raise ValueError("boom")
            return original(self, data, *args, **kwargs)

        monkeypatch.setattr(RejectionSampler, "run_with_samples", flaky)
        run.run_rejection(shard=(0, 2))  # half the sources stay pending

        # summarize reads neither the data nor any samples.
        def forbidden(*args: object, **kwargs: object):
            raise AssertionError("summarize must not read this")

        monkeypatch.setattr(PreparedData, "read", forbidden)
        monkeypatch.setattr(results_module, "read_source_samples", forbidden)
        summary = pq.read_table(run.summarize())

        names = set(summary.column_names)
        spec_columns = {
            "source_id",
            "n_obs",
            "time_baseline",
            "rejection_status",
            "rejection_error",
            "n_prior_samples",
            "ln_Z_int",
            "ln_Z_int_mcse",
            "ln_Z_int_ess",
            "max_ln_likelihood",
            "weight_captured",
            "well_resolved",
            "mcmc_status",
            "r_hat_max",
            "ess_bulk_min",
            "final",
            "period_unimodal",
            "max_phase_gap",
            "phase_coverage",
            "periods_spanned",
            "map_period",
            "period_p16",
            "period_p50",
            "period_p84",
        }
        assert spec_columns <= names
        assert summary.num_rows == len(rv_run.kept)
        assert sorted(summary["source_id"].to_pylist()) == sorted(rv_run.kept)
        statuses = set(summary["rejection_status"].to_pylist())
        assert statuses == {"ok", "failed", "pending"}
        assert set(summary["mcmc_status"].to_pylist()) == {"not_selected"}
        assert set(summary["final"].to_pylist()) == {"rejection"}
        rows = summary.to_pylist()
        assert all(
            r["period_p50"] is None for r in rows if r["rejection_status"] != "ok"
        )
        assert all(
            r["period_p50"] is not None for r in rows if r["rejection_status"] == "ok"
        )
        assert (
            pq.read_schema(run.run_dir / "summary.parquet")
            .field("period_p50")
            .metadata[b"unit"]
            == b"d"
        )

    def test_a_converged_mcmc_result_becomes_final(self, rv_run):
        run = rejected_run(rv_run)
        sid = next(iter(rv_run.kept))
        row = {
            "source_id": sid,
            "status": "ok",
            "mcmc_status": "converged",
            "r_hat_max": 1.001,
            "ess_bulk_min": 900.0,
            "period_p50": 123.0,
            "finished": datetime.now(UTC),
        }
        (run.run_dir / "results" / "mcmc").mkdir()
        write_part(
            run.run_dir / "results" / "mcmc",
            "0000-of-0001-mcmc0000",
            [Payload(row=row)],
            {"stage": "mcmc"},
        )
        summary = {
            r["source_id"]: r for r in pq.read_table(run.summarize()).to_pylist()
        }
        assert summary[sid]["final"] == "mcmc"
        assert summary[sid]["period_p50"] == 123.0
        assert summary[sid]["r_hat_max"] == 1.001
        others = [r for k, r in summary.items() if k != sid]
        assert {r["final"] for r in others} == {"rejection"}
        assert all(r["period_p50"] is not None for r in others)

    def test_load_source_returns_the_summary_row(self, rv_run):
        run = rejected_run(rv_run)
        sid = next(iter(rv_run.kept))
        assert run.load_source(sid).summary is None
        run.summarize()
        summary = Run(run.run_dir).load_source(sid).summary
        assert summary["source_id"] == sid
        assert summary["rejection_status"] == "ok"


def test_status_counts_pending_sources(rv_run):
    config = rv_run.run_dir / "hq.toml"
    config.write_text(config.read_text() + "\n[results]\nflush_n_sources = 4\n")
    run = Run(rv_run.run_dir)
    run.prepare()
    run.make_prior_cache()
    assert run.status() == {"rejection": {"pending": len(rv_run.kept)}, "mcmc": {}}
    run.run_rejection(shard=(0, 2))
    counts = Run(run.run_dir).status()["rejection"]
    assert counts["ok"] + counts["pending"] == len(rv_run.kept)
    assert 0 < counts["pending"] < len(rv_run.kept)


def test_cli(rv_run, capsys):
    run = rejected_run(rv_run)
    run_dir = str(run.run_dir)
    assert main(["compact", "--run-dir", run_dir]) == 0
    assert "superseded rows dropped" in capsys.readouterr().out
    assert main(["summarize", "--run-dir", run_dir]) == 0
    assert (run.run_dir / "summary.parquet").exists()
    assert main(["status", "--run-dir", run_dir]) == 0
    out = capsys.readouterr().out
    assert f"ok {len(rv_run.kept)}" in out
    assert "mcmc" in out
