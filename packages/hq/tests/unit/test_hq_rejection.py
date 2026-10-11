"""Tests for ``hq prior-cache`` and ``hq run``: result parts, resume, provenance."""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime

import jax
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from unxt import Q

from harv.samplers import RejectionSampler, Samples
from harv_hq import ProvenanceError, Run
from harv_hq import rejection as rejection_module
from harv_hq import run as run_module
from harv_hq.cli import main
from harv_hq.execution import Progress
from harv_hq.ids import source_key
from harv_hq.prepare import PreparedData
from harv_hq.results import (
    PartWriter,
    Payload,
    ResultsIndex,
    samples_table,
    supersede,
)

TOP_K = 64  # the fixtures' [rejection] top_k


def prepared_run(fixture):
    """Prepare the fixture's data and prior cache, with 4-source result parts."""
    config = fixture.run_dir / "hq.toml"
    config.write_text(config.read_text() + "\n[results]\nflush_n_sources = 4\n")
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    return run


def stage_dir(run):
    return run.run_dir / "results" / "rejection"


def index(run):
    return ResultsIndex.build(stage_dir(run))


class TestEndToEnd:
    def test_every_source_gets_a_result_in_several_parts(self, rv_run):
        run = prepared_run(rv_run)
        run.run_rejection()
        records = index(run).records
        assert set(records) == set(rv_run.kept)
        assert {r.status for r in records.values()} == {"ok"}
        assert {r.n_samples for r in records.values()} == {TOP_K}
        parts = sorted(stage_dir(run).glob("*.sources.parquet"))
        assert len(parts) == -(-len(records) // 4)  # ceil: flush_n_sources = 4
        assert all(p.name.startswith("0000-of-0001-") for p in parts)
        assert (run.run_dir / "logs" / "rejection-0000-of-0001.log").exists()

    def test_sources_table_columns(self, rv_run):
        run = prepared_run(rv_run)
        run.run_rejection()
        path = next(stage_dir(run).glob("*.sources.parquet"))
        schema = pq.read_schema(path)
        for name in (
            "status",
            "error",
            "warnings",
            "seed_hash",
            "n_obs",
            "started",
            "finished",
            "wall_time_s",
            "samples_row_start",
            "n_samples",
            "time_ref",
            "ln_Z_int",
            "ln_Z_int_ess",
            "weight_captured",
            "well_resolved",
            "period_unimodal",
            "max_phase_gap",
            "map_period",
            "period_p16",
            "period_p50",
            "period_p84",
            "rv_semiamp_p50",
        ):
            assert name in schema.names, name
        assert schema.field("period_p50").metadata[b"unit"] == b"d"
        assert str(schema.field("finished").type) == "timestamp[us, tz=UTC]"

    def test_load_source_matches_a_fresh_run(self, rv_run):
        """Parquet round trip, and the same seed gives the same samples."""
        run = prepared_run(rv_run)
        run.run_rejection()
        sid = "J1234/5678"
        result = run.load_source(sid)
        assert result.rejection_status == "ok"
        assert result.mcmc is None
        assert result.mcmc_status == "pending"

        prior, model = run.model_file.setup("rv")
        sampler_key, _ = jax.random.split(
            source_key(run.config.run.seed, sid, "rejection")
        )
        fresh = RejectionSampler(prior, model).run_with_samples(
            PreparedData(run.run_dir).read([sid])[sid],
            run.run_dir / "prior_cache.h5",
            key=sampler_key,
            top_k=TOP_K,
        )
        loaded = result.rejection
        for name in fresh.nonlinear:
            np.testing.assert_array_equal(
                np.asarray(loaded[name].value), np.asarray(fresh[name].value)
            )
        np.testing.assert_array_equal(
            np.asarray(loaded.ln_likelihood), np.asarray(fresh.ln_likelihood)
        )
        assert loaded.metadata == fresh.metadata
        assert loaded.model_type == fresh.model_type

    def test_population_read(self, rv_run):
        run = prepared_run(rv_run)
        run.run_rejection()
        table = ds.dataset(
            sorted(map(str, stage_dir(run).glob("*.samples.parquet"))), format="parquet"
        ).to_table(columns=["source_id", "period", "weight"])
        assert table.num_rows == len(set(rv_run.kept)) * TOP_K

    def test_gaia(self, gaia_run):
        run = prepared_run(gaia_run)
        run.run_rejection()
        sid = next(iter(gaia_run.n_obs))
        result = run.load_source(sid)
        assert result.rejection_status == "ok"
        assert "semi_major_axis" in result.rejection.linear


class TestFailuresAndResume:
    def test_a_failing_source_is_recorded_and_the_run_continues(
        self, rv_run, monkeypatch
    ):
        run = prepared_run(rv_run)
        bad = "J1234/5678"
        original = RejectionSampler.run_with_samples

        def flaky(self, data, *args: object, **kwargs: object):
            if data.n_obs == rv_run.n_obs[bad]:
                raise ValueError("boom: missing prior keys")
            return original(self, data, *args, **kwargs)

        monkeypatch.setattr(RejectionSampler, "run_with_samples", flaky)
        run.run_rejection()
        records = index(run).records
        failed = {sid for sid, r in records.items() if r.status == "failed"}
        assert bad in failed
        assert len(records) == len(set(rv_run.kept))
        table = pq.read_table(records[bad].sources_path).to_pylist()
        row = next(r for r in table if r["source_id"] == bad)
        assert "boom: missing prior keys" in row["error"]
        assert row["n_samples"] == 0

        # Failed sources are not done: the next resume reruns them, and only them.
        monkeypatch.setattr(RejectionSampler, "run_with_samples", original)
        calls = []
        process = rejection_module.process_rejection
        monkeypatch.setattr(
            rejection_module,
            "process_rejection",
            lambda sid, *a, **k: calls.append(sid) or process(sid, *a, **k),
        )
        run.run_rejection()
        assert set(calls) == failed
        assert index(run).status_counts() == {"ok": len(set(rv_run.kept))}

    def test_resume_after_an_interruption(self, rv_run, monkeypatch):
        run = prepared_run(rv_run)
        calls = []
        original = rejection_module.process_rejection

        def interrupted(source_id, *args: object, **kwargs: object):
            if len(calls) == 6:
                raise KeyboardInterrupt
            calls.append(source_id)
            return original(source_id, *args, **kwargs)

        monkeypatch.setattr(rejection_module, "process_rejection", interrupted)
        # Caches the (empty) index on this Run.
        assert run.load_source(next(iter(rv_run.kept))).rejection_status == "pending"
        with pytest.raises(KeyboardInterrupt):
            run.run_rejection()
        # Everything processed before the interruption was written on the way out.
        assert set(index(run).records) == set(calls)
        # ...and the cached index was dropped, so this Run sees it too.
        assert run.load_source(calls[0]).rejection_status == "ok"

        monkeypatch.setattr(rejection_module, "process_rejection", original)
        rerun = []
        monkeypatch.setattr(
            rejection_module,
            "process_rejection",
            lambda sid, *a, **k: rerun.append(sid) or original(sid, *a, **k),
        )
        run.run_rejection()
        assert set(rerun) == set(rv_run.kept) - set(calls)
        assert set(index(run).records) == set(rv_run.kept)

    def test_an_orphaned_samples_file_is_ignored(self, rv_run):
        run = prepared_run(rv_run)
        run.run_rejection()
        sources = min(stage_dir(run).glob("*.sources.parquet"))
        lost = set(
            pq.read_table(sources, columns=["source_id"])["source_id"].to_pylist()
        )
        sources.unlink()  # a crash between the two renames
        assert lost.isdisjoint(index(run).records)
        run.run_rejection()
        assert set(index(run).records) == set(rv_run.kept)

    def test_a_crash_loses_only_the_buffer(self, tmp_path):
        writer = PartWriter(
            tmp_path / "rejection",
            stage="rejection",
            shard=(0, 1),
            provenance={},
            flush_n_sources=3,
            flush_seconds=1e9,
        )
        for i in range(5):
            row = {
                "source_id": str(i),
                "status": "failed",
                "error": "x",
                "finished": datetime.now(UTC),
            }
            writer.add(Payload(row=row))
        # No close(): the process died. One part (3 sources) is on disk.
        assert set(ResultsIndex.build(tmp_path / "rejection").records) == {
            "0",
            "1",
            "2",
        }

    def test_overwrite_moves_results_aside(self, rv_run):
        run = prepared_run(rv_run)
        run.run_rejection()
        run.run_rejection(overwrite=True)
        superseded = list((run.run_dir / "results").glob("superseded-*-rejection"))
        assert len(superseded) == 1
        assert any(superseded[0].glob("*.sources.parquet"))
        assert set(index(run).records) == set(rv_run.kept)

    def test_each_overwrite_gets_its_own_archive(self, tmp_path):
        stage = tmp_path / "results" / "rejection"
        for _ in range(3):  # well within one second
            stage.mkdir(parents=True)
            (stage / "part").touch()
            supersede(stage)
        archives = list((tmp_path / "results").glob("superseded-*-rejection"))
        assert len(archives) == 3
        assert all((a / "part").exists() for a in archives)


STRUCTURE_A = {
    "model_type": "RVModel",
    "linear_extension_names": [],
    "nonlinear_names": ["period"],
    "linear_names": [],
}
STRUCTURE_B = {**STRUCTURE_A, "nonlinear_names": ["period", "e"]}


class TestSchema:
    def _write(self, stage, structure, source_id):
        writer = PartWriter(
            stage,
            stage="rejection",
            shard=(0, 1),
            provenance={},
            flush_n_sources=1,
            flush_seconds=1e9,
        )
        row = {"source_id": source_id, "status": "ok", "finished": datetime.now(UTC)}
        writer.add(Payload(row=row, structure=structure))

    def test_a_resumed_writer_refuses_a_new_structure(self, tmp_path):
        self._write(tmp_path, STRUCTURE_A, "a")
        existing = ResultsIndex.build(tmp_path)
        assert existing.structure == STRUCTURE_A
        writer = PartWriter(
            tmp_path,
            stage="rejection",
            shard=(0, 1),
            provenance={},
            flush_n_sources=1,
            flush_seconds=1e9,
            structure=existing.structure,
        )
        row = {"source_id": "b", "status": "ok", "finished": datetime.now(UTC)}
        with pytest.raises(ValueError, match="one Samples structure"):
            writer.add(Payload(row=row, structure=STRUCTURE_B))

    def test_parts_of_different_structure_are_refused(self, tmp_path):
        self._write(tmp_path, STRUCTURE_A, "a")
        self._write(tmp_path, STRUCTURE_B, "b")
        with pytest.raises(ValueError, match="--overwrite"):
            ResultsIndex.build(tmp_path)

    @pytest.mark.parametrize("name", ["source_id", "sample_index", "chain", "weight"])
    def test_a_parameter_named_like_a_bookkeeping_column_is_refused(self, name):
        samples = Samples(
            nonlinear={name: Q(np.ones(3), "")},
            linear={},
            model_type="RVModel",
        )
        with pytest.raises(ValueError, match="clash"):
            samples_table(samples, "a", stage="rejection")


def all_samples(run):
    """Every result's samples, in a fixed order, for comparing runs."""
    paths = sorted(str(p) for p in stage_dir(run).glob("*.samples.parquet"))
    return (
        ds.dataset(paths)
        .to_table()
        .sort_by([("source_id", "ascending"), ("sample_index", "ascending")])
    )


def twin_runs(fixture, tmp_path):
    """Two prepared copies of one run directory (same seed, data, and cache)."""
    run = prepared_run(fixture)
    shutil.copytree(run.run_dir, tmp_path / "twin")
    return run, Run(tmp_path / "twin")


class TestExecutionModes:
    def test_shards_match_a_serial_run(self, rv_run, tmp_path):
        serial, sharded = twin_runs(rv_run, tmp_path)
        serial.run_rejection()
        for i in range(3):
            sharded.run_rejection(shard=(i, 3))
        assert all_samples(sharded).equals(all_samples(serial))
        names = {p.name[:12] for p in stage_dir(sharded).glob("*.sources.parquet")}
        assert names == {f"000{i}-of-0003" for i in range(3)}

    def test_a_pool_matches_a_serial_run(self, rv_run, tmp_path):
        serial, pooled = twin_runs(rv_run, tmp_path)
        serial.run_rejection()
        environ = dict(os.environ)
        pooled.run_rejection(workers=2)
        # The workers' thread pinning does not leak into this process.
        assert dict(os.environ) == environ
        assert all_samples(pooled).equals(all_samples(serial))
        assert index(pooled).status_counts() == {"ok": len(rv_run.kept)}
        assert all(
            p.name.startswith("0000-of-0001-")
            for p in stage_dir(pooled).glob("*.parquet")
        )

    def test_changing_the_shard_count_recomputes_nothing(self, rv_run, monkeypatch):
        run = prepared_run(rv_run)
        run.run_rejection(shard=(0, 4))
        run.run_rejection(shard=(1, 4))
        first = set(index(run).records)
        assert first
        assert first != set(rv_run.kept)
        calls = []
        process = rejection_module.process_rejection
        monkeypatch.setattr(
            rejection_module,
            "process_rejection",
            lambda sid, *a, **k: calls.append(sid) or process(sid, *a, **k),
        )
        run.run_rejection()
        assert set(calls) == set(rv_run.kept) - first
        assert set(index(run).records) == set(rv_run.kept)

    def test_two_processes_on_one_slice_resolve_to_the_newest(
        self, rv_run, monkeypatch
    ):
        run = prepared_run(rv_run)
        run.run_rejection()
        first_parts = set(stage_dir(run).glob("*.sources.parquet"))
        # A second process that started before the first wrote anything.
        monkeypatch.setattr(ResultsIndex, "done", lambda _self: set())
        run.run_rejection()
        second_parts = set(stage_dir(run).glob("*.sources.parquet")) - first_parts
        assert len(second_parts) == len(first_parts)  # distinct names, none lost
        records = index(run).records
        assert set(records) == set(rv_run.kept)
        assert {r.sources_path for r in records.values()} <= second_parts

    def test_mpi_excludes_shard_and_workers(self, rv_run):
        run = Run(rv_run.run_dir)
        with pytest.raises(ValueError, match="omit shard and workers"):
            run.run_rejection(mpi=True, shard=(0, 2))
        with pytest.raises(ValueError, match="omit shard and workers"):
            run.run_rejection(mpi=True, workers=2)
        with pytest.raises(ValueError, match="at least 1"):
            run.run_rejection(workers=0)

    def test_mpi_ranks_take_their_slices(self, rv_run, monkeypatch):
        """The MPI path, with a stand-in communicator: ranks run one at a time."""
        run = prepared_run(rv_run)
        run.run_rejection()
        barriers = []

        class FakeComm:
            def __init__(self, rank):
                self.rank = rank

            def Get_rank(self):
                return self.rank

            def Get_size(self):
                return 2

            def Barrier(self):
                barriers.append(self.rank)

        # --overwrite: only rank 0 moves results aside, so rank 1 does not move
        # rank 0's new parts.
        for rank in (0, 1):
            monkeypatch.setattr(run_module, "mpi_comm", lambda r=rank: FakeComm(r))
            run.run_rejection(mpi=True, overwrite=True)
        assert barriers == [0, 0, 1, 1]  # after the overwrite, and at the end
        assert len(list((run.run_dir / "results").glob("superseded-*"))) == 1
        assert index(run).status_counts() == {"ok": len(rv_run.kept)}
        names = {p.name[:12] for p in stage_dir(run).glob("*.sources.parquet")}
        assert names == {"0000-of-0002", "0001-of-0002"}
        logs = {p.name for p in (run.run_dir / "logs").iterdir()}
        assert {"rejection-0000-of-0002.log", "rejection-0001-of-0002.log"} <= logs

    def test_mpi_without_mpi4py(self, rv_run, monkeypatch):
        monkeypatch.setitem(sys.modules, "mpi4py", None)
        with pytest.raises(ImportError, match="harv-hq\\[mpi\\]"):
            Run(rv_run.run_dir).run_rejection(mpi=True)

    @pytest.mark.skipif(
        importlib.util.find_spec("mpi4py") is None or shutil.which("mpirun") is None,
        reason="needs mpi4py and mpirun",
    )
    def test_mpi(self, rv_run):
        run = prepared_run(rv_run)
        subprocess.run(  # noqa: S603 -- fixed argv
            [
                *("mpirun", "-n", "2", sys.executable),
                *("-m", "harv_hq.cli", "run", "--mpi", "--run-dir", str(run.run_dir)),
            ],
            check=True,
            timeout=600,
        )
        assert index(run).status_counts() == {"ok": len(rv_run.kept)}
        names = {p.name[:12] for p in stage_dir(run).glob("*.sources.parquet")}
        assert names == {"0000-of-0002", "0001-of-0002"}


def test_progress_logs_about_every_five_percent(caplog):
    progress = Progress("rejection", 40, ranks=4)
    with caplog.at_level(logging.INFO, logger="harv_hq"):
        for _ in range(40):
            progress.tick()
    assert len(caplog.records) == 20
    assert caplog.messages[-1] == (
        "rejection: 40/40 sources (about 160/160 over 4 ranks)"
    )


class TestProvenance:
    def test_editing_the_model_file_is_refused(self, rv_run):
        run = prepared_run(rv_run)
        prior = run.run_dir / "prior.py"
        prior.write_text(prior.read_text() + "\n# edited\n")
        with pytest.raises(ProvenanceError, match="model_file_sha256"):
            Run(run.run_dir).run_rejection()

    def test_a_rebuilt_prior_cache_is_refused_by_existing_parts(self, rv_run):
        run = prepared_run(rv_run)
        run.run_rejection(shard=(0, 2))
        run.make_prior_cache(overwrite=True)
        with pytest.raises(ProvenanceError, match="prior_cache_id"):
            run.run_rejection(shard=(1, 2))

    def test_missing_prior_cache(self, rv_run):
        run = prepared_run(rv_run)
        (run.run_dir / "prior_cache.h5").unlink()
        with pytest.raises(FileNotFoundError, match="hq prior-cache"):
            run.run_rejection()

    def test_prior_cache_overwrite_refusal(self, rv_run):
        run = prepared_run(rv_run)
        with pytest.raises(FileExistsError, match="overwrite"):
            run.make_prior_cache()


def test_cli(rv_run):
    run_dir = str(rv_run.run_dir)
    assert main(["prepare", "--run-dir", run_dir]) == 0
    assert main(["prior-cache", "--run-dir", run_dir]) == 0
    assert main(["run", "--run-dir", run_dir, "--shard", "0/1"]) == 0
    assert set(
        ResultsIndex.build(rv_run.run_dir / "results" / "rejection").records
    ) == set(rv_run.kept)
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--run-dir", run_dir, "--mpi", "--workers", "2"])
    assert "mutually exclusive" in str(excinfo.value.code)


def test_copy_of_a_run_directory_still_resumes(rv_run, tmp_path):
    """Results are found by globbing the stage directory, not by absolute paths."""
    run = prepared_run(rv_run)
    run.run_rejection(shard=(0, 2))
    copy = tmp_path / "copy"
    shutil.copytree(run.run_dir, copy)
    moved = Run(copy)
    moved.run_rejection(shard=(1, 2))
    assert set(ResultsIndex.build(copy / "results" / "rejection").records) == set(
        rv_run.kept
    )
