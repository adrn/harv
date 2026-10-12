"""Tests for ``hq mcmc``: selection, convergence, results, resume, provenance."""

import shutil

import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from unxt import Q

from harv.samplers import NumpyroSampler, Samples
from harv_hq import ConfigError, ProvenanceError, Run
from harv_hq import mcmc as mcmc_module
from harv_hq._model_file import ModelFile
from harv_hq._parquet_io import read_parquet, write_parquet
from harv_hq.cli import main
from harv_hq.config import MCMCConfig
from harv_hq.mcmc import RejectionSamples, convergence, select_sources
from harv_hq.prepare import PreparedData
from harv_hq.results import (
    SAMPLES_SUFFIX,
    SOURCES_SUFFIX,
    ResultsIndex,
    read_source_rows,
)
from harv_hq.run import PRIOR_CACHE_FILE
from harv_hq.summarize import SUMMARY_FILE

# The model file's select_for_mcmc picks these; the [mcmc] rule alone would
# not pick 2M00000016+0000112 (well resolved, multimodal).
SELECTED = ("2M00000003+0000021", "J1234/5678", "2M00000016+0000112")
FLAKY = "J1234/5678"  # the only selected source with 5 observations
HOOK = f"""

def select_for_mcmc(row):
    return row["source_id"] in {SELECTED!r}
"""
NUM_CHAINS, NUM_SAMPLES = 2, 20
# Thresholds every run passes, so the status logic is tested separately.
MCMC = f"""
[mcmc]
num_chains = {NUM_CHAINS}
num_warmup = 20
num_samples = {NUM_SAMPLES}
max_r_hat = 1e6
min_ess_bulk = 1
"""


@pytest.fixture(scope="module")
def summarized(tmp_path_factory, rv_run_factory):
    """An RV run with rejection results and a summary, built once per module."""
    fixture = rv_run_factory(tmp_path_factory.mktemp("mcmc"))
    model_file = fixture.run_dir / "prior.py"
    model_file.write_text(model_file.read_text() + HOOK)
    config = fixture.run_dir / "hq.toml"
    config.write_text(config.read_text() + MCMC + "\n[results]\nflush_n_sources = 2\n")
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection()
    run.summarize()
    return fixture.run_dir


@pytest.fixture
def run(summarized, tmp_path):
    """A fresh copy of the summarized run."""
    shutil.copytree(summarized, tmp_path / "run")
    return Run(tmp_path / "run")


def mcmc_dir(run):
    return run.run_dir / "results" / "mcmc"


def index(run):
    return ResultsIndex.build(mcmc_dir(run))


def summary_rows(run):
    return {r["source_id"]: r for r in pq.read_table(run.summarize()).to_pylist()}


def edit_config(run, old, new):
    path = run.run_dir / "hq.toml"
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))
    return Run(run.run_dir)


def all_samples(run):
    tables = [pq.read_table(p) for p in mcmc_dir(run).glob("*.samples.parquet")]
    return pa.concat_tables(tables).sort_by(
        [("source_id", "ascending"), ("sample_index", "ascending")]
    )


class TestEndToEnd:
    def test_selected_sources_get_mcmc_results(self, run, monkeypatch):
        loads = []
        load = ModelFile.load
        monkeypatch.setattr(
            ModelFile, "load", lambda path: loads.append(path) or load(path)
        )
        run.run_mcmc()
        assert len(loads) == 1  # the parent's import is reused, not repeated
        records = index(run).records
        assert set(records) == set(SELECTED)
        assert {r.status for r in records.values()} == {"ok"}
        # Compacted at the end of a whole-stage run.
        (part,) = mcmc_dir(run).glob("*.sources.parquet")
        assert part.name.startswith("c-")
        # MCMC parts record [mcmc] on top of what rejection parts record.
        provenance = read_parquet(part).metadata["provenance"]
        assert "mcmc" in provenance["config_sha256"]

        samples = all_samples(run)
        assert "weight" not in samples.column_names
        for sid in SELECTED:
            chains = samples.filter(pa.compute.equal(samples["source_id"], sid))
            counts = np.bincount(chains["chain"].to_numpy())
            assert counts.tolist() == [NUM_SAMPLES] * NUM_CHAINS

        summary = summary_rows(run)
        for sid in SELECTED:
            assert summary[sid]["mcmc_status"] == "converged"
            assert summary[sid]["final"] == "mcmc"
            assert np.isfinite(summary[sid]["r_hat_max"])
            assert summary[sid]["ess_bulk_min"] >= 1
        others = [r for sid, r in summary.items() if sid not in SELECTED]
        assert {r["mcmc_status"] for r in others} == {"not_selected"}
        assert {r["final"] for r in others} == {"rejection"}
        # Selection reads the rejection value even where MCMC is final.
        rejection = ResultsIndex.build(run.run_dir / "results" / "rejection")
        rows, _ = read_source_rows([rejection.records[sid] for sid in SELECTED])
        expected = dict(zip(SELECTED, rows["period_unimodal"].to_pylist(), strict=True))
        for sid in SELECTED:
            assert summary[sid]["rejection_period_unimodal"] == expected[sid]

        result = Run(run.run_dir).load_source(FLAKY)
        assert result.mcmc_status == "ok"
        assert result.mcmc.n_samples == NUM_CHAINS * NUM_SAMPLES
        assert result.mcmc.metadata["num_chains"] == NUM_CHAINS
        calls = []
        inputs = Run._inputs
        monkeypatch.setattr(
            Run, "_inputs", lambda self, **k: calls.append(k) or inputs(self, **k)
        )
        assert run.status()["mcmc"] == {"ok": len(SELECTED)}
        assert len(calls) == 1  # inputs checked once, not once per stage

    def test_failing_thresholds_are_not_converged(self, run):
        run = edit_config(run, "max_r_hat = 1e6", "max_r_hat = 0.5")
        assert main(["mcmc", "--run-dir", str(run.run_dir)]) == 0
        summary = summary_rows(run)
        for sid in SELECTED:
            assert summary[sid]["mcmc_status"] == "not_converged"
            assert summary[sid]["final"] == "rejection"
        # Not converged is still a finished result: a rerun does nothing.
        assert run.status()["mcmc"] == {"ok": len(SELECTED)}


class TestFailuresAndResume:
    def test_a_failing_source_is_recorded_and_rerun(self, run, monkeypatch):
        original = NumpyroSampler.run

        def flaky(self, data, **kwargs: object):
            if data.n_obs == 5:
                raise ValueError("boom: chains diverged")
            return original(self, data, **kwargs)

        monkeypatch.setattr(NumpyroSampler, "run", flaky)
        run.run_mcmc()
        records = index(run).records
        assert records[FLAKY].status == "failed"
        summary = summary_rows(run)
        assert summary[FLAKY]["mcmc_status"] == "failed"
        assert summary[FLAKY]["final"] == "rejection"
        (row,) = [
            r
            for r in pq.read_table(records[FLAKY].sources_path).to_pylist()
            if r["source_id"] == FLAKY
        ]
        assert "boom: chains diverged" in row["error"]

        monkeypatch.setattr(NumpyroSampler, "run", original)
        calls = []
        process = mcmc_module.process_mcmc
        monkeypatch.setattr(
            mcmc_module,
            "process_mcmc",
            lambda sid, *a, **k: calls.append(sid) or process(sid, *a, **k),
        )
        run.run_mcmc()
        assert calls == [FLAKY]
        assert index(run).status_counts() == {"ok": len(SELECTED)}

    def test_shards_then_a_whole_run_recomputes_nothing(self, run, monkeypatch):
        run.run_mcmc(shard=(0, 2))
        first = set(index(run).records)
        assert first
        assert first < set(SELECTED)
        names = {p.name[:12] for p in mcmc_dir(run).glob("*.sources.parquet")}
        assert names == {"0000-of-0002"}
        calls = []
        process = mcmc_module.process_mcmc
        monkeypatch.setattr(
            mcmc_module,
            "process_mcmc",
            lambda sid, *a, **k: calls.append(sid) or process(sid, *a, **k),
        )
        run.run_mcmc()
        assert set(calls) == set(SELECTED) - first
        assert set(index(run).records) == set(SELECTED)

    def test_a_pool_matches_a_serial_run(self, run, tmp_path):
        shutil.copytree(run.run_dir, tmp_path / "twin")
        pooled = Run(tmp_path / "twin")
        run.run_mcmc()
        pooled.run_mcmc(workers=2)
        assert index(pooled).status_counts() == {"ok": len(SELECTED)}
        assert all_samples(pooled).equals(all_samples(run))


class TestInputs:
    def test_without_an_mcmc_table(self, run):
        run = edit_config(run, MCMC, "")
        with pytest.raises(ConfigError, match=r"no \[mcmc\] table"):
            run.run_mcmc()

    def test_without_a_summary(self, run):
        (run.run_dir / SUMMARY_FILE).unlink()
        with pytest.raises(FileNotFoundError, match="run hq summarize first"):
            run.run_mcmc()

    @staticmethod
    def rewrite_summary(run, *, created=None, pending=None) -> None:
        path = run.run_dir / SUMMARY_FILE
        summary = read_parquet(path)
        provenance = summary.metadata["provenance"]
        if created is not None:
            provenance = {**provenance, "created": created}
        table = summary.table
        if pending is not None:
            status = [
                "pending" if sid == pending else s
                for sid, s in zip(
                    table["source_id"].to_pylist(),
                    table["rejection_status"].to_pylist(),
                    strict=True,
                )
            ]
            index = table.schema.get_field_index("rejection_status")
            table = table.set_column(index, "rejection_status", pa.array(status))
        write_parquet(
            path, table, units=summary.units, metadata={"provenance": provenance}
        )

    def test_a_summary_written_before_the_last_result_is_refused(self, run):
        # Parts record when their run started, so a summary written while a
        # run was still going is newer than every part, but not than every
        # result.
        rejection = ResultsIndex.build(run.run_dir / "results" / "rejection")
        part_created = rejection.provenance["created"]
        self.rewrite_summary(run, created=part_created)
        with pytest.raises(ProvenanceError, match="rerun hq summarize"):
            run.run_mcmc()

    def test_a_summary_missing_a_result_is_refused(self, run):
        # A result that finished before the summary started, but whose part
        # was written after the summary read the stage.
        self.rewrite_summary(run, pending=SELECTED[0])
        with pytest.raises(ProvenanceError, match="rerun hq summarize"):
            run.run_mcmc()

    def test_the_prior_cache_is_not_needed_after_rejection(self, run):
        (run.run_dir / PRIOR_CACHE_FILE).unlink()
        run.run_mcmc(shard=(0, 3))
        assert set(index(run).records)
        assert run.status()["rejection"]["ok"]
        run.summarize()
        with pytest.raises(FileNotFoundError, match="hq prior-cache"):
            run.run_rejection()

    def test_rejection_compacted_during_the_stage(self, run):
        # Compaction replaces a part with one under a new name; records taken
        # before it point at the old one.
        stage = run.run_dir / "results" / "rejection"
        records = ResultsIndex.build(stage).records
        for path in stage.glob(f"*{SOURCES_SUFFIX}"):
            stem = path.name.removesuffix(SOURCES_SUFFIX)
            for suffix in (SAMPLES_SUFFIX, SOURCES_SUFFIX):
                (stage / (stem + suffix)).rename(stage / (f"c-moved-{stem}" + suffix))
        read = RejectionSamples()
        for sid in SELECTED:
            assert read(sid, records[sid]).n_samples == records[sid].n_samples

    def test_editing_mcmc_invalidates_only_mcmc_results(self, run):
        run.run_mcmc()
        # Selection keys change which sources run, not any result.
        run = edit_config(run, "num_warmup = 20", 'num_warmup = 20\nselect = "all"')
        assert run.status()["mcmc"] == {"ok": len(SELECTED)}
        run = edit_config(run, "num_warmup = 20", "num_warmup = 21")
        with pytest.raises(ProvenanceError):
            run.status()
        with pytest.raises(ProvenanceError):
            run.run_mcmc()
        # The rejection results are still current.
        ResultsIndex.build(
            run.run_dir / "results" / "rejection",
            expected_provenance=run._stage_provenance("rejection"),
        )
        run.run_mcmc(overwrite=True)
        assert run.status()["mcmc"] == {"ok": len(SELECTED)}

    def test_slices_are_ordered_by_n_obs(self, run):
        prepared = PreparedData(run.run_dir)
        ordered = prepared.slice_ids((0, 1), among=[*SELECTED, "not prepared"])
        assert ordered == ["2M00000016+0000112", "J1234/5678", "2M00000003+0000021"]
        assert prepared.slice_ids((1, 2), among=SELECTED) == ["J1234/5678"]


def _summary(**columns: list):
    n = len(columns["source_id"])
    defaults = {
        "rejection_status": ["ok"] * n,
        "well_resolved": [False] * n,
        "rejection_period_unimodal": [True] * n,
    }
    return pa.table({**defaults, **columns})


class TestSelectSources:
    def test_under_resolved_and_unimodal(self):
        summary = _summary(
            source_id=["a", "b", "c", "d", "e"],
            rejection_status=["ok", "ok", "ok", "failed", "pending"],
            well_resolved=[False, True, False, None, None],
            rejection_period_unimodal=[True, True, False, None, None],
        )
        assert select_sources(summary, MCMCConfig()) == {"a"}
        assert select_sources(summary, MCMCConfig(require_unimodal=False)) == {
            "a",
            "c",
        }
        assert select_sources(summary, MCMCConfig(select="all")) == {"a", "b"}
        assert select_sources(
            summary, MCMCConfig(select="all", require_unimodal=False)
        ) == {"a", "b", "c"}

    def test_the_hook_replaces_the_rule_but_needs_an_ok_result(self):
        summary = _summary(
            source_id=["a", "b", "c"],
            rejection_status=["ok", "ok", "failed"],
            well_resolved=[False, True, None],
            n_obs=[5, 12, 20],
        )
        seen = []

        def hook(row):
            seen.append(row["source_id"])
            return row["n_obs"] > 10

        assert select_sources(summary, MCMCConfig(), hook) == {"b"}
        assert seen == ["a", "b"]  # the hook sees whole rows of ok sources only

    def test_no_ok_rejection_result(self):
        # Every rejection source failed: the statistic columns are all null.
        summary = pa.table(
            {
                "source_id": ["a", "b"],
                "rejection_status": ["failed", "pending"],
                "well_resolved": pa.nulls(2),
                "rejection_period_unimodal": pa.nulls(2),
            }
        )
        for config in (MCMCConfig(), MCMCConfig(select="all", require_unimodal=False)):
            assert select_sources(summary, config) == set()


def _chains(values, num_chains):
    values = np.asarray(values, dtype=float)
    return Samples(
        nonlinear={"period": Q(values, "day"), "e": Q(np.zeros_like(values), "")},
        linear={},
        model_type="RVModel",
        metadata={"num_chains": num_chains},
        ln_likelihood=jnp.zeros(values.size),
        ln_prior=jnp.zeros(values.size),
    )


class TestConvergence:
    def test_well_mixed_chains(self):
        draws = np.random.default_rng(0).normal(size=4000)
        r_hat, ess = convergence(_chains(draws, 4))
        assert r_hat == pytest.approx(1.0, abs=0.01)
        assert ess > 2000  # a constant parameter ("e") is skipped, not NaN

    def test_chains_stuck_in_different_places(self):
        rng = np.random.default_rng(0)
        draws = np.concatenate([rng.normal(size=500), 10 + rng.normal(size=500)])
        r_hat, ess = convergence(_chains(draws, 2))
        assert r_hat > 1.5
        assert ess < 50

    def test_nan_draws_never_converge(self):
        draws = np.random.default_rng(0).normal(size=4000)
        draws[17] = np.nan
        r_hat, ess = convergence(_chains(draws, 4))
        assert np.isnan(r_hat)
        assert np.isnan(ess)

    def test_only_constant_parameters(self):
        r_hat, ess = convergence(_chains(np.ones(100), 2))
        assert np.isnan(r_hat)
        assert np.isnan(ess)


def test_process_mcmc_status(run, monkeypatch):
    """The thresholds decide the status; NaN diagnostics never converge."""
    rejection = ResultsIndex.build(run.run_dir / "results" / "rejection").records
    data = PreparedData(run.run_dir).read([FLAKY])[FLAKY]
    process = mcmc_module.mcmc_setup(run.config)
    for diagnostics, status in [
        ((1.01, 500.0), "converged"),
        ((1.2, 500.0), "not_converged"),
        ((1.01, 10.0), "not_converged"),
        ((float("nan"), float("nan")), "not_converged"),
    ]:
        monkeypatch.setattr(mcmc_module, "convergence", lambda _s, d=diagnostics: d)
        config = MCMCConfig(num_chains=2, num_warmup=5, num_samples=5)
        payload = mcmc_module.process_mcmc(
            FLAKY,
            (data, rejection[FLAKY]),
            sampler=process.keywords["sampler"],
            read_rejection=RejectionSamples(),
            config=config,
            seed=run.config.run.seed,
        )
        assert payload.row["mcmc_status"] == status
