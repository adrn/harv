"""Tests for SB2 runs (``kind = "sb2"``): config, model file, prepare, the
pipeline end to end, and the derived ``mass_ratio``."""

import numpy as np
import pyarrow.parquet as pq
import pytest
from astropy.time import Time
from unxt import ustrip

from harv.data import SystemData
from harv_hq import Config, ConfigError, Run, read_source
from harv_hq._model_file import ModelFile
from harv_hq._parquet_io import read_parquet

# conftest's SB2_SEMIAMPS: |K_primary| / |K_secondary|.
SB2_MASS_RATIO = 28.0 / 41.0

SB2_DATA = """
[run]
name = "test"
kind = "sb2"
seed = 1

[data]
file = "obs.fits"
source_id = "id"
time = "jd"
rv = "rv"
rv_err = "rv_err"
component = "comp"
{components}
[prior_cache]
n_samples = 1000

[rejection]
top_k = 16
"""
MCMC = """
[mcmc]
select = "all"
num_chains = 2
num_warmup = 20
num_samples = 20
"""


def load(tmp_path, components=""):
    path = tmp_path / "hq.toml"
    path.write_text(SB2_DATA.format(components=components))
    return Config.from_file(path)


class TestConfig:
    def test_defaults_and_integer_labels(self, tmp_path):
        assert load(tmp_path).data.components == {"primary": "1", "secondary": "2"}
        config = load(tmp_path, "components = {primary = 2, secondary = 1}")
        assert config.data.components == {"primary": "2", "secondary": "1"}

    @pytest.mark.parametrize(
        ("components", "match"),
        [
            ('components = {a = "1", b = "2"}', "must name exactly"),
            ('components = {primary = "1", secondary = "1"}', "distinct labels"),
            ("components = {primary = 1.5, secondary = 2}", "components must be"),
            ('components = "12"', "components must be"),
        ],
    )
    def test_bad_components(self, tmp_path, components, match):
        with pytest.raises(ConfigError, match=match):
            load(tmp_path, components)

    def test_component_keys_are_sb2_only(self, tmp_path):
        text = SB2_DATA.format(components="").replace('kind = "sb2"', 'kind = "rv"')
        (tmp_path / "hq.toml").write_text(text)
        with pytest.raises(ConfigError, match=r"\['component'\] are for kind = 'sb2'"):
            Config.from_file(tmp_path / "hq.toml")

    def test_component_is_required(self, tmp_path):
        text = SB2_DATA.format(components="").replace('component = "comp"\n', "")
        (tmp_path / "hq.toml").write_text(text)
        with pytest.raises(ConfigError, match=r"missing.*component"):
            Config.from_file(tmp_path / "hq.toml")


class TestModelFile:
    @pytest.mark.parametrize(
        ("setup", "match"),
        [
            ("prior, hm.RVModel()", "requires a JointModel"),
            (
                "prior, hm.JointModel.for_sb2(prior=named, component_names=('a', 'b'))",
                "RVModel components named",
            ),
        ],
    )
    def test_the_model_must_be_an_sb2_joint_model(self, tmp_path, setup, match):
        path = tmp_path / "prior.py"
        path.write_text(
            "from unxt import Q\nimport harv.models as hm\n"
            "def make_setup():\n"
            "    kw = dict(period_min=Q(2, 'day'), period_max=Q(100, 'day'),\n"
            "              sigma_K0=Q(30, 'km/s'), sigma_v0=Q(30, 'km/s'))\n"
            "    prior = hm.default_sb2_prior(**kw)\n"
            "    named = hm.default_sb2_prior(component_names=('a', 'b'), **kw)\n"
            f"    return {setup}\n"
        )
        with pytest.raises(ConfigError, match=match):
            ModelFile.load(path).setup("sb2")


class TestPrepare:
    def test_sources_and_catalog_counts(self, sb2_run):
        Run(sb2_run.run_dir).prepare()
        catalog = read_parquet(sb2_run.run_dir / "catalog.parquet").table.to_pydict()
        # 103 has two secondary RVs but enough primary ones; 104 has no
        # secondary, and 105 too few of either.
        kept = [101, 102, 103]
        assert catalog["source_id"] == kept
        table = sb2_run.table
        for name, label in (("primary", 1), ("secondary", 2)):
            expected = [
                int(np.sum((table["source_id"] == s) & (table["component"] == label)))
                for s in kept
            ]
            assert catalog[f"n_obs_{name}"] == expected
        assert catalog["n_obs"] == [sb2_run.kept[s] for s in kept]
        assert catalog["n_obs_secondary"][2] == 2  # kept on its primary alone

    def test_component_column_and_sort_order(self, sb2_run):
        Run(sb2_run.run_dir).prepare()
        table = pq.read_table(sb2_run.run_dir / "data.parquet")
        assert table.schema.field("component").type == "int8"
        rows = table.to_pydict()
        keys = list(
            zip(rows["source_id"], rows["component"], rows["time"], strict=True)
        )
        assert keys == sorted(keys)

    def test_read_returns_synchronized_system_data(self, sb2_run):
        Run(sb2_run.run_dir).prepare()
        data = read_source(sb2_run.run_dir, 103)
        assert isinstance(data, SystemData)
        assert list(data.keys()) == ["primary", "secondary"]
        assert data["primary"].time_ref == data["secondary"].time_ref

        table = sb2_run.table[sb2_run.table["source_id"] == 103]
        for name, label in (("primary", 1), ("secondary", 2)):
            rows = table[table["component"] == label]
            rows.sort("time")
            expected = Time(rows["time"], format="jd", scale="tdb").tcb.mjd
            np.testing.assert_allclose(
                ustrip("day", data[name].time), expected, rtol=0, atol=1e-9
            )
            np.testing.assert_array_equal(ustrip("km/s", data[name].rv), rows["rv"])

    def test_labels_are_mapped_by_the_config(self, sb2_run):
        table = sb2_run.table.copy()
        table["component"] = np.where(table["component"] == 1, "A", "B")
        table.write(sb2_run.run_dir / "observations.ecsv", overwrite=True)
        config = sb2_run.run_dir / "hq.toml"
        text = config.read_text()
        config.write_text(
            text.replace(
                'primary = "1", secondary = "2"', 'primary = "B", secondary = "A"'
            )
        )
        Run(sb2_run.run_dir).prepare()
        swapped = read_source(sb2_run.run_dir, 103)
        assert (swapped["primary"].n_obs, swapped["secondary"].n_obs) == (2, 8)

    def test_an_unmapped_label_is_refused(self, sb2_run):
        table = sb2_run.table.copy()
        table["component"][0] = 3
        table.write(sb2_run.run_dir / "observations.ecsv", overwrite=True)
        with pytest.raises(ConfigError, match=r"label\(s\) \['3'\]"):
            Run(sb2_run.run_dir).prepare()


@pytest.fixture(scope="module")
def sb2_results(tmp_path_factory, sb2_run_factory):
    """An SB2 run through rejection and summarize, built once per module."""
    fixture = sb2_run_factory(tmp_path_factory.mktemp("sb2"))
    config = fixture.run_dir / "hq.toml"
    config.write_text(config.read_text() + MCMC + "\n[results]\nflush_n_sources = 2\n")
    model_file = fixture.run_dir / "prior.py"
    model_file.write_text(
        model_file.read_text()
        + "\n\ndef select_for_mcmc(row):\n    return row['source_id'] == 102\n"
    )
    run = Run(fixture.run_dir)
    run.prepare()
    run.make_prior_cache()
    run.run_rejection(compact=False)
    run.summarize()
    return fixture


class TestPipeline:
    def test_every_source_succeeds(self, sb2_results):
        run = Run(sb2_results.run_dir)
        assert run.status()["rejection"] == {"ok": 3, "pending": 0}
        # Resuming finds nothing to do.
        run.run_rejection(compact=False)
        assert run.status()["rejection"] == {"ok": 3, "pending": 0}

    def test_summary_columns_and_mass_ratio(self, sb2_results):
        summary = pq.read_table(sb2_results.run_dir / "summary.parquet").to_pydict()
        for column in (
            "map_primary.rv_semiamp",
            "secondary.rv_semiamp_p50",
            "map_v_sys",
            "periods_spanned",
            "mass_ratio_p16",
            "mass_ratio_p84",
        ):
            assert column in summary
        well_sampled = summary["source_id"].index(101)
        assert summary["mass_ratio_p50"][well_sampled] == pytest.approx(
            SB2_MASS_RATIO, rel=0.05
        )
        assert np.all(np.asarray(summary["secondary.rv_semiamp_p50"]) < 0)

    def test_load_source(self, sb2_results):
        result = Run(sb2_results.run_dir).load_source(103)
        assert isinstance(result.data, SystemData)
        assert result.rejection_status == "ok"
        assert "primary.rv_semiamp" in result.rejection.linear

    def test_compact(self, sb2_results):
        run = Run(sb2_results.run_dir)
        report = run.compact()
        assert report.n_sources == 3
        assert run.status()["rejection"] == {"ok": 3, "pending": 0}

    def test_mcmc_on_one_source(self, sb2_results):
        run = Run(sb2_results.run_dir)
        run.run_mcmc()
        run.summarize()
        result = run.load_source(102)
        assert result.mcmc_status == "ok"
        assert "secondary.rv_semiamp" in result.mcmc.linear
        summary = pq.read_table(sb2_results.run_dir / "summary.parquet").to_pydict()
        assert summary["mcmc_status"] == [
            "not_selected",
            result.mcmc_row["mcmc_status"],
            "not_selected",
        ]
        assert summary["mass_ratio_p50"][1] == pytest.approx(SB2_MASS_RATIO, rel=0.1)
