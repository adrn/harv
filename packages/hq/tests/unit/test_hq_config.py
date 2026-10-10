"""Tests for ``harv_hq.config``: loading and validating ``hq.toml``."""

import dataclasses
import re
from pathlib import Path

import pytest

from harv_hq import Config, ConfigError
from harv_hq.config import _TABLES

SPEC = Path(__file__).parents[2] / "docs" / "spec.md"

MINIMAL_RV = """
[run]
name = "test"
kind = "rv"
seed = 1

[data]
file = "obs.fits"
source_id = "id"
time = "jd"
rv = "rv"
rv_err = "rv_err"

[prior_cache]
n_samples = 1000

[rejection]
top_k = 16
"""

MINIMAL_GAIA = """
[run]
name = "test"
kind = "gaia_astrometry"
seed = 1

[data]
file = "obs.fits"
source_id = "id"
time = "t"
al_position = "x"
al_position_err = "x_err"
scan_angle = "psi"
parallax_factor = "plx_factor"

[prior_cache]
n_samples = 1000

[rejection]
top_k = 16
"""


def load(tmp_path, text):
    path = tmp_path / "hq.toml"
    path.write_text(text)
    return Config.from_file(path)


def with_table(base, table, body):
    """Replace or append ``[table]`` in a TOML string."""
    pattern = rf"\[{table}\]\n(?:(?!\n\[).)*"
    replacement = f"[{table}]\n{body}"
    if re.search(rf"^\[{table}\]$", base, flags=re.MULTILINE):
        return re.sub(pattern, replacement, base, flags=re.DOTALL)
    return f"{base}\n{replacement}\n"


def spec_config_section():
    text = SPEC.read_text()
    start = text.index("## Configuration (`hq.toml`)")
    return text[start : text.index("## The model file", start)]


class TestSpecAgreement:
    def test_spec_example_parses(self, tmp_path):
        section = spec_config_section()
        example = section[section.index("### Example") :]
        toml = example.split("```toml\n", 1)[1].split("```", 1)[0]
        config = load(tmp_path, toml)
        assert config.run.name == "apogee-dr17-binaries"
        assert config.data.rv == "VHELIO"
        assert config.catalog.columns == ["TEFF", "LOGG", "M_H", "J", "K"]
        assert config.prior_cache.n_samples == 100_000_000
        assert config.mcmc.select == "under_resolved"

    def test_every_spec_key_is_a_field_and_vice_versa(self):
        """The spec's key tables and the dataclasses list exactly the same keys."""
        section = spec_config_section()
        spec_keys: dict[str, set[str]] = {}
        for block in re.split(r"^### ", section, flags=re.MULTILINE)[1:]:
            match = re.match(r"`\[(\w+)\]`", block)
            if match is None:
                continue
            keys = set(re.findall(r"^\| `(\w+)`", block, flags=re.MULTILINE))
            spec_keys[match.group(1)] = keys

        assert set(spec_keys) == set(_TABLES)
        for table, (table_cls, _) in _TABLES.items():
            fields = {f.name for f in dataclasses.fields(table_cls)}
            assert fields == spec_keys[table], table


class TestDefaults:
    def test_minimal_rv(self, tmp_path):
        config = load(tmp_path, MINIMAL_RV)
        assert config.run.model_file == tmp_path.resolve() / "prior.py"
        assert config.data.rv_unit == "km/s"
        assert config.data.time_format == "jd"
        assert config.data.time_scale == "tdb"
        assert config.data.al_position is None
        assert config.prepare.min_n_obs == 3
        assert config.catalog is None
        assert config.mcmc is None
        assert config.results.flush_n_sources == 1000
        assert config.results.compact_n_sources == 100_000
        assert config.serve.host == "127.0.0.1"
        assert config.rejection.randomize_prior_order is True
        assert config.run_dir == tmp_path.resolve()

    def test_minimal_gaia(self, tmp_path):
        config = load(tmp_path, MINIMAL_GAIA)
        assert config.data.al_position_unit == "mas"
        assert config.data.scan_angle_unit == "deg"
        assert config.data.rv is None

    def test_empty_mcmc_table_takes_defaults(self, tmp_path):
        config = load(tmp_path, MINIMAL_RV + "\n[mcmc]\n")
        assert config.mcmc.num_chains == 4
        assert config.mcmc.max_r_hat == 1.05

    def test_frozen(self, tmp_path):
        config = load(tmp_path, MINIMAL_RV)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.run.seed = 2


class TestPaths:
    def test_relative_paths_resolve_against_run_dir(self, tmp_path):
        config = load(
            tmp_path,
            with_table(MINIMAL_RV, "catalog", 'file = "cat.fits"\nsource_id = "id"'),
        )
        assert config.data.file == tmp_path.resolve() / "obs.fits"
        assert config.catalog.file == tmp_path.resolve() / "cat.fits"

    def test_absolute_paths_are_kept(self, tmp_path):
        # Outside the run directory, and absolute on every platform: on
        # Windows "/scratch/..." has no drive, so it is not absolute there.
        data_file = (tmp_path / "scratch" / "obs.fits").resolve()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        text = MINIMAL_RV.replace(
            'file = "obs.fits"', f'file = "{data_file.as_posix()}"'
        )
        assert load(run_dir, text).data.file == data_file


class TestErrors:
    @pytest.mark.parametrize(
        ("text", "match"),
        [
            (
                MINIMAL_RV.replace('rv_err = "rv_err"', 'rv_er = "rv_err"'),
                r"\[data\] unknown key.*rv_er",
            ),
            (MINIMAL_RV + "\n[extra]\nx = 1\n", r"unknown table.*extra"),
            (
                MINIMAL_RV.replace('name = "test"\n', ""),
                r"\[run\] missing required key.*name",
            ),
            (
                MINIMAL_RV.replace("[rejection]\ntop_k = 16\n", ""),
                r"missing required table \[rejection\]",
            ),
            (
                MINIMAL_RV.replace("seed = 1", 'seed = "1"'),
                r"\[run\] seed must be an integer",
            ),
            (
                MINIMAL_RV.replace("seed = 1", "seed = true"),
                r"\[run\] seed must be an integer",
            ),
            (
                MINIMAL_RV.replace("seed = 1", "seed = 1.5"),
                r"\[run\] seed must be an integer",
            ),
            (
                MINIMAL_RV.replace('kind = "rv"', 'kind = "astrometry"'),
                r"\[run\] kind must be one of",
            ),
            (
                MINIMAL_RV.replace("top_k = 16", "top_k = 0"),
                r"\[rejection\] top_k must be positive",
            ),
            (
                MINIMAL_RV.replace('rv_err = "rv_err"\n', ""),
                r"\[data\] missing.*rv_err.*'rv'",
            ),
            (
                MINIMAL_RV.replace('time = "jd"', 'time = "jd"\nscan_angle = "psi"'),
                r"for kind = 'gaia_astrometry'",
            ),
            (
                MINIMAL_RV.replace('time = "jd"', 'time = "jd"\ntime_scale = "bjd"'),
                r"\[data\] time_scale must be one of",
            ),
            (
                MINIMAL_RV.replace(
                    'time = "jd"', 'time = "jd"\ntime_format = "julian"'
                ),
                r"\[data\] time_format must be one of",
            ),
            (
                MINIMAL_RV.replace('time = "jd"', 'time = "jd"\nhdu = 1.5'),
                r"\[data\] hdu must be an integer or a string",
            ),
            (
                MINIMAL_RV + '\n[mcmc]\nselect = "everything"\n',
                r"\[mcmc\] select must be one of",
            ),
            (
                MINIMAL_RV + '\n[mcmc]\nchain_method = "threads"\n',
                r"\[mcmc\] chain_method must be one of",
            ),
            (
                MINIMAL_RV
                + '\n[catalog]\nfile = "c.fits"\nsource_id = "id"'
                + '\ncolumns = ["a", 1]\n',
                r"\[catalog\] columns must be a list of strings",
            ),
            (
                MINIMAL_RV.replace('file = "obs.fits"', "file = 3"),
                r"\[data\] file must be a path string",
            ),
            ("name = 'x'\n" + MINIMAL_RV, r"unknown table.*name"),
            (MINIMAL_RV + "\n[run\n", r"not valid TOML"),
        ],
    )
    def test_invalid_config_names_the_key(self, tmp_path, text, match):
        with pytest.raises(ConfigError, match=match):
            load(tmp_path, text)

    def test_hdu_accepts_int_and_str(self, tmp_path):
        as_int = MINIMAL_RV.replace('time = "jd"', 'time = "jd"\nhdu = 1')
        as_str = MINIMAL_RV.replace('time = "jd"', 'time = "jd"\nhdu = "VISITS"')
        assert load(tmp_path, as_int).data.hdu == 1
        assert load(tmp_path, as_str).data.hdu == "VISITS"

    def test_float_field_accepts_integer(self, tmp_path):
        config = load(
            tmp_path,
            MINIMAL_RV.replace("top_k = 16", "top_k = 16\nmin_evidence_ess = 5"),
        )
        assert config.rejection.min_evidence_ess == 5.0
        assert type(config.rejection.min_evidence_ess) is float
