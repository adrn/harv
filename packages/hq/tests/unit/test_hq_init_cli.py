"""Tests for ``hq init`` (``harv_hq.init_run``) and the CLI skeleton."""

import argparse

import pytest

from harv_hq import Config, init_run
from harv_hq._model_file import ModelFile
from harv_hq.cli import main, parse_shard


@pytest.mark.parametrize("kind", ["rv", "gaia_astrometry"])
def test_init_templates_load_and_set_up(tmp_path, kind):
    run_dir = init_run(tmp_path / "run", kind=kind)
    config = Config.from_file(run_dir / "hq.toml")
    assert config.run.kind == kind
    # The template model file builds a valid prior and model for its kind.
    ModelFile.load(config.run.model_file).setup(kind)


def test_init_into_existing_empty_directory(tmp_path):
    assert init_run(tmp_path, kind="rv") == tmp_path
    assert (tmp_path / "hq.toml").exists()


def test_init_refuses_non_empty_directory(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not empty"):
        init_run(tmp_path, kind="rv")


def test_init_unknown_kind(tmp_path):
    with pytest.raises(ValueError, match="kind must be one of"):
        init_run(tmp_path / "run", kind="sb2")


def test_cli_init(tmp_path, capsys):
    assert main(["init", str(tmp_path / "run"), "--kind", "gaia_astrometry"]) == 0
    assert "hq.toml" in capsys.readouterr().out
    config = Config.from_file(tmp_path / "run" / "hq.toml")
    assert config.run.kind == "gaia_astrometry"


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare"],
        ["prior-cache"],
        ["run", "--shard", "1/4"],
        ["mcmc", "--mpi"],
        ["compact", "--stage", "mcmc"],
        ["summarize"],
        ["status"],
        ["serve"],
    ],
)
def test_cli_unimplemented_subcommands_exit_with_a_message(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert "not implemented yet" in str(excinfo.value.code)


def test_cli_shard_and_mpi_are_exclusive(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--shard", "0/2", "--mpi"])
    assert excinfo.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_parse_shard():
    assert parse_shard("3/16") == (3, 16)
    for bad in ["16/16", "-1/4", "3", "a/b", "1/0"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_shard(bad)
