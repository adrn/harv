"""Tests for ``harv_hq._model_file``: importing and checking ``prior.py``."""

import textwrap

import pytest

from harv.models import HarvPrior, RVModel
from harv_hq import ConfigError
from harv_hq._model_file import ModelFile

RV_SETUP = """
from unxt import Q
import harv.models as hm

N_CALLS = 0

def make_setup():
    global N_CALLS
    N_CALLS += 1
    prior = hm.StandardRV().default_prior(
        period_min=Q(2, "day"),
        period_max=Q(100, "day"),
        sigma_K0=Q(30, "km/s"),
        sigma_v0=Q(100, "km/s"),
    )
    return prior, hm.RVModel()
"""


def write(tmp_path, text, name="prior.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


def test_setup_returns_prior_and_model(tmp_path):
    prior, model = ModelFile.load(write(tmp_path, RV_SETUP)).setup("rv")
    assert isinstance(prior, HarvPrior)
    assert isinstance(model, RVModel)


def test_make_setup_runs_once(tmp_path):
    model_file = ModelFile.load(write(tmp_path, RV_SETUP))
    first = model_file.setup("rv")
    second = model_file.setup("rv")
    assert all(a is b for a, b in zip(first, second, strict=True))
    assert model_file._module.N_CALLS == 1


def test_wrong_model_for_kind(tmp_path):
    model_file = ModelFile.load(write(tmp_path, RV_SETUP))
    with pytest.raises(ConfigError, match="requires a GaiaAstrometryModel"):
        model_file.setup("gaia_astrometry")


def test_missing_make_setup(tmp_path):
    with pytest.raises(ConfigError, match="does not define make_setup"):
        ModelFile.load(write(tmp_path, "x = 1\n"))


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        ModelFile.load(tmp_path / "nope.py")


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("def make_setup():\n    return 1\n", r"must return \(prior, model\)"),
        (
            (
                "import harv.models as hm\n"
                "def make_setup():\n    return 'prior', hm.RVModel()\n"
            ),
            "expected a HarvPrior",
        ),
    ],
)
def test_bad_return_value(tmp_path, body, match):
    with pytest.raises(ConfigError, match=match):
        ModelFile.load(write(tmp_path, body)).setup("rv")


def test_optional_hooks(tmp_path):
    without = ModelFile.load(write(tmp_path, RV_SETUP))
    assert without.select_rows is None
    assert without.select_for_mcmc is None

    hooks = RV_SETUP + (
        "\ndef select_rows(table):\n    return table['ok']\n"
        "\ndef select_for_mcmc(row):\n    return row['n_obs'] > 3\n"
    )
    with_hooks = ModelFile.load(write(tmp_path, hooks, name="prior2.py"))
    assert with_hooks.select_rows({"ok": [True]}) == [True]
    assert with_hooks.select_for_mcmc({"n_obs": 5}) is True


def test_dataclass_in_model_file(tmp_path):
    """Classes defined in the file can find their module (it is in sys.modules)."""
    body = RV_SETUP + (
        "\nfrom dataclasses import dataclass\n"
        "@dataclass\nclass Settings:\n    scale: float = 1.0\n"
        "SETTINGS = Settings()\n"
    )
    model_file = ModelFile.load(write(tmp_path, body))
    assert model_file._module.SETTINGS.scale == 1.0
