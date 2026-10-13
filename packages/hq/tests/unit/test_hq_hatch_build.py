"""Tests for ``packages/hq/hatch_build.py``: the harv pin in released builds."""

import importlib.util
from pathlib import Path

import pytest

HOOK_FILE = Path(__file__).parents[2] / "hatch_build.py"


@pytest.fixture(scope="module")
def hook_class():
    spec = importlib.util.spec_from_file_location("hq_hatch_build", HOOK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PinHarvHook


@pytest.mark.parametrize(
    ("version", "harv"),
    [
        ("0.5.0", "harv==0.5.0"),  # a release: pinned
        ("0.5.0rc1", "harv==0.5.0rc1"),  # a pre-release is a release too
        ("0.5.1.dev3+gabc1234", "harv"),  # between tags
        ("0.5.0+d20261012", "harv"),  # a tagged but dirty checkout
        ("0.5.1.dev3", "harv"),
    ],
)
def test_harv_is_pinned_only_in_releases(hook_class, version, harv):
    hook = hook_class(str(HOOK_FILE.parent), {"dependencies": ["numpy>=2.0"]})
    metadata = {"version": version}
    hook.update(metadata)
    assert metadata["dependencies"] == ["numpy>=2.0", harv]
