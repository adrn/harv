"""Test that the package can be imported."""

import subprocess
import sys

import harv_hq


def test_import():
    assert harv_hq.__version__


def test_import_is_quiet_and_float64():
    """hq enables float64 itself, so harv's warning about it never fires."""
    code = "import harv_hq, jax; assert jax.config.jax_enable_x64"
    result = subprocess.run(  # noqa: S603 -- fixed argv
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={"JAX_ENABLE_X64": "0", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_the_web_stack_is_not_imported():
    code = (
        "import sys, harv_hq, harv_hq.cli, harv_hq.run; "
        "assert 'fastapi' not in sys.modules and 'uvicorn' not in sys.modules"
    )
    result = subprocess.run(  # noqa: S603 -- fixed argv
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
