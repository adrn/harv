"""Test that the package can be imported."""

import os
import subprocess
import sys

import harv_hq


def test_import():
    assert harv_hq.__version__


def _python(code: str, **env: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter, with JAX_ENABLE_X64 unset unless given."""
    environ = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    return subprocess.run(  # noqa: S603 -- fixed argv
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**environ, **env},
    )


def test_import_is_quiet_float64_and_without_the_web_stack():
    """With JAX_ENABLE_X64 unset, hq enables float64 before harv can warn about
    it, and importing hq (and its CLI) does not load the web stack."""
    result = _python(
        "import sys, harv_hq, harv_hq.cli, harv_hq.run, jax; "
        "assert jax.config.jax_enable_x64; "
        "assert 'fastapi' not in sys.modules and 'uvicorn' not in sys.modules"
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_an_explicit_jax_enable_x64_is_left_alone():
    result = _python(
        "import harv_hq, jax; assert not jax.config.jax_enable_x64",
        JAX_ENABLE_X64="0",
    )
    assert result.returncode == 0, result.stderr
