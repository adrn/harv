"""harv enables JAX's float64 mode on import, and says so.

Each case runs in a subprocess: the flag is global and process-wide, and the
whole point of the feature is what happens at the moment `harv` is first
imported, which cannot be re-staged inside an already-running interpreter.

See ``docs/spec.md`` -> Core design principles, Double precision.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

PROBE = textwrap.dedent("""
    import json, warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import harv  # noqa: F401
    import jax
    print(json.dumps({
        "x64": bool(jax.config.read("jax_enable_x64")),
        "warned": [
            str(w.message) for w in caught if "jax_enable_x64" in str(w.message)
        ],
    }))
""")


def _probe(**env_overrides: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(  # noqa: S603  -- argv is this module's own constant
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_enables_x64_and_warns():
    """The default: harv turns float64 on and tells the user it did."""
    out = _probe()
    assert out["x64"] is True
    assert len(out["warned"]) == 1
    assert "JAX_ENABLE_X64" in out["warned"][0]


def test_explicit_opt_in_is_silent():
    """Saying yes first means harv has nothing to announce."""
    out = _probe(JAX_ENABLE_X64="1")
    assert out["x64"] is True
    assert out["warned"] == []


def test_explicit_opt_out_is_respected():
    """`JAX_ENABLE_X64=0` is a deliberate choice, so harv leaves it alone.

    No warning either: the user already knows, and nagging on every import is
    how warnings get filtered out wholesale.
    """
    out = _probe(JAX_ENABLE_X64="0")
    assert out["x64"] is False
    assert out["warned"] == []


@pytest.mark.parametrize("value", ["1", "0"])
def test_the_suite_itself_never_triggers_the_warning(value):
    """`conftest.py` sets the flag before harv is imported, on purpose.

    `filterwarnings = ["error"]` would otherwise turn harv's own advisory into
    a collection error, and the fix for that must not be an ignore line.
    """
    out = _probe(JAX_ENABLE_X64=value)
    assert out["warned"] == []
