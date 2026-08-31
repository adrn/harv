"""Root conftest.py: configure runtime type checking via beartype + jaxtyping."""

import os

from jaxtyping import install_import_hook
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser

pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser()],
    patterns=["README.md"],
).pytest()

# Enable runtime shape/dtype checking for modules with complete type annotations.
# When adding a new subpackage or module, add it here once its type hints are
# consistent and use the ScalarQ*/ScalarFloat conventions from harv.custom_types.
#
# HARV_NO_TYPECHECK skips the hooks entirely. It exists for the benchmark harness
# (benchmarks/, see docs/running-benchmarks.md): beartype decorates Python-level
# functions, and `model.log_prob` is called inside `jax.vmap` under
# `eqx.filter_jit`, so the checks run once at *trace* time. That does not touch
# warm timings, but it does inflate the first-call compile number the benchmarks
# report. Leave it unset everywhere else -- the test suite wants the checks.
#
# TODO: extend to other subpackages as typing is completed:
#   - harv.simulate
#   - harv.samplers
if not os.environ.get("HARV_NO_TYPECHECK"):
    install_import_hook("harv.kepler", "beartype.beartype")
    install_import_hook("harv.likelihood", "beartype.beartype")
    install_import_hook("harv.models", "beartype.beartype")
    install_import_hook("harv.periodogram", "beartype.beartype")
