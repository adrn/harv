"""Root conftest.py: configure runtime type checking via beartype + jaxtyping."""

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
# TODO: extend to other subpackages as typing is completed:
#   - harv.simulate
#   - harv.samplers
install_import_hook("harv.kepler", "beartype.beartype")
install_import_hook("harv.likelihood", "beartype.beartype")
install_import_hook("harv.models", "beartype.beartype")
install_import_hook("harv.periodogram", "beartype.beartype")
