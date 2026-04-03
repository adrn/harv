"""Root conftest.py: configure runtime type checking via beartype + jaxtyping."""

from beartype import beartype
from jaxtyping import install_import_hook

# Enable runtime shape/dtype checking for modules with complete type annotations.
# When adding a new subpackage or module, add it here once its type hints are
# consistent and use the ScalarQ*/ScalarFloat conventions from harv.custom_types.
#
# TODO: extend to other subpackages as typing is completed:
#   - harv.likelihood
#   - harv.simulate
#   - harv.samplers
#   - harv.priors
install_import_hook("harv.kepler", typechecker=beartype)
