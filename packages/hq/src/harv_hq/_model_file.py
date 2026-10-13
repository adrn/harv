"""Load the user's model file (``prior.py``) and check what it returns.

See ``packages/hq/docs/spec.md``, "The model file (``prior.py``)".
"""

__all__ = ("ModelFile",)

import hashlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, final

from harv.models import GaiaAstrometryModel, HarvPrior, JointModel, RVModel
from harv_hq.config import SB2_COMPONENTS, ConfigError

_MODEL_TYPES = {
    "rv": RVModel,
    "gaia_astrometry": GaiaAstrometryModel,
    "sb2": JointModel,
}
Model = RVModel | GaiaAstrometryModel | JointModel


@final
class ModelFile:
    """An imported model file, with its required and optional hooks.

    ``make_setup`` is called at most once per instance (see :meth:`setup`), so
    each process pays for building the prior and model once.
    """

    def __init__(self, path: Path, module: ModuleType) -> None:
        self.path = path
        self._module = module
        self._setup: tuple[HarvPrior, Model] | None = None
        if not callable(getattr(module, "make_setup", None)):
            msg = f"model file {path} does not define make_setup()"
            raise ConfigError(msg)

    @classmethod
    def load(cls, path: str | Path) -> "ModelFile":
        """Import the model file at ``path``.

        Parameters
        ----------
        path
            Path to the model file.

        Returns
        -------
            The loaded model file.

        Raises
        ------
        ConfigError
            If the file does not exist or does not define ``make_setup``.
        """
        path = Path(path).resolve()
        if not path.is_file():
            msg = f"model file {path} does not exist"
            raise ConfigError(msg)
        # A name unique to the file, registered in sys.modules so that classes
        # defined in it (dataclasses, equinox modules) can find their module.
        name = "hq_model_file_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            msg = f"model file {path} cannot be imported as Python"
            raise ConfigError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return cls(path, module)

    @property
    def select_rows(self) -> Callable[[Any], Any] | None:
        """The optional ``select_rows(table)`` hook, or ``None``."""
        return getattr(self._module, "select_rows", None)

    @property
    def select_for_mcmc(self) -> Callable[[dict[str, Any]], bool] | None:
        """The optional ``select_for_mcmc(row)`` hook, or ``None``."""
        return getattr(self._module, "select_for_mcmc", None)

    def setup(self, kind: str) -> tuple[HarvPrior, Model]:
        """Call ``make_setup()`` once, check its result, and cache it.

        Parameters
        ----------
        kind
            The run kind (``"rv"``, ``"gaia_astrometry"`` or ``"sb2"``) the
            model must match.

        Returns
        -------
            The ``(prior, model)`` pair.

        Raises
        ------
        ConfigError
            If ``make_setup`` does not return a ``(HarvPrior, model)`` pair
            whose model matches ``kind``. For ``"sb2"``, the model must be a
            ``JointModel`` of two ``RVModel`` components named ``primary``
            and ``secondary``.
        """
        if self._setup is None:
            result = self._module.make_setup()
            if not (isinstance(result, tuple) and len(result) == 2):
                msg = (
                    f"make_setup() in {self.path} must return (prior, model), "
                    f"got {type(result).__name__}"
                )
                raise ConfigError(msg)
            self._setup = result
        prior, model = self._setup
        if not isinstance(prior, HarvPrior):
            msg = (
                f"make_setup() in {self.path} returned a {type(prior).__name__} "
                "as the prior; expected a HarvPrior"
            )
            raise ConfigError(msg)
        expected = _MODEL_TYPES[kind]
        if not isinstance(model, expected):
            msg = (
                f"make_setup() in {self.path} returned a {type(model).__name__}, "
                f"but kind = {kind!r} requires a {expected.__name__}"
            )
            raise ConfigError(msg)
        if isinstance(model, JointModel) and not (
            sorted(model.components) == sorted(SB2_COMPONENTS)
            and all(isinstance(c, RVModel) for c in model.components.values())
        ):
            found = {k: type(c).__name__ for k, c in model.components.items()}
            msg = (
                f"make_setup() in {self.path} returned a JointModel with "
                f"components {found}; kind = 'sb2' requires RVModel components "
                f"named {list(SB2_COMPONENTS)} (JointModel.for_sb2)"
            )
            raise ConfigError(msg)
        return prior, model
