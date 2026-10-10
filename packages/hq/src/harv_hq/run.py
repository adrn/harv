"""``Run``: one run directory, and the pipeline stages as methods.

See ``packages/hq/docs/spec.md``, "Stages" and "Public API". Each CLI
subcommand is a thin wrapper around one method here.
"""

__all__ = ("Run",)

import os
from pathlib import Path
from typing import final

from harv_hq._model_file import ModelFile
from harv_hq.config import Config
from harv_hq.prepare import prepare


@final
class Run:
    """A run directory: its configuration, model file, and stages.

    Parameters
    ----------
    run_dir
        The directory holding ``hq.toml``.

    Raises
    ------
    ConfigError
        If ``hq.toml`` is invalid.
    """

    def __init__(self, run_dir: str | os.PathLike) -> None:
        self.config = Config.from_file(Path(run_dir) / "hq.toml")
        self._model_file: ModelFile | None = None

    @property
    def model_file(self) -> ModelFile:
        """The run's model file, imported on first use."""
        if self._model_file is None:
            self._model_file = ModelFile.load(self.config.run.model_file)
        return self._model_file

    def prepare(self, *, overwrite: bool = False) -> None:
        """Prepare per-source data from the input table (``hq prepare``).

        Parameters
        ----------
        overwrite
            Replace existing prepared data.
        """
        prepare(
            self.config, select_rows=self.model_file.select_rows, overwrite=overwrite
        )
