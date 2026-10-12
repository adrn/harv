"""Hatch metadata hook: pin harv to this package's own version when released.

harv and harv-hq are released in lockstep from one git tag, and hatch-vcs gives
both the same version, so a released harv-hq requires exactly the harv it was
tested with. Development versions (a ``.dev`` or local ``+g<sha>`` part) leave
harv unpinned, so the workspace lock does not change with every commit.
"""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.version import Version


class PinHarvHook(MetadataHookInterface):
    """Add ``harv`` to the dependencies listed in this hook's config."""

    def update(self, metadata: dict[str, Any]) -> None:
        """Set ``dependencies`` to the configured list plus ``harv``."""
        version = Version(metadata["version"])
        released = not (version.is_devrelease or version.local)
        harv = f"harv=={version}" if released else "harv"
        metadata["dependencies"] = [*self.config["dependencies"], harv]
