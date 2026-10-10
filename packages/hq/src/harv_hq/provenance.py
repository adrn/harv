"""Run provenance: what every hq output records, and the check that refuses stale ones.

See ``packages/hq/docs/spec.md``, "Provenance".
"""

__all__ = ("ProvenanceError", "check_provenance", "make_provenance", "sha256_file")

import hashlib
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax

import harv
from harv_hq._version import __version__ as _hq_version
from harv_hq.config import Config

logger = logging.getLogger(__name__)

# Fields whose mismatch means an output was built from different inputs, so a
# stage refuses to build on it. Versions are only logged.
_REFUSED_FIELDS = ("config_sha256", "model_file_sha256", "data_id", "prior_cache_id")
_VERSION_FIELDS = ("hq_version", "harv_version", "jax_version")


class ProvenanceError(RuntimeError):
    """An output was built from different inputs than the current run's."""


def sha256_file(path: str | os.PathLike) -> str:
    """The hex sha256 of a file's bytes.

    Parameters
    ----------
    path
        The file to hash.

    Returns
    -------
        The hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_provenance(
    config: Config,
    *,
    data_id: str | None = None,
    prior_cache_id: str | None = None,
) -> dict[str, Any]:
    """The provenance record for an output of this run, as of now.

    Parameters
    ----------
    config
        The run configuration; ``hq.toml`` and the model file are hashed.
        Every output records both, the prepared data included: the model
        file's ``select_rows`` decides which rows were prepared.
    data_id
        The prepared data's ``data_id``, when the output depends on it.
    prior_cache_id
        The prior cache's ``prior_cache_id``, for result parts.

    Returns
    -------
        A JSON-friendly dict with the keys listed in the spec; ``data_id``
        and ``prior_cache_id`` appear only when given.
    """
    record: dict[str, Any] = {
        "hq_version": _hq_version,
        "harv_version": harv.__version__,
        "jax_version": jax.__version__,
        "config_sha256": sha256_file(config.config_path),
        "model_file_sha256": sha256_file(config.run.model_file),
    }
    if data_id is not None:
        record["data_id"] = data_id
    if prior_cache_id is not None:
        record["prior_cache_id"] = prior_cache_id
    record["created"] = datetime.now(UTC).isoformat()
    return record


def check_provenance(
    found: dict[str, Any], expected: dict[str, Any], *, path: str | os.PathLike
) -> None:
    """Refuse an output built from different inputs; log version differences.

    Every refused field (``config_sha256``, ``model_file_sha256``,
    ``data_id``, ``prior_cache_id``) present in ``expected`` must equal its
    value in ``found``; a field missing from ``found`` counts as a mismatch.

    Parameters
    ----------
    found
        The provenance recorded in the output being built on.
    expected
        The current provenance, holding the fields relevant to that output.
    path
        The output's path, for the error message.

    Raises
    ------
    ProvenanceError
        Naming the first mismatched field and the file. The fix is
        ``--overwrite`` on the stage that produced the output.
    """
    for field in _REFUSED_FIELDS:
        if field in expected and found.get(field) != expected[field]:
            msg = (
                f"{path} was built with {field} = {found.get(field)!r}, but the "
                f"current run has {expected[field]!r}. Rerun the stage that "
                "wrote it with --overwrite."
            )
            raise ProvenanceError(msg)
    for field in _VERSION_FIELDS:
        if field in expected and found.get(field) != expected[field]:
            logger.warning(
                "%s was written with %s %s; this process has %s",
                path,
                field.removesuffix("_version"),
                found.get(field),
                expected[field],
            )
