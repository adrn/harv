"""Run provenance: what every hq output records, and the check that refuses stale ones.

See ``packages/hq/docs/spec.md``, "Provenance".
"""

__all__ = (
    "ALL_TABLES",
    "CATALOG_TABLES",
    "PREPARED_TABLES",
    "PRIOR_CACHE_TABLES",
    "REJECTION_TABLES",
    "ProvenanceError",
    "check_provenance",
    "config_hashes",
    "make_provenance",
    "sha256_file",
)

import hashlib
import json
import logging
import os
import tomllib
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
_REFUSED_FIELDS = ("model_file_sha256", "data_id", "prior_cache_id")
# The hq.toml tables each output depends on. [results] and [serve] change only
# how results are stored or shown, so no output records them.
PREPARED_TABLES = ("run", "data", "prepare")
# catalog.parquet also depends on [catalog]; the data do not, so editing the
# catalog columns leaves the data and every result built on them valid.
CATALOG_TABLES = (*PREPARED_TABLES, "catalog")
PRIOR_CACHE_TABLES = ("run", "prior_cache")
REJECTION_TABLES = ("run", "data", "prepare", "prior_cache", "rejection")
MCMC_TABLES = (*REJECTION_TABLES, "mcmc")
ALL_TABLES = ("run", "data", "prepare", "catalog", "prior_cache", "rejection", "mcmc")
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


def config_hashes(
    config_path: str | os.PathLike, tables: tuple[str, ...]
) -> dict[str, str]:
    """The sha256 of each named ``hq.toml`` table's parsed values.

    Each table is hashed as sorted-key JSON, so comments, formatting, key
    order, and every other table leave its hash unchanged. A table absent from
    the file hashes as ``null``.

    Parameters
    ----------
    config_path
        The ``hq.toml``.
    tables
        Table names to hash.

    Returns
    -------
        Table name to hex digest.
    """
    with Path(config_path).open("rb") as f:
        raw = tomllib.load(f)
    return {
        name: hashlib.sha256(
            json.dumps(raw.get(name), sort_keys=True, default=str).encode()
        ).hexdigest()
        for name in tables
    }


def make_provenance(
    config: Config,
    *,
    tables: tuple[str, ...] = ALL_TABLES,
    data_id: str | None = None,
    prior_cache_id: str | None = None,
) -> dict[str, Any]:
    """The provenance record for an output of this run, as of now.

    Parameters
    ----------
    config
        The run configuration; the model file is hashed, and so are the
        ``hq.toml`` ``tables``. Every output records the model file, the
        prepared data included: its ``select_rows`` decides which rows were
        prepared.
    tables
        The ``hq.toml`` tables the output depends on (``PREPARED_TABLES``,
        ``CATALOG_TABLES``, ``PRIOR_CACHE_TABLES``, ``REJECTION_TABLES``).
        The default, every table, is for checking: a check compares only the
        tables the output recorded.
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
        "config_sha256": config_hashes(config.config_path, tables),
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

    Every ``hq.toml`` table hash ``found`` recorded (``config_sha256``) must
    equal the current one in ``expected``, and every other refused field
    (``model_file_sha256``, ``data_id``, ``prior_cache_id``) present in
    ``expected`` must equal its value in ``found``; a field missing from
    ``found`` counts as a mismatch.

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
    tables = found.get("config_sha256")
    current = expected.get("config_sha256", {})
    if not isinstance(tables, dict):
        changed = ["(unrecorded)"]
    else:
        changed = [
            name for name, digest in tables.items() if current.get(name) != digest
        ]
    if "config_sha256" in expected and changed:
        msg = (
            f"{path} was built with different hq.toml tables {changed} "
            "(config_sha256) than the current run. Rerun the stage that wrote "
            "it with --overwrite."
        )
        raise ProvenanceError(msg)
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
