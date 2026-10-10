"""Source IDs: the stable hash and the PRNG keys derived from it.

See ``packages/hq/docs/spec.md``, "Per-source randomness". Every key depends
only on the run seed and the source ID, never on shard layout, process count,
or execution order.
"""

__all__ = ("prior_cache_key", "source_key", "stable_hash")

import hashlib
from typing import Any, Literal

import jax
import numpy as np

_STAGES = {"rejection": 0, "mcmc": 1}
_PRIOR_CACHE_FOLD = 2**32 - 1


def stable_hash(source_id: Any) -> int:
    """A deterministic 32-bit hash of a source ID.

    The first 4 bytes of the sha256 of ``str(source_id)`` (UTF-8), read as a
    big-endian unsigned integer. Unlike Python's ``hash``, it is the same in
    every process and on every machine, so it can seed per-source randomness
    and select reproducible subsets (e.g. ``stable_hash(id) % 100 == 0`` in a
    model file's ``select_rows``).

    Parameters
    ----------
    source_id
        The source ID; integer and string IDs that print the same hash the
        same.

    Returns
    -------
        An integer in ``[0, 2**32)``.

    Examples
    --------
    >>> stable_hash("2M00000002+7417074")
    807116804
    >>> stable_hash(4295806720) == stable_hash("4295806720")
    True
    """
    digest = hashlib.sha256(str(source_id).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def source_key(
    seed: int, source_id: Any, stage: Literal["rejection", "mcmc"]
) -> jax.Array:
    """The PRNG key for one source in one stage.

    ``fold_in(fold_in(key(seed), stable_hash(source_id)), stage)`` with stage 0
    for rejection and 1 for MCMC.

    Parameters
    ----------
    seed
        The run seed (``[run] seed``).
    source_id
        The source ID.
    stage
        ``"rejection"`` or ``"mcmc"``.

    Returns
    -------
        A JAX PRNG key.
    """
    key = jax.random.fold_in(jax.random.key(seed), np.uint32(stable_hash(source_id)))
    return jax.random.fold_in(key, _STAGES[stage])


def prior_cache_key(seed: int) -> jax.Array:
    """The PRNG key for building the shared prior cache.

    Parameters
    ----------
    seed
        The run seed (``[run] seed``).

    Returns
    -------
        ``fold_in(key(seed), 2**32 - 1)``.
    """
    return jax.random.fold_in(jax.random.key(seed), np.uint32(_PRIOR_CACHE_FOLD))
