"""Which sources a process handles: slice ``i`` of ``N``.

See ``packages/hq/docs/spec.md``, "Execution modes". Sources are ordered by
``(n_obs, source_id)`` and slice ``i`` is ``order[i::N]``: round-robin balances
total ``n_obs`` across slices, and each slice stays sorted by ``n_obs`` so
consecutive sources reuse JIT compilations.
"""

__all__ = ("slice_ids",)

from typing import Any

from harv_hq.prepare import PreparedData


def slice_ids(prepared: PreparedData, shard: tuple[int, int]) -> list[Any]:
    """The source IDs in slice ``i`` of ``N``, in processing order.

    Parameters
    ----------
    prepared
        The run's prepared data.
    shard
        ``(i, N)`` with ``0 <= i < N``.

    Returns
    -------
        The slice's source IDs, sorted by ``(n_obs, source_id)``.

    Raises
    ------
    ValueError
        If ``shard`` is not ``(i, N)`` with ``0 <= i < N``.
    """
    i, n = shard
    if not 0 <= i < n:
        msg = f"shard must be (i, N) with 0 <= i < N, got {shard}"
        raise ValueError(msg)
    order = sorted(prepared.source_ids, key=lambda sid: (prepared.n_obs(sid), sid))
    return order[i::n]
