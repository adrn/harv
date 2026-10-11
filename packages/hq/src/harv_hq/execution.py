"""Drivers that run a per-source function over a slice of sources.

See ``packages/hq/docs/spec.md``, "Execution modes". A driver gets the
slice's IDs and data, a picklable ``setup`` that builds the per-source
function (``setup() -> process(source_id, data) -> Payload``), and a ``sink``
that receives every payload. Only the calling process writes results, so
drivers know nothing about parts. The MPI mode is the serial driver run on
slice ``rank/size`` (:func:`mpi_comm`).
"""

__all__ = ("Progress", "mpi_comm", "run_pool", "run_serial")

import logging
import multiprocessing
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any, cast, final
from unittest.mock import patch

from harv_hq.results import Payload

logger = logging.getLogger("harv_hq")

ProcessFn = Callable[[Any, Any], Payload]
Setup = Callable[[], ProcessFn]
Sink = Callable[[Payload], None]

# One CPU thread per pool worker, so W workers use W cores. Set in the
# environment the workers are spawned with: by the time a worker's own code
# runs, importing harv_hq has already loaded NumPy (BLAS) and JAX.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_XLA_SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


@final
class Progress:
    """Log a progress line about every 5% of a slice.

    Parameters
    ----------
    label
        The stage, for the log line.
    total
        Sources in this process's slice.
    ranks
        MPI ranks; above 1, the line also extrapolates this rank's count to
        the whole run (rank 0 only logs this).
    """

    def __init__(self, label: str, total: int, *, ranks: int = 1) -> None:
        self.label = label
        self.total = total
        self.ranks = ranks
        self.done = 0
        self._every = max(1, total // 20)

    def tick(self) -> None:
        """Count one finished source."""
        self.done += 1
        if self.done % self._every and self.done != self.total:
            return
        overall = (
            f" (about {self.done * self.ranks}/{self.total * self.ranks} "
            f"over {self.ranks} ranks)"
            if self.ranks > 1
            else ""
        )
        logger.info("%s: %d/%d sources%s", self.label, self.done, self.total, overall)


def run_serial(
    ids: Sequence[Any], data: Mapping[Any, Any], setup: Setup, sink: Sink
) -> None:
    """Process every source in this process, in order.

    Parameters
    ----------
    ids
        The sources to process, in order.
    data
        Source ID to its data.
    setup
        Builds the per-source function.
    sink
        Receives each payload.
    """
    process = setup()
    for source_id in ids:
        sink(process(source_id, data[source_id]))


def run_pool(
    ids: Sequence[Any],
    data: Mapping[Any, Any],
    setup: Setup,
    sink: Sink,
    *,
    workers: int,
) -> None:
    """Process sources on ``workers`` spawned processes; payloads return here.

    Each worker runs ``setup`` once (its initializer) and is pinned to one
    CPU thread. At most ``2 * workers`` sources are in flight, so memory stays
    flat however long ``ids`` is. Payloads reach ``sink`` in completion order.

    Parameters
    ----------
    ids
        The sources to process; submitted in this order.
    data
        Source ID to its data; each task carries one source's data.
    setup
        Builds the per-source function; must be picklable.
    sink
        Receives each payload, in this process.
    workers
        Number of worker processes.
    """
    changes = {**_THREAD_ENV, "XLA_FLAGS": _XLA_SINGLE_THREAD}
    if flags := os.environ.get("XLA_FLAGS"):
        changes["XLA_FLAGS"] = f"{flags} {_XLA_SINGLE_THREAD}"
    # Set only while the workers are spawned; this process's environment is
    # restored afterwards.
    with patch.dict(os.environ, changes):
        pool = ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(setup,),
        )
        try:
            pending: set[Future[Payload]] = set()
            for source_id in ids:
                if len(pending) >= 2 * workers:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        sink(future.result())
                pending.add(pool.submit(_process_in_worker, source_id, data[source_id]))
            for future in wait(pending).done:
                sink(future.result())
        finally:
            # On an interruption, drop queued work rather than wait for it.
            pool.shutdown(wait=True, cancel_futures=True)


def mpi_comm() -> Any:
    """``MPI.COMM_WORLD``, importing ``mpi4py`` only when ``--mpi`` is used.

    Returns
    -------
        The world communicator.

    Raises
    ------
    ImportError
        If ``mpi4py`` is not installed.
    """
    try:
        from mpi4py import MPI  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as err:
        msg = "hq run --mpi needs mpi4py: pip install 'harv-hq[mpi]'"
        raise ImportError(msg) from err
    return MPI.COMM_WORLD


# The per-source function, built once per worker by _init_worker.
_worker_process: ProcessFn | None = None


def _init_worker(setup: Setup) -> None:
    global _worker_process  # noqa: PLW0603 -- one per worker process
    _worker_process = setup()


def _process_in_worker(source_id: Any, data: Any) -> Payload:
    return cast("ProcessFn", _worker_process)(source_id, data)
