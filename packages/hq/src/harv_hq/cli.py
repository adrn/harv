"""The ``hq`` command-line interface: a thin wrapper around :mod:`harv_hq`.

Every subcommand maps to one public function or ``Run`` method (spec,
"Stages"). Subcommands whose stage is not implemented yet exit with a message
saying so.
"""

__all__ = ("main",)

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from harv_hq.config import KINDS, ConfigError
from harv_hq.init import init_run
from harv_hq.provenance import ProvenanceError
from harv_hq.run import Run

# Errors that mean "the run directory or config is not in a state for this";
# the CLI reports them in one line instead of a traceback.
_USER_ERRORS = (ConfigError, ProvenanceError, FileExistsError, FileNotFoundError)


def parse_shard(text: str) -> tuple[int, int]:
    """Parse ``--shard i/N`` into ``(i, N)``, with ``0 <= i < N``.

    Parameters
    ----------
    text
        The argument, e.g. ``"3/16"``.

    Returns
    -------
        The slice index and slice count.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not ``i/N`` with ``0 <= i < N``.
    """
    try:
        i, n = (int(part) for part in text.split("/"))
    except ValueError:
        msg = f"expected i/N (e.g. 3/16), got {text!r}"
        raise argparse.ArgumentTypeError(msg) from None
    if not 0 <= i < n:
        msg = f"need 0 <= i < N, got {text!r}"
        raise argparse.ArgumentTypeError(msg)
    return i, n


def _not_implemented(args: argparse.Namespace) -> int:
    sys.exit(f"hq {args.command}: not implemented yet")


def _prepare(args: argparse.Namespace) -> int:
    Run(args.run_dir).prepare(overwrite=args.overwrite)
    return 0


def _prior_cache(args: argparse.Namespace) -> int:
    Run(args.run_dir).make_prior_cache(overwrite=args.overwrite)
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.workers < 1:
        sys.exit("hq run: --workers must be at least 1")
    Run(args.run_dir).run_rejection(
        shard=args.shard, workers=args.workers, mpi=args.mpi, overwrite=args.overwrite
    )
    return 0


def _init(args: argparse.Namespace) -> int:
    run_dir = init_run(args.run_dir, kind=args.kind)
    print(f"Created {run_dir / 'hq.toml'} and {run_dir / 'prior.py'}")  # noqa: T201
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hq", description="Run harv over large samples (harv-hq)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(
        name: str, help_: str, handler: Callable[[argparse.Namespace], int]
    ) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(handler=handler)
        if name != "init":
            p.add_argument(
                "--run-dir",
                default=".",
                help="run directory holding hq.toml (default: current directory)",
            )
        return p

    p = add("init", "create a run directory from templates", _init)
    p.add_argument("run_dir", help="directory to create (must be new or empty)")
    p.add_argument("--kind", choices=KINDS, default="rv", help="run kind")

    p = add("prepare", "prepare per-source data from the input table", _prepare)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing prepared data (invalidates every later stage)",
    )

    p = add("prior-cache", "build the shared prior cache", _prior_cache)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild an existing cache (invalidates every result built on it)",
    )

    for name, help_, handler in (
        ("run", "run the rejection sampler on every source", _run),
        ("mcmc", "run MCMC follow-up on selected sources", _not_implemented),
    ):
        p = add(name, help_, handler)
        mode = p.add_mutually_exclusive_group()
        mode.add_argument(
            "--shard", type=parse_shard, default=(0, 1), help="process slice i of N"
        )
        mode.add_argument("--mpi", action="store_true", help="one slice per MPI rank")
        p.add_argument("--workers", type=int, default=1, help="local worker processes")
        p.add_argument(
            "--overwrite",
            action="store_true",
            help="move existing results to results/superseded-* and start over",
        )
        p.add_argument(
            "--no-compact",
            dest="compact",
            action="store_false",
            help="skip compaction at the end of a whole-stage run",
        )

    p = add("compact", "merge a stage's result parts", _not_implemented)
    p.add_argument("--stage", choices=("rejection", "mcmc"), default="rejection")

    add("summarize", "write summary.parquet", _not_implemented)
    add("status", "count sources by stage and status", _not_implemented)
    add("serve", "serve the web viewer", _not_implemented)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``hq`` command line.

    Parameters
    ----------
    argv
        Arguments, without the program name; ``sys.argv[1:]`` when ``None``.

    Returns
    -------
        The process exit status.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    try:
        return args.handler(args)
    except _USER_ERRORS as err:
        sys.exit(f"hq {args.command}: {err}")


if __name__ == "__main__":
    raise SystemExit(main())
