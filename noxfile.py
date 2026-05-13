"""Nox setup."""

import argparse

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"


@session(uv_groups=["docs"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    s.notify("build_api_docs")
    s.notify("sphinx_build")


@session(uv_groups=["docs"], reuse_venv=True)
def sphinx_build(s: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b",
        dest="builder",
        default="html",
        help="Build target (default: html)",
    )
    parser.add_argument("--output-dir", dest="output_dir", default="_build")
    args, posargs = parser.parse_known_args(s.posargs)

    if args.builder != "html" and args.serve:
        s.error("Must not specify non-HTML builder with --serve")

    s.chdir("docs")

    if args.builder == "linkcheck":
        s.run(
            "sphinx-build",
            "-b",
            "linkcheck",
            ".",
            "_build/linkcheck",
            *posargs,
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        f"-d={args.output_dir}/doctrees",
        "-D",
        "language=en",
        ".",
        f"{args.output_dir}/{args.builder}",
        *posargs,
    )

    if args.serve:
        s.run("sphinx-autobuild", *shared_args)
    else:
        s.run("sphinx-build", "--keep-going", *shared_args)


@session(uv_groups=["docs"], reuse_venv=True)
def build_api_docs(s: nox.Session, /) -> None:
    """Build (regenerate) API docs."""
    s.chdir("docs")
    s.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--module-first",
        "--no-toc",
        "--force",
        "../src/harv",
    )
