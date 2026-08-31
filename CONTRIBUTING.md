# Contributing

Contributions of all kinds are welcome here, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

## Example Contributions

You can contribute in many ways, for example:

- [Report bugs](#report-bugs)
- [Fix Bugs](#fix-bugs)
- [Implement Features](#implement-features)
- [Write Documentation](#write-documentation)
- [Submit Feedback](#submit-feedback)

### Report Bugs

Report bugs at https://github.com/adrn/harv/issues.

**If you are reporting a bug, please follow the template guidelines. The more
detailed your report, the easier and thus faster we can help you.**

### Fix Bugs

Look through the GitHub issues for bugs. Anything labelled with `bug` and
`help wanted` is open to whoever wants to implement it. When you decide to work on such
an issue, please assign yourself to it and add a comment that you'll be working on that,
too. If you see another issue without the `help wanted` label, just post a comment, the
maintainers are usually happy for any support that they can get.

### Implement Features

Look through the GitHub issues for features. Anything labelled with
`enhancement` and `help wanted` is open to whoever wants to implement it. As
for [fixing bugs](#fix-bugs), please assign yourself to the issue and add a comment that
you'll be working on that, too. If another enhancement catches your fancy, but it
doesn't have the `help wanted` label, just post a comment, the maintainers are usually
happy for any support that they can get.

### Write Documentation

harv could always use more documentation, whether as
part of the official documentation, in docstrings, or even on the web in blog
posts, articles, and such. Just
[open an issue](https://github.com/adrn/harv/issues)
to let us know what you will be working on so that we can provide you with guidance.

### Submit Feedback

The best way to send feedback is to file an issue at
https://github.com/adrn/harv/issues. If your feedback fits the format of one of
the issue templates, please use that. Remember that this is a volunteer-driven
project and everybody has limited time.

## Get Started!

Ready to contribute? Here's how to set up harv for
local development.

1. Fork the https://github.com/adrn/harv
   repository on GitHub.

1. Clone your fork locally (*if you want to work locally*)

   ```shell
   git clone git@github.com:your_name_here/harv.git
   ```

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), then
   create the development environment:

   ```shell
   uv sync --group dev
   ```

   That gives you the CPU build of jax. On a Linux machine with an NVIDIA GPU,
   this keeps a second GPU-enabled environment beside it for interactive work.
   Both resolve from the same `uv.lock`, so they cannot drift:

   ```shell
   UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --group dev --extra cuda13
   ```

   Use `cuda12` in place of `cuda13` if the machine's driver predates CUDA 13.
   For running the tests you do not need this — see the `nox` sessions below.

1. Create a branch for local development using the default branch (typically `main`) as a starting point. Use `fix` or `feat` as a prefix for your branch name.

   ```shell
   git checkout main
   git checkout -b fix-name-of-your-bugfix
   ```

   Now you can make your changes locally.

1. When you're done making changes, check that they pass our test suite:

   ```shell
   uv run pytest
   ```

   `nox` runs the same suite in its own throwaway environment, and is how you
   test against the GPU build without managing a second venv by hand:

   ```shell
   nox -s tests       # CPU
   nox -s tests_gpu   # CUDA 13; fails fast if jax cannot see a GPU
   ```

1. Commit your changes and push your branch to GitHub. Please use [semantic
   commit messages](https://www.conventionalcommits.org/).

   ```shell
   git add .
   git commit -m "fix: summarize your changes"
   git push -u origin fix-name-of-your-bugfix
   ```

1. Open the link displayed in the message when pushing your new branch in order
   to submit a pull request.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
1. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring.
1. Your pull request will automatically be checked by the full test suite.
   It needs to pass all of them before it can be considered for merging.
