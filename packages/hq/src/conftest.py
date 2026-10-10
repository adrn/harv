"""Sybil configuration for collecting doctests from harv_hq source modules."""

from sybil import Sybil
from sybil.parsers.rest import DocTestParser

pytest_collect_file = Sybil(
    parsers=[DocTestParser()],
    # "**/" needs at least one subdirectory, so top-level modules are listed too.
    patterns=["harv_hq/*.py", "harv_hq/**/*.py"],
).pytest()
