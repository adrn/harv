"""Sybil configuration for collecting doctests from source modules."""

from sybil import Sybil
from sybil.parsers.rest import DocTestParser

pytest_collect_file = Sybil(
    parsers=[DocTestParser()],
    patterns=["harv/**/*.py"],
).pytest()
