"""Rejection sampling for Keplerian orbits.

This module provides the rejection sampling infrastructure including design matrices,
the main RejectionSampler class, and the Samples container for posterior samples.
"""

from .rejection import RejectionSampler
from .samples import Samples

__all__ = [
    "RejectionSampler",
    "Samples",
]
