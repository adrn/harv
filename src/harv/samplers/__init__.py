"""Sampling infrastructure for Keplerian orbits.

This module provides the rejection sampling infrastructure, prior distributions,
the main RejectionSampler class, and the Samples container for posterior samples.
"""

from .base import AbstractSampler
from .conversion import convert_parameterization
from .numpyro import NumpyroSampler
from .prior_cache import make_prior_cache
from .rejection import RejectionSampler
from .samples import Samples, pad_and_stack_samples

# NOTE: ``QD`` / ``QuantityDistribution`` live in :mod:`harv.distributions` and
# ``HarvPrior`` / ``default_sb2_prior`` in :mod:`harv.models.priors`.  They are
# deliberately *not* re-exported here -- importing them from the sampler package
# blurs the layer boundary.  All four are reachable from the top level
# (``harv.QD``, ``harv.HarvPrior``, ``harv.models.default_sb2_prior``).
__all__ = (
    "AbstractSampler",
    "NumpyroSampler",
    "RejectionSampler",
    "Samples",
    "convert_parameterization",
    "make_prior_cache",
    "pad_and_stack_samples",
)
