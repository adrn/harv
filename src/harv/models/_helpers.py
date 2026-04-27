"""Internal helpers shared across the models subpackage."""

__all__: tuple[str, ...] = ()

from collections.abc import Callable

import numpyro.distributions as dist

from harv.distributions import QuantityDistribution

type PriorDist = dist.Distribution | QuantityDistribution
"""Union type for prior distributions (bare numpyro or unit-aware)."""

type LinearPriorCallable = Callable[..., QuantityDistribution | dist.Normal]
"""Callable that returns a Normal-like prior given nonlinear parameter values."""

type LinearPriorDist = dict[str, PriorDist | LinearPriorCallable]
"""Per-parameter linear prior dictionary."""


def _unwrap_dist(v: PriorDist) -> dist.Distribution:
    """Extract the underlying numpyro distribution from a PriorDist."""
    if isinstance(v, QuantityDistribution):
        return v.distribution
    return v


def _needs_explicit_sampling(d: PriorDist | LinearPriorCallable) -> bool:
    """True if a linear prior entry must be sampled explicitly by the sampler.

    Returns ``False`` for entries the likelihood can handle analytically:
    ``dist.Normal``, ``dist.Delta``, their ``QuantityDistribution`` wrappers, and
    ``LinearPriorCallable`` (assumed to produce ``Normal``). Returns ``True`` for
    everything else (e.g. ``dist.HalfNormal``).
    """
    if isinstance(d, QuantityDistribution):
        return not isinstance(d.distribution, (dist.Normal, dist.Delta))
    if isinstance(d, (dist.Normal, dist.Delta)):
        return False
    # LinearPriorCallable (or any other callable) -> returns Normal, handled by
    # _resolve_prior_to_mvn.
    return not (callable(d) and not isinstance(d, dist.Distribution))
