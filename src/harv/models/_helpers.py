"""Internal helpers shared across the models subpackage."""

__all__: tuple[str, ...] = ()

import types
from collections.abc import Callable
from typing import Any, cast

import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt import Q
from unxt.quantity import AllowValue, ustrip

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
    everything else (e.g., ``dist.HalfNormal``).
    """
    if isinstance(d, QuantityDistribution):
        return not isinstance(d.distribution, (dist.Normal, dist.Delta))
    if isinstance(d, (dist.Normal, dist.Delta)):
        return False
    # LinearPriorCallable (or any other callable) -> returns Normal, handled by
    # _resolve_prior_to_mvn.
    return not (callable(d) and not isinstance(d, dist.Distribution))


def _resolve_prior_to_mvn(
    prior_dict: dict[str, PriorDist | LinearPriorCallable],
    nl_values: dict[str, Any],
    unit_dict: dict[str, str],
    extra_values: dict[str, Any] | None = None,
) -> dist.MultivariateNormal:
    """Build diagonal MVN from per-parameter priors."""
    locs: list[Any] = []
    scales: list[Any] = []
    # Build a namespace proxy for any LinearPriorCallable that needs it.
    # Include explicit linear values so that callables depending on
    # explicitly-sampled linear params (e.g. parallax) can resolve.
    proxy_values = dict(nl_values)
    if extra_values:
        proxy_values.update(extra_values)
    params_proxy = types.SimpleNamespace(**proxy_values)
    for name, prior in prior_dict.items():
        target_u = unit_dict.get(name, "")
        resolved = None

        if isinstance(prior, (dist.Distribution, QuantityDistribution)):
            resolved = prior
        elif callable(prior):
            resolved = prior(params_proxy)

        expected_msg = (
            f"Expected Normal inside QuantityDistribution for {name}, "
            f"got {type(resolved)}"
        )
        if isinstance(resolved, QuantityDistribution):
            prior_unit = cast("str", resolved.unit)
            inner = resolved.distribution
            if not isinstance(inner, dist.Normal):
                raise TypeError(expected_msg)
            loc = ustrip(AllowValue, target_u, Q(inner.loc, prior_unit))
            scale = ustrip(AllowValue, target_u, Q(inner.scale, prior_unit))
        elif isinstance(resolved, dist.Normal):
            loc = resolved.loc
            scale = resolved.scale
        else:
            raise TypeError(expected_msg)
        locs.append(loc)
        scales.append(scale)

    return dist.MultivariateNormal(
        loc=jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in locs]),
        scale_tril=jnp.diag(jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in scales])),
    )
