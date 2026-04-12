"""Helper functions shared across the likelihood modules."""

__all__ = (
    "LinearPriorCallable",
    "_needs_explicit_sampling",
    "_solve_kepler",
    "_resolve_linear_prior_mvn",
)

from typing import Protocol, cast, runtime_checkable

import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt.quantity import AllowValue, Quantity, ustrip

from harv.data import AbstractData
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.likelihood.params import AbstractParameters, MarginalizedParameters
from harv.quantity_distribution import QuantityDistribution

type PriorDist = dist.Distribution | QuantityDistribution
type LinearPriorDist = (
    dict[str, PriorDist | LinearPriorCallable]
    # QuantityDistribution
    # | dist.MultivariateNormal
    # | LinearPriorCallable
    # | dict[str, PriorDist | LinearPriorCallable]
    # | dict[
    #     tuple[str, ...], PriorDist | LinearPriorCallable
    # ]  # TODO: we could also implement - e.g., ("rv_semiamp", "v_sys") -> MultivariateNormal
)


@runtime_checkable
class LinearPriorCallable(Protocol):
    """Returns a ``Normal`` given nonlinear params.

    Any ``eqx.Module`` (or other callable) whose ``__call__`` matches this signature
    satisfies the protocol.  The rejection sampler and likelihood classes accept either
    a fixed ``dist.Normal`` *or* a ``LinearPriorCallable`` for each ``linear_prior``
    field.
    """

    def __call__(
        self, params: AbstractParameters | MarginalizedParameters
    ) -> QuantityDistribution | dist.Normal:
        """Returns a Normal distribution given nonlinear parameters."""


def _unwrap_dist(v: PriorDist) -> dist.Distribution:
    """Extract the underlying numpyro distribution from a PriorDist."""
    if isinstance(v, QuantityDistribution):
        return v.distribution
    return v


def _needs_explicit_sampling(d: PriorDist | LinearPriorCallable) -> bool:
    """True if a linear prior entry must be sampled explicitly by the sampler.

    Returns ``False`` for entries the likelihood can handle analytically:
    ``dist.Normal``, ``dist.Delta``, their ``QuantityDistribution`` wrappers,
    and ``LinearPriorCallable`` (assumed to produce ``Normal``).
    Returns ``True`` for everything else (e.g. ``dist.HalfNormal``).
    """
    if isinstance(d, QuantityDistribution):
        return not isinstance(d.distribution, (dist.Normal, dist.Delta))
    if isinstance(d, (dist.Normal, dist.Delta)):
        return False
    # LinearPriorCallable (or any other callable) -> returns Normal, handled by
    # _resolve_linear_prior_mvn.
    if callable(d) and not isinstance(d, dist.Distribution):
        return False
    return True


def _resolve_linear_prior_mvn(
    linear_prior: LinearPriorDist,
    params: AbstractParameters | MarginalizedParameters,
    expected_units: dict[str, str],
) -> dist.MultivariateNormal:
    """Build a diagonal MVN from per-parameter priors, converting units as needed."""
    locs = []
    scales = []
    for name, prior in linear_prior.items():
        target_u = expected_units.get(name, "")
        resolved = None

        # 1. Resolve callables (param-dependent priors)
        if isinstance(prior, dist.Distribution | QuantityDistribution):
            resolved = prior
        elif callable(prior):
            resolved = prior(params)

        # 2. Unwrap QuantityDistribution -> bare Normal + unit conversion
        expected_msg = (
            f"Expected Normal inside QuantityDistribution for {name}, got "
            f"{type(resolved)}"
        )
        if isinstance(resolved, QuantityDistribution):
            prior_unit = cast("str", resolved.unit)
            inner = resolved.distribution
            if not isinstance(inner, dist.Normal):
                raise TypeError(expected_msg)
            loc = ustrip(AllowValue, target_u, Quantity(inner.loc, prior_unit))
            scale = ustrip(AllowValue, target_u, Quantity(inner.scale, prior_unit))
        elif isinstance(resolved, dist.Normal):
            loc = resolved.loc
            scale = resolved.scale
        else:
            raise TypeError(expected_msg)

        locs.append(loc)
        scales.append(scale)

    return dist.MultivariateNormal(
        loc=jnp.array(locs), scale_tril=jnp.diag(jnp.array(scales))
    )


def _solve_kepler(
    data: AbstractData,
    params: AbstractParameters | MarginalizedParameters,
) -> tuple[jax.Array, jax.Array]:  # TODO: improve type to be Float with a batch shape
    """Solve Kepler's equation; return (sin_f, cos_f)."""
    # phase_peri in [0, 1] is a dimensionless fractional phase;
    # t_peri = phase_peri * period gives the pericenter time relative to t=0.
    t_peri = params.phase_peri * params.period
    dt = data.time - t_peri
    M = mean_anomaly(dt, params.period)
    sinf, cosf = true_anomaly_from_mean(M, params.eccentricity)
    return (ustrip(AllowValue, "", sinf), ustrip(AllowValue, "", cosf))
