"""Helper functions shared across the likelihood modules."""

__all__ = ["_solve_kepler"]

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
    # ]  # TODO: we could also implement - e.g., ("K", "v0") -> MultivariateNormal
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
    ) -> dist.Normal: ...


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
        if isinstance(prior, (dist.Distribution, QuantityDistribution)):
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


# TODO: this shouldn't be here! why is it here? we should be able to import this from
# the kepler module
def _solve_kepler(
    data: AbstractData,
    params: AbstractParameters | MarginalizedParameters,
) -> tuple[jax.Array, jax.Array]:  # TODO: improve type to be Float with a batch shape
    """Solve Kepler's equation; return (sin_f, cos_f)."""
    t_peri = params.phase_peri * params.period
    dt = data.time - t_peri
    M = mean_anomaly(dt, params.period)
    sinf, cosf = true_anomaly_from_mean(M, params.eccentricity)
    return (ustrip(AllowValue, "", sinf), ustrip(AllowValue, "", cosf))
