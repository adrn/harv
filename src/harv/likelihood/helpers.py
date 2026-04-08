"""Helper functions shared across the likelihood modules."""

__all__ = ["_solve_kepler"]

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from unxt.quantity import AllowValue, ustrip

from harv.data import AbstractData
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.likelihood.params import AbstractParameters, MarginalizedParameters

# ---------------------------------------------------------------------------
# Linear prior Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LinearPriorCallable(Protocol):
    """Callable that returns a ``MultivariateNormal`` given nonlinear params.

    Any ``eqx.Module`` (or other callable) whose ``__call__`` matches this
    signature satisfies the protocol.  The rejection sampler and likelihood
    classes accept either a fixed ``dist.MultivariateNormal`` *or* a
    ``LinearPriorCallable`` for the ``linear_prior`` field.
    """

    def __call__(self, params: Any) -> dist.MultivariateNormal: ...


# ---------------------------------------------------------------------------
# Sub-distribution extraction
# ---------------------------------------------------------------------------


def _sub_mvn(
    mvn: dist.MultivariateNormal,
    indices: tuple[int, ...],
) -> dist.MultivariateNormal:
    """Extract a sub-block from a block-diagonal ``MultivariateNormal``.

    Selects the given rows/columns of ``loc`` and ``scale_tril`` via fancy
    indexing.  This is correct when the joint distribution is block-diagonal
    with respect to the selected indices (i.e. the selected parameters are
    independent from the unselected ones).

    Parameters
    ----------
    mvn :
        Joint multivariate normal distribution.
    indices :
        Parameter indices to retain.  Static at JAX trace time.

    Returns
    -------
    dist.MultivariateNormal
        Sub-distribution over the selected parameters.
    """
    idx = jnp.array(indices)
    return dist.MultivariateNormal(
        loc=mvn.loc[idx],
        scale_tril=mvn.scale_tril[jnp.ix_(idx, idx)],
    )


# ---------------------------------------------------------------------------
# Indexed callable & resolve helper
# ---------------------------------------------------------------------------


class _IndexedCallable(eqx.Module):
    """Wraps a joint callable prior, selecting a subset of parameters by index.

    Used when a single callable prior covers multiple likelihood components
    (e.g. combined astrometry + RV) and each component needs its own sub-prior.
    The ``indices`` tuple may be non-contiguous, so the covariance sub-block is
    extracted via fancy indexing rather than slicing.

    Parameters
    ----------
    wrapped : LinearPriorCallable
        Callable returning a ``dist.MultivariateNormal`` given orbit params.
    indices : tuple[int, ...]
        Indices of the linear parameters for this component within the full
        joint prior. Static so JAX can trace through without recompilation.
    """

    wrapped: LinearPriorCallable
    indices: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, params: Any) -> dist.MultivariateNormal:
        full = self.wrapped(params)
        return _sub_mvn(full, self.indices)


def _resolve_linear_prior(
    linear_prior: dist.MultivariateNormal | LinearPriorCallable,
    params: Any,
) -> dist.MultivariateNormal:
    """Resolve a fixed or callable linear prior.

    If *linear_prior* is already a ``dist.MultivariateNormal`` it is returned
    unchanged. If it is a callable ``LinearPriorCallable`` it is called with
    *params* and must return a ``dist.MultivariateNormal``.

    This check is a static Python isinstance test; it is evaluated at JAX
    trace time (not at runtime of the compiled function) so it is safe inside
    ``jax.vmap`` and ``jax.jit``.
    """
    if isinstance(linear_prior, dist.MultivariateNormal):
        return linear_prior
    return linear_prior(params)


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
