"""Helper functions shared across the likelihood modules."""

from __future__ import annotations

__all__ = ["_solve_kepler"]

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxoplanet.core.kepler import kepler
from unxt import ustrip

if TYPE_CHECKING:
    from harv.data import AbstractData
    from harv.likelihood._params import AbstractBaseKeplerParameters


class _IndexedCallable(eqx.Module):
    """Wraps a joint callable prior, selecting a subset of parameters by index.

    Used when a single callable prior covers multiple likelihood components
    (e.g. combined astrometry + RV) and each component needs its own sub-prior.
    The ``indices`` tuple may be non-contiguous, so the covariance sub-block is
    extracted via fancy indexing rather than slicing.

    Parameters
    ----------
    wrapped : eqx.Module
        Callable returning a ``dist.MultivariateNormal`` given orbit params.
    indices : tuple[int, ...]
        Indices of the linear parameters for this component within the full
        joint prior. Static so JAX can trace through without recompilation.
    """

    wrapped: eqx.Module
    indices: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, params: Any) -> dist.MultivariateNormal:
        full = self.wrapped(params)  # type: ignore[operator]
        idx = jnp.array(self.indices)
        return dist.MultivariateNormal(
            loc=full.loc[idx],
            scale_tril=full.scale_tril[jnp.ix_(idx, idx)],
        )


def _resolve_linear_prior(
    linear_prior: dist.MultivariateNormal | eqx.Module,
    params: Any,
) -> dist.MultivariateNormal:
    """Resolve a fixed or callable linear prior.

    If *linear_prior* is already a ``dist.MultivariateNormal`` it is returned
    unchanged. If it is a callable ``eqx.Module`` it is called with *params*
    and must return a ``dist.MultivariateNormal``.

    This check is a static Python isinstance test; it is evaluated at JAX
    trace time (not at runtime of the compiled function) so it is safe inside
    ``jax.vmap`` and ``jax.jit``.
    """
    if isinstance(linear_prior, dist.MultivariateNormal):
        return linear_prior
    return linear_prior(params)  # type: ignore[operator]


def _solve_kepler(
    data: AbstractData,
    params: AbstractBaseKeplerParameters,
) -> tuple[jax.Array, jax.Array]:
    """Solve Kepler's equation; return (sin_f, cos_f)."""
    t_peri = params.phase_peri * params.period
    dt = data.time - t_peri
    M = 2 * jnp.pi * ustrip("", dt / params.period)
    return kepler(M, params.eccentricity)
