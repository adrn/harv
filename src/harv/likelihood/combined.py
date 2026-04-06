"""Composite likelihood for combining heterogeneous data sources."""

from typing import Any

import equinox as eqx
import jax
import quaxed.numpy as jnp

from harv.likelihood.base import AbstractLikelihood

__all__ = ("CompositeLikelihood",)


class CompositeLikelihood(eqx.Module):
    """Sum of heterogeneous likelihood components.

    Parameters with the same name across components are automatically shared
    via duck typing: ``log_prob`` passes the same params struct to every
    component, and each component reads only the fields it needs.  There is
    no explicit sharing mechanism — a struct that carries the union of all
    required fields satisfies every component simultaneously.

    For combined astrometry + RV data the natural params struct is
    ``GaiaAstrometryMarginalizedParameters`` (6 nonlinear params).
    ``MarginalizedRVLikelihood`` only reads ``period``, ``eccentricity``,
    ``phase_peri``, and ``arg_peri`` from it; the extra ``cos_i`` and
    ``lon_asc_node`` fields are silently ignored.  This means ``period`` (and
    the other shared orbital elements) is passed once and consumed by both
    components without any extra wiring.

    The required parameter names and their count are inferred from the union of
    each component's ``param_names``.

    Parameters
    ----------
    **components : AbstractLikelihood
        Named likelihood components. Names are arbitrary labels (e.g. ``rv``,
        ``astro``) used to identify each component.

    Examples
    --------
    Combining marginalized astrometry and RV likelihoods.  Both components
    share ``period``, ``eccentricity``, ``phase_peri``, and ``arg_peri``
    automatically — pass a ``GaiaAstrometryMarginalizedParameters`` and each component
    reads what it needs::

        import numpyro.distributions as dist
        import jax.numpy as jnp
        from harv.likelihood.combined import CompositeLikelihood
        from harv.likelihood.gaia_astrometry import MarginalizedGaiaAstrometryLikelihood
        from harv.likelihood.rv import MarginalizedRVLikelihood
        from harv.likelihood._params import GaiaAstrometryMarginalizedParameters

        astro_prior = dist.MultivariateNormal(
            loc=jnp.zeros(6), covariance_matrix=jnp.eye(6) * 1000.0**2
        )
        rv_prior = dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * 100.0**2
        )

        composite = CompositeLikelihood(
            astro=MarginalizedGaiaAstrometryLikelihood(gaia_data, astro_prior),
            rv=MarginalizedRVLikelihood(rv_data, rv_prior),
        )

        # Union of both param_names: 6 unique orbital parameters
        composite.param_names
        # ('period', 'eccentricity', 'phase_peri', 'cos_i', 'arg_peri', 'lon_asc_node')
        composite.n_params
        # 6

        # Evaluate at a single point — GaiaAstrometryMarginalizedParameters satisfies both
        # components via duck typing
        log_lik = composite.log_prob(params)  # params: GaiaAstrometryMarginalizedParameters

        # Batch evaluation over prior samples
        log_liks = jax.jit(jax.vmap(composite.log_prob))(params_batch)

    A pure-RV composite (e.g. two instruments, one marginalized each)::

        composite_rv = CompositeLikelihood(
            keck=MarginalizedRVLikelihood(keck_data, rv_prior),
            espresso=MarginalizedRVLikelihood(espresso_data, rv_prior),
        )
        # period and other orbital params shared across both instruments
        composite_rv.param_names
        # ('period', 'eccentricity', 'phase_peri', 'arg_peri')
    """

    _components: dict[str, AbstractLikelihood[Any]]

    def __init__(self, **components: AbstractLikelihood[Any]) -> None:
        self._components = components

    @property
    def param_names(self) -> tuple[str, ...]:
        """Union of component param_names, preserving first-seen order."""
        seen: dict[str, None] = {}
        for comp in self._components.values():
            for name in comp.param_names:
                seen[name] = None
        return tuple(seen)

    @property
    def n_params(self) -> int:
        """Total number of unique nonlinear parameters."""
        return len(self.param_names)

    def __getitem__(self, key: str) -> AbstractLikelihood[Any]:
        return self._components[key]

    def __len__(self) -> int:
        return len(self._components)

    def keys(self) -> Any:
        """Return component names."""
        return self._components.keys()

    def values(self) -> Any:
        """Return likelihood components."""
        return self._components.values()

    def items(self) -> Any:
        """Return (name, component) pairs."""
        return self._components.items()

    def log_prob(self, params: eqx.Module) -> jax.Array:
        """Sum log-likelihoods from all components.

        Each component reads only the fields it needs from ``params``. The
        params struct must have at least the fields listed in ``param_names``.
        """
        log_probs = [comp.log_prob(params) for comp in self._components.values()]
        return jnp.sum(jnp.stack(log_probs))
