"""Composite likelihood for combining heterogeneous data sources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx

if TYPE_CHECKING:
    import jax

    from harv.likelihood.base import AbstractLikelihood

__all__ = ["CompositeLikelihood"]


class CompositeLikelihood(eqx.Module):
    """Sum of heterogeneous likelihood components.

    Parameters with the same name across components are automatically shared:
    each component reads only the fields it needs from the combined params struct,
    so shared parameters (e.g. ``log_period``) are passed once and used by all.

    The required parameter names and their count are inferred from the union of
    each component's ``param_names``.

    Parameters
    ----------
    **components : AbstractLikelihood
        Named likelihood components. Names are arbitrary labels (e.g. ``rv``,
        ``astro``) used to identify each component.

    Examples
    --------
    >>> composite = CompositeLikelihood(
    ...     rv=RVLikelihood(data=rv_data, linear_prior=rv_prior),
    ...     astro=GaiaAstrometryLikelihood(data=gaia_data, linear_prior=astro_prior),
    ... )
    >>> composite.param_names
    ('log_period', 'eccentricity', 'phase_peri', 'arg_peri', 'cos_i', 'lon_asc_node')
    >>> composite.n_params
    6
    >>> log_lik = composite.log_prob(params)
    >>> log_liks = jax.jit(jax.vmap(composite.log_prob))(params_batch)
    """

    _components: dict[str, AbstractLikelihood]

    def __init__(self, **components: AbstractLikelihood) -> None:
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

    def __getitem__(self, key: str) -> AbstractLikelihood:
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
        return sum(  # type: ignore[return-value]
            comp.log_prob(params) for comp in self._components.values()
        )
