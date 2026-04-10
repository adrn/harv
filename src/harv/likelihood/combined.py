"""Composite likelihood for combining heterogeneous data sources."""

from typing import Any

import equinox as eqx
import jax
import quaxed.numpy as jnp

from harv.likelihood.base import AbstractLikelihood

__all__ = ("CompositeLikelihood",)


class CompositeLikelihood(eqx.Module):
    """Sum of heterogeneous likelihood components.

    Each component receives its own parameter struct via a dict keyed by
    component name. Shared orbital parameters (e.g. period) are duplicated
    across structs by the caller.  This avoids name collisions and allows
    per-component marginalization decisions.

    Parameters
    ----------
    **components : AbstractLikelihood
        Named likelihood components. Names are arbitrary labels (e.g. ``rv``,
        ``astro``) used to identify each component.

    Examples
    --------
    Combining astrometry and RV likelihoods::

        import numpyro.distributions as dist
        import jax.numpy as jnp
        from harv.likelihood.combined import CompositeLikelihood
        from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
        from harv.likelihood.rv import RVLikelihood
        from harv.likelihood.params import (
            GaiaAstrometryParameters, RVParameters,
        )

        astro_prior = dist.MultivariateNormal(
            loc=jnp.zeros(6), covariance_matrix=jnp.eye(6) * 1000.0**2
        )
        rv_prior = dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * 100.0**2
        )

        composite = CompositeLikelihood(
            astro=GaiaAstrometryLikelihood(gaia_data, astro_prior),
            rv=RVLikelihood(rv_data, rv_prior),
        )

        # Build per-component params (orbital params shared by construction)
        astro_params = GaiaAstrometryParameters.marginalized(
            period=period, eccentricity=ecc, phase_peri=ph, arg_peri=w,
            cos_i=ci, lon_asc_node=lo,
        )
        rv_params = RVParameters.marginalized(
            period=period, eccentricity=ecc, phase_peri=ph, arg_peri=w,
        )
        log_lik = composite.log_prob({"astro": astro_params, "rv": rv_params})
    """

    components: dict[str, AbstractLikelihood[Any, Any]]

    def __init__(self, **components: AbstractLikelihood[Any, Any]) -> None:
        self.components = components

    def __getitem__(self, key: str) -> AbstractLikelihood[Any, Any]:
        return self.components[key]

    def __len__(self) -> int:
        return len(self.components)

    def keys(self) -> Any:
        """Return component names."""
        return self.components.keys()

    def values(self) -> Any:
        """Return likelihood components."""
        return self.components.values()

    def items(self) -> Any:
        """Return (name, component) pairs."""
        return self.components.items()

    def log_prob(self, params: dict[str, eqx.Module]) -> jax.Array:
        """Sum log-likelihoods from all components.

        Parameters
        ----------
        params : dict[str, eqx.Module]
            Per-component parameter structs, keyed by component name.
            Each component's ``log_prob`` is called with its own params.
        """
        log_probs = [
            comp.log_prob(params[name]) for name, comp in self.components.items()
        ]
        return jnp.sum(jnp.stack(log_probs))
