"""Composite likelihood for combining heterogeneous data sources."""

from typing import Any

import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import AbstractQuantity

from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.params import AbstractParameters, MarginalizedParameters

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
        from harv.likelihood.combined import CompositeLikelihood
        from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
        from harv.likelihood.rv import RVLikelihood
        from harv.likelihood.params import (
            GaiaAstrometryParameters, RVParameters,
        )
        from harv.quantity_distribution import QuantityDistribution

        composite = CompositeLikelihood(
            astro=GaiaAstrometryLikelihood(
                data=gaia_data,
                linear_marginalized_prior={
                    "ra0": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
                    "dec0": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
                    "pmra": QuantityDistribution(dist.Normal(0., 1e3), "mas/yr"),
                    "pmdec": QuantityDistribution(dist.Normal(0., 1e3), "mas/yr"),
                    "parallax": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
                    "semi_major_axis": QuantityDistribution(
                        dist.Normal(0., 1e3), "mas"
                    ),
                },
            ),
            rv=RVLikelihood(
                data=rv_data,
                linear_marginalized_prior={
                    "K": QuantityDistribution(dist.Normal(0., 100.), "km/s"),
                    "v0": QuantityDistribution(dist.Normal(0., 100.), "km/s"),
                },
            ),
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

    def log_prob(
        self,
        params: dict[str, AbstractParameters | MarginalizedParameters],
        offsets: dict[str, dict[str, AbstractQuantity]] | None = None,
    ) -> jax.Array:
        """Sum log-likelihoods from all components.

        Parameters
        ----------
        params : dict[str, eqx.Module]
            Per-component parameter structs, keyed by component name.
            Each component's ``log_prob`` is called with its own params.
        offsets : dict[str, dict[str, AbstractQuantity]] or None
            Per-component, per-instrument offsets.  Outer keys are component
            names matching ``params``; inner dicts are passed to the
            corresponding component's ``log_prob``.  ``None`` (or missing
            keys) means no offsets for that component.
        """
        _offsets = offsets or {}
        log_probs = [
            comp.log_prob(params[name], _offsets.get(name))
            for name, comp in self.components.items()
        ]
        return jnp.sum(jnp.stack(log_probs))
