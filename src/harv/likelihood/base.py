"""Abstract base class for likelihood components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
    import jax


class AbstractLikelihood(eqx.Module):
    """Abstract base class for likelihood components.

    Subclasses store their data and priors as fields, and expose a ``log_prob``
    method that takes only the nonlinear parameters. This makes batching clean::

        batched = jax.jit(jax.vmap(likelihood.log_prob))
        log_liks = batched(params_batch)  # params_batch is a pytree

    """

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the nonlinear parameters this likelihood requires."""
        raise NotImplementedError  # pragma: no cover

    def log_prob(self, params: eqx.Module) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        raise NotImplementedError  # pragma: no cover
