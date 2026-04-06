"""Abstract base class for likelihood components."""

import equinox as eqx
import jax


class AbstractLikelihood[ParamT: eqx.Module](eqx.Module):
    """Abstract base class for likelihood components.

    Generic over the parameter struct type ``_ParamT``. Subclasses declare
    their expected parameter type explicitly::

        class MarginalizedRVLikelihood(AbstractLikelihood[RVMarginalizedParameters]):
            ...

    This makes batching clean::

        batched = jax.jit(jax.vmap(likelihood.log_prob))
        log_liks = batched(params_batch)  # params_batch is a pytree

    """

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the nonlinear parameters this likelihood requires."""
        raise NotImplementedError  # pragma: no cover

    def log_prob(self, params: ParamT) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        raise NotImplementedError  # pragma: no cover
