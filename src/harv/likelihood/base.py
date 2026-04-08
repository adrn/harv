"""Abstract base class for likelihood components."""

import equinox as eqx
import jax


class AbstractLikelihood[ParamT: eqx.Module](eqx.Module):
    """Abstract base class for likelihood components.

    Generic over the parameter struct type ``ParamT``. Subclasses declare
    their expected parameter type explicitly, for example::

        class RVLikelihood(AbstractLikelihood[MarginalizedParameters | RVParameters]):
            ...

    """

    def log_prob(self, params: ParamT) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        raise NotImplementedError  # pragma: no cover
