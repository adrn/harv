"""Unit-aware wrapper for numpyro distributions.

This module provides :class:`QuantityDistribution`, a thin wrapper that
pairs a numpyro distribution with a physical unit string so that samples
carry explicit units via ``unxt.Q``.

It also exports the :data:`PriorDist` type alias and the private
:func:`_unwrap_dist` helper used throughout ``harv.samplers``.
"""

from typing import Any

import equinox as eqx
import jax
import numpyro.distributions as dist
from unxt import Q, ustrip

__all__ = ("QD", "QuantityDistribution")


class QuantityDistribution(eqx.Module):
    """Pairs a numpyro distribution with the physical unit of its samples.

    For scalar distributions, ``unit`` is a single string.
    For multivariate distributions (e.g. ``MultivariateNormal`` over parameters
    with mixed units), ``unit`` is a tuple of strings -- one per dimension.

    Parameters
    ----------
    distribution : dist.Distribution
        The underlying numpyro distribution (works with bare floats).
    unit : str or tuple[str, ...]
        Physical unit of the samples.  A single string for scalar
        distributions; a tuple for multivariate distributions where each
        element may have a different unit.

    Examples
    --------
    Scalar (period in days):

    >>> import jax
    >>> import numpyro.distributions as dist
    >>> from harv import QuantityDistribution
    >>> qd = QuantityDistribution(dist.LogUniform(50., 2000.), "day")
    >>> sample = qd.sample(jax.random.key(0))
    >>> sample.unit
    Unit("d")

    Multivariate (astrometric linear parameters with mixed units)::

        qd = QuantityDistribution(
            dist.MultivariateNormal(loc=jnp.zeros(6), ...),
            ("mas", "mas", "mas/yr", "mas/yr", "mas", "mas"),
        )
        sample = qd.sample(key)  # -> raw jax.Array (consumer splits by name)
    """

    distribution: dist.Distribution
    unit: str | tuple[str, ...]

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> Any:
        """Sample from the distribution, attaching units when possible.

        For scalar units (``str``), returns ``Quantity``.
        For tuple units (multivariate with mixed units), returns a raw array --
        the consumer splits by parameter name and attaches per-element units.
        """
        raw = self.distribution.sample(key, sample_shape)
        if isinstance(self.unit, str):
            return Q(raw, self.unit)
        return raw

    def log_prob(self, value: Any) -> jax.Array:
        """Evaluate log-probability, stripping units if present."""
        if isinstance(self.unit, str) and isinstance(value, Q):
            return self.distribution.log_prob(ustrip(self.unit, value))
        return self.distribution.log_prob(value)


QD = QuantityDistribution
# Shorthand alias for :class:`QuantityDistribution`.
