"""Unit-aware wrapper for numpyro distributions.

This module provides :class:`QuantityDistribution`, a thin wrapper that
pairs a numpyro distribution with a physical unit string so that samples
carry explicit units via ``unxt.Q``.
"""

from collections.abc import Sequence
from typing import Any, final

import equinox as eqx
import jax
import numpyro.distributions as dist
from unxt import Q, ustrip

__all__ = ("QD", "JointPrior", "QuantityDistribution", "joint")


class QuantityDistribution(eqx.Module):
    """Pairs a numpyro distribution with the physical unit of its samples.

    For scalar distributions, ``unit`` is a single string.
    For multivariate distributions (e.g. ``MultivariateNormal`` over parameters
    with mixed units), ``unit`` is a tuple of strings -- one per dimension.

    Parameters
    ----------
    distribution
        The underlying numpyro distribution (works with bare floats).
    unit
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
        raw = self.distribution.sample(
            key,  # ty: ignore[invalid-argument-type]
            sample_shape,
        )
        if isinstance(self.unit, str):
            return Q(raw, self.unit)
        return raw

    def log_prob(self, value: Any) -> Any:
        """Evaluate log-probability, stripping units if present."""
        if isinstance(self.unit, str) and isinstance(value, Q):
            return self.distribution.log_prob(ustrip(self.unit, value))
        return self.distribution.log_prob(value)


QD = QuantityDistribution
# Shorthand alias for :class:`QuantityDistribution`.


def _names_tuple(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, str):  # a bare string would split into characters
        msg = f"JointPrior names must be a sequence of names, got the string {names!r}."
        raise TypeError(msg)
    return tuple(names)


@final
class JointPrior(eqx.Module):
    """One distribution shared by several nonlinear parameters.

    Build these with :func:`joint`, which places the same ``JointPrior`` under each
    of ``names`` in ``HarvPrior.nonlinear_priors``. The distribution is sampled once
    and its trailing axis split across ``names``; its ``log_prob`` is counted once.
    See ``docs/spec.md`` §Joint nonlinear priors.

    Examples
    --------
    >>> import jax
    >>> from harv.distributions import JointPrior
    >>> from harv.stats import UnitDisk
    >>> jp = JointPrior(("ecosw", "esinw"), UnitDisk())
    >>> jp.distribution.sample(jax.random.key(0), (5,)).shape
    (5, 2)

    Dimensioned names take a ``QD`` with one unit per name:

    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from harv.distributions import QD
    >>> d = dist.Uniform(jnp.array([1.0, 0.0]), jnp.array([100.0, 1.0])).to_event(1)
    >>> JointPrior(("period", "eccentricity"), QD(d, ("day", ""))).names
    ('period', 'eccentricity')
    """

    names: tuple[str, ...] = eqx.field(static=True, converter=_names_tuple)
    distribution: dist.Distribution | QuantityDistribution

    def __check_init__(self) -> None:
        if len(self.names) < 2 or len(set(self.names)) != len(self.names):
            msg = f"JointPrior needs at least two distinct names, got {self.names}."
            raise ValueError(msg)
        d = self.distribution
        if isinstance(d, QuantityDistribution):
            if not (
                isinstance(d.unit, tuple)
                and len(d.unit) == len(self.names)
                and all(isinstance(u, str) for u in d.unit)
            ):
                msg = (
                    f"A QD in a JointPrior over {self.names} needs a tuple of "
                    f"{len(self.names)} unit strings, one per name; got {d.unit!r}."
                )
                raise TypeError(msg)
            d = d.distribution
        if not isinstance(d, dist.Distribution):
            msg = (
                "JointPrior needs a numpyro distribution (optionally in a QD), got "
                f"{type(d).__name__}."
            )
            raise TypeError(msg)
        if tuple(d.batch_shape) != () or tuple(d.event_shape) != (len(self.names),):
            msg = (
                f"JointPrior over {self.names} needs batch_shape () and event_shape "
                f"({len(self.names)},), got {d.batch_shape} and {d.event_shape}."
            )
            raise ValueError(msg)


def joint(
    names: Sequence[str], distribution: dist.Distribution | QuantityDistribution
) -> dict[str, JointPrior]:
    """Return ``nonlinear_priors`` entries for a prior shared by ``names``.

    Examples
    --------
    >>> import harv
    >>> from harv.stats import UnitDisk
    >>> sorted(harv.joint(("ecosw", "esinw"), UnitDisk()))
    ['ecosw', 'esinw']
    """
    jp = JointPrior(names, distribution)
    return dict.fromkeys(jp.names, jp)
