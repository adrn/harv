"""The Kipping (2013) eccentricity prior on ``(sqrt(e) cos w, sqrt(e) sin w)``.

See ``docs/spec.md`` ("Statistical utilities" -> ``KippingDisk``).
"""

__all__ = ("KIPPING_2013_ECC", "KippingDisk")

from typing import final

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.util import validate_sample

from harv.stats.numpyro_ext import unit_disk

# The Kipping (2013) eccentricity prior, shared with StandardRV's default.
KIPPING_2013_ECC = dist.Beta(0.867, 3.03)


@final
class KippingDisk(dist.Distribution):
    """Kipping (2013) eccentricity prior with uniform ``omega``, on the unit disk.

    The coordinates are ``(sqrt(e) cos(omega), sqrt(e) sin(omega))``, with
    ``e ~ Beta(0.867, 3.03)`` and ``omega ~ Uniform(-pi, pi)``. The area element
    there is ``de domega / 2``, so the density is ``p(e) / pi`` with
    ``e = x**2 + y**2``: only a mild cusp at the origin, which NUTS samples
    correctly. It shares ``UnitDisk``'s support, so NUTS reuses its bijector.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from harv.stats import KippingDisk
    >>> x = KippingDisk().sample(jax.random.key(0), (1000,))
    >>> x.shape
    (1000, 2)
    >>> bool(jnp.all(jnp.sum(x**2, axis=-1) < 1))
    True
    >>> bool(jnp.isneginf(KippingDisk().log_prob(jnp.array([0.8, 0.8]))))
    True
    """

    support = unit_disk

    def __init__(self, *, validate_args: bool | None = None) -> None:
        super().__init__(batch_shape=(), event_shape=(2,), validate_args=validate_args)

    def sample(  # ty: ignore[invalid-method-override]
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Draw ``e`` from the Beta and ``omega`` uniformly, then project."""
        key_e, key_w = jax.random.split(key)
        sqrt_ecc = jnp.sqrt(KIPPING_2013_ECC.sample(key_e, sample_shape))
        omega = jax.random.uniform(key_w, sample_shape, minval=-jnp.pi, maxval=jnp.pi)
        return jnp.stack([sqrt_ecc * jnp.cos(omega), sqrt_ecc * jnp.sin(omega)], -1)

    @validate_sample
    def log_prob(self, value: ArrayLike) -> jax.Array:
        """``Beta.log_prob(x**2 + y**2) - log(pi)`` inside the disk, else ``-inf``."""
        ecc = jnp.sum(jnp.asarray(value) ** 2, axis=-1)
        # Clamp so the origin gives a finite value (and gradient) instead of +inf.
        eps = jnp.finfo(ecc.dtype)
        ecc_c = jnp.clip(ecc, eps.tiny, 1.0 - eps.eps)
        lp = KIPPING_2013_ECC.log_prob(ecc_c) - jnp.log(jnp.pi)
        return jnp.where(ecc < 1.0, lp, -jnp.inf)
