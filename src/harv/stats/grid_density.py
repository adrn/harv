"""Grid-based density distribution over a positive variable (e.g. period).

This module implements :class:`LogGridDensity`, the numpyro distribution that
backs periodogram-informed interim period priors (see ``docs/spec.md``,
"Statistical utilities" and "Periodogram and interim period priors"). The pdf
is piecewise-linear in ``u = ln(x)`` on a fixed knot grid, with exact
(trapezoid) normalization and inverse-CDF sampling — all shape-static and safe
under ``jax.jit`` and ``jax.vmap``.
"""

__all__ = ("LogGridDensity",)

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key


@final
class LogGridDensity(Distribution):
    r"""Distribution over ``x > 0`` with a pdf piecewise-linear in ``ln(x)``.

    The density is defined by knots ``u_j = ln_grid[j]`` (strictly increasing)
    and unnormalized log-densities ``g_j = log_density[j]`` *with respect to
    the* ``d(ln x)`` *measure*. Between knots the (normalized) density
    ``rho(u)`` interpolates linearly; outside ``[u_0, u_{n-1}]`` the density is
    zero. Normalization uses the trapezoid rule, which is exact for a
    piecewise-linear pdf.

    ``log_prob(x)`` returns the log-density **per unit x** (matching the
    convention of ``numpyro.distributions.LogUniform``); use
    :meth:`log_prob_ln` for the log-density per unit ``ln x``, which is
    invariant under a change of the unit that ``x`` is measured in.

    Sampling is by inverse-CDF: the CDF is piecewise-quadratic in ``u`` and is
    inverted in closed form per segment. All operations use static shapes, so
    two instances with equal knot counts share a pytree structure (and hence a
    single JIT trace).

    Parameters
    ----------
    ln_grid
        Strictly increasing knots in ``ln(x / unit)``, shape ``(n,)`` with
        ``n >= 2``. The unit convention is the caller's responsibility (wrap
        the distribution in a `~harv.distributions.QuantityDistribution` to
        make it explicit).
    log_density
        Unnormalized log-density at each knot w.r.t. ``d(ln x)``, shape
        ``(n,)``.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from harv.stats import LogGridDensity
    >>> d = LogGridDensity(jnp.log(jnp.array([1.0, 10.0, 100.0])), jnp.zeros(3))
    >>> x = d.sample(jax.random.key(0), (4,))
    >>> bool(jnp.all((x >= 1.0) & (x <= 100.0)))
    True

    A flat ``log_density`` reproduces a log-uniform distribution:

    >>> import numpyro.distributions as dist
    >>> lu = dist.LogUniform(1.0, 100.0)
    >>> bool(jnp.allclose(d.log_prob(10.0), lu.log_prob(10.0)))
    True
    """

    # ``Distribution`` declares these as instance-level attributes, so they
    # cannot be narrowed to ``ClassVar`` here (that would violate LSP).
    arg_constraints: dict[str, Any] = {  # noqa: RUF012
        "ln_grid": constraints.real_vector,
        # NOT real_vector: that rejects every non-finite value, but -inf is a
        # documented, deliberately-produced input here (a zero-density knot --
        # see harv.periodogram.priors._to_prior). less_than(inf) admits -inf
        # while still rejecting +inf and NaN.
        "log_density": constraints.independent(constraints.less_than(jnp.inf), 1),
    }
    reparametrized_params: list[str] = []  # noqa: RUF012
    pytree_data_fields: tuple[str, ...] = (
        "ln_grid",
        "log_density",
        "_rho",
        "_cdf_knots",
        "_support",
    )

    def __init__(
        self,
        ln_grid: jax.Array,
        log_density: jax.Array,
        *,
        validate_args: bool | None = None,
    ) -> None:
        ln_grid = jnp.asarray(ln_grid)
        log_density = jnp.asarray(log_density)
        if ln_grid.ndim != 1 or ln_grid.shape != log_density.shape:
            raise ValueError(
                "ln_grid and log_density must be 1-d arrays of equal shape; "
                f"got {ln_grid.shape} and {log_density.shape}"
            )
        if ln_grid.shape[0] < 2:
            raise ValueError("ln_grid must have at least 2 knots")
        ln_grid = eqx.error_if(
            ln_grid,
            jnp.any(jnp.diff(ln_grid) <= 0),
            "ln_grid must be strictly increasing",
        )
        self.ln_grid = ln_grid
        self.log_density = log_density

        # Normalized knot densities w.r.t. d(ln x) and CDF at the knots.
        du = jnp.diff(ln_grid)
        rho_tilde = jnp.exp(log_density - jnp.max(log_density))
        segment_mass = 0.5 * (rho_tilde[:-1] + rho_tilde[1:]) * du
        norm = jnp.sum(segment_mass)
        norm = eqx.error_if(
            norm, ~(norm > 0), "log_density must have positive total mass"
        )
        self._rho = rho_tilde / norm
        cdf_knots = jnp.concatenate([jnp.zeros(1), jnp.cumsum(segment_mass) / norm])
        self._cdf_knots = cdf_knots.at[-1].set(1.0)
        self._support = constraints.interval(jnp.exp(ln_grid[0]), jnp.exp(ln_grid[-1]))
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        """Interval constraint ``[exp(ln_grid[0]), exp(ln_grid[-1])]``."""
        return self._support

    @property
    def low(self) -> jax.Array:
        """Lower edge of the support, ``exp(ln_grid[0])``."""
        return jnp.exp(self.ln_grid[0])

    @property
    def high(self) -> jax.Array:
        """Upper edge of the support, ``exp(ln_grid[-1])``."""
        return jnp.exp(self.ln_grid[-1])

    def _segment(self, u: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Locate the knot segment containing ``u``: ``(j, u_j, du_j)``."""
        n = self.ln_grid.shape[0]
        j = jnp.clip(jnp.searchsorted(self.ln_grid, u, side="right") - 1, 0, n - 2)
        u0 = self.ln_grid[j]
        du = self.ln_grid[j + 1] - u0
        return j, u0, du

    def _log_rho_ln(self, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Log-density per unit ``ln x`` at ``value`` (masked to ``-inf`` outside)."""
        value = jnp.asarray(value)
        positive = value > 0
        u = jnp.log(jnp.where(positive, value, 1.0))
        j, u0, du = self._segment(u)
        t = (u - u0) / du
        rho = self._rho[j] + (self._rho[j + 1] - self._rho[j]) * t
        inside = positive & (u >= self.ln_grid[0]) & (u <= self.ln_grid[-1]) & (rho > 0)
        log_rho = jnp.where(inside, jnp.log(jnp.where(rho > 0, rho, 1.0)), -jnp.inf)
        return log_rho, u

    @validate_sample
    def log_prob(self, value: ArrayLike) -> jax.Array:
        """Log-density per unit ``x`` (``-inf`` outside the support)."""
        log_rho, u = self._log_rho_ln(jnp.asarray(value))
        return log_rho - u

    def log_prob_ln(self, value: ArrayLike) -> jax.Array:
        """Log-density per unit ``ln x`` — unit-of-``x`` independent.

        Equals ``log_prob(value) + ln(value)`` inside the support and ``-inf``
        outside. This is the natural quantity for interim-prior bookkeeping in
        hierarchical reweighting (see ``docs/spec.md``).
        """
        log_rho, _ = self._log_rho_ln(jnp.asarray(value))
        return log_rho

    def cdf(self, value: ArrayLike) -> jax.Array:
        """Cumulative distribution function (piecewise-quadratic in ``ln x``)."""
        value = jnp.asarray(value)
        positive = value > 0
        u = jnp.log(jnp.where(positive, value, 1.0))
        j, u0, du = self._segment(u)
        a = self._rho[j]
        b = (self._rho[j + 1] - a) / du
        t = jnp.clip(u - u0, 0.0, du)
        out = jnp.clip(self._cdf_knots[j] + a * t + 0.5 * b * t * t, 0.0, 1.0)
        below = ~positive | (u < self.ln_grid[0])
        above = positive & (u > self.ln_grid[-1])
        return jnp.where(below, 0.0, jnp.where(above, 1.0, out))

    def icdf(self, q: ArrayLike) -> jax.Array:
        """Inverse CDF, in closed form per knot segment.

        Solves ``(b/2) t^2 + a t = r`` on the located segment using the
        "citardauq" form ``t = 2r / (a + sqrt(a^2 + 2 b r))``, which is stable
        as the density slope ``b -> 0``.
        """
        q = jnp.asarray(q)
        n = self.ln_grid.shape[0]
        j = jnp.clip(jnp.searchsorted(self._cdf_knots, q, side="right") - 1, 0, n - 2)
        u0 = self.ln_grid[j]
        du = self.ln_grid[j + 1] - u0
        a = self._rho[j]
        b = (self._rho[j + 1] - a) / du
        r = jnp.clip(q - self._cdf_knots[j], 0.0, None)
        disc = jnp.sqrt(jnp.maximum(a * a + 2.0 * b * r, 0.0))
        denom = a + disc
        t = jnp.where(denom > 0, 2.0 * r / jnp.where(denom > 0, denom, 1.0), 0.0)
        return jnp.exp(u0 + jnp.clip(t, 0.0, du))

    def sample(  # ty: ignore[invalid-method-override]
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """Draw samples by inverse-CDF transform of uniform variates.

        ``key`` is annotated ``jax.Array`` rather than ``Distribution.sample``'s
        ``jax.dtypes.prng_key | None``: the latter is a dtype class, not the
        runtime type of a PRNG key, and beartype rejects real keys against it.
        """
        if not is_prng_key(key):
            raise TypeError("key must be a JAX PRNG key")
        q = jax.random.uniform(key, shape=sample_shape + self.batch_shape)
        return self.icdf(q)

    @property
    def mean(self) -> jax.Array:
        """Closed-form mean, ``sum_j int_{u_j}^{u_{j+1}} e^u rho(u) du``."""
        u = self.ln_grid
        du = jnp.diff(u)
        a = self._rho[:-1]
        b = (self._rho[1:] - a) / du
        seg = jnp.exp(u[:-1]) * (jnp.exp(du) * (self._rho[1:] - b) - (a - b))
        return jnp.sum(seg)
