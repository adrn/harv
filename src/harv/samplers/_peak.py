"""Likelihood-only peak refinement.

The rejection sampler normalizes its acceptance probabilities by the maximum
marginal log-likelihood **over the drawn library**, so that maximum is a
property of the draw rather than of the posterior. This module finds the nearby
maximum of the marginal log-likelihood itself, which answers two questions the
library maximum cannot:

- how far short of the peak the library fell (``max lnL`` is only meaningful
  once it has converged; see ``docs/spec.md``, "Interpreting acceptance"), and
- what a run-independent acceptance threshold would be.

**The objective is the marginal log-likelihood alone, not the log-posterior.**
:meth:`~harv.samplers.NumpyroSampler.optimize` maximizes
``log_prior + marginal_log_likelihood``, which is the right target for seeding
MCMC but the wrong one here: its likelihood value has no ordering relation to
the library maximum, so under an informative prior — exactly what a
periodogram-informed period prior is — it can land *below* it and report a
negative shortfall.

Optimization runs in the unconstrained reparametrization given by each prior's
``biject_to(support)``, so period stays positive, ``eccentricity`` stays in
``[0, 1)`` and ``cos_i`` in ``[-1, 1]`` without the solver needing bounds.
"""

__all__ = ("maximize_log_likelihood",)

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from numpyro.distributions.transforms import biject_to

from harv.distributions import QuantityDistribution


def _bijector(prior_dist: Any) -> Any:
    """Unconstrained -> constrained transform for one parameter's prior.

    ``QuantityDistribution`` wraps a numpyro distribution whose support is
    expressed in unit-stripped values, which is the space this module works in.
    """
    d = (
        prior_dist.distribution
        if isinstance(prior_dist, QuantityDistribution)
        else prior_dist
    )
    return biject_to(d.support)


def maximize_log_likelihood(
    log_likelihood_fn: Callable[[dict[str, Any]], jax.Array],
    starts: dict[str, jax.Array],
    nonlinear_priors: dict[str, Any],
    *,
    max_passes: int = 8,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Maximize the marginal log-likelihood by BFGS from several starting points.

    Parameters
    ----------
    log_likelihood_fn
        Maps a dict of unit-stripped scalar parameter values to the marginal
        log-likelihood. The caller builds this over its own model, data,
        effective linear priors and marginalized names, which keeps this module
        free of any dependency on the sampler modules.
    starts
        Starting points, one array of leading length ``N`` per parameter. Keys
        absent from *nonlinear_priors* (e.g. explicitly sampled non-Gaussian
        linear parameters) are held fixed at their starting values rather than
        optimized, and are passed through to ``log_likelihood_fn`` unchanged.
    nonlinear_priors
        Prior per parameter to be optimized, used only for its support. Pass the
        base nonlinear priors plus any nonlinear extension priors (jitter).
    max_passes
        BFGS restarts per starting point. Restarts matter because
        ``jax.scipy.optimize.minimize``'s line search often aborts early;
        restarting from the previous result usually makes further progress. The
        best pass is kept, so extra passes can never make the answer worse.

    Returns
    -------
        ``(ln_likelihood, params)`` at the best point found over all starting
        points, with *params* unit-stripped and in the constrained space.

    Raises
    ------
    ValueError
        If no parameter in *starts* has a prior in *nonlinear_priors*.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from harv.samplers._peak import maximize_log_likelihood
    >>> # A quadratic in x, maximized at x = 2, with x constrained positive.
    >>> lnl, params = maximize_log_likelihood(
    ...     lambda v: -((v["x"] - 2.0) ** 2),
    ...     {"x": jnp.array([0.5, 5.0])},
    ...     {"x": dist.LogUniform(0.01, 100.0)},
    ... )
    >>> bool(jnp.isclose(params["x"], 2.0, atol=1e-3)), bool(lnl > -1e-6)
    (True, True)
    """
    names = tuple(n for n in sorted(starts) if n in nonlinear_priors)
    if not names:
        raise ValueError(
            "No parameter to optimize: none of the starting-point keys "
            f"{sorted(starts)} has a prior in nonlinear_priors "
            f"{sorted(nonlinear_priors)}."
        )
    bij = {n: _bijector(nonlinear_priors[n]) for n in names}
    fixed = {k: jnp.asarray(v) for k, v in starts.items() if k not in names}

    def to_constrained(u: jax.Array) -> dict[str, Any]:
        return {n: bij[n](u[i]) for i, n in enumerate(names)}

    # (N, d) in the unconstrained space.
    u0 = jnp.stack(
        [bij[n].inv(jnp.asarray(starts[n])) for n in names],
        axis=-1,
    )

    def one_start(
        u_row: jax.Array, fixed_row: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        def neg(u: jax.Array) -> jax.Array:
            values = to_constrained(u)
            values.update(fixed_row)
            return -log_likelihood_fn(values)

        def body(
            _: int, carry: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            u, best_f, best_u = carry
            res = jax.scipy.optimize.minimize(neg, u, method="BFGS")
            # Keep the best pass rather than the last: a restart that diverges
            # must not lose ground already made.
            better = res.fun < best_f
            return (
                res.x,
                jnp.where(better, res.fun, best_f),
                jnp.where(better, res.x, best_u),
            )

        f0 = neg(u_row)
        _, best_f, best_u = jax.lax.fori_loop(0, max_passes, body, (u_row, f0, u_row))
        return best_f, best_u

    best_f, best_u = jax.vmap(one_start)(u0, fixed)

    # A diverged start yields NaN; never let it win the argmin.
    ranked = jnp.where(jnp.isnan(best_f), jnp.inf, best_f)
    i = jnp.argmin(ranked)
    params = to_constrained(best_u[i])
    params.update({k: v[i] for k, v in fixed.items()})
    return -best_f[i], params
