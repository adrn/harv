r"""Support-constrained and mixture priors for analytic linear marginalization.

:class:`~harv.stats.MarginalizedLinear` (vendored, see
:mod:`harv.stats.numpyro_ext`) integrates out linear parameters under a
*Gaussian* prior. This module widens that to two further prior families while
leaving the vendored code untouched, by composing around it:

- a **support-constrained** (truncated) Gaussian prior, e.g. ``parallax > 0`` or
  the signed SB2 semi-amplitudes ``K_1 > 0``, ``K_2 < 0``;
- a **mixture** of Gaussians sharing one support.

See ``docs/spec.md`` §Support-constrained and mixture linear priors for the
normative description. The identity both classes rest on is, for a prior
restricted to a box ``S``,

.. math::

    \ln p(y) = \ln p_\mathrm{untrunc}(y) + \ln Z_\mathrm{post} - \ln Z_\mathrm{prior}

with :math:`Z_\mathrm{prior} = \int_S N(\beta\,|\,\mu, \mathrm{diag}(s^2))` and
:math:`Z_\mathrm{post} = \int_S N(\beta\,|\,\hat\beta, \Sigma)`, where
:math:`\hat\beta` is the conditional mean and :math:`\Sigma = \Lambda^{-1}` the
conditional covariance.

``Z_prior`` factorizes exactly, for any number of constrained parameters,
because the prior covariance is diagonal. ``Z_post`` does not: the conditional
posterior couples the linear parameters through ``X^T C^-1 X``, so it is a
Gaussian rectangle probability -- closed form for one constrained parameter, a
smooth one-dimensional integral for two, and unsupported above that.
"""

__all__ = (
    "GeneralizedMarginalizedLinear",
    "ResolvedLinearPrior",
    "build_marginalized",
)

import math
from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.linalg import cho_solve
from jax.scipy.special import log_ndtr, logsumexp

from harv.stats.numpyro_ext import MarginalizedLinear

# Gauss-Legendre rule for the Plackett integral in :func:`_ln_phi2`. Built once
# at import so the quadrature is a static constant: the rule must not depend on
# traced values or the whole module stops being jit/vmap-safe.
#
# Held as *numpy* rather than JAX arrays, so that only the node *values* are
# fixed at import and the dtype is still decided where they are used.
# ``jnp.asarray`` here would bake in whatever precision JAX happened to be
# configured for at import, pinning the quadrature to float32 for the life of
# any process that does ``import harv`` before
# ``jax.config.update("jax_enable_x64", True)`` -- degrading ``_ln_bvn_rect``
# from 7e-14 to 5e-6 nats while :func:`_signed_logsumexp_guarded` still sized
# its budget from the float64 operands, leaving the guard ~20 nats too
# permissive in exactly the regime it exists for.
_GL_N = 64
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GL_N)
_LN_GL_WEIGHTS = np.log(_GL_WEIGHTS)

# Bisection steps used to invert a conditional CDF in
# :meth:`_TruncatedConditional.sample`. 60 steps bisects a 12-sigma bracket to
# below float64 resolution; the count is static so the loop has a fixed trip
# count under ``vmap``.
_BISECT_STEPS = 60
_BISECT_HALF_WIDTH = 12.0

# Held back from the float-precision budget in :func:`_signed_logsumexp_guarded`
# so that any surviving value stands ~100x above the cancellation noise floor.
_CANCELLATION_MARGIN = math.log(100.0)


# -------------------------------------------------------------------------
# Normalizer numerics
# -------------------------------------------------------------------------


def _signed_logsumexp_guarded(terms: jax.Array, weights: jax.Array) -> jax.Array:
    r"""Log of a signed sum of exponentials, or ``-inf`` if it cancelled away.

    An alternating sum is only meaningful while the total stays within the
    working precision of its largest term. Past that the surviving digits are
    roundoff, and the result comes out *finite and far too high* (in float32, a
    true ``ln P`` of -44 reported as -17), which would over-weight a draw the
    truncation is meant to forbid. ``-inf`` errs the other way and discards
    only weight already below the precision floor.

    The budget is ``-ln(eps)`` for the working dtype, derived rather than
    hard-coded so it tightens in single precision, less
    ``_CANCELLATION_MARGIN``. The margin is not padding: the test can only be
    made against the *computed* total, which is itself what the cancellation
    corrupts, so a bare ``-ln(eps)`` threshold lets noise-floor values through.

    ``weights`` carries each term's sign and may carry magnitude too (see
    :func:`_ln_phi2`), so the largest term is measured as
    ``terms + ln|weights|`` under ``stop_gradient`` -- it feeds a comparison
    only, where a ``-ln 0`` is harmless. The sum is formed by hand rather than
    with ``logsumexp(terms, b=weights)`` because that masks zero weights when
    choosing its shift, and so reports a *zero* derivative with respect to a
    weight that vanishes -- silently wrong at the one point :func:`_ln_phi2`
    needs it (``rho = 0``).
    """
    amax = jax.lax.stop_gradient(jnp.max(terms))
    total = jnp.sum(weights * jnp.exp(terms - amax))
    budget = -jnp.log(jnp.finfo(terms.dtype).eps) - _CANCELLATION_MARGIN
    scaled = jax.lax.stop_gradient(terms + jnp.log(jnp.abs(weights)))
    # The substituted 1.0 keeps ``log`` off a non-positive total: a cancelled
    # sum is discarded below, and without it the unselected branch of the
    # ``where`` would hand back an infinite derivative.
    positive = total > 0
    out = jnp.log(jnp.where(positive, total, 1.0)) + amax
    return jnp.where(positive & (out > jnp.max(scaled) - budget), out, -jnp.inf)


def _ln_ndtr_interval(lo_z: jax.Array, hi_z: jax.Array) -> jax.Array:
    r"""Return :math:`\ln[\Phi(\mathrm{hi}) - \Phi(\mathrm{lo})]`, elementwise.

    Evaluated as ``log_ndtr(hi) + log(-expm1(log_ndtr(lo) - log_ndtr(hi)))``,
    which stays accurate both when the interval covers almost all the mass
    (the difference of the two ``log_ndtr`` values is near zero) and when it
    sits deep in a tail. Infinite bounds fall out of the same expression:
    ``lo = -inf`` gives ``log_ndtr(lo) = -inf`` and hence ``log(-expm1(-inf)) = 0``.

    A zero-width interval correctly returns ``-inf``; callers are responsible
    for rejecting ``low >= high`` before it reaches the likelihood (see
    :func:`harv.models._helpers._resolve_linear_priors`).
    """
    ln_hi = log_ndtr(hi_z)
    ln_lo = log_ndtr(lo_z)
    return ln_hi + jnp.log(-jnp.expm1(ln_lo - ln_hi))


def _ln_phi2(a: jax.Array, b: jax.Array, rho: jax.Array) -> jax.Array:
    r"""Return :math:`\ln \Phi_2(a, b; \rho)` for **finite** ``a`` and ``b``.

    Uses Plackett's identity,

    .. math::

        \Phi_2(a, b; \rho) = \Phi(a)\Phi(b)
            + \frac{1}{2\pi} \int_0^{\arcsin \rho}
              \exp\!\left(-\frac{a^2 - 2ab\sin\theta + b^2}{2\cos^2\theta}\right)
              \mathrm{d}\theta

    The integral is written in :math:`\theta` rather than in the correlation
    :math:`t = \sin\theta`, which is what makes a fixed Gauss-Legendre rule
    adequate: the :math:`(1-t^2)^{-1/2}` factor of the ``t``-form is exactly
    cancelled by :math:`\mathrm{d}t = \cos\theta\,\mathrm{d}\theta`, leaving a
    bounded, smooth integrand over the whole range of :math:`\rho`.

    The two contributions are combined by a signed ``logsumexp`` (the integral
    is negative for :math:`\rho < 0`), so the result stays accurate when
    :math:`\Phi_2` is small.

    Infinite arguments are *not* handled here -- the integrand is indeterminate
    for them. :func:`_ln_bvn_rect` reduces those corners analytically first,
    branching on static finiteness flags.
    """
    # ``arcsin`` is evaluated at the correlation, so keep it inside (-1, 1).
    rho = jnp.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)
    half = jnp.arcsin(rho)

    # Map the fixed [-1, 1] rule onto [0, half].
    theta = 0.5 * half * (1.0 + _GL_NODES)
    sin_t = jnp.sin(theta)
    cos2 = jnp.cos(theta) ** 2
    expo = -(a**2 - 2.0 * a * b * sin_t + b**2) / (2.0 * cos2)

    # The integral is ``half * J`` with J > 0. ``half`` is carried as this
    # term's signed *weight* rather than folded in as ``ln|half|``: the two are
    # equal, but ``log|half|`` has an infinite derivative at ``half = 0``, and
    # ``rho`` is exactly zero whenever the two constrained columns share no
    # design column, so the gradient there has to stay finite.
    ln_j = -jnp.log(4.0 * jnp.pi) + logsumexp(_LN_GL_WEIGHTS + expo)

    # The two contributions have opposite signs for rho < 0 and cancel hard
    # there, so this sum needs the same guard as the outer inclusion-exclusion.
    return _signed_logsumexp_guarded(
        jnp.stack([log_ndtr(a) + log_ndtr(b), ln_j]),
        jnp.stack([jnp.ones_like(half), half]),
    )


def _ln_bvn_rect(
    lo: jax.Array,
    hi: jax.Array,
    rho: jax.Array,
    *,
    lo_finite: tuple[bool, ...],
    hi_finite: tuple[bool, ...],
) -> jax.Array:
    r"""Log probability that a standard bivariate normal lies in a rectangle.

    ``lo``/``hi`` are shape ``(2,)`` standardized bounds and ``rho`` is the
    correlation. Assembled by inclusion-exclusion over the four corners,

    .. math::

        \Phi_2(h_1,h_2) - \Phi_2(l_1,h_2) - \Phi_2(h_1,l_2) + \Phi_2(l_1,l_2)

    combined with a signed ``logsumexp``. The finiteness flags are static, so
    every corner is reduced here, before the quadrature: a corner taking an
    unbounded *lower* bound is exactly zero, one taking a single ``+inf``
    reduces to a univariate ``log_ndtr``, and :func:`_ln_phi2` therefore only
    ever sees finite arguments.

    Scalar-only in ``lo``/``hi``/``rho`` -- :func:`_ln_phi2` uses the trailing
    axis for its quadrature nodes. Batch with ``jax.vmap``.

    The sum cancels once the rectangle probability drops far below its largest
    corner, so :func:`_signed_logsumexp_guarded` returns ``-inf`` past the
    precision floor (near ``ln P = -32`` in float64) rather than a value it
    cannot justify; accuracy above that is pinned by
    ``test_ln_bvn_rect_matches_scipy``.
    """
    axis0 = ((True, lo[0], lo_finite[0]), (False, hi[0], hi_finite[0]))
    axis1 = ((True, lo[1], lo_finite[1]), (False, hi[1], hi_finite[1]))

    terms: list[jax.Array] = []
    signs: list[float] = []
    for is_lo0, x0, finite0 in axis0:
        for is_lo1, x1, finite1 in axis1:
            if (is_lo0 and not finite0) or (is_lo1 and not finite1):
                continue  # an unbounded lower bound: Phi_2 is zero there
            if not finite0 and not finite1:
                term = jnp.zeros_like(rho)
            elif not finite0:
                term = log_ndtr(x1)
            elif not finite1:
                term = log_ndtr(x0)
            else:
                term = _ln_phi2(x0, x1, rho)
            terms.append(term)
            signs.append(1.0 if is_lo0 == is_lo1 else -1.0)

    return _signed_logsumexp_guarded(jnp.stack(terms), jnp.asarray(signs))


@final
class ResolvedLinearPrior(eqx.Module):
    """A per-parameter linear prior resolved to arrays, ready to marginalize.

    One object covers all three supported prior families, so the marginalization
    path has a single shape to consume:

    - a plain Gaussian is ``C = 1`` with infinite bounds;
    - a truncated Gaussian sets finite ``low`` and/or ``high``;
    - a mixture has ``C > 1`` components with log-weights summing to zero in
      probability.

    ``low_finite`` / ``high_finite`` are ``static`` pytree metadata, not derived
    from the arrays with ``jnp.isfinite``: which side of which parameter is
    bounded is structural (it comes from the prior's support) and must be known
    at trace time, because it selects the closed form used for ``Z_post``. The
    bound *values* stay ordinary leaves, so a ``LinearPriorCallable`` may return
    a traced bound.

    Examples
    --------
    The signed SB2 pair ``K_1 > 0``, ``K_2 < 0``, under a zero-mean prior:

    >>> import jax.numpy as jnp
    >>> from harv.stats.marginalized import ResolvedLinearPrior
    >>> prior = ResolvedLinearPrior(
    ...     loc=jnp.zeros((1, 2)),
    ...     scale=jnp.full((1, 2), 30.0),
    ...     ln_weights=jnp.zeros(1),
    ...     low=jnp.array([0.0, -jnp.inf]),
    ...     high=jnp.array([jnp.inf, 0.0]),
    ...     low_finite=(True, False),
    ...     high_finite=(False, True),
    ...     names=("K_1", "K_2"),
    ... )
    >>> prior.n_components, prior.constrained, prior.is_plain_gaussian
    (1, (0, 1), False)
    """

    loc: jax.Array  # (C, k) prior means
    scale: jax.Array  # (C, k) prior standard deviations
    ln_weights: jax.Array  # (C,) normalized mixture log-weights
    low: jax.Array  # (k,) lower bounds, -inf where unbounded
    high: jax.Array  # (k,) upper bounds, +inf where unbounded
    low_finite: tuple[bool, ...] = eqx.field(static=True)
    high_finite: tuple[bool, ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)  # for error messages

    @property
    def n_components(self) -> int:
        """Number of mixture components (1 for a non-mixture prior)."""
        return len(self.ln_weights)

    @property
    def constrained(self) -> tuple[int, ...]:
        """Static indices of the parameters carrying a finite bound."""
        return tuple(
            i
            for i, (lf, hf) in enumerate(
                zip(self.low_finite, self.high_finite, strict=True)
            )
            if lf or hf
        )

    @property
    def is_plain_gaussian(self) -> bool:
        """Whether this reduces to the untruncated single-Gaussian fast path."""
        return self.n_components == 1 and not self.constrained


def _standardize_bounds(
    prior: ResolvedLinearPrior, center: jax.Array, scale: jax.Array
) -> tuple[jax.Array, jax.Array]:
    r"""Standardize the truncation box, keeping infinities out of the arithmetic.

    An unbounded side is never *computed*: ``(-inf - loc) / scale`` would
    differentiate to ``NaN`` via ``0 * inf``, and a ``jnp.where`` does not help
    because it evaluates both branches under ``grad``. Since the finiteness
    flags are static, those slots are filled with a literal infinity, which is
    a constant carrying no gradient path. ``log_ndtr(+-inf)`` is exact, and
    :func:`_ln_bvn_rect` reduces the infinite corners before the quadrature.
    """

    def standardize(
        bounds: jax.Array, finite: tuple[bool, ...], unbounded: float
    ) -> jax.Array:
        return jnp.stack(
            [
                (bounds[i] - center[..., i]) / scale[..., i]
                if f
                else jnp.full_like(scale[..., i], unbounded)
                for i, f in enumerate(finite)
            ],
            axis=-1,
        )

    return (
        standardize(prior.low, prior.low_finite, -jnp.inf),
        standardize(prior.high, prior.high_finite, jnp.inf),
    )


def _ln_prior_normalizer(prior: ResolvedLinearPrior) -> jax.Array:
    r"""Return ``ln Z_prior`` per mixture component, shape ``(C,)``.

    The prior covariance is diagonal, so the box probability factorizes over
    parameters *exactly*, for any number of constrained ones -- this is the half
    of the correction that has no dimensional limit.

    No branching on which parameters are bounded is needed: an unbounded
    parameter has ``low = -inf``, ``high = +inf``, which
    :func:`_ln_ndtr_interval` maps to ``0`` and so drops out of the sum.
    """
    z_lo, z_hi = _standardize_bounds(prior, prior.loc, prior.scale)
    return jnp.sum(_ln_ndtr_interval(z_lo, z_hi), axis=-1)


def _ln_post_normalizer(
    mean: jax.Array, cov: jax.Array, prior: ResolvedLinearPrior
) -> jax.Array:
    r"""Return ``ln Z_post``: the conditional posterior mass inside the box.

    ``mean`` is the conditional mean (shape ``(k,)``) and ``cov`` the
    conditional covariance (shape ``(k, k)``) of the *untruncated* Gaussian
    conditional. Unlike ``Z_prior`` this does **not** factorize: ``cov`` is
    dense because ``X^T C^-1 X`` couples the linear parameters, so the integral
    is a Gaussian rectangle probability over the constrained sub-block.

    Dispatches on the number of constrained parameters, which is static:

    - 0: no truncation, returns ``0``.
    - 1: exact, a single :func:`_ln_ndtr_interval` on that parameter's marginal.
    - 2: exact up to quadrature, via :func:`_ln_bvn_rect`.
    - 3+: :class:`NotImplementedError`.

    Unconstrained parameters need no special handling: they integrate to one, so
    marginalizing down to the constrained sub-block is exactly taking the
    corresponding entries of ``mean`` and ``cov``.

    Scalar in the batch: ``jax.vmap`` it over a mixture's component axis.
    """
    idx = prior.constrained
    if not idx:
        return jnp.zeros(())

    sd = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
    z_lo, z_hi = _standardize_bounds(prior, mean, sd)

    if len(idx) == 1:
        i = idx[0]
        return _ln_ndtr_interval(z_lo[i], z_hi[i])

    if len(idx) == 2:
        i, j = idx
        rho = cov[i, j] / (sd[i] * sd[j])
        return _ln_bvn_rect(
            jnp.stack([z_lo[i], z_lo[j]]),
            jnp.stack([z_hi[i], z_hi[j]]),
            rho,
            lo_finite=(prior.low_finite[i], prior.low_finite[j]),
            hi_finite=(prior.high_finite[i], prior.high_finite[j]),
        )

    constrained_names = tuple(prior.names[i] for i in idx)
    msg = (
        f"Analytic marginalization over {len(idx)} support-constrained linear "
        f"parameters {constrained_names} is not supported. The conditional "
        "posterior couples the linear parameters through X^T C^-1 X, so the "
        "required normalization is a Gaussian orthant probability: closed form "
        "for one constrained parameter, a one-dimensional quadrature for two, "
        "and a higher-dimensional integral above that. Constrain at most two, "
        "or sample the others explicitly by leaving them out of "
        "`marginalized_names`."
    )
    raise NotImplementedError(msg)


# -------------------------------------------------------------------------
# Exact truncated-Gaussian sampling helpers
# -------------------------------------------------------------------------


def _trunc_normal_draw(
    key: jax.Array, mean: jax.Array, sd: jax.Array, lo: jax.Array, hi: jax.Array
) -> jax.Array:
    """Exact draw from a scalar normal truncated to ``[lo, hi]``.

    numpyro's truncated normal reflects around the base location rather than
    inverting the CDF on the probability scale, so it stays exact tens of sigma
    into a tail; infinite bounds pass through unchanged. The clip is not
    redundant: :func:`_sample_truncated_gaussian` conditions the remaining
    coordinates on these values, so "inside the box" has to hold absolutely and
    rounding can land a draw a hair outside.
    """
    draw = dist.TruncatedNormal(mean, sd, low=lo, high=hi).sample(key)
    return jnp.clip(draw, lo, hi)


def _gaussian_conditional(
    mean: jax.Array,
    cov: jax.Array,
    obs_idx: tuple[int, ...],
    obs_vals: jax.Array,
    free_idx: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Condition a Gaussian on a subset of its coordinates, exactly.

    Returns ``(mean_free, cov_free)`` for the standard Gaussian conditional
    ``mu_f + S_fo S_oo^-1 (x_o - mu_o)``, ``S_ff - S_fo S_oo^-1 S_of``. Index
    tuples are static, so this is plain fancy indexing rather than a gather.
    """
    f = jnp.asarray(free_idx)
    s_ff = cov[f[:, None], f[None, :]]
    if not obs_idx:
        return mean[f], s_ff
    o = jnp.asarray(obs_idx)
    s_oo = cov[o[:, None], o[None, :]]
    s_fo = cov[f[:, None], o[None, :]]
    mean_f = mean[f] + s_fo @ jnp.linalg.solve(s_oo, obs_vals - mean[o])
    cov_f = s_ff - s_fo @ jnp.linalg.solve(s_oo, s_fo.T)
    return mean_f, cov_f


def _bisect_inverse_cdf(
    ln_cdf: Callable[[jax.Array], jax.Array],
    target_ln: jax.Array,
    lo_b: jax.Array,
    hi_b: jax.Array,
) -> jax.Array:
    """Invert a monotone log-CDF on ``[lo_b, hi_b]`` by fixed-step bisection.

    ``_BISECT_STEPS`` is static, so the loop has a fixed trip count and stays
    ``vmap``-safe -- a ``while_loop`` on a convergence test would not, and one
    slow draw inside a vmapped batch would stall every lane with it.
    """

    def body(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        a, b = bounds
        mid = 0.5 * (a + b)
        go_right = ln_cdf(mid) < target_ln
        return (jnp.where(go_right, mid, a), jnp.where(go_right, b, mid))

    a, b = jax.lax.fori_loop(0, _BISECT_STEPS, body, (lo_b, hi_b))
    return 0.5 * (a + b)


def _sample_truncated_gaussian(
    key: jax.Array, mean: jax.Array, cov: jax.Array, prior: ResolvedLinearPrior
) -> jax.Array:
    r"""Exact draw from ``N(mean, cov)`` restricted to the prior's box.

    The constrained coordinates are drawn first, in sequence, then the
    unconstrained remainder follows from one exact Gaussian conditional. For one
    constrained coordinate every step is closed-form; for two, the first
    coordinate's marginal-within-the-box has the closed-form CDF
    ``_ln_bvn_rect`` already provides, so it is inverted by bisection rather
    than approximated (no Gibbs, no rejection loop, no bias).
    """
    k = mean.shape[-1]
    idx = prior.constrained
    free = tuple(i for i in range(k) if i not in idx)
    sd = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
    out = jnp.zeros((k,), dtype=mean.dtype)

    key_c, key_f = jax.random.split(key)
    drawn: list[jax.Array] = []

    if len(idx) == 1:
        i = idx[0]
        drawn.append(
            _trunc_normal_draw(key_c, mean[i], sd[i], prior.low[i], prior.high[i])
        )
    elif len(idx) == 2:
        i, j = idx
        key_i, key_j = jax.random.split(key_c)
        rho = cov[i, j] / (sd[i] * sd[j])
        z_lo, z_hi = _standardize_bounds(prior, mean, sd)
        lo_fin = (prior.low_finite[i], prior.low_finite[j])
        hi_fin = (prior.high_finite[i], prior.high_finite[j])

        def ln_rect_to(t: jax.Array) -> jax.Array:
            """Ln P(Z_i <= t, Z_j in box_j) -- monotone increasing in ``t``."""
            return _ln_bvn_rect(
                jnp.stack([z_lo[i], z_lo[j]]),
                jnp.stack([t, z_hi[j]]),
                rho,
                lo_finite=lo_fin,
                hi_finite=(True, hi_fin[1]),
            )

        ln_total = _ln_bvn_rect(
            jnp.stack([z_lo[i], z_lo[j]]),
            jnp.stack([z_hi[i], z_hi[j]]),
            rho,
            lo_finite=lo_fin,
            hi_finite=hi_fin,
        )
        # Bisection bracket. The coordinates are standardized on the
        # conditional mean, so the mass sits at z = 0 and an open end is
        # anchored there rather than on the bound: a bracket running
        # ``[bound, bound + width]`` excludes the whole bulk whenever the bound
        # lies further than that below the mean, which is the ordinary case for
        # a constraint the data do not fight. Normal mass beyond 24 sigma of
        # the mean is below ``exp(-288)``, so nothing representable is lost.
        width = 2.0 * _BISECT_HALF_WIDTH
        lo_b = z_lo[i] if lo_fin[0] else jnp.minimum(z_hi[i], 0.0) - width
        hi_b = z_hi[i] if hi_fin[0] else jnp.maximum(z_lo[i], 0.0) + width
        u = jax.random.uniform(key_i, dtype=mean.dtype)
        z_i = _bisect_inverse_cdf(ln_rect_to, ln_total + jnp.log(u), lo_b, hi_b)

        # When the box holds less mass than the precision can resolve,
        # ``ln_total`` is -inf by design and bisection has no target. Fall back
        # to coordinate i's own truncated marginal: it ignores the coupling to
        # j but stays inside the box, and such a draw has ``log_prob = -inf``
        # so the sampler rejects it before it reaches output.
        z_i = jnp.where(
            jnp.isfinite(ln_total),
            z_i,
            _trunc_normal_draw(
                key_i, jnp.zeros_like(z_i), jnp.ones_like(z_i), z_lo[i], z_hi[i]
            ),
        )
        x_i = mean[i] + sd[i] * z_i

        # Coordinate j given i is an ordinary 1-D truncated normal.
        mean_j = mean[j] + rho * sd[j] * z_i
        sd_j = sd[j] * jnp.sqrt(jnp.clip(1.0 - rho**2, 1e-300, None))
        x_j = _trunc_normal_draw(key_j, mean_j, sd_j, prior.low[j], prior.high[j])
        drawn.extend([x_i, x_j])

    if drawn:
        out = out.at[jnp.asarray(idx)].set(jnp.stack(drawn))

    if free:
        obs_vals = jnp.stack(drawn) if drawn else jnp.zeros((0,))
        mean_f, cov_f = _gaussian_conditional(mean, cov, idx, obs_vals, free)
        # The Schur complement is symmetric in exact arithmetic but not quite in
        # floating point, and a hair-negative eigenvalue makes the default
        # Cholesky path return NaN. Symmetrizing plus an SVD draw costs nothing
        # at these sizes (k is a handful) and cannot fail that way.
        cov_f = 0.5 * (cov_f + cov_f.T)
        draw_f = jax.random.multivariate_normal(
            key_f, mean_f, cov_f, dtype=mean.dtype, method="svd"
        )
        out = out.at[jnp.asarray(free)].set(draw_f)

    return out


# -------------------------------------------------------------------------
# Public wrapper
# -------------------------------------------------------------------------


@final
class _GeneralizedConditional(eqx.Module):
    """Conditional posterior of the linear parameters, possibly truncated.

    Duck-types the part of :class:`numpyro.distributions.MultivariateNormal`
    that harv's marginalization call sites use -- ``.mean`` and
    ``.sample(key)`` -- so :class:`GeneralizedMarginalizedLinear` drops into
    them unchanged.
    """

    mean_untruncated: jax.Array  # (C, k) conditional means, before truncation
    cov: jax.Array  # (C, k, k) conditional covariances
    ln_component_weights: jax.Array  # (C,) normalized posterior log-weights
    prior: ResolvedLinearPrior

    @property
    def mean(self) -> jax.Array:
        r"""The (truncated) conditional mean, mixed over components.

        The per-component truncated mean comes from autodiff rather than a
        closed-form truncated-moment formula. For
        :math:`Z(m) = \int_S N(\beta\,|\,m, \Sigma)\,\mathrm{d}\beta`,

        .. math::

            \nabla_m \ln Z = \Sigma^{-1}\left(E_S[\beta] - m\right)
            \quad\Longrightarrow\quad
            E_S[\beta] = m + \Sigma \nabla_m \ln Z

        so differentiating :func:`_ln_post_normalizer` -- which we need anyway --
        gives the exact truncated mean for every supported number of constrained
        parameters, including the correct shift of the *unconstrained*
        coordinates. An untruncated prior has ``grad = 0`` and returns the
        ordinary conditional mean.
        """
        grad_ln_z = jax.vmap(
            jax.grad(lambda m, c: _ln_post_normalizer(m, c, self.prior)),
            in_axes=(0, 0),
        )(self.mean_untruncated, self.cov)
        # When the box holds less conditional mass than the working precision
        # can resolve, ``ln Z_post`` is -inf by design and its gradient is not
        # defined. Fall back to the *untruncated* conditional mean, which is
        # finite and inside no box in particular; such a parameter set has
        # ``log_prob = -inf``, so the sampler rejects it and the fallback never
        # reaches output -- the same argument :func:`_sample_truncated_gaussian`
        # makes for its own fallback. Without this, a NaN here propagates
        # through ``linear_log_prior_correction`` and poisons the whole batch's
        # evidence and top-k selection.
        grad_ln_z = jnp.where(jnp.isfinite(grad_ln_z), grad_ln_z, 0.0)
        means = self.mean_untruncated + jnp.einsum("cij,cj->ci", self.cov, grad_ln_z)
        return jnp.einsum("c,ci->i", jnp.exp(self.ln_component_weights), means)

    def sample(self, key: jax.Array) -> jax.Array:
        """Draw one exact sample of the linear parameters.

        Picks a mixture component from its posterior weights, then draws from
        that component's conditional restricted to the prior's support. Only the
        selected component is sampled -- the gather is by traced index, so the
        cost does not scale with the number of components.
        """
        key_c, key_draw = jax.random.split(key)
        c = jax.random.categorical(key_c, self.ln_component_weights)
        return _sample_truncated_gaussian(
            key_draw,
            jnp.take(self.mean_untruncated, c, axis=0),
            jnp.take(self.cov, c, axis=0),
            self.prior,
        )


@final
class GeneralizedMarginalizedLinear(eqx.Module):
    r"""Linear marginalization under a truncated and/or mixture Gaussian prior.

    Wraps the vendored :class:`~harv.stats.MarginalizedLinear` rather than
    editing it, and exposes the same two methods harv's marginalization path
    uses -- ``log_prob(y)`` and ``conditional(y)`` -- so every call site is
    unchanged.

    Mixture components live on the *batch* axis of the inner distribution, which
    :class:`~harv.stats.MarginalizedLinear` already supports, so a mixture needs
    no new linear algebra at all: ``inner.log_prob(y)`` returns one value per
    component and the answer is a weighted ``logsumexp``.

    The mixture and the truncation interact in one place that is easy to get
    silently wrong. Because every component shares the same support, the
    indicator factors out of the mixture sum, so the prior normalizer is the
    *mixture* normalizer

    .. math::

        Z_\mathrm{mix} = \sum_c w_c Z_{\mathrm{prior},c}

    subtracted **once**. Building this from per-component truncated log-probs --
    each of which has already divided by its own :math:`Z_{\mathrm{prior},c}` --
    gives a different and wrong answer, and :math:`Z_{\mathrm{prior},c}` varies
    across components even under a shared box, so it does not cancel.

    See ``docs/spec.md`` §Support-constrained and mixture linear priors.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from harv.stats.marginalized import ResolvedLinearPrior, build_marginalized
    >>> design = jnp.array([[1.0, 1.0], [0.5, 1.0], [-0.5, 1.0], [-1.0, 1.0]])
    >>> err = jnp.full(4, 0.2)
    >>> obs = design @ jnp.array([2.0, -1.0])

    A prior that forbids the sign the data want costs marginal likelihood:

    >>> def marg(low):
    ...     prior = ResolvedLinearPrior(
    ...         jnp.zeros((1, 2)), jnp.full((1, 2), 5.0), jnp.zeros(1),
    ...         jnp.asarray(low), jnp.full(2, jnp.inf),
    ...         tuple(bool(v) for v in jnp.isfinite(jnp.asarray(low))),
    ...         (False, False), ("slope", "offset"),
    ...     )
    ...     return build_marginalized(design, prior, dist.Normal(0.0, err))
    >>> allowed = marg([0.0, -jnp.inf]).log_prob(obs)
    >>> forbidden = marg([4.0, -jnp.inf]).log_prob(obs)
    >>> bool(allowed > forbidden)
    True

    ``conditional`` respects the same support:

    >>> import jax
    >>> draw = marg([0.0, -jnp.inf]).conditional(obs).sample(jax.random.key(0))
    >>> bool(draw[0] >= 0.0)
    True
    """

    inner: MarginalizedLinear
    prior: ResolvedLinearPrior

    def _conditional_moments(self, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Untruncated conditional mean ``(C, k)`` and covariance ``(C, k, k)``."""
        loc = jnp.atleast_2d(self.inner.conditional(value).loc)
        tril = self.inner.conditional_inv_tril
        tril = tril if tril.ndim == 3 else tril[None]
        eye = jnp.broadcast_to(jnp.eye(tril.shape[-1], dtype=tril.dtype), tril.shape)
        return loc, cho_solve((tril, True), eye)

    def log_prob(self, value: jax.Array) -> jax.Array:
        """Marginal log-likelihood with the truncation and mixture corrections."""
        ln_p = jnp.atleast_1d(self.inner.log_prob(value))
        mean, cov = self._conditional_moments(value)
        ln_z_post = jax.vmap(_ln_post_normalizer, in_axes=(0, 0, None))(
            mean, cov, self.prior
        )
        ln_z_prior = _ln_prior_normalizer(self.prior)
        ln_w = self.prior.ln_weights
        # ``ln Z_prior`` collapses to -inf once a finite bound standardizes past
        # ~38 sigma of the prior mean, where both ``log_ndtr`` values underflow
        # to -0.0. Subtracting it would give +inf, or NaN when ``ln Z_post``
        # underflowed too, and either poisons the sampler's ``logsumexp``
        # evidence and top-k comparison across the whole batch. -inf is the
        # honest answer and the direction this module always errs in: a prior
        # keeping ``e**-316`` of its own mass inside the box describes a draw
        # that should be rejected. Written as a double ``where`` so the
        # unselected branch cannot contribute a ``0 * inf`` gradient (the same
        # reason :func:`_standardize_bounds` sanitizes before dividing).
        den = logsumexp(ln_w + ln_z_prior)
        ok = jnp.isfinite(den)
        num = logsumexp(ln_w + ln_p + ln_z_post)
        return jnp.where(ok, num - jnp.where(ok, den, 0.0), -jnp.inf)

    def conditional(self, value: jax.Array) -> _GeneralizedConditional:
        """Conditional posterior over the linear parameters given ``value``."""
        ln_p = jnp.atleast_1d(self.inner.log_prob(value))
        mean, cov = self._conditional_moments(value)
        ln_z_post = jax.vmap(_ln_post_normalizer, in_axes=(0, 0, None))(
            mean, cov, self.prior
        )
        # A component's posterior weight is its prior weight times the evidence
        # it gives the data, times the share of its conditional that survives
        # the truncation.
        ln_w = self.prior.ln_weights + ln_p + ln_z_post
        # Every component's ``ln Z_post`` can underflow to -inf at once (by
        # design, see :func:`_signed_logsumexp_guarded`), and ``-inf - (-inf)``
        # is NaN. Fall back to equal weights: the draw is rejected on its
        # ``log_prob`` either way, and a NaN escaping here reaches users through
        # ``sample_conditional_linear(use_mean=True)`` and through the
        # marginalized log-prob's Jacobian correction.
        total = logsumexp(ln_w)
        ln_w = jnp.where(
            jnp.isfinite(total),
            ln_w - jnp.where(jnp.isfinite(total), total, 0.0),
            -jnp.log(ln_w.shape[0]),
        )
        return _GeneralizedConditional(mean, cov, ln_w, self.prior)


def build_marginalized(
    design_matrix: jax.Array,
    prior: ResolvedLinearPrior,
    data_distribution: dist.Distribution,
) -> MarginalizedLinear | GeneralizedMarginalizedLinear:
    """Assemble the marginalized likelihood for a resolved linear prior.

    Returns a bare :class:`~harv.stats.MarginalizedLinear` for an untruncated
    single Gaussian, so that the overwhelmingly common path is bit-identical to
    the pre-existing code and costs nothing extra. Anything else gets the
    :class:`GeneralizedMarginalizedLinear` wrapper.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpyro.distributions as dist
    >>> from harv.stats import MarginalizedLinear
    >>> from harv.stats.marginalized import ResolvedLinearPrior, build_marginalized
    >>> design = jnp.array([[1.0, 1.0], [0.5, 1.0], [-1.0, 1.0]])
    >>> data_dist = dist.Normal(0.0, jnp.full(3, 0.2))
    >>> def prior(low_finite):
    ...     low = jnp.array([0.0, -jnp.inf]) if low_finite else jnp.full(2, -jnp.inf)
    ...     return ResolvedLinearPrior(
    ...         jnp.zeros((1, 2)), jnp.ones((1, 2)), jnp.zeros(1),
    ...         low, jnp.full(2, jnp.inf), (low_finite, False), (False, False),
    ...         ("a", "b"),
    ...     )
    >>> type(build_marginalized(design, prior(False), data_dist)) is MarginalizedLinear
    True
    >>> type(build_marginalized(design, prior(True), data_dist)).__name__
    'GeneralizedMarginalizedLinear'
    """
    if prior.is_plain_gaussian:
        return MarginalizedLinear(
            design_matrix=design_matrix,
            prior_distribution=dist.MultivariateNormal(
                loc=prior.loc[0], scale_tril=jnp.diag(prior.scale[0])
            ),
            data_distribution=data_distribution,
        )
    return GeneralizedMarginalizedLinear(
        inner=MarginalizedLinear(
            design_matrix=design_matrix,
            prior_distribution=dist.MultivariateNormal(
                loc=prior.loc, scale_tril=jax.vmap(jnp.diag)(prior.scale)
            ),
            data_distribution=data_distribution,
        ),
        prior=prior,
    )
