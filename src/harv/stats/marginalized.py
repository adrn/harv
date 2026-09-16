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

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.scipy.linalg import cho_solve
from jax.scipy.special import log_ndtr, logsumexp, ndtr, ndtri

from harv.stats.numpyro_ext import MarginalizedLinear

# Gauss-Legendre rule for the Plackett integral in :func:`_ln_phi2`. Fixed at
# import time so the quadrature is a static constant: the rule must not depend
# on traced values or the whole module stops being jit/vmap-safe.
_GL_N = 64
_GL_NODES, _GL_WEIGHTS = (
    jnp.asarray(x) for x in np.polynomial.legendre.leggauss(_GL_N)
)
_LN_GL_WEIGHTS = jnp.log(_GL_WEIGHTS)

# ``asin`` is evaluated at the correlation, so keep it strictly inside (-1, 1).
_RHO_MAX = 1.0 - 1e-12

# Bisection steps used to invert a conditional CDF in
# :meth:`_TruncatedConditional.sample`. 60 steps bisects a 12-sigma bracket to
# below float64 resolution; the count is static so the loop has a fixed trip
# count under ``vmap``.
_BISECT_STEPS = 60
_BISECT_HALF_WIDTH = 12.0

# Stand-in for an infinite standardized bound. Phi(-40) underflows to zero even
# in float64, so this is indistinguishable from infinity in every expression it
# feeds, while keeping actual infinities out of the arithmetic -- see
# :func:`_standardize_bounds`.
_Z_SENTINEL = 40.0

# Held back from the float-precision budget in :func:`_signed_logsumexp_guarded`
# so that any surviving value stands ~100x above the cancellation noise floor.
_CANCELLATION_MARGIN = jnp.log(100.0)


# -------------------------------------------------------------------------
# Normalizer numerics
# -------------------------------------------------------------------------


def _signed_logsumexp_guarded(terms: jax.Array, signs: jax.Array) -> jax.Array:
    r"""Log of a signed sum of exponentials, or ``-inf`` if it cancelled away.

    A probability built as an alternating sum is only meaningful while the total
    stays within the working precision of its largest term. Past that the
    surviving digits are roundoff, and measured in float32 the result comes out
    *finite and far too high* -- a true ``ln P`` of -44 reported as -17 -- which
    would over-weight a draw the truncation is meant to forbid. Returning
    ``-inf`` errs the other way and only ever discards weight that was already
    below the precision floor.

    The budget is ``-ln(eps)`` for the working dtype -- about 16 nats in float32
    and 36 in float64 -- less ``_CANCELLATION_MARGIN``. The margin is not
    padding: the test has to be made against the *computed* total, which is
    itself the quantity corrupted by the cancellation, so a bare ``-ln(eps)``
    threshold lets values sitting on the noise floor pass by a hair (measured:
    a true -26.7 surviving as -18.9). Holding back ``ln(100)`` means anything
    returned stands a factor of 100 above the noise floor, and so carries at
    most ~1% relative error, or ~0.01 nats.
    """
    out, sign = logsumexp(terms, b=signs, return_sign=True)
    budget = -jnp.log(jnp.finfo(terms.dtype).eps) - _CANCELLATION_MARGIN
    trustworthy = out > jnp.max(terms) - budget
    return jnp.where((sign > 0) & trustworthy, out, -jnp.inf)


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
    rho = jnp.clip(rho, -_RHO_MAX, _RHO_MAX)
    half = jnp.arcsin(rho)

    # Map the fixed [-1, 1] rule onto [0, half].
    theta = 0.5 * half * (1.0 + _GL_NODES)
    sin_t = jnp.sin(theta)
    cos2 = jnp.cos(theta) ** 2
    expo = -(a**2 - 2.0 * a * b * sin_t + b**2) / (2.0 * cos2)

    # ln|I|: the integrand is strictly positive and the weights are positive,
    # so the only sign in the integral comes from the orientation of [0, half].
    ln_abs_integral = jnp.log(jnp.abs(half) / (4.0 * jnp.pi)) + logsumexp(
        _LN_GL_WEIGHTS + expo
    )

    # The two contributions have opposite signs for rho < 0 and cancel hard
    # there, so this sum needs the same precision guard as the outer
    # inclusion-exclusion in :func:`_ln_bvn_rect`.
    return _signed_logsumexp_guarded(
        jnp.stack([log_ndtr(a) + log_ndtr(b), ln_abs_integral]),
        jnp.stack([jnp.ones_like(half), jnp.sign(half)]),
    )


def _corner_kinds(
    lo_finite: tuple[bool, ...], hi_finite: tuple[bool, ...]
) -> list[tuple[float, tuple[bool, bool], tuple[int, int]]]:
    """Enumerate the surviving corners of the inclusion-exclusion sum.

    Returns ``(sign, (axis0_is_inf, axis1_is_inf), (side0, side1))`` per corner,
    where a side indexes ``lo`` (0) or ``hi`` (1). A corner taking an *unbounded
    lower* bound contributes exactly zero, so it is dropped here and never
    reaches the quadrature; the flags that survive can therefore only mean
    ``+inf``. Everything here is static: the *finiteness* of a truncation bound
    is structural (it comes from the prior's support), even though its value may
    be traced.
    """
    finite = (lo_finite, hi_finite)
    out = []
    for sign, sides in ((1.0, (1, 1)), (-1.0, (0, 1)), (-1.0, (1, 0)), (1.0, (0, 0))):
        if any(sides[ax] == 0 and not finite[0][ax] for ax in (0, 1)):
            continue  # an unbounded lower bound: Phi_2 is zero there
        out.append((sign, tuple(not finite[sides[ax]][ax] for ax in (0, 1)), sides))
    return out


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

    combined with a signed ``logsumexp``. Corners involving ``-inf`` vanish and
    are dropped statically, and corners with a ``+inf`` reduce to a univariate
    ``log_ndtr``, so :func:`_ln_phi2` only ever sees finite arguments.

    Scalar-only in ``lo``/``hi``/``rho`` -- :func:`_ln_phi2` uses the trailing
    axis for its quadrature nodes. Batch with ``jax.vmap``.

    ponytail: this cancels once the rectangle probability drops far enough below
    its largest corner, so :func:`_signed_logsumexp_guarded` returns ``-inf``
    past the precision floor rather than a value it cannot justify. Measured
    against ``scipy`` over ``rho`` in (-0.999, 0.999): every *finite* result is
    within 0.004 nats, and in harv's default float32 the floor sits near
    ``ln P = -12`` (near -32 under ``jax_enable_x64``). See
    ``tests/unit/stats/test_marginalized.py::test_ln_bvn_rect_matches_scipy``.
    The floor is benign here -- a rectangle holding ``e**-12`` or less of the
    conditional mass is a draw the sampler rejects either way, and erring to
    ``-inf`` under-weights it rather than over-weighting it. Upgrade path if a
    deeper floor is ever needed: integrate
    ``phi(z) * [Phi(beta(h2,z)) - Phi(beta(l2,z))]`` over coordinate 1, which
    never cancels, but needs an ``|rho|``-dependent node count to resolve the
    inner step -- a different accuracy cliff, not a free win.
    """
    terms: list[jax.Array] = []
    signs: list[float] = []
    for sign, (inf0, inf1), sides in _corner_kinds(lo_finite, hi_finite):
        x0 = (lo, hi)[sides[0]][0]
        x1 = (lo, hi)[sides[1]][1]
        if inf0 and inf1:
            term = jnp.zeros_like(rho)
        elif inf0:
            term = log_ndtr(x1)
        elif inf1:
            term = log_ndtr(x0)
        else:
            term = _ln_phi2(x0, x1, rho)
        terms.append(term)
        signs.append(sign)

    return _signed_logsumexp_guarded(jnp.stack(terms), jnp.asarray(signs))


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

    Returning ``(low - center) / scale`` directly would put ``-inf`` in the
    graph, and differentiating it gives ``NaN`` rather than zero: the derivative
    of ``(-inf - loc) / scale`` with respect to ``scale`` is infinite, and an
    unconstrained parameter contributes an upstream derivative of zero, so the
    chain rule produces ``0 * inf``. A ``jnp.where`` on the bound value does not
    help, because ``where`` evaluates both branches when differentiated.

    So the unbounded sides are neutralized *before* dividing -- the sanitized
    bound array is built from the static finiteness flags and holds no
    infinities -- and the sentinel is substituted afterwards. Both branches of
    the resulting ``where`` are finite, so gradients flow cleanly.
    """
    low_sane = jnp.asarray(
        [prior.low[i] if f else 0.0 for i, f in enumerate(prior.low_finite)]
    )
    high_sane = jnp.asarray(
        [prior.high[i] if f else 0.0 for i, f in enumerate(prior.high_finite)]
    )
    z_lo = jnp.where(
        jnp.asarray(prior.low_finite), (low_sane - center) / scale, -_Z_SENTINEL
    )
    z_hi = jnp.where(
        jnp.asarray(prior.high_finite), (high_sane - center) / scale, _Z_SENTINEL
    )
    return z_lo, z_hi


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
    """Exact draw from a scalar normal truncated to ``[lo, hi]``, by inverse CDF.

    Infinite bounds work unchanged: ``ndtr(-inf) = 0`` and ``ndtr(inf) = 1``.
    """
    p_lo = ndtr((lo - mean) / sd)
    p_hi = ndtr((hi - mean) / sd)
    u = jax.random.uniform(key, dtype=mean.dtype)
    # Clamp into the representable *open* unit interval: in float32 a tail
    # truncation makes p_lo round up to 1.0 for some u, and ndtri(1.0) is +inf,
    # which then poisons the conditional draw of the remaining coordinates. The
    # clamp caps a draw at roughly +/-5.2 sigma in float32 (+/-8.2 in float64).
    eps = jnp.finfo(mean.dtype).eps
    prob = jnp.clip(p_lo + u * (p_hi - p_lo), eps, 1.0 - eps)
    draw = mean + sd * ndtri(prob)
    # Callers rely on "every draw is inside the box" absolutely, and rounding in
    # ndtr/ndtri can land a hair outside it. Clipping makes the invariant exact.
    #
    # ponytail: the probability-scale inverse CDF loses resolution when the
    # truncation sits many sigma from the mean (p_lo -> 1), which in float32
    # starts to matter past ~4 sigma and piles draws near the bound. That regime
    # is one where the data contradict the constraint, so the draw is rejected
    # on its likelihood anyway. Upgrade path: invert in log space via the upper
    # tail, which needs a hand-rolled ndtri_exp -- JAX has no such primitive.
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
    if not free_idx:
        return jnp.zeros((0,)), jnp.zeros((0, 0))
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


def _bracket(
    z_lo: jax.Array, z_hi: jax.Array, lo_finite: bool, hi_finite: bool
) -> tuple[jax.Array, jax.Array]:
    """A finite bisection bracket covering the support, in standardized units.

    Extends 2 * ``_BISECT_HALF_WIDTH`` sigma past whichever bound is finite; the
    standard normal mass beyond that is below ``exp(-288)``, so nothing
    representable is lost.
    """
    width = 2.0 * _BISECT_HALF_WIDTH
    if lo_finite and hi_finite:
        return z_lo, z_hi
    if lo_finite:
        return z_lo, z_lo + width
    if hi_finite:
        return z_hi - width, z_hi
    return (
        jnp.full_like(z_lo, -_BISECT_HALF_WIDTH),
        jnp.full_like(z_hi, _BISECT_HALF_WIDTH),
    )


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
        u = jax.random.uniform(key_i, dtype=mean.dtype)
        lo_b, hi_b = _bracket(z_lo[i], z_hi[i], lo_fin[0], hi_fin[0])
        z_i = _bisect_inverse_cdf(ln_rect_to, ln_total + jnp.log(u), lo_b, hi_b)

        # When the box holds less conditional mass than the working precision can
        # resolve, ``ln_total`` is -inf by design (see
        # :func:`_signed_logsumexp_guarded`) and the bisection has no target to
        # aim at. Fall back to coordinate ``i``'s own truncated marginal, which
        # ignores the coupling to ``j`` but still lands inside the box -- the
        # invariant callers depend on. Such a draw has ``log_prob = -inf``, so
        # the sampler rejects it and the approximation never reaches output.
        z_i = jnp.where(
            jnp.isfinite(ln_total),
            z_i,
            (
                _trunc_normal_draw(
                    key_i,
                    jnp.zeros_like(z_i),
                    jnp.ones_like(z_i),
                    z_lo[i],
                    z_hi[i],
                )
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
        # float32, and a hair-negative eigenvalue makes the default Cholesky
        # path return NaN. Symmetrizing plus an SVD draw costs nothing at these
        # sizes (k is a handful) and cannot fail that way.
        cov_f = 0.5 * (cov_f + cov_f.T)
        draw_f = jax.random.multivariate_normal(
            key_f, mean_f, cov_f, dtype=mean.dtype, method="svd"
        )
        out = out.at[jnp.asarray(free)].set(draw_f)

    return out


# -------------------------------------------------------------------------
# Public wrapper
# -------------------------------------------------------------------------


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
        return logsumexp(ln_w + ln_p + ln_z_post) - logsumexp(ln_w + ln_z_prior)

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
        return _GeneralizedConditional(mean, cov, ln_w - logsumexp(ln_w), self.prior)


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
