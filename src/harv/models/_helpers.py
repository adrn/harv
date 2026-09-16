"""Internal helpers shared across the models subpackage."""

__all__: tuple[str, ...] = ()

import itertools
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro.distributions.truncated import (
    LeftTruncatedDistribution,
    RightTruncatedDistribution,
    TwoSidedTruncatedDistribution,
)
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.distributions import QuantityDistribution
from harv.stats.marginalized import ResolvedLinearPrior

PriorDist = dist.Distribution | QuantityDistribution
# Union type for prior distributions (bare numpyro or unit-aware).

LinearPriorCallable = Callable[
    [dict[str, Any]], QuantityDistribution | dist.Distribution
]
# Callable that returns a marginalizable prior given a dict of parameter values.
# The dict is keyed by bare parameter name and holds the already-sampled nonlinear
# values plus any explicit (non-marginalized) linear values. Values carrying units
# are ``unxt.Q``-wrapped; dimensionless ones (e.g. ``eccentricity``) are bare arrays.
#
# The return may be any of the families ``_parse_linear_prior`` accepts: a Normal,
# a truncated Normal, or a Gaussian mixture, optionally ``QD``-wrapped. A callable
# may also declare a ``requires`` attribute naming the parameters it reads; see
# ``pinned_linear_names``.

LinearPriorDist = PriorDist | LinearPriorCallable

LinearPriorDict = dict[str, LinearPriorDist]
# Per-parameter linear prior dictionary.


def _unwrap_dist(v: PriorDist) -> Any:
    """Extract the underlying numpyro distribution from a PriorDist."""
    if isinstance(v, QuantityDistribution):
        return v.distribution
    return v


def _evaluate_nonlinear_log_prior(
    priors: dict[str, PriorDist], samples: dict[str, jax.Array]
) -> jax.Array:
    """Sum nonlinear-prior log-densities over all sampled parameters.

    ``samples`` holds the bare (unit-stripped) sampled arrays, matching the
    space the prior distributions sample in (see ``HarvPrior``). Used by
    the samplers' ``return_logprobs`` path to record per-sample ``ln_prior``.
    """
    total: jax.Array | None = None
    for name, prior in priors.items():
        if name not in samples:
            continue
        lp = _unwrap_dist(prior).log_prob(samples[name])
        total = lp if total is None else total + lp
    if total is None:
        n = len(next(iter(samples.values()))) if samples else 0
        return jnp.zeros(n)
    return total


def _is_callable_prior(p: Any) -> bool:
    """True iff ``p`` is a callable prior factory (e.g. ``PeriodDependentKPrior``).

    Plain ``dist.Distribution`` and :class:`QuantityDistribution` instances are
    *not* considered callable priors here even if their classes happen to be
    callable: they are sampled directly without being resolved against
    nonlinear parameter values first.
    """
    return callable(p) and not isinstance(p, dist.Distribution | QuantityDistribution)


class _ParsedPrior(NamedTuple):
    """One linear parameter's prior, parsed into arrays in the model's unit.

    ``loc``/``scale``/``ln_weights`` carry a component axis (length 1 for a
    non-mixture prior). The truncation bounds are shared across components --
    that is what lets the indicator factor out of a mixture sum -- and their
    *finiteness* is static Python, not array data.
    """

    loc: jax.Array  # (C,)
    scale: jax.Array  # (C,)
    ln_weights: jax.Array  # (C,)
    low: jax.Array  # ()
    high: jax.Array  # ()
    low_finite: bool
    high_finite: bool


# Prior families the analytic marginalization can integrate out. ``dist.Delta`` is
# handled separately (reclassified as a fixed explicit value), and a
# ``LinearPriorCallable`` is judged by what it returns.
_MARGINALIZABLE = (
    dist.Normal,
    dist.HalfNormal,
    LeftTruncatedDistribution,
    RightTruncatedDistribution,
    TwoSidedTruncatedDistribution,
    dist.MixtureSameFamily,
)


def _can_marginalize(d: LinearPriorDist) -> bool:
    """Whether the likelihood *can* integrate this linear prior out analytically.

    This is a statement about the math only -- see
    :func:`_needs_explicit_sampling` for what the auto-classification actually
    chooses to do. Truncated and mixture Gaussians became marginalizable when
    :mod:`harv.stats.marginalized` was added; before that only plain Normals
    were.
    """
    if isinstance(d, QuantityDistribution):
        return isinstance(d.distribution, (*_MARGINALIZABLE, dist.Delta))
    if isinstance(d, (*_MARGINALIZABLE, dist.Delta)):
        return True
    # A LinearPriorCallable is resolved at evaluation time; assume it returns
    # something marginalizable and let ``_parse_linear_prior`` complain if not.
    return _is_callable_prior(d)


def pinned_linear_names(prior_dict: LinearPriorDict) -> frozenset[str]:
    """Linear params that must stay explicitly sampled because something reads them.

    A :data:`LinearPriorCallable` may declare ``requires``, naming the parameter
    values it looks up in the dict it is called with. Any linear parameter named
    there has to be drawn explicitly: a marginalized parameter has no sampled
    value for the callable to read, and it would raise ``KeyError`` mid-trace.

    This is why ``parallax`` stays explicit under the Gaia defaults even though
    its ``HalfNormal`` prior is now marginalizable --
    ``PeriodDependentSemiMajorAxisPrior`` and
    ``ParallaxDependentProperMotionPrior`` both read it. Encoding the dependency
    beats special-casing the name: nothing reads ``rv_semiamp``, so signed SB2
    semi-amplitudes marginalize without an exception being carved out for them.

    Keys may be component-qualified (``"astro.parallax"`` in a
    :class:`~harv.models.JointModel`) while ``requires`` names the bare
    parameter, so a dependency is resolved inside the declaring prior's own
    namespace first and only then at the top level. Without that, a joint
    RV + Gaia model would silently fail to pin ``astro.parallax`` and the
    callable would raise ``KeyError`` mid-trace.

    A user-supplied callable that reads a linear parameter without declaring
    ``requires`` gets the same ``KeyError`` it would have got before, with the
    same explanatory message. Callables are not introspected.
    """
    pinned: set[str] = set()
    for name, entry in prior_dict.items():
        if not _is_callable_prior(entry):
            continue
        namespace = name.rsplit(".", 1)[0] + "." if "." in name else ""
        for dep in getattr(entry, "requires", ()):
            for candidate in (f"{namespace}{dep}", dep):
                if candidate != name and candidate in prior_dict:
                    pinned.add(candidate)
                    break
    return frozenset(pinned)


def _is_plain_gaussian_prior(d: LinearPriorDist) -> bool:
    """Whether a linear prior can join the non-marginalized model's ``_linear`` MVN.

    Distinct from both :func:`_can_marginalize` and
    :func:`_needs_explicit_sampling`, and the distinction is load-bearing. On the
    ``marginalized=False`` path the Gaussian linear parameters are drawn from one
    joint ``MultivariateNormal`` site, which can only represent an *untruncated*
    single Gaussian. A truncated or mixture prior is perfectly samplable there --
    numpyro has a bijector for a truncated Normal -- just not as part of that one
    MVN, so it gets its own site instead.
    """
    inner = d.distribution if isinstance(d, QuantityDistribution) else d
    if isinstance(inner, dist.Normal):
        return True
    # Callables are assumed to return a plain Normal (the default ones do);
    # ``_parse_linear_prior`` raises clearly at trace time if one does not.
    return _is_callable_prior(d)


def _needs_explicit_sampling(
    d: PriorDist | LinearPriorCallable,
    *,
    name: str = "",
    pinned_names: frozenset[str] = frozenset(),
) -> bool:
    """True if a linear prior entry must be sampled explicitly by the sampler.

    This is the *auto-mode default*. It returns ``False`` for everything
    :func:`_can_marginalize` accepts, except names in ``pinned_names`` -- those
    are read by another prior's callable and so must be drawn explicitly (see
    :func:`pinned_linear_names`).

    ``dist.Delta`` deliberately answers ``False`` here even though its value is
    fixed rather than integrated. It enters the marginalized set and is then
    reclassified, with its value extracted, by
    ``AbstractComponentModel._handle_delta_priors``. Short-circuiting it to
    explicit here would skip that extraction and leave the value unset.

    A user may still override this per sampler via ``marginalized_names``, which
    is gated on :func:`_can_marginalize` instead.
    """
    if name and name in pinned_names:
        return True
    return not _can_marginalize(d)


def _with_derived_eccentricity(
    param_values: dict[str, Any],
    parameterization: Any | None,
) -> dict[str, Any]:
    """Add ``eccentricity`` when the parameterization implies one but does not carry it.

    ``LinearPriorCallable`` implementations are written against the standard parameter
    names, so ``EcoswEsinwRV`` -- which carries ``(ecosw, esinw)`` instead -- would
    otherwise raise ``KeyError: 'eccentricity'`` for the default ``rv_semiamp`` prior.
    The parameterization already owns the conversion; this only applies it.

    Parameters
    ----------
    param_values
        Values that will be handed to a callable prior.
    parameterization
        The parameterization the values came from, or ``None`` to skip.

    Returns
    -------
        ``param_values`` unchanged, or a copy with ``eccentricity`` added.
    """
    if parameterization is None or "eccentricity" in param_values:
        return param_values
    ecc = parameterization.derived_eccentricity(param_values)
    if ecc is None:
        return param_values
    return {**param_values, "eccentricity": ecc}


def _parse_linear_prior(resolved: Any, target_unit: str, name: str) -> _ParsedPrior:
    """Parse one resolved linear prior into loc/scale/bounds in ``target_unit``.

    Accepts a ``Normal``, a ``HalfNormal``, any of numpyro's three truncated
    flavours, or a ``MixtureSameFamily`` over those -- each optionally wrapped in
    a :class:`~harv.distributions.QuantityDistribution` to declare its unit.

    ``HalfNormal(s)`` is parsed as ``TruncatedNormal(0, s, low=0)`` rather than
    special-cased: they are the same distribution, and the truncation machinery
    needs no help to handle it.
    """
    prior_unit: str | None = None
    if isinstance(resolved, QuantityDistribution):
        prior_unit = cast("str", resolved.unit)
        resolved = resolved.distribution

    def conv(value: Any) -> jax.Array:
        """Convert a value from the prior's declared unit to the model's."""
        if prior_unit is None:
            return jnp.asarray(value)
        return jnp.asarray(ustrip(AllowValue, target_unit, Q(value, prior_unit)))

    # -- Mixtures: recurse on the component distribution, which carries the
    # -- per-component loc/scale as a batch axis.
    if isinstance(resolved, dist.MixtureSameFamily):
        probs = jnp.asarray(resolved.mixing_distribution.probs)
        inner = resolved.component_distribution
        wrapped = (
            QuantityDistribution(inner, prior_unit) if prior_unit is not None else inner
        )
        parsed = _parse_linear_prior(wrapped, target_unit, name)
        n_comp = probs.shape[-1]
        return _ParsedPrior(
            loc=jnp.broadcast_to(parsed.loc, (n_comp,)),
            scale=jnp.broadcast_to(parsed.scale, (n_comp,)),
            ln_weights=jnp.log(probs),
            low=parsed.low,
            high=parsed.high,
            low_finite=parsed.low_finite,
            high_finite=parsed.high_finite,
        )

    neg_inf = jnp.asarray(-jnp.inf)
    pos_inf = jnp.asarray(jnp.inf)

    if isinstance(resolved, dist.Normal):
        loc, scale = conv(resolved.loc), conv(resolved.scale)
        low, high, low_fin, high_fin = neg_inf, pos_inf, False, False
    elif isinstance(resolved, dist.HalfNormal):
        loc = jnp.zeros_like(jnp.asarray(resolved.scale))
        scale = conv(resolved.scale)
        low, high, low_fin, high_fin = jnp.zeros_like(scale), pos_inf, True, False
    elif isinstance(
        resolved,
        (
            LeftTruncatedDistribution,
            RightTruncatedDistribution,
            TwoSidedTruncatedDistribution,
        ),
    ):
        base = resolved.base_dist
        if not isinstance(base, dist.Normal):
            msg = (
                f"Linear prior for {name!r} truncates a "
                f"{type(base).__name__}; only a truncated Normal can be "
                "analytically marginalized."
            )
            raise TypeError(msg)
        loc, scale = conv(base.loc), conv(base.scale)
        low_fin = hasattr(resolved, "low")
        high_fin = hasattr(resolved, "high")
        low = conv(resolved.low) if low_fin else neg_inf
        high = conv(resolved.high) if high_fin else pos_inf
    else:
        msg = (
            f"Linear prior for {name!r} is a {type(resolved).__name__}, which "
            "cannot be analytically marginalized. Supported: Normal, "
            "HalfNormal, TruncatedNormal, and MixtureSameFamily over those "
            "(optionally wrapped in a QuantityDistribution)."
        )
        raise TypeError(msg)

    return _ParsedPrior(
        loc=jnp.atleast_1d(jnp.squeeze(loc)),
        scale=jnp.atleast_1d(jnp.squeeze(scale)),
        ln_weights=jnp.zeros(1),
        low=jnp.squeeze(low),
        high=jnp.squeeze(high),
        low_finite=low_fin,
        high_finite=high_fin,
    )


def _resolve_linear_priors(
    prior_dict: dict[str, PriorDist | LinearPriorCallable],
    nonlinear_values: dict[str, Any],
    unit_dict: dict[str, str],
    extra_values: dict[str, Any] | None = None,
    parameterization: Any | None = None,
) -> ResolvedLinearPrior:
    """Resolve per-parameter linear priors into one marginalizable prior.

    Callables are invoked here, against the nonlinear values plus any explicitly
    sampled linear values, then every entry is parsed into loc/scale/bounds in
    the model's units.

    When more than one parameter carries a mixture prior, the joint prior is the
    **Cartesian product** of their components with product weights -- the number
    of components is ``prod(K_i)``, which is static and becomes the batch axis of
    the marginalized likelihood. That is exact, but the cost grows
    multiplicatively, so putting mixtures on many parameters at once is
    expensive by construction rather than by accident.
    """
    # Values passed to any LinearPriorCallable. Include explicit linear values so
    # that callables depending on explicitly-sampled linear params (e.g. parallax)
    # can resolve.
    param_values = dict(nonlinear_values)
    if extra_values:
        param_values.update(extra_values)
    param_values = _with_derived_eccentricity(param_values, parameterization)

    names = tuple(prior_dict)
    parsed: list[_ParsedPrior] = []
    for name in names:
        prior = prior_dict[name]
        resolved = (
            cast("LinearPriorCallable", prior)(param_values)
            if _is_callable_prior(prior)
            else prior
        )
        parsed.append(_parse_linear_prior(resolved, unit_dict.get(name, ""), name))

    low = jnp.stack([p.low for p in parsed])
    high = jnp.stack([p.high for p in parsed])

    # A reversed or empty interval would otherwise show up only as every sample
    # being rejected, which is a confusing way to learn about a typo. Bounds may
    # be traced (a callable can return them), so this has to be ``error_if``
    # rather than a Python ``if`` -- see spec, Trace-friendly validation.
    finite_both = jnp.asarray(
        [p.low_finite and p.high_finite for p in parsed], dtype=bool
    )
    low = eqx.error_if(
        low,
        jnp.any(finite_both & (low >= high)),
        "A linear prior has an empty truncation interval (low >= high); "
        f"parameters in order: {names}",
    )

    combos = list(itertools.product(*(range(p.loc.shape[0]) for p in parsed)))
    loc = jnp.stack(
        [jnp.stack([parsed[i].loc[c[i]] for i in range(len(names))]) for c in combos]
    )
    scale = jnp.stack(
        [jnp.stack([parsed[i].scale[c[i]] for i in range(len(names))]) for c in combos]
    )
    ln_weights = jnp.stack(
        [sum(parsed[i].ln_weights[c[i]] for i in range(len(names))) for c in combos]
    )

    return ResolvedLinearPrior(
        loc=loc,
        scale=scale,
        ln_weights=ln_weights,
        low=low,
        high=high,
        low_finite=tuple(p.low_finite for p in parsed),
        high_finite=tuple(p.high_finite for p in parsed),
        names=names,
    )


def _explicit_scalar_dist(resolved: ResolvedLinearPrior) -> dist.Distribution:
    """A numpyro distribution for sampling ONE linear parameter explicitly.

    Used on the numpyro paths that draw a linear parameter as its own sample
    site instead of marginalizing it -- a pinned callable prior, or one the user
    excluded via ``marginalized_names``. Truncation is preserved (numpyro's
    truncated Normal has a ``biject_to``, so NUTS handles it), which matters
    because otherwise a positivity constraint would silently vanish on this path
    while being honoured on the marginalized one.
    """
    name = resolved.names[0]
    if resolved.n_components > 1:
        msg = (
            f"Linear prior for {name!r} is a Gaussian mixture, which cannot be "
            "sampled as a single explicit site. Marginalize it instead (leave "
            "it out of `marginalized_names`), where mixtures are supported."
        )
        raise NotImplementedError(msg)
    loc, scale = resolved.loc[0, 0], resolved.scale[0, 0]
    if not resolved.constrained:
        return dist.Normal(loc, scale)
    return dist.TruncatedNormal(
        loc,
        scale,
        low=resolved.low[0] if resolved.low_finite[0] else None,
        high=resolved.high[0] if resolved.high_finite[0] else None,
    )


def _explicit_joint_mvn(resolved: ResolvedLinearPrior) -> dist.MultivariateNormal:
    """The joint MVN for the non-marginalized numpyro model's ``_linear`` site.

    Only the untruncated single-Gaussian case is representable as one MVN.
    Truncated and mixture priors are rejected rather than quietly dropped: a
    truncated MVN is not a numpyro distribution, and silently sampling the
    untruncated one would return out-of-support linear values while the
    marginalized path honoured the bound.
    """
    if resolved.n_components > 1 or resolved.constrained:
        offenders = tuple(resolved.names[i] for i in resolved.constrained) or (
            resolved.names
        )
        msg = (
            "Non-marginalized numpyro models (`marginalized=False`) support only "
            "untruncated Gaussian linear priors; got a truncated or mixture "
            f"prior for {offenders}. Use `marginalized=True`, where both are "
            "supported analytically."
        )
        raise NotImplementedError(msg)
    return dist.MultivariateNormal(
        loc=resolved.loc[0], scale_tril=jnp.diag(resolved.scale[0])
    )
