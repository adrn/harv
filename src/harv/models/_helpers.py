"""Internal helpers shared across the models subpackage."""

__all__: tuple[str, ...] = ()

import itertools
from collections.abc import Callable
from enum import Enum
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import numpyro
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


class LinearRole(Enum):
    """How the samplers will treat one linear parameter.

    One value per parameter, decided by :func:`classify_linear_prior`. Having a
    single classifier rather than a family of booleans is deliberate: the
    *precedence* between these cases is the part that is easy to get wrong, and
    it now lives in exactly one ``if`` chain. ``RejectionSampler.summary()`` and
    the ``verbose=True`` advisory both render this enum, so the two
    introspection surfaces cannot disagree.
    """

    MARGINALIZED = "marginalized"
    """Integrated out analytically."""

    PINNED = "pinned"
    """Marginalizable, but another prior's callable reads its sampled value."""

    FIXED = "fixed"
    """A ``Delta``: the value is fixed rather than integrated or sampled."""

    EXPLICIT = "explicit"
    """Outside the family the likelihood can integrate out."""


def classify_linear_prior(
    d: LinearPriorDist,
    *,
    name: str = "",
    pinned_names: frozenset[str] = frozenset(),
) -> LinearRole:
    """Classify one linear prior into the role the samplers will give it.

    The order of the tests is the whole content of this function:

    1. A prior outside the Gaussian family cannot be integrated out at all, so
       it is ``EXPLICIT`` **first**. Testing ``pinned`` before this would label a
       ``Gamma`` prior on a pinned parameter as "read by another prior", whose
       documented fix -- drop the dependency -- would change nothing.
    2. A name another callable reads is ``PINNED``: a marginalized parameter has
       no sampled value for the callable to read, so it must stay explicit (see
       :func:`pinned_linear_names`). This is the only exception to "marginalize
       whatever you can".
    3. A ``Delta`` is ``FIXED``. It joins the marginalized set and is then
       reclassified, with its value extracted, by
       ``AbstractComponentModel._handle_delta_priors``; short-circuiting it to
       explicit here would skip that extraction and leave the value unset. Under
       ``marginalized=False`` it becomes a ``numpyro.deterministic`` site.
    4. Everything else is ``MARGINALIZED``: a ``Normal``, a truncated Normal
       (including ``HalfNormal``), a Gaussian mixture, or a callable returning
       one of those. Truncated and mixture priors became marginalizable when
       :mod:`harv.stats.marginalized` was added; before that only plain Normals
       were.

    Passing no ``name``/``pinned_names`` answers the question "*can* the math
    integrate this out", ignoring the pinning policy -- which is what an explicit
    ``marginalized_names`` request is gated on.
    """
    inner = d.distribution if isinstance(d, QuantityDistribution) else d
    marginalizable = isinstance(inner, (*_MARGINALIZABLE, dist.Delta)) or (
        # A callable is resolved at evaluation time; assume it returns something
        # marginalizable and let ``_parse_linear_prior`` complain if it does not.
        _is_callable_prior(d)
    )
    if not marginalizable:
        return LinearRole.EXPLICIT
    if name and name in pinned_names:
        return LinearRole.PINNED
    if isinstance(inner, dist.Delta):
        return LinearRole.FIXED
    return LinearRole.MARGINALIZED


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


def _needs_explicit_sampling(
    d: PriorDist | LinearPriorCallable,
    *,
    name: str = "",
    pinned_names: frozenset[str] = frozenset(),
) -> bool:
    """True if a linear prior entry must be sampled explicitly by the sampler.

    The *auto-mode default*: the two roles that cannot be integrated out are
    ``EXPLICIT`` and ``PINNED``. ``FIXED`` answers ``False`` because it is
    reclassified later with its value extracted; see
    :func:`classify_linear_prior`. A user may override this per sampler via
    ``marginalized_names``, which is gated on the role alone.
    """
    role = classify_linear_prior(d, name=name, pinned_names=pinned_names)
    return role in (LinearRole.EXPLICIT, LinearRole.PINNED)


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


def _parse_linear_prior(
    resolved: Any, target_unit: str, name: str, *, _batched: bool = False
) -> _ParsedPrior:
    """Parse one resolved linear prior into loc/scale/bounds in ``target_unit``.

    Accepts a ``Normal``, a ``HalfNormal``, any of numpyro's three truncated
    flavours, or a ``MixtureSameFamily`` over those -- each optionally wrapped in
    a :class:`~harv.distributions.QuantityDistribution` to declare its unit.

    ``HalfNormal(s)`` is parsed as ``TruncatedNormal(0, s, low=0)`` rather than
    special-cased: they are the same distribution, and the truncation machinery
    needs no help to handle it.

    ``_batched`` is set only by the mixture branch's recursive call, where a
    batched ``loc``/``scale`` *is* the component axis. Everywhere else a batch
    is a mistake: one linear parameter gets one prior, and quietly reading a
    length-``C`` batch as ``C`` equally-weighted components would inflate the
    total prior weight to ``C``.
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
        parsed = _parse_linear_prior(wrapped, target_unit, name, _batched=True)
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

    loc, scale = jnp.atleast_1d(jnp.squeeze(loc)), jnp.atleast_1d(jnp.squeeze(scale))
    if not _batched and (loc.size != 1 or scale.size != 1):
        msg = (
            f"Linear prior for {name!r} is batched (loc/scale of shape "
            f"{tuple(loc.shape)}/{tuple(scale.shape)}), but one linear "
            "parameter takes one scalar prior. Use `dist.MixtureSameFamily` "
            "for a mixture, which carries explicit weights."
        )
        raise ValueError(msg)

    return _ParsedPrior(
        loc=loc,
        scale=scale,
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


def _sample_explicit_linear_prior(
    name: str,
    prior_dist: Any,
    target_unit: str,
    nonlinear_values: dict[str, Any],
    extra_values: dict[str, Any] | None = None,
    *,
    site_name: str | None = None,
    parameterization: Any | None = None,
) -> jax.Array:
    """Sample one explicit (non-marginalized) linear prior in a numpyro model.

    Unifies the cases that would otherwise need separate code paths:

    * Plain ``dist.Distribution`` or :class:`QuantityDistribution` priors are sampled
      directly via ``numpyro.sample``; if a ``QuantityDistribution`` is provided the
      result is unit-stripped to ``target_unit``.
    * Callable priors (e.g. :class:`PeriodDependentKPrior`) are resolved at the
      current ``nonlinear_values`` / ``extra_values`` and then sampled. The
      resolver returns values already expressed in ``target_unit``, so no further
      unit-strip is performed, and truncation survives -- which is what lets a
      callable declaring ``support="positive"`` be sampled on the
      ``marginalized=False`` path instead of being forced into the joint
      ``_linear`` MVN, where a truncated Gaussian has no representation.
    * ``Delta`` priors are recorded with ``numpyro.deterministic``: the value is
      fixed, not distributed.

    Parameters
    ----------
    name
        Site name passed to ``numpyro.sample``.
    prior_dist
        The prior specification.
    target_unit
        Unit string the returned value must be expressed in.  ``""`` for
        dimensionless.
    nonlinear_values
        Already-sampled nonlinear (and previously-sampled explicit-linear)
        values, keyed by bare parameter name.  Used by callable priors.
    extra_values
        Optional ``Q``-wrapped versions of values that callable priors may
        consume to evaluate unit-aware dependencies; ``None`` when no such
        values are needed (e.g. for shared explicit-linear priors).
    site_name
        Optional site name to use within numpyro.
    parameterization
        Parameterization the values came from, used to derive ``eccentricity`` for
        callable priors that need it.  ``None`` for *shared* joint priors, where the
        components need not agree on a parameterization and there is no unambiguous
        answer; per-component priors pass their own.

    Returns
    -------
        The sampled value, unit-stripped to ``target_unit``.
    """
    _site = site_name if site_name is not None else name

    # A ``Delta`` prior fixes the value rather than distributing it. Sampling it
    # as a site would hand NUTS a log-prob that is -inf almost everywhere --
    # stuck chains, no error -- so it is recorded as what the classification
    # table calls it, *Fixed* (see ``docs/spec.md`` -> Linear prior
    # classification). The marginalized path reaches the same place through
    # ``AbstractComponentModel._handle_delta_priors``.
    inner = _unwrap_dist(prior_dist) if not _is_callable_prior(prior_dist) else None
    if isinstance(inner, dist.Delta):
        value = jnp.asarray(inner.v)
        if isinstance(prior_dist, QuantityDistribution) and target_unit:
            value = jnp.asarray(
                ustrip(target_unit, Q(value, cast("str", prior_dist.unit)))
            )
        return cast("jax.Array", numpyro.deterministic(_site, value))

    if _is_callable_prior(prior_dist):
        resolved = _resolve_linear_priors(
            {name: prior_dist},
            nonlinear_values,
            {name: target_unit},
            extra_values=extra_values,
            parameterization=parameterization,
        )
        return cast(
            "jax.Array",
            numpyro.sample(_site, _explicit_scalar_dist(resolved)),
        )

    # Direct sampling for plain Distribution / QuantityDistribution priors.
    raw = cast("jax.Array", numpyro.sample(_site, _unwrap_dist(prior_dist)))
    if isinstance(prior_dist, QuantityDistribution) and target_unit:
        raw = jnp.asarray(ustrip(target_unit, Q(raw, cast("str", prior_dist.unit))))
    return raw
