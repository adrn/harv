"""Prior/model resolution helpers shared by the prior sampler and rejection sampler.

These functions walk a model's extensions to resolve which extension and linear
parameters need priors, and to classify which linear priors can be analytically
marginalized.  They are kept in their own module so both
:class:`~harv.models.priors.HarvPrior` (in :meth:`HarvPrior.sample`) and
:class:`~harv.samplers.RejectionSampler` can call them without forming an import
cycle.
"""

import warnings
from collections.abc import Mapping
from typing import Any

from harv.models._helpers import (
    LinearRole,
    _can_marginalize,
    _needs_explicit_sampling,
    classify_linear_prior,
    pinned_linear_names,
)
from harv.models.component import AbstractComponentModel
from harv.models.joint import JointModel
from harv.models.priors import HarvPrior

__all__ = (
    "effective_linear_prior_from_prior",
    "explicit_linear_names",
    "extension_model_key",
    "iter_component_extensions",
    "lookup_extension_prior",
    "nonlinear_extension_priors_from_model",
    "resolve_effective_marginalized_names",
    "validate_extension_priors",
)


def explicit_linear_names(
    effective_linear_prior: Mapping[str, Any],
    effective_marginalized_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Linear params the sampler draws explicitly (not analytically marginalized).

    When ``effective_marginalized_names`` is ``None`` the auto-classification
    decides (see :func:`~harv.models._helpers._needs_explicit_sampling`): a prior
    is explicit if it is a ``Delta``, if the marginalization cannot handle it, or
    if another prior's callable reads its value. When it is set, every linear
    param not in that set is explicit (even ones that could have been
    marginalized).

    Both :meth:`RejectionSampler._expected_prior_keys` and :meth:`HarvPrior.sample`
    call this so the explicit-linear key set stays consistent between the cache
    producer and consumer.
    """
    if effective_marginalized_names is None:
        pinned = pinned_linear_names(dict(effective_linear_prior))
        return tuple(
            name
            for name, d in effective_linear_prior.items()
            if _needs_explicit_sampling(d, name=name, pinned_names=pinned)
        )
    marg = set(effective_marginalized_names)
    return tuple(name for name in effective_linear_prior if name not in marg)


def lookup_extension_prior(
    extension_priors: Mapping[str, Any],
    param_name: str,
    *,
    component_name: str = "",
) -> Any | None:
    """Get an extension prior by component-qualified or bare parameter name."""
    if component_name:
        qualified_name = f"{component_name}.{param_name}"
        if qualified_name in extension_priors:
            return extension_priors[qualified_name]
    return extension_priors.get(param_name)


def iter_component_extensions(
    model: AbstractComponentModel | JointModel,
) -> list[tuple[str, Any]]:
    """Return ``(component_name, extension)`` pairs for a sampler model."""
    if isinstance(model, JointModel):
        return [
            (comp_name, ext)
            for comp_name, comp in model.components.items()
            for ext in comp.extensions
        ]
    return [("", ext) for ext in model.extensions]


def extension_model_key(component_name: str, param_name: str) -> str:
    """Return the flattened sampler/model key for an extension parameter."""
    return f"{component_name}.{param_name}" if component_name else param_name


def resolve_effective_marginalized_names(
    effective_linear_prior: dict[str, Any] | None,
    marginalized_names: tuple[str, ...] | None,
    *,
    verbose: bool = False,
) -> tuple[str, ...] | None:
    """Resolve and validate the effective marginalized linear parameter subset.

    A name the user asked to marginalize is honoured whenever the math allows it
    (:func:`~harv.models._helpers._can_marginalize`), which is a weaker test than
    the auto-mode default: a truncated ``parallax`` pinned by a callable stays
    explicit *by default*, but an explicit ``marginalized_names=("parallax",)``
    is respected -- and then the dependent callable raises its own ``KeyError``,
    which is the documented behaviour.

    Dropping genuinely non-marginalizable priors from the set is silent unless
    ``verbose=True``: it is expected behaviour, not a defect, and
    :meth:`~harv.samplers.RejectionSampler.summary` already reports the resulting
    per-parameter classification.
    """
    if effective_linear_prior is None:
        return marginalized_names

    if marginalized_names is not None:
        unknown = set(marginalized_names) - set(effective_linear_prior)
        if unknown:
            msg = (
                "marginalized_names contains unknown linear parameter(s): "
                f"{unknown}. Valid names: {tuple(effective_linear_prior.keys())}"
            )
            raise ValueError(msg)

    # Only check names the user actually wants to marginalize.
    names_to_check = (
        set(effective_linear_prior)
        if marginalized_names is None
        else set(marginalized_names)
    )

    pinned = pinned_linear_names(dict(effective_linear_prior))
    explicit = {
        name
        for name in names_to_check
        if (
            # An explicit request only needs the math to support it...
            not _can_marginalize(effective_linear_prior[name])
            if marginalized_names is not None
            # ...whereas auto mode also respects Delta and `requires` pinning.
            else _needs_explicit_sampling(
                effective_linear_prior[name], name=name, pinned_names=pinned
            )
        )
    }
    if not explicit:
        return marginalized_names

    if marginalized_names is None:
        resolved_names = tuple(
            name for name in effective_linear_prior if name not in explicit
        )
    else:
        resolved_names = tuple(
            name for name in marginalized_names if name not in explicit
        )

    if verbose:
        # The explicit set has two populations and they want different fixes.
        # Saying "cannot be analytically marginalized" about a pinned name is
        # simply false -- `parallax` under the Gaia defaults is a HalfNormal,
        # which this package marginalizes fine; it stays explicit only because
        # another prior's callable reads its sampled value. Both surfaces render
        # the role rather than re-deriving the precedence, so this warning and
        # `RejectionSampler.summary()` cannot disagree.
        roles = {
            n: classify_linear_prior(
                effective_linear_prior[n], name=n, pinned_names=pinned
            )
            for n in explicit
        }
        read_by_prior = sorted(n for n, r in roles.items() if r is LinearRole.PINNED)
        unmarginalizable = sorted(
            n for n, r in roles.items() if r is LinearRole.EXPLICIT
        )
        clauses = []
        if unmarginalizable:
            clauses.append(f"{unmarginalizable} cannot be analytically marginalized")
        if read_by_prior:
            clauses.append(
                f"{read_by_prior} is read by another prior's callable "
                "(see its `requires`), so it has no marginalized value to read"
            )
        warnings.warn(
            f"Linear prior(s) will be sampled explicitly: {' and '.join(clauses)}. "
            f"Marginalized parameters: {resolved_names}",
            stacklevel=3,
        )
    return resolved_names


def effective_linear_prior_from_prior(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
) -> dict[str, Any] | None:
    """Build the effective linear prior from prior.linear_priors + extensions.

    Models are templates and carry no ``linear_priors`` themselves.  The sampler
    computes it at run-time by merging ``prior.linear_priors`` with any
    linear-extension parameters declared on the model's extensions.
    """
    effective: dict[str, Any] | None = (
        dict(prior.linear_priors)
        if isinstance(prior.linear_priors, dict)
        else prior.linear_priors
    )
    if effective is None:
        return None
    # Merge linear extension params from the model.
    for comp_name, ext in iter_component_extensions(model):
        for p in ext.extra_params():
            if not p.linear:
                continue
            extension_prior = lookup_extension_prior(
                prior.extension_priors,
                p.name,
                component_name=comp_name,
            )
            if extension_prior is not None:
                effective[extension_model_key(comp_name, p.name)] = extension_prior
    return effective


def validate_extension_priors(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
    effective_linear_prior: dict[str, Any] | None,
) -> None:
    """Ensure every extension-declared parameter has a prior before sampling."""
    linear_names = (
        set(effective_linear_prior)
        if isinstance(effective_linear_prior, dict)
        else set()
    )
    missing: list[str] = []

    for comp_name, ext in iter_component_extensions(model):
        for p in ext.extra_params():
            model_key = extension_model_key(comp_name, p.name)
            if p.linear:
                if model_key not in linear_names:
                    missing.append(model_key)
                continue

            extension_prior = lookup_extension_prior(
                prior.extension_priors,
                p.name,
                component_name=comp_name,
            )
            if extension_prior is None:
                missing.append(model_key)

    if missing:
        msg = (
            "Missing required prior(s) for extension parameter(s): "
            f"{tuple(missing)}. Add priors for every parameter declared by "
            "model.extensions."
        )
        raise ValueError(msg)


def nonlinear_extension_priors_from_model(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Derive nonlinear extension priors and linear extension names from a model.

    Walks the model's extensions, looks up matching priors in
    ``prior.extension_priors``, and routes them by the ``linear`` flag on
    each param. Component-qualified names (for example ``"rv.jitter"``)
    are preferred when the model is a :class:`JointModel`, but bare names are
    accepted as a fallback for backward compatibility.

    Returns
    -------
    nonlinear_extension_priors : dict[str, PriorDist]
        Extension nonlinear params, keyed by model-key convention.
    linear_extension_names : tuple[str, ...]
        Names of extension linear (offset) params.
    """
    nonlinear_extension_priors: dict[str, Any] = {}
    linear_extension_names: list[str] = []

    for comp_name, ext in iter_component_extensions(model):
        for p in ext.extra_params():
            extension_prior = lookup_extension_prior(
                prior.extension_priors,
                p.name,
                component_name=comp_name,
            )
            if extension_prior is None:
                continue
            model_key = extension_model_key(comp_name, p.name)
            if p.linear:
                linear_extension_names.append(model_key)
            else:
                nonlinear_extension_priors[model_key] = extension_prior

    return nonlinear_extension_priors, tuple(linear_extension_names)
