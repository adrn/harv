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

from harv.models._helpers import _needs_explicit_sampling
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

    When ``effective_marginalized_names`` is ``None`` only non-Gaussian priors are
    explicit (decided by :func:`~harv.models._helpers._needs_explicit_sampling`).
    When it is set, every linear param not in that set is explicit (even Gaussian
    ones the user chose to sample rather than marginalize).

    Both :meth:`RejectionSampler._expected_prior_keys` and :meth:`HarvPrior.sample`
    call this so the explicit-linear key set stays consistent between the cache
    producer and consumer.
    """
    if effective_marginalized_names is None:
        return tuple(
            name
            for name, d in effective_linear_prior.items()
            if _needs_explicit_sampling(d)
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
) -> tuple[str, ...] | None:
    """Resolve and validate the effective marginalized linear parameter subset."""
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

    explicit = {
        name
        for name in names_to_check
        if _needs_explicit_sampling(effective_linear_prior[name])
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

    warnings.warn(
        f"Non-Gaussian linear prior(s) {sorted(explicit)} cannot be analytically "
        f"marginalized and will be sampled explicitly. Marginalized parameters: "
        f"{resolved_names}",
        stacklevel=3,
    )
    return resolved_names


def effective_linear_prior_from_prior(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
) -> dict[str, Any] | None:
    """Build the effective linear prior from prior.linear_priors + extensions.

    Models are templates and carry no ``linear_prior`` themselves.  The sampler
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
