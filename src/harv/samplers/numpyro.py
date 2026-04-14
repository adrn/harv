"""Numpyro model builders and MCMC sampler for harv.

This module provides the ``_ModelContext`` dataclass, the numpyro model builder
functions, and the ``NumpyroSampler`` class that wraps MCMC warm-start
initialization.  The builders share pre-computed state via
``_build_model_context`` to avoid duplicating setup logic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import infer as _numpyro_infer
from unxt import Q
from unxt.quantity import ustrip

from harv.distributions import QuantityDistribution
from harv.likelihood.composite import CompositeLikelihood
from harv.likelihood.helpers import _resolve_linear_prior_mvn, _unwrap_dist

if TYPE_CHECKING:
    from harv.model import Model
    from harv.samplers.rejection_prior import RejectionPrior

__all__ = ("NumpyroSampler",)

# ---------------------------------------------------------------------------
# Shared pre-computed context
# ---------------------------------------------------------------------------


def _iter_sub_likelihoods(lik: Any) -> tuple[tuple[str, Any], ...]:
    """Return ``(name, sub_lik)`` pairs for any likelihood.

    For a ``CompositeLikelihood`` this yields one entry per sub-likelihood.
    For a single-component likelihood it yields one ``("primary", lik)`` pair.
    """
    if isinstance(lik, CompositeLikelihood):
        return tuple(lik.components.items())
    return (("primary", lik),)


class _MergedParams:
    """Attribute-access proxy that merges multiple sub-params objects.

    For composite strategies, ``_sample_nonlinear`` returns a
    ``dict[str, MarginalizedParameters]``.  Callable linear priors (e.g.
    ``ParallaxDependentProperMotionPrior``) expect a single object with
    attribute access like ``params.parallax``.  This wrapper delegates
    attribute lookups to the first sub-params that has the requested field.
    """

    def __init__(self, sub_params: dict[str, Any]) -> None:
        object.__setattr__(self, "_sub_params", sub_params)

    def __getattr__(self, name: str) -> Any:
        for p in object.__getattribute__(self, "_sub_params").values():
            try:
                return getattr(p, name)
            except AttributeError:
                continue
        raise AttributeError(name)


@dataclasses.dataclass(frozen=True)
class _ModelContext:
    """Pre-computed shared state used by all numpyro model builders.

    Component-generic: data-type-specific details live in the ``Model`` and
    its likelihood object.
    """

    prior: "RejectionPrior"
    model: "Model"
    time_unit: str
    nonlinear_priors: dict[str, Any]
    lik: Any  # AbstractLikelihood or CompositeLikelihood
    all_linear_names: tuple[str, ...]
    linear_param_units: dict[str, str]


def _build_model_context(
    model: "Model",
) -> _ModelContext:
    """Pre-compute shared state for numpyro model builders from a Model."""
    return _ModelContext(
        prior=model.prior,
        model=model,
        time_unit=model.time_unit,
        nonlinear_priors=model.prior.nonlinear_priors,
        lik=model.likelihood,
        all_linear_names=model.all_linear_names,
        linear_param_units=model.linear_param_units,
    )


def _sample_nonlinear(
    ctx: _ModelContext,
) -> tuple[dict[str, Any], Any]:
    """Sample nonlinear parameters inside a numpyro model closure.

    Must be called within an active numpyro model context.  Returns the raw
    sampled values dict *and* params suitable for passing to
    ``lik.log_prob()``.  For single-component strategies this is a
    ``MarginalizedParameters``; for composite strategies it is a
    ``dict[str, MarginalizedParameters]``.

    When ``prior.marginalize_names`` is not ``None``, any linear params
    **not** listed are sampled explicitly from their numpyro priors and
    included in ``values``.
    """
    values: dict[str, Any] = {}
    for name, d in ctx.nonlinear_priors.items():
        values[name] = numpyro.sample(name, _unwrap_dist(d))

    # Convert period from prior unit to data time unit (when the prior
    # carries an explicit unit).  Bare-float priors are assumed to be
    # in the data's time unit already.
    _p_prior = ctx.prior.nonlinear_priors.get("period")
    if isinstance(_p_prior, QuantityDistribution):
        period_in_data_unit = ustrip(
            ctx.time_unit, Q(values["period"], str(_p_prior.unit))
        )
        values["period"] = period_in_data_unit

    # Sample explicit linear params (those not being marginalized).
    marg_names = ctx.prior.marginalize_names
    if isinstance(ctx.prior.linear_prior, dict) and marg_names is not None:
        marg_set = set(marg_names)
        for name, d in ctx.prior.linear_prior.items():
            if name not in marg_set:
                raw = numpyro.sample(name, _unwrap_dist(d))
                # Convert to data units if the prior carries a unit.
                target_u = ctx.linear_param_units.get(name, "")
                if isinstance(d, QuantityDistribution) and target_u:
                    raw = ustrip(target_u, Q(raw, str(d.unit)))
                values[name] = raw

    # Sample jitter parameters (if any)
    if ctx.prior.jitter_priors is not None:
        for dt_label, qd in ctx.prior.jitter_priors.items():
            jitter_key = f"_jitter_{dt_label}"
            values[jitter_key] = numpyro.sample(f"jitter_{dt_label}", _unwrap_dist(qd))

    params = ctx.model._build_params_raw(values)
    return values, params


# ---------------------------------------------------------------------------
# Numpyro model builders
# ---------------------------------------------------------------------------


def _component_jitter(
    comp_name: str, values: dict[str, Any], prior: "RejectionPrior"
) -> Any:
    """Return the jitter value for a likelihood component (0.0 if none)."""
    if prior.jitter_priors is None:
        return 0.0
    if comp_name == "primary":
        # Single-component model: use the sole jitter prior
        for dt_label in prior.jitter_priors:
            return values.get(f"_jitter_{dt_label}", 0.0)
        return 0.0
    return values.get(f"_jitter_{comp_name}", 0.0)


def _build_marginalized_numpyro_model(
    model: "Model",
) -> Callable[[], None]:
    """Build a marginalized numpyro model for MCMC.

    The returned callable samples each nonlinear parameter from its prior and
    evaluates the analytically-marginalized log-likelihood via ``numpyro.factor``.
    Linear parameters (rv_semiamp, v_sys, astrometric solution, etc.) are integrated out
    analytically; MCMC explores only the nonlinear subspace.

    Parameters
    ----------
    model : Model
        The model containing the prior, data, and pre-built likelihood.

    Returns
    -------
    model_fn : callable
        A numpyro model with no required arguments.
    """
    ctx = _build_model_context(model)

    def model_fn() -> None:
        _, params = _sample_nonlinear(ctx)
        numpyro.factor("log_lik", ctx.lik.log_prob(params))

    return model_fn


def _build_full_numpyro_model(
    model: "Model",
) -> Callable[[], None]:
    """Build a full (unmarginalized) numpyro model for MCMC.

    The returned callable samples both nonlinear and linear parameters explicitly.
    Linear parameters are sampled jointly as a single latent site ``"_linear"``
    from the prior's ``MultivariateNormal``.

    Parameters
    ----------
    model : Model
        The model containing the prior, data, and pre-built likelihood.

    Returns
    -------
    model_fn : callable
        A numpyro model with no required arguments.
    """
    ctx = _build_model_context(model)

    # Identify which linear priors are Gaussian (can go into the joint MVN)
    # vs. non-Gaussian (sampled as separate sites by _sample_nonlinear).
    marg_names = ctx.prior.marginalize_names
    if isinstance(ctx.prior.linear_prior, dict) and marg_names is not None:
        _marg_set = set(marg_names)
        _gaussian_lp = {
            n: d for n, d in ctx.prior.linear_prior.items() if n in _marg_set
        }
    elif isinstance(ctx.prior.linear_prior, dict):
        _gaussian_lp = dict(ctx.prior.linear_prior)  # all Gaussian
    else:
        _gaussian_lp = ctx.prior.linear_prior  # pre-built MVN
    _gaussian_names = (
        list(_gaussian_lp.keys())
        if isinstance(_gaussian_lp, dict)
        else ctx.all_linear_names
    )
    _gaussian_units = {n: ctx.linear_param_units[n] for n in _gaussian_names}

    def model_fn() -> None:
        values, params = _sample_nonlinear(ctx)

        # --- linear parameters ---
        # _sample_nonlinear already samples non-Gaussian linear priors (e.g.
        # HalfNormal parallax) as separate numpyro sites.  Build the joint
        # MVN only for the remaining Gaussian linear priors.
        if isinstance(params, dict):
            # Composite models: pre-resolve callable priors with a merged
            # namespace, then pass a real sub-params to satisfy the type hint.
            merged = _MergedParams(params)
            _lp: dict[str, Any] = {}
            for _name, _prior in _gaussian_lp.items():
                if callable(_prior) and not isinstance(
                    _prior, dist.Distribution | QuantityDistribution
                ):
                    _lp[_name] = _prior(merged)
                else:
                    _lp[_name] = _prior
            any_sub = next(iter(params.values()))
            resolved_lp = _resolve_linear_prior_mvn(_lp, any_sub, _gaussian_units)
        else:
            resolved_lp = _resolve_linear_prior_mvn(
                _gaussian_lp, params, _gaussian_units
            )
        linear_vec = numpyro.sample("_linear", resolved_lp)
        # Expose each Gaussian column as a named deterministic site.
        for i, lname in enumerate(_gaussian_names):
            numpyro.deterministic(lname, linear_vec[i])

        # --- data log-likelihood (component loop) ---
        log_lik: jax.Array = jnp.zeros(())

        for _comp_name, comp_lik in _iter_sub_likelihoods(ctx.lik):
            comp_names = tuple(comp_lik.linear_param_units.keys())
            # Assemble this component's linear values from _linear (Gaussian)
            # and values dict (explicit non-Gaussian, already sampled).
            comp_vals = []
            for n in comp_names:
                if n in _gaussian_names:
                    comp_vals.append(linear_vec[_gaussian_names.index(n)])
                else:
                    comp_vals.append(values[n])
            comp_linear = jnp.stack(comp_vals)

            sub_params = params[_comp_name] if isinstance(params, dict) else params
            dm = comp_lik.design_matrix(sub_params)
            prediction = dm @ comp_linear
            obs = ustrip(comp_lik.data._get_obs())
            err = ustrip(comp_lik.data._get_obs_err())
            jitter_val = _component_jitter(_comp_name, values, ctx.prior)
            err = jnp.sqrt(err**2 + jitter_val**2)
            log_lik = log_lik + dist.Normal(prediction, err).log_prob(obs).sum()

        numpyro.factor("log_lik", log_lik)

    return model_fn


def _build_extra_numpyro_model(
    model: "Model",
    extra_model_fn: Callable[[dict[str, Any]], dict[str, Any]],
    marginalized: bool,
) -> Callable[[], None]:
    """Build a numpyro model with an ``extra_model`` reparameterization.

    Allows users to replace specific linear parameters (e.g. ``rv_semiamp``) with
    deterministic functions of additional physically-motivated parameters
    (e.g. stellar masses and inclination).

    Parameters
    ----------
    model :
        The model containing the prior, data, and pre-built likelihood.
    extra_model_fn :
        Callable ``(pars: dict[str, scalar]) -> dict[str, scalar]``.
    marginalized :
        If ``True``, analytically marginalize the free linear parameters.
        If ``False``, sample them explicitly from their marginal prior.

    Returns
    -------
    model_fn : callable
        Numpyro model with no required arguments.
    """
    ctx = _build_model_context(model)

    def model_fn() -> None:
        # --- nonlinear parameters ---
        values, base_params = _sample_nonlinear(ctx)

        # --- extra model: sample physical params, get fixed linear values ---
        fixed_linear: dict[str, Any] = extra_model_fn(values)

        # Validate returned keys at trace time (string comparison is static).
        unknown = set(fixed_linear.keys()) - set(ctx.all_linear_names)
        if unknown:
            msg = (
                f"extra_model returned unknown linear parameter name(s): {unknown}. "
                f"Valid names for this data type: {ctx.all_linear_names}"
            )
            raise ValueError(msg)

        for name, val in fixed_linear.items():
            numpyro.deterministic(name, val)

        free_names = tuple(n for n in ctx.all_linear_names if n not in fixed_linear)

        if marginalized and free_names:
            # Analytically marginalize free linear params.
            params = ctx.model._build_params_with_fixed_linear_raw(
                values,
                fixed_linear,
            )
            numpyro.factor("log_lik", ctx.lik.log_prob(params))
            return

        # --- Component-loop fallback ---
        # Used when marginalized=False (sample free params explicitly) or
        # when all linear params are fixed (no marginalization needed).
        fixed_idx = [i for i, n in enumerate(ctx.all_linear_names) if n in fixed_linear]
        free_idx = [
            i for i, n in enumerate(ctx.all_linear_names) if n not in fixed_linear
        ]

        log_lik: jax.Array = jnp.zeros(())

        for comp_name, comp_lik in _iter_sub_likelihoods(ctx.lik):
            comp_names = tuple(comp_lik.linear_param_units.keys())
            global_indices = [ctx.all_linear_names.index(n) for n in comp_names]

            c_fixed_global = [i for i in global_indices if i in set(fixed_idx)]
            c_free_global = [i for i in global_indices if i in set(free_idx)]

            global_to_local_idx = {g: loc for loc, g in enumerate(global_indices)}
            c_fixed_local = [global_to_local_idx[i] for i in c_fixed_global]
            c_free_local = [global_to_local_idx[i] for i in c_free_global]

            dm = comp_lik.design_matrix(base_params)
            obs = ustrip(comp_lik.data._get_obs())
            err = ustrip(comp_lik.data._get_obs_err())
            jitter_val = _component_jitter(comp_name, values, ctx.prior)
            err = jnp.sqrt(err**2 + jitter_val**2)

            y = obs
            if c_fixed_local:
                fv = jnp.stack(
                    [fixed_linear[ctx.all_linear_names[i]] for i in c_fixed_global]
                )
                y = y - dm[:, jnp.array(c_fixed_local)] @ fv

            if c_free_local:
                # Build the marginal prior for this component's free params.
                c_free_names = [ctx.all_linear_names[i] for i in c_free_global]
                filtered_prior = {
                    k: v for k, v in ctx.prior.linear_prior.items() if k in c_free_names
                }
                filtered_units = {
                    k: v for k, v in ctx.linear_param_units.items() if k in c_free_names
                }
                free_mvn = _resolve_linear_prior_mvn(
                    filtered_prior, base_params, filtered_units
                )
                free_vals = numpyro.sample(
                    f"_{comp_name}_linear_free",
                    free_mvn,
                )
                for j, col in enumerate(c_free_global):
                    numpyro.deterministic(ctx.all_linear_names[col], free_vals[j])
                prediction = dm[:, jnp.array(c_free_local)] @ free_vals
                if c_fixed_local:
                    fv = jnp.stack(
                        [fixed_linear[ctx.all_linear_names[i]] for i in c_fixed_global]
                    )
                    prediction = prediction + dm[:, jnp.array(c_fixed_local)] @ fv
                log_lik = log_lik + dist.Normal(prediction, err).log_prob(obs).sum()
            else:
                log_lik = (
                    log_lik + dist.Normal(jnp.zeros_like(obs), err).log_prob(y).sum()
                )

        numpyro.factor("log_lik", log_lik)

    return model_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unconstrain_init_params(
    init_params: dict[str, Any],
    prior: RejectionPrior,
) -> dict[str, Any]:
    """Transform init_params from constrained to unconstrained space.

    Numpyro's HMC/NUTS operates in unconstrained space and applies
    ``biject_to(constraint)`` as the forward transform.  The init values
    we build from the rejection-sampler posterior are in *constrained*
    (natural) space, so we must apply the inverse transform before
    passing them to ``WarmStartMCMC``.
    """
    from numpyro.distributions import biject_to

    # Build a mapping: numpyro site name -> bare numpyro distribution.
    site_dists: dict[str, dist.Distribution] = {}
    for name, d in prior.nonlinear_priors.items():
        site_dists[name] = _unwrap_dist(d)
    if isinstance(prior.linear_prior, dict):
        for name, d in prior.linear_prior.items():
            if isinstance(d, (dist.Distribution, QuantityDistribution)):
                site_dists[name] = _unwrap_dist(d)
    if prior.jitter_priors is not None:
        for dt_label, qd in prior.jitter_priors.items():
            site_dists[f"jitter_{dt_label}"] = _unwrap_dist(qd)

    out: dict[str, Any] = {}
    for name, val in init_params.items():
        d = site_dists.get(name)
        if d is not None:
            transform = biject_to(d.support)
            out[name] = transform.inv(jnp.asarray(val))
        else:
            out[name] = val
    return out


# ---------------------------------------------------------------------------
# NumpyroSampler
# ---------------------------------------------------------------------------

# Lazy import to avoid circular import at module level.
# Resolved at instance creation time (after all modules are loaded).
from harv.samplers.samples import Samples, WarmStartMCMC  # noqa: E402


class NumpyroSampler(eqx.Module):
    """MCMC sampler for Keplerian orbital parameters using numpyro.

    Builds a numpyro model from a :class:`~harv.model.Model` and provides
    warm-started MCMC initialization from rejection-sampler output.

    Parameters
    ----------
    model : Model
        A pre-built model combining prior and data.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv import Model
    >>> from harv.samplers import RejectionPrior, RejectionSampler, NumpyroSampler
    >>> prior = RejectionPrior.default_rv(
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> model = Model(prior, rv_data)  # doctest: +SKIP
    >>> sampler = RejectionSampler(model)  # doctest: +SKIP
    >>> samples = sampler.run(n_prior_samples=100_000)  # doctest: +SKIP
    >>> mcmc_sampler = NumpyroSampler(model)  # doctest: +SKIP
    >>> mcmc = mcmc_sampler.init_mcmc(samples)  # doctest: +SKIP
    """

    model: Model

    def init_mcmc(
        self,
        samples: Samples,
        *,
        marginalized: bool = True,
        extra_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        extra_init_params: dict[str, Any] | None = None,
        kernel: type | None = None,
        num_chains: int = 4,
        **mcmc_kwargs: Any,
    ) -> WarmStartMCMC:
        """Construct a numpyro MCMC object warm-started from rejection-sampler output.

        Builds a numpyro model from this sampler's model and observed data,
        draws one starting position per chain from ``samples``, and returns a
        :class:`~harv.samplers.samples.WarmStartMCMC` whose
        :meth:`~harv.samplers.samples.WarmStartMCMC.run` injects those positions
        automatically.

        Three model variants are supported:

        - **Marginalized** (``marginalized=True``, default): MCMC explores only
          the nonlinear subspace; linear parameters are analytically marginalized.
        - **Full** (``marginalized=False``): all parameters sampled jointly.
        - **Extra model** (``extra_model`` provided): some linear parameters are
          replaced by deterministic functions of new physical parameters sampled
          inside ``extra_model``; the remaining linear parameters are either
          analytically marginalized (``marginalized=True``) or sampled from their
          marginal prior (``marginalized=False``).

        Parameters
        ----------
        samples : Samples
            Posterior samples produced by rejection sampling.  One sample per
            chain is used as the MCMC warm-start position.
        marginalized : bool, optional
            If ``True`` (default) use the analytically-marginalized likelihood
            for any linear parameters not provided by ``extra_model``.
            If ``False``, sample those parameters explicitly from their
            marginal prior.
        extra_model : callable, optional
            A function ``(pars: dict[str, scalar]) -> dict[str, scalar]`` that
            is called inside the numpyro model after the nonlinear parameters
            have been sampled.  ``pars`` contains the raw scalar nonlinear
            parameter values keyed by name (e.g. ``pars["period"]`` in the
            data's time unit, ``pars["eccentricity"]``, ...).  The function may
            call ``numpyro.sample`` for any number of new sites (e.g. stellar
            masses, inclination) and must return a dict mapping linear
            parameter names (e.g. ``"rv_semiamp"``) to their computed values.
            Any linear parameter not in the returned dict is handled by
            ``marginalized``.
        extra_init_params : dict, optional
            Initial values for the parameters introduced by ``extra_model``,
            one entry per chain.  Required when ``extra_model`` is provided.
        kernel : type, optional
            A numpyro MCMC kernel *class* (not an instance).
            Defaults to ``numpyro.infer.NUTS``.
        num_chains : int, optional
            Number of independent MCMC chains.  Default: 4.
        **mcmc_kwargs :
            Forwarded unchanged to ``numpyro.infer.MCMC``.

        Returns
        -------
        mcmc : WarmStartMCMC
            Configured MCMC wrapper.  Call ``mcmc.run(jr.key(seed))`` to
            begin sampling.

        Raises
        ------
        ValueError
            If there are no posterior samples, fewer samples than chains, or
            ``extra_model`` is provided without ``extra_init_params``.
        """
        if samples.n_samples == 0:
            msg = "Cannot initialise MCMC: no posterior samples available."
            raise ValueError(msg)
        if extra_model is not None and extra_init_params is None:
            msg = (
                "extra_init_params is required when extra_model is provided. "
                "Provide initial values for each parameter introduced by extra_model "
                "(one entry per chain, shape (num_chains,))."
            )
            raise ValueError(msg)

        if kernel is None:
            kernel = _numpyro_infer.NUTS

        prior = self.model.prior

        _broadcast = samples.n_samples < num_chains
        _scalar_init = num_chains == 1 or _broadcast
        indices = list(range(min(num_chains, samples.n_samples)))
        if _scalar_init:
            init_params: dict[str, Any] = {
                key_name: jnp.asarray(ustrip(str(qty.unit), qty)[0])
                for key_name, qty in samples.nonlinear.items()
            }
        else:
            init_params = {
                key_name: jnp.stack([ustrip(str(qty.unit), qty)[i] for i in indices])
                for key_name, qty in samples.nonlinear.items()
            }

        # Include init values for explicit (non-marginalized) linear params.
        marg_names = prior.marginalize_names
        if marg_names is not None and isinstance(prior.linear_prior, dict):
            marg_set = set(marg_names)
            for name, d in prior.linear_prior.items():
                if name not in marg_set and name in samples.linear:
                    qty = samples.linear[name]
                    if isinstance(d, QuantityDistribution):
                        prior_unit = str(d.unit)
                        vals = ustrip(prior_unit, qty)
                    else:
                        vals = np.asarray(qty.value)
                    if _scalar_init:
                        init_params[name] = jnp.asarray(vals[0])
                    else:
                        init_params[name] = jnp.stack(
                            [jnp.asarray(vals[i]) for i in indices]
                        )

        if extra_model is not None:
            numpyro_model = _build_extra_numpyro_model(
                self.model, extra_model, marginalized
            )
            init_params.update(extra_init_params)  # type: ignore[arg-type]
        elif marginalized:
            numpyro_model = _build_marginalized_numpyro_model(self.model)
        else:
            numpyro_model = _build_full_numpyro_model(self.model)
            if isinstance(prior.linear_prior, dict):
                if marg_names is not None:
                    _marg_set = set(marg_names)
                else:
                    _marg_set = set(prior.linear_prior.keys())
                lin_names = [n for n in samples.linear if n in _marg_set]
            else:
                lin_names = list(samples.linear.keys())
            if lin_names:
                lin_arr = np.column_stack(
                    [np.asarray(samples.linear[n].value) for n in lin_names]
                )
                if _scalar_init:
                    init_params["_linear"] = jnp.asarray(lin_arr[0])
                else:
                    init_params["_linear"] = jnp.stack(
                        [jnp.asarray(lin_arr[i]) for i in indices]
                    )

        init_params = _unconstrain_init_params(init_params, prior)

        kernel_instance = kernel(numpyro_model)
        return WarmStartMCMC(
            kernel_instance,
            _init_params=init_params,
            num_chains=num_chains,
            **mcmc_kwargs,
        )
