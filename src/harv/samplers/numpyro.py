"""Numpyro model builders for the rejection sampler.

This module provides the ``_ModelContext`` dataclass and the three numpyro
model builder functions used by ``RejectionSampler.init_mcmc``.  The builders
share pre-computed state via ``_build_model_context`` to avoid duplicating
setup logic.
"""

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from unxt import Q
from unxt.quantity import ustrip

from harv.data import InputData
from harv.distributions import QuantityDistribution
from harv.likelihood.helpers import _resolve_linear_prior_mvn, _unwrap_dist
from harv.samplers.strategies import DataTypeStrategy

if TYPE_CHECKING:
    from harv.samplers.rejection_prior import RejectionPrior

# ---------------------------------------------------------------------------
# Shared pre-computed context
# ---------------------------------------------------------------------------


def _iter_sub_likelihoods(lik: Any) -> tuple[tuple[str, Any], ...]:
    """Return ``(name, sub_lik)`` pairs for any likelihood.

    For a ``CompositeLikelihood`` this yields one entry per sub-likelihood.
    For a single-component likelihood it yields one ``("primary", lik)`` pair.
    """
    from harv.likelihood.composite import CompositeLikelihood

    if isinstance(lik, CompositeLikelihood):
        return tuple(lik.components.items())
    return (("primary", lik),)


@dataclasses.dataclass(frozen=True)
class _ModelContext:
    """Pre-computed shared state used by all numpyro model builders.

    Component-generic: data-type-specific details live in the likelihood
    object itself.  Adding a new data type requires only a new strategy --
    no changes here.
    """

    prior: "RejectionPrior"
    strategy: DataTypeStrategy
    time_unit: str
    nonlinear_priors: dict[str, Any]
    lik: Any  # AbstractLikelihood or CompositeLikelihood
    all_linear_names: tuple[str, ...]
    linear_param_units: dict[str, str]


def _build_model_context(
    sampler: Any,  # RejectionSampler (avoids circular import)
    data: InputData,
) -> _ModelContext:
    """Pre-compute shared state for numpyro model builders.

    Extracts the prior, strategy, data splits, time unit, likelihood,
    component infos, linear parameter names, and unit info -- all the setup
    that every builder needs (or a superset thereof).
    """
    prior = sampler.prior
    strategy = sampler._infer_strategy(data)
    datasets = strategy.extract_data(data)
    _ref = next(iter(datasets.values()))
    time_unit = str(_ref.time.unit)
    nonlinear_priors = prior.nonlinear_priors

    all_linear_names = strategy.all_linear_names(prior, data)

    # Build likelihood and derive linear-param units for the model builders.
    lik = strategy.build_likelihood(datasets, prior, data)
    lp_units = lik.linear_param_units

    return _ModelContext(
        prior=prior,
        strategy=strategy,
        time_unit=time_unit,
        nonlinear_priors=nonlinear_priors,
        lik=lik,
        all_linear_names=all_linear_names,
        linear_param_units=lp_units,
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

    params = ctx.strategy.build_marginalized_params(
        values, ctx.time_unit, ctx.prior.marginalize_names, ctx.linear_param_units
    )
    return values, params


# ---------------------------------------------------------------------------
# Numpyro model builders
# ---------------------------------------------------------------------------


def _build_marginalized_numpyro_model(
    sampler: Any,
    data: InputData,
) -> Callable[[], None]:
    """Build a marginalized numpyro model for MCMC.

    The returned callable samples each nonlinear parameter from its prior and
    evaluates the analytically-marginalized log-likelihood via ``numpyro.factor``.
    Linear parameters (rv_semiamp, v_sys, astrometric solution, etc.) are integrated out
    analytically; MCMC explores only the nonlinear subspace.

    Parameters
    ----------
    sampler : RejectionSampler
        The rejection sampler whose prior is used for the model.
    data : AbstractData or SourceData
        Observed data.  The data type determines which marginalized likelihood
        class is instantiated.

    Returns
    -------
    model : callable
        A numpyro model with no required arguments.  Sample sites: the keys of
        ``sampler.prior.nonlinear_priors`` (e.g. ``"period"``, ``"eccentricity"``).
    """
    ctx = _build_model_context(sampler, data)

    def model() -> None:
        _, params = _sample_nonlinear(ctx)
        numpyro.factor("log_lik", ctx.lik.log_prob(params))

    return model


def _build_full_numpyro_model(
    sampler: Any,
    data: InputData,
) -> Callable[[], None]:
    """Build a full (unmarginalized) numpyro model for MCMC.

    The returned callable samples both nonlinear and linear parameters explicitly.
    Linear parameters are sampled jointly as a single latent site ``"_linear"``
    from the prior's ``MultivariateNormal`` (so the correlation structure of the
    prior is preserved), then exposed as named ``deterministic`` sites (e.g.
    ``"rv_semiamp"``, ``"v_sys"``) for convenient access via ``get_samples()``.  The
    Gaussian data log-likelihood is evaluated directly at the sampled values.

    Parameters
    ----------
    sampler : RejectionSampler
        The rejection sampler whose prior is used for the model.
    data : AbstractData or SourceData
        Observed data.  The data type determines the design matrix and noise
        model used for the likelihood.

    Returns
    -------
    model : callable
        A numpyro model with no required arguments.  Sample sites: keys of
        ``sampler.prior.nonlinear_priors`` plus ``"_linear"`` (the joint linear
        vector).  Deterministic sites: individual linear parameter names (e.g.
        ``"rv_semiamp"``, ``"v_sys"``, ``"semi_major_axis"``).
    """
    ctx = _build_model_context(sampler, data)

    def model() -> None:
        _, params = _sample_nonlinear(ctx)

        # --- linear parameters ---
        # Sample the full vector jointly to preserve the prior's covariance.
        resolved_lp = _resolve_linear_prior_mvn(
            ctx.prior.linear_prior, params, ctx.linear_param_units
        )
        linear_vec = numpyro.sample("_linear", resolved_lp)
        # Expose each column as a named deterministic site.
        for i, lname in enumerate(ctx.all_linear_names):
            numpyro.deterministic(lname, linear_vec[i])

        # --- data log-likelihood (component loop) ---
        log_lik: jax.Array = jnp.zeros(())

        for _comp_name, comp_lik in _iter_sub_likelihoods(ctx.lik):
            comp_names = tuple(comp_lik.linear_param_units.keys())
            global_indices = jnp.array(
                [ctx.all_linear_names.index(n) for n in comp_names]
            )
            dm = comp_lik.design_matrix(params)
            prediction = dm @ linear_vec[global_indices]
            obs = ustrip(comp_lik.data._get_obs())
            err = ustrip(comp_lik.data._get_obs_err())
            log_lik = log_lik + dist.Normal(prediction, err).log_prob(obs).sum()

        numpyro.factor("log_lik", log_lik)

    return model


def _build_extra_numpyro_model(
    sampler: Any,
    data: InputData,
    extra_model_fn: Callable[[dict[str, Any]], dict[str, Any]],
    marginalized: bool,
) -> Callable[[], None]:
    """Build a numpyro model with an ``extra_model`` reparameterization.

    Allows users to replace specific linear parameters (e.g. ``rv_semiamp``) with
    deterministic functions of additional physically-motivated parameters
    (e.g. stellar masses and inclination).  ``extra_model_fn`` is called
    inside the numpyro model after the nonlinear parameters have been
    sampled; it may call ``numpyro.sample`` for any number of new sites and
    must return a dict mapping linear parameter names to their computed values.

    Linear parameters *not* returned by ``extra_model_fn`` are handled
    according to ``marginalized``:

    - ``True``: analytically marginalized over the residual observations
      ``y - D_fixed @ fixed_vals``, using the marginal prior extracted from
      ``sampler.prior.linear_prior``.
    - ``False``: sampled explicitly as a joint latent site
      ``"_linear_free"``; each component is also exposed as a named
      ``deterministic`` site.

    Parameters
    ----------
    sampler :
        Rejection sampler providing the prior and strategy.
    data :
        Observed data.
    extra_model_fn :
        Callable ``(pars: dict[str, scalar]) -> dict[str, scalar]``.
        ``pars`` contains the already-sampled nonlinear parameter values
        keyed by name (e.g. ``pars["period"]`` in the data's time unit,
        ``pars["eccentricity"]``, ...).  The callable may call
        ``numpyro.sample`` internally.  It must return a dict whose keys
        are a subset of the linear parameter names for this data type
        (e.g. ``"rv_semiamp"`` or ``"v_sys"`` for RV data).
    marginalized :
        If ``True``, analytically marginalize the free linear parameters.
        If ``False``, sample them explicitly from their marginal prior.

    Returns
    -------
    model : callable
        Numpyro model with no required arguments.

    Notes
    -----
    The ``pars`` dict passed to ``extra_model_fn`` uses raw scalar values in
    the same units as the prior.  In particular, ``pars["period"]`` is in the
    time unit of the input data (e.g. days if ``data.time`` is in days).

    Example -- replace ``rv_semiamp`` with a mass-function reparameterization::

        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        # Semi-amplitude constant: rv_semiamp [km/s] = K_FACTOR * f(masses, inc, P, e)
        # (Lovis & Fischer 2010, converted to km/s with period in days)
        _K_FACTOR = 28.4329  # km/s * day^(1/3) * M_sun^(-1/3)

        def K_from_masses(m1, m2, inc, period_days, ecc):
            return (
                _K_FACTOR
                * (m2 * jnp.sin(inc))
                * (m1 + m2) ** (-2.0 / 3.0)
                * (period_days / 365.25) ** (-1.0 / 3.0)
                / jnp.sqrt(1.0 - ecc**2)
            )

        def mass_model(pars):
            m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
            m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
            inc = numpyro.sample("inc", dist.Uniform(0.0, jnp.pi / 2))
            K   = K_from_masses(m1, m2, inc,
                                 pars["period"], pars["eccentricity"])
            return {"rv_semiamp": K}

    With this ``extra_model_fn``, ``rv_semiamp`` becomes a deterministic site in
    ``get_samples()``; ``v_sys`` is analytically marginalized (if
    ``marginalized=True``) or sampled from its marginal prior.
    """
    ctx = _build_model_context(sampler, data)

    def model() -> None:
        # --- nonlinear parameters ---
        values, base_params = _sample_nonlinear(ctx)

        # --- extra model: sample physical params, get fixed linear values ---
        # ``values`` contains raw scalar nonlinear parameters; period is in
        # ``time_unit`` (the unit of data.time).
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
            # Analytically marginalize free linear params by delegating to
            # the likelihood's _build_marginalized_linear infrastructure.
            # Fixed values are stored as non-marginalized fields (Quantities);
            # the likelihood classifies them as "explicit" and subtracts
            # their contribution before building MarginalizedLinear.
            params = ctx.strategy.build_params_with_fixed_linear(
                values, fixed_linear, ctx.linear_param_units, ctx.time_unit
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

    return model
