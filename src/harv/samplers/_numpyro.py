"""Numpyro model builders for the rejection sampler.

This module provides the ``_ModelContext`` dataclass and the three numpyro
model builder functions used by ``RejectionSampler.init_mcmc``.  The builders
share pre-computed state via ``_build_model_context`` to avoid duplicating
setup logic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Quantity, ustrip

from harv.data import (
    GaiaAstrometryData,
    InputData,
    RadialVelocityData,
    SourceData,
)
from harv.likelihood._params import (
    GaiaAstrometryParameters,
    MarginalizedParameters,
)
from harv.likelihood.combined import CompositeLikelihood
from harv.likelihood.helpers import _resolve_linear_prior
from harv.samplers._strategies import _DataTypeStrategy

if TYPE_CHECKING:
    from harv.priors.rejection import RejectionPrior


# ---------------------------------------------------------------------------
# Shared pre-computed context
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ModelContext:
    """Pre-computed shared state used by all numpyro model builders."""

    prior: "RejectionPrior"
    strategy: _DataTypeStrategy
    astro_data: GaiaAstrometryData | None
    rv_data: RadialVelocityData | None
    time_unit: str
    nonlinear_cls: type
    nonlinear_priors: dict[str, Any]
    lik: Any  # AbstractLikelihood or CompositeLikelihood
    astro_lik: Any  # component likelihood or None
    rv_lik: Any  # component likelihood or None
    n_astro: int
    all_linear_names: tuple[str, ...]
    astro_obs: jax.Array | None
    astro_err: jax.Array | None
    rv_obs: jax.Array | None
    rv_err: jax.Array | None


def _build_model_context(
    sampler: Any,  # RejectionSampler (avoids circular import)
    data: InputData,
) -> _ModelContext:
    """Pre-compute shared state for numpyro model builders.

    Extracts the prior, strategy, data splits, time unit, likelihood
    components, linear parameter names, and unit-stripped data arrays — all
    the setup that every builder needs (or a superset thereof).
    """
    prior = sampler.prior
    strategy = sampler._infer_strategy(data)
    astro_data, rv_data = strategy.extract_data(data)
    time_unit = str(
        astro_data.time.unit if astro_data is not None else rv_data.time.unit  # type: ignore[union-attr]
    )
    nonlinear_cls = strategy.nonlinear_cls
    nonlinear_priors = prior.nonlinear_priors

    # Linear parameter names, in the same column order as samples._linear.
    all_linear_names: tuple[str, ...] = sum(
        (cls.linear_param_names for cls in strategy.full_cls),  # type: ignore[attr-defined]
        (),
    )
    offset_names: tuple[str, ...] = ()
    if prior.offsets is not None and isinstance(data, SourceData) and data.n_rv() > 1:
        offset_names = tuple(k for k, v in prior.offsets.items() if v is not None)
    all_linear_names = all_linear_names + offset_names

    # Build likelihood + extract per-component objects for DM construction.
    lik = strategy.build_likelihood(astro_data, rv_data, prior, data)
    if isinstance(lik, CompositeLikelihood):
        astro_lik = lik["astro"]
        rv_lik = lik["rv"]
    else:
        astro_lik = lik if astro_data is not None else None
        rv_lik = lik if rv_data is not None else None

    # Slice boundary for combined data (astro columns come first).
    n_astro = (
        len(GaiaAstrometryParameters.linear_param_names)
        if astro_data is not None
        else 0
    )

    # Pre-strip units from data arrays (outside the model closure for efficiency).
    astro_obs = astro_err = rv_obs = rv_err = None
    if astro_data is not None:
        astro_obs = ustrip(str(astro_data.al_position.unit), astro_data.al_position)
        astro_err = ustrip(str(astro_data.al_position.unit), astro_data.al_position_err)
    if rv_data is not None:
        rv_obs = ustrip(str(rv_data.rv.unit), rv_data.rv)
        rv_err = ustrip(str(rv_data.rv.unit), rv_data.rv_err)

    return _ModelContext(
        prior=prior,
        strategy=strategy,
        astro_data=astro_data,
        rv_data=rv_data,
        time_unit=time_unit,
        nonlinear_cls=nonlinear_cls,
        nonlinear_priors=nonlinear_priors,
        lik=lik,
        astro_lik=astro_lik,
        rv_lik=rv_lik,
        n_astro=n_astro,
        all_linear_names=all_linear_names,
        astro_obs=astro_obs,
        astro_err=astro_err,
        rv_obs=rv_obs,
        rv_err=rv_err,
    )


def _sample_nonlinear(
    ctx: _ModelContext,
) -> tuple[dict[str, Any], MarginalizedParameters]:
    """Sample nonlinear parameters inside a numpyro model closure.

    Must be called within an active numpyro model context.  Returns the raw
    sampled values dict *and* a ``MarginalizedParameters`` instance suitable
    for passing to ``lik.design_matrix()`` or ``lik.log_prob()``.
    """
    import numpyro

    values: dict[str, Any] = {}
    for name, d in ctx.nonlinear_priors.items():
        values[name] = numpyro.sample(name, d)

    orbit_kwargs = {k: v for k, v in values.items() if k != "period"}
    orbit_kwargs["period"] = Quantity(values["period"], ctx.time_unit)
    params = ctx.nonlinear_cls.marginalized(**orbit_kwargs)
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
    Linear parameters (K, v0, astrometric solution, etc.) are integrated out
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
    import numpyro

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
    ``"K"``, ``"v0"``) for convenient access via ``get_samples()``.  The
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
        ``"K"``, ``"v0"``, ``"semi_major_axis"``).
    """
    import numpyro

    ctx = _build_model_context(sampler, data)

    def model() -> None:
        _, params = _sample_nonlinear(ctx)

        # --- linear parameters ---
        # Sample the full vector jointly to preserve the prior's covariance.
        resolved_lp = _resolve_linear_prior(ctx.prior.linear_prior, params)
        linear_vec = numpyro.sample("_linear", resolved_lp)
        # Expose each column as a named deterministic site.
        for i, lname in enumerate(ctx.all_linear_names):
            numpyro.deterministic(lname, linear_vec[i])

        # --- data log-likelihood ---
        log_lik: jax.Array = jnp.zeros(())

        if ctx.astro_lik is not None:
            dm = ctx.astro_lik.design_matrix(params)
            prediction = dm @ linear_vec[: ctx.n_astro]
            log_lik = (
                log_lik
                + dist.Normal(prediction, ctx.astro_err).log_prob(ctx.astro_obs).sum()
            )

        if ctx.rv_lik is not None:
            dm = ctx.rv_lik.design_matrix(params)
            prediction = dm @ linear_vec[ctx.n_astro :]
            log_lik = (
                log_lik + dist.Normal(prediction, ctx.rv_err).log_prob(ctx.rv_obs).sum()
            )

        numpyro.factor("log_lik", log_lik)

    return model


def _marginal_mvn(
    mvn: dist.MultivariateNormal,
    indices: list[int],
) -> dist.MultivariateNormal:
    """Extract the marginal ``MultivariateNormal`` for the given column indices.

    Parameters
    ----------
    mvn :
        Joint multivariate normal distribution.
    indices :
        Column indices of the parameters to retain.  Must be a Python list
        (static at JAX trace time) so the indexing is resolved at trace time.

    Returns
    -------
    dist.MultivariateNormal
        Marginal distribution over the selected parameters.
    """
    idx = jnp.array(indices)
    cov = mvn.scale_tril @ mvn.scale_tril.T
    return dist.MultivariateNormal(
        loc=mvn.loc[idx],
        covariance_matrix=cov[idx][:, idx],
    )


def _build_extra_numpyro_model(
    sampler: Any,
    data: InputData,
    extra_model_fn: Callable[[dict[str, Any]], dict[str, Any]],
    marginalized: bool,
) -> Callable[[], None]:
    """Build a numpyro model with an ``extra_model`` reparameterization.

    Allows users to replace specific linear parameters (e.g. ``K``) with
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
        ``pars["eccentricity"]``, …).  The callable may call
        ``numpyro.sample`` internally.  It must return a dict whose keys
        are a subset of the linear parameter names for this data type
        (e.g. ``"K"`` or ``"v0"`` for RV data).
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

    Example — replace ``K`` with a mass-function reparameterization::

        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        # Semi-amplitude constant: K [km/s] = K_FACTOR * f(masses, inc, P, e)
        # (Lovis & Fischer 2010, converted to km/s with period in days)
        _K_FACTOR = 28.4329  # km/s · day^(1/3) · M_sun^(-1/3)

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
            return {"K": K}

    With this ``extra_model_fn``, ``K`` becomes a deterministic site in
    ``get_samples()``; ``v0`` is analytically marginalized (if
    ``marginalized=True``) or sampled from its marginal prior.
    """
    import numpyro

    ctx = _build_model_context(sampler, data)

    def model() -> None:
        # --- nonlinear parameters ---
        values, params = _sample_nonlinear(ctx)

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

        # Determine fixed/free column split (static Python-level at trace time).
        fixed_idx = [i for i, n in enumerate(ctx.all_linear_names) if n in fixed_linear]
        free_idx = [
            i for i, n in enumerate(ctx.all_linear_names) if n not in fixed_linear
        ]

        # Resolve the linear prior once (may depend on orbit params if callable).
        resolved_lp = _resolve_linear_prior(ctx.prior.linear_prior, params)

        log_lik: jax.Array = jnp.zeros(())

        # --- astrometry component ---
        if ctx.astro_lik is not None:
            dm_a = ctx.astro_lik.design_matrix(params)

            a_fixed = [i for i in fixed_idx if i < ctx.n_astro]
            a_free = [i for i in free_idx if i < ctx.n_astro]

            y_a = ctx.astro_obs
            if a_fixed:
                fv = jnp.stack([fixed_linear[ctx.all_linear_names[i]] for i in a_fixed])
                y_a = y_a - dm_a[:, jnp.array(a_fixed)] @ fv

            if a_free and marginalized:
                marg = MarginalizedLinear(
                    design_matrix=dm_a[:, jnp.array(a_free)],
                    prior_distribution=_marginal_mvn(resolved_lp, a_free),
                    data_distribution=dist.Normal(0.0, ctx.astro_err),
                )
                log_lik = log_lik + marg.log_prob(y_a)
            elif a_free:
                free_vals = numpyro.sample(
                    "_astro_linear_free", _marginal_mvn(resolved_lp, a_free)
                )
                for j, col in enumerate(a_free):
                    numpyro.deterministic(ctx.all_linear_names[col], free_vals[j])
                prediction = dm_a[:, jnp.array(a_free)] @ free_vals
                if a_fixed:
                    fv = jnp.stack(
                        [fixed_linear[ctx.all_linear_names[i]] for i in a_fixed]
                    )
                    prediction = prediction + dm_a[:, jnp.array(a_fixed)] @ fv
                log_lik = (
                    log_lik
                    + dist.Normal(prediction, ctx.astro_err)
                    .log_prob(ctx.astro_obs)
                    .sum()
                )
            else:
                log_lik = (
                    log_lik
                    + dist.Normal(jnp.zeros_like(ctx.astro_obs), ctx.astro_err)
                    .log_prob(y_a)
                    .sum()
                )

        # --- RV component ---
        if ctx.rv_lik is not None:
            dm_r = ctx.rv_lik.design_matrix(params)

            # Shift column indices into the RV block (starts at n_astro in the
            # joint linear vector, but the RV design matrix is zero-indexed).
            r_fixed_global = [i for i in fixed_idx if i >= ctx.n_astro]
            r_free_global = [i for i in free_idx if i >= ctx.n_astro]
            r_fixed_local = [i - ctx.n_astro for i in r_fixed_global]
            r_free_local = [i - ctx.n_astro for i in r_free_global]

            y_r = ctx.rv_obs
            if r_fixed_local:
                fv = jnp.stack(
                    [fixed_linear[ctx.all_linear_names[i]] for i in r_fixed_global]
                )
                y_r = y_r - dm_r[:, jnp.array(r_fixed_local)] @ fv

            if r_free_local and marginalized:
                marg = MarginalizedLinear(
                    design_matrix=dm_r[:, jnp.array(r_free_local)],
                    prior_distribution=_marginal_mvn(resolved_lp, r_free_global),
                    data_distribution=dist.Normal(0.0, ctx.rv_err),
                )
                log_lik = log_lik + marg.log_prob(y_r)
            elif r_free_local:
                free_vals = numpyro.sample(
                    "_rv_linear_free", _marginal_mvn(resolved_lp, r_free_global)
                )
                for j, col in enumerate(r_free_global):
                    numpyro.deterministic(ctx.all_linear_names[col], free_vals[j])
                prediction = dm_r[:, jnp.array(r_free_local)] @ free_vals
                if r_fixed_local:
                    fv = jnp.stack(
                        [fixed_linear[ctx.all_linear_names[i]] for i in r_fixed_global]
                    )
                    prediction = prediction + dm_r[:, jnp.array(r_fixed_local)] @ fv
                log_lik = (
                    log_lik
                    + dist.Normal(prediction, ctx.rv_err).log_prob(ctx.rv_obs).sum()
                )
            else:
                log_lik = (
                    log_lik
                    + dist.Normal(jnp.zeros_like(ctx.rv_obs), ctx.rv_err)
                    .log_prob(y_r)
                    .sum()
                )

        numpyro.factor("log_lik", log_lik)

    return model
