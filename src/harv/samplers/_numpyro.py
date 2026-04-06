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
from unxt import Quantity

from harv.data import InputData
from harv.likelihood._params import MarginalizedParameters
from harv.likelihood.helpers import _resolve_linear_prior
from harv.samplers._strategies import _ComponentSlice, _DataTypeStrategy

if TYPE_CHECKING:
    from harv.priors.rejection import RejectionPrior


# ---------------------------------------------------------------------------
# Shared pre-computed context
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ModelContext:
    """Pre-computed shared state used by all numpyro model builders.

    Component-generic: data-type-specific details live in
    ``components`` (a tuple of ``_ComponentSlice``), not in named fields.
    Adding a new data type requires only a new strategy — no changes here.
    """

    prior: "RejectionPrior"
    strategy: _DataTypeStrategy
    time_unit: str
    nonlinear_cls: type
    nonlinear_priors: dict[str, Any]
    lik: Any  # AbstractLikelihood or CompositeLikelihood
    components: tuple[_ComponentSlice, ...]
    all_linear_names: tuple[str, ...]


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
    datasets = strategy.extract_data(data)
    _ref = next(iter(datasets.values()))
    time_unit = str(_ref.time.unit)
    nonlinear_cls = strategy.nonlinear_cls
    nonlinear_priors = prior.nonlinear_priors

    all_linear_names = strategy.all_linear_names(prior, data)

    # Build likelihood + component slices for the model builders.
    lik = strategy.build_likelihood(datasets, prior, data)
    components = strategy.build_component_slices(lik, datasets, prior, data)

    return _ModelContext(
        prior=prior,
        strategy=strategy,
        time_unit=time_unit,
        nonlinear_cls=nonlinear_cls,
        nonlinear_priors=nonlinear_priors,
        lik=lik,
        components=components,
        all_linear_names=all_linear_names,
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

        # --- data log-likelihood (component loop) ---
        log_lik: jax.Array = jnp.zeros(())

        for comp in ctx.components:
            dm = comp.lik.design_matrix(params)
            idx = jnp.array(comp.global_col_indices)
            prediction = dm @ linear_vec[idx]
            log_lik = (
                log_lik + dist.Normal(prediction, comp.err).log_prob(comp.obs).sum()
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

        # --- per-component likelihood (generic loop) ---
        for comp in ctx.components:
            dm = comp.lik.design_matrix(params)
            global_idx_set = set(comp.global_col_indices)

            c_fixed_global = [i for i in fixed_idx if i in global_idx_set]
            c_free_global = [i for i in free_idx if i in global_idx_set]

            # Map global linear-vector indices to local DM column indices.
            g2l = {g: l for l, g in enumerate(comp.global_col_indices)}
            c_fixed_local = [g2l[i] for i in c_fixed_global]
            c_free_local = [g2l[i] for i in c_free_global]

            y = comp.obs
            if c_fixed_local:
                fv = jnp.stack(
                    [fixed_linear[ctx.all_linear_names[i]] for i in c_fixed_global]
                )
                y = y - dm[:, jnp.array(c_fixed_local)] @ fv

            if c_free_local and marginalized:
                marg = MarginalizedLinear(
                    design_matrix=dm[:, jnp.array(c_free_local)],
                    prior_distribution=_marginal_mvn(resolved_lp, c_free_global),
                    data_distribution=dist.Normal(0.0, comp.err),
                )
                log_lik = log_lik + marg.log_prob(y)
            elif c_free_local:
                free_vals = numpyro.sample(
                    f"_{comp.name}_linear_free",
                    _marginal_mvn(resolved_lp, c_free_global),
                )
                for j, col in enumerate(c_free_global):
                    numpyro.deterministic(ctx.all_linear_names[col], free_vals[j])
                prediction = dm[:, jnp.array(c_free_local)] @ free_vals
                if c_fixed_local:
                    fv = jnp.stack(
                        [fixed_linear[ctx.all_linear_names[i]] for i in c_fixed_global]
                    )
                    prediction = prediction + dm[:, jnp.array(c_fixed_local)] @ fv
                log_lik = (
                    log_lik + dist.Normal(prediction, comp.err).log_prob(comp.obs).sum()
                )
            else:
                log_lik = (
                    log_lik
                    + dist.Normal(jnp.zeros_like(comp.obs), comp.err).log_prob(y).sum()
                )

        numpyro.factor("log_lik", log_lik)

    return model
