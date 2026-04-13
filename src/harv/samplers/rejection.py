"""Rejection sampler for orbital parameter inference.

This module implements rejection sampling with analytical marginalization over
linear parameters. The sampler draws samples from the prior distribution over
nonlinear parameters, evaluates the marginalized likelihood, and performs
rejection sampling to obtain posterior samples.

Data-type-specific logic (param struct construction, likelihood building,
linear sampling) is encapsulated in ``DataTypeStrategy`` descriptors in
``_strategies.py``; numpyro model builder helpers live in ``_numpyro.py``.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
from numpyro import infer as _numpyro_infer
from unxt import AbstractQuantity, Q, ustrip

from harv.data import (
    AbstractAstrometryData,
    InputData,
    RVData,
    SourceData,
)
from harv.distributions import QuantityDistribution
from harv.likelihood.helpers import _unwrap_dist
from harv.samplers._strategies import (
    _STRATEGIES,
    DataTypeStrategy,
    _jitter_units_from_prior,
)
from harv.samplers.numpyro import (
    _build_extra_numpyro_model,
    _build_full_numpyro_model,
    _build_marginalized_numpyro_model,
)
from harv.samplers.rejection_prior import RejectionPrior
from harv.samplers.samples import Samples, WarmStartMCMC

__all__ = ["RejectionSampler"]


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
    # NOTE: the site names here (e.g. "jitter_{dt_label}") must match those
    # created by _sample_nonlinear() in numpyro.py.  If naming conventions
    # change in one place, they must change in the other.  Consider extracting
    # a shared ``prior.site_distributions()`` method to eliminate the coupling.
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
            # Unknown site (e.g. "_linear", extra_model params) — pass through.
            out[name] = val
    return out


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class RejectionSampler(eqx.Module):
    """Rejection sampler for Keplerian orbital parameters.

    This class implements rejection sampling with analytical marginalization
    over linear parameters. It supports both astrometric and radial velocity
    data.

    Parameters
    ----------
    prior : RejectionPrior
        Prior distribution for nonlinear and linear parameters.
    batch_size : int, optional
        Number of samples to process per batch. Smaller values use less memory
        but may be slower. Default: 100_000.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers import RejectionPrior, RejectionSampler
    >>> prior = RejectionPrior.default_rv(
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> sampler = RejectionSampler(prior)
    >>> sampler.batch_size
    100000

    Run rejection sampling on data (expensive):

    >>> samples = sampler.run(data, n_prior_samples=100_000)  # doctest: +SKIP
    """

    prior: RejectionPrior
    batch_size: int = eqx.field(static=True, default=100_000)

    def _infer_strategy(self, data: InputData) -> DataTypeStrategy:  # noqa: C901
        """Infer data type from ``data`` and return the matching strategy.

        Also validates that the prior has all required parameters for the
        inferred data type (derived from the orbit param class fields).

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters for the data type.
        """
        if isinstance(data, SourceData):
            has_rv = data._n_rv() > 0
            has_astro = data._n_astrometry() > 0
            if has_astro and has_rv:
                data_type = "combined"
            elif has_astro:
                data_type = "astrometry"
            elif has_rv:
                data_type = "rv"
            else:
                msg = "SourceData must contain at least one dataset"
                raise ValueError(msg)
        elif isinstance(data, AbstractAstrometryData):
            data_type = "astrometry"
        elif isinstance(data, RVData):
            data_type = "rv"
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        strategy = _STRATEGIES[data_type]

        # Validate prior -- required params derived from orbit param class fields
        # TODO: audit how we do this validation
        all_prior_keys = set(self.prior.nonlinear_priors)
        if isinstance(self.prior.linear_prior, dict):
            all_prior_keys |= set(self.prior.linear_prior)
        missing = [
            p
            for p in strategy.required_prior_params(self.prior)
            if p not in all_prior_keys
        ]
        if missing:
            msg = (
                f"Prior missing required parameters for {data_type} data: {missing}. "
                f"Use RejectionPrior.default_{data_type}() or provide these parameters."
            )
            raise ValueError(msg)

        # Validate that dimensioned parameters use QuantityDistribution.
        # Bare numpyro distributions lack unit metadata and can cause silent
        # unit-mismatch bugs downstream.
        dimensioned: set[str] = set()
        for param_cls in strategy.full_cls:
            dimensioned.update(param_cls._dimensioned_param_names)

        bad: list[str] = []
        for name in dimensioned:
            d = self.prior.nonlinear_priors.get(name)
            if d is None and isinstance(self.prior.linear_prior, dict):
                d = self.prior.linear_prior.get(name)
            if d is None:
                continue
            # QDistribution and callables (LinearPriorCallable) are OK.
            if isinstance(d, QuantityDistribution):
                continue
            if callable(d) and not isinstance(d, dist.Distribution):
                continue
            bad.append(name)

        if bad:
            msg = (
                f"Parameters {sorted(bad)} have physical dimensions and require "
                f"a QDistribution prior, but received bare "
                f"numpyro distributions. Wrap each in "
                f"QuantityDistribution(dist, unit_str)."
            )
            raise TypeError(msg)

        return strategy

    def run(
        self,
        data: InputData,
        n_prior_samples: int,
        *,
        max_posterior_samples: int | None = None,
        seed: int = 0,
    ) -> Samples:
        """Run rejection sampling.

        Parameters
        ----------
        data
            Observational data.
        n_prior_samples
            Number of samples to draw from the prior.
        max_posterior_samples
            Maximum number of posterior samples to return. If None, returns all
            accepted samples.
        seed
            Random seed for reproducibility. Default: 0.

        Returns
        -------
        samples
            Posterior samples container.

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters.

        Examples
        --------
        >>> from unxt import Q  # doctest: +SKIP
        >>> from harv.samplers import RejectionPrior, RejectionSampler  # doctest: +SKIP
        >>> from harv.simulate.rv import simulate_rv_sb1_data  # doctest: +SKIP
        >>> rv_data, _ = simulate_rv_sb1_data()  # doctest: +SKIP
        >>> prior = RejectionPrior.default_rv(  # doctest: +SKIP
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> sampler = RejectionSampler(prior)  # doctest: +SKIP
        >>> samples = sampler.run(rv_data, n_prior_samples=100_000)  # doctest: +SKIP
        >>> samples.n_samples  # doctest: +SKIP
        42
        """
        strategy = self._infer_strategy(data)
        datasets = strategy.extract_data(data)
        lik = strategy.build_likelihood(datasets, self.prior, data)

        key = jr.key(seed)
        sample_key, rej_key = jr.split(key)

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            sample_key, data, n_prior_samples, lik, strategy
        )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)
        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}

        linear_key = jr.fold_in(key, 2)
        linear_samples = self._sample_linear_parameters(
            linear_key, accepted_nonlinear, datasets, strategy, data, lik
        )

        if max_posterior_samples is not None:
            n_accepted = len(next(iter(accepted_nonlinear.values())))
            if n_accepted > max_posterior_samples:
                idx_key = jr.fold_in(key, 3)
                idx = jr.choice(
                    idx_key,
                    n_accepted,
                    shape=(max_posterior_samples,),
                    replace=False,
                )
                accepted_nonlinear = {k: v[idx] for k, v in accepted_nonlinear.items()}
                linear_samples = {k: v[idx] for k, v in linear_samples.items()}

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        t_ref = _ref.t_ref
        time_unit = str(_ref.time.unit)

        # Convert t_ref to a plain Python float (in time_unit) so it can be stored
        # safely in the static metadata dict without "JAX array set as static"
        # warnings. Samples.__getitem__ reads metadata["t_ref"] as a scalar in
        # the period's unit when computing t_peri.
        if isinstance(t_ref, Q):
            t_ref_stored: float | None = float(ustrip(time_unit, t_ref))
        elif t_ref is not None:
            t_ref_stored = float(t_ref)
        else:
            t_ref_stored = None

        extra_linear_names: tuple[str, ...] = ()
        if self.prior.offsets is not None:
            extra_linear_names = tuple(
                k for k, v in self.prior.offsets.items() if v is not None
            )

        # Build nonlinear dict as Quantities with units baked in.
        # Only include actual nonlinear parameters (not explicit-linear ones
        # that may also be in accepted_nonlinear due to partial marginalization).
        _nl_units: dict[str, str] = {
            "period": time_unit,
            "eccentricity": "",
            "phase_peri": "",
            "arg_peri": "rad",
            "cos_i": "",
            "lon_asc_node": "rad",
        }
        _nl_keys = set(self.prior.nonlinear_priors)
        nonlinear_q: dict[str, AbstractQuantity] = {
            k: Q(v, _nl_units.get(k, ""))
            for k, v in accepted_nonlinear.items()
            if k in _nl_keys
        }

        # Include jitter samples (keyed by user-friendly names like
        # "jitter_rv", "jitter_astrometry") in the nonlinear dict.
        if self.prior.jitter_priors is not None:
            for dt_label, d in self.prior.jitter_priors.items():
                values_key = f"_jitter_{dt_label}"
                user_key = f"jitter_{dt_label}"
                if values_key in accepted_nonlinear:
                    unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
                    nonlinear_q[user_key] = Q(accepted_nonlinear[values_key], unit)

        return Samples(
            nonlinear=nonlinear_q,
            linear=linear_samples,
            orbit_cls=strategy.full_cls[0],
            full_cls=strategy.full_cls,
            data_type=strategy.data_type,
            metadata={"t_ref": t_ref_stored},
            extra_linear_names=extra_linear_names,
        )

    def init_mcmc(
        self,
        samples: Samples,
        data: InputData,
        *,
        marginalized: bool = True,
        extra_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        extra_init_params: dict[str, Any] | None = None,
        kernel: type | None = None,
        num_chains: int = 4,
        **mcmc_kwargs: Any,
    ) -> WarmStartMCMC:
        """Construct a numpyro MCMC object warm-started from rejection-sampler output.

        Builds a numpyro model from this sampler's prior and the observed data,
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
            Posterior samples produced by :meth:`run`.  One sample per chain
            is used as the MCMC warm-start position.
        data : AbstractData or SourceData
            The observed data passed to :meth:`run`.
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
            parameter names (e.g. ``"rv_semiamp"``) to their computed values.  Any
            linear parameter not in the returned dict is handled by
            ``marginalized``.

            Minimal pattern::

                import numpyro
                import numpyro.distributions as dist

                def extra_model(pars):
                    # pars["period"] is in the data's time unit (e.g. days)
                    m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
                    m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
                    inc = numpyro.sample("inc", dist.Uniform(0, jnp.pi / 2))
                    K   = _K_FACTOR * (m2 * jnp.sin(inc)) * (m1 + m2) ** (-2/3) \
                          * (pars["period"] / 365.25) ** (-1/3) \
                          / jnp.sqrt(1 - pars["eccentricity"] ** 2)
                    return {"rv_semiamp": K}

        extra_init_params : dict, optional
            Initial values for the parameters introduced by ``extra_model``,
            one entry per chain.  Required when ``extra_model`` is provided,
            since harv cannot automatically invert rv_semiamp -> (m1, m2, inc).
            Each value must be a 1-D array of length ``num_chains``::

                extra_init_params={
                    "m1":  jnp.full(4, 1.0),
                    "m2":  jnp.full(4, 0.5),
                    "inc": jnp.full(4, 1.0),
                }

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
        ImportError
            If numpyro is not installed.

        Examples
        --------
        **Marginalized (default)** -- MCMC over nonlinear parameters only,
        ``rv_semiamp`` and ``v_sys`` analytically marginalized:

        >>> import jax.random as jr  # doctest: +SKIP
        >>> prior = RejectionPrior.default_rv(...)  # doctest: +SKIP
        >>> sampler = RejectionSampler(prior)  # doctest: +SKIP
        >>> samples = sampler.run(rv_data, n_prior_samples=500_000)  # doctest: +SKIP
        >>> mcmc = sampler.init_mcmc(samples, rv_data,  # doctest: +SKIP
        ...                          num_chains=4, num_warmup=500,
        ...                          num_samples=2000)
        >>> mcmc.run(jr.key(0))  # doctest: +SKIP
        >>> posterior = mcmc.get_samples()  # doctest: +SKIP

        **Full model** -- all parameters sampled jointly:

        >>> mcmc = sampler.init_mcmc(  # doctest: +SKIP
        ...     samples, rv_data, marginalized=False,
        ...     num_chains=4, num_warmup=500,
        ...                          num_samples=2000)
        >>> mcmc.run(jr.key(0))  # doctest: +SKIP
        >>> posterior = mcmc.get_samples()  # doctest: +SKIP

        **Physical reparameterization** -- replace ``rv_semiamp`` with stellar masses
        and inclination; ``v_sys`` is analytically marginalized:

        >>> import jax.numpy as jnp  # doctest: +SKIP
        >>> import numpyro  # doctest: +SKIP
        >>> import numpyro.distributions as dist  # doctest: +SKIP
        >>>
        >>> _K_FACTOR = 28.4329  # doctest: +SKIP
        >>>
        >>> def K_from_masses(m1, m2, inc, period_days, ecc):  # doctest: +SKIP
        ...     return (
        ...         _K_FACTOR
        ...         * (m2 * jnp.sin(inc))
        ...         * (m1 + m2) ** (-2.0 / 3.0)
        ...         * (period_days / 365.25) ** (-1.0 / 3.0)
        ...         / jnp.sqrt(1.0 - ecc**2)
        ...     )
        >>>
        >>> def mass_model(pars):  # doctest: +SKIP
        ...     m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
        ...     m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
        ...     inc = numpyro.sample("inc", dist.Uniform(0.0, jnp.pi / 2))
        ...     K   = K_from_masses(m1, m2, inc,
        ...                         pars["period"], pars["eccentricity"])
        ...     return {"rv_semiamp": K}
        >>>
        >>> mcmc = sampler.init_mcmc(  # doctest: +SKIP
        ...     samples, rv_data,
        ...     extra_model=mass_model,
        ...     extra_init_params={
        ...         "m1":  jnp.full(4, 1.0),
        ...         "m2":  jnp.full(4, 0.5),
        ...         "inc": jnp.full(4, 1.0),
        ...     },
        ...     num_chains=4, num_warmup=500, num_samples=2000,
        ... )
        >>> mcmc.run(jr.key(0))  # doctest: +SKIP
        >>> posterior = mcmc.get_samples()  # doctest: +SKIP
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

        # Take the first min(num_chains, n_samples) posterior samples as
        # starting positions.  When there are fewer samples than chains,
        # we use scalar (0-d) init values and let numpyro broadcast the
        # single starting point to all chains.
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
        # These are linear params whose priors are non-Gaussian (e.g.
        # HalfNormal) and are sampled as numpyro sites rather than
        # analytically marginalized.  The numpyro model samples them in the
        # *prior* unit, so we convert from the stored linear-sample unit back
        # to the prior unit.
        marg_names = self.prior.marginalize_names
        if marg_names is not None and isinstance(self.prior.linear_prior, dict):
            marg_set = set(marg_names)
            for name, d in self.prior.linear_prior.items():
                if name not in marg_set and name in samples.linear:
                    qty = samples.linear[name]
                    # The numpyro site value is in the prior's native unit.
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
            model = _build_extra_numpyro_model(self, data, extra_model, marginalized)
            # Warm-start the extra physical parameters from user-provided values.
            init_params.update(extra_init_params)  # type: ignore[arg-type]
        elif marginalized:
            model = _build_marginalized_numpyro_model(self, data)
        else:
            model = _build_full_numpyro_model(self, data)
            # Warm-start the joint linear site from the rejection-sampler draws.
            # Stack linear params into a 2D array (n_chains, n_linear).
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

        # Transform init_params from constrained (natural) space to
        # unconstrained space.  Numpyro's HMC/NUTS operates in unconstrained
        # space and applies the forward transform internally, so the init
        # values must already be unconstrained.
        init_params = _unconstrain_init_params(init_params, self.prior)

        kernel_instance = kernel(model)
        return WarmStartMCMC(
            kernel_instance,
            _init_params=init_params,
            num_chains=num_chains,
            **mcmc_kwargs,
        )

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(
        self,
        key: jax.Array,
        data: InputData,
        n_prior_samples: int,
        lik: Any,
        strategy: DataTypeStrategy,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        The pre-built ``lik`` (a single marginalized likelihood or a
        ``CompositeLikelihood``) is evaluated with ``jax.vmap`` inside a
        ``fori_loop`` over batches of ``batch_size`` samples.  ``strategy`` is
        a static value (hashed by class identity) so ``build_marginalized_params``
        dispatches to the correct param type at trace time.

        The likelihood handles all linear parameter classification (Gaussian,
        Delta, explicit) internally via ``_build_marginalized_linear``.

        Instead of zero-padding the last batch, we oversample so that every
        evaluation uses a real prior draw.  The returned arrays are trimmed to
        ``n_prior_samples``.
        """
        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        n_total = n_batches * self.batch_size

        key, nl_key = jr.split(key)
        prior_samples = self.prior.sample_nonlinear(nl_key, n_total)

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        time_unit = _ref.time.unit

        # Convert period samples from the prior's unit to the data's time
        # unit.  When the period prior is a bare distribution (not wrapped in
        # QDistribution), assume values are already in time_unit.
        _p_prior = self.prior.nonlinear_priors.get("period")
        _p_unit = (
            str(_p_prior.unit) if isinstance(_p_prior, QuantityDistribution) else ""
        )
        if _p_unit:
            period_converted = ustrip(time_unit, Q(prior_samples["period"], _p_unit))
            prior_samples["period"] = period_converted

        # Sample explicit linear params (those not being marginalized).
        marg_names = self.prior.marginalize_names
        if isinstance(self.prior.linear_prior, dict) and marg_names is not None:
            marg_set = set(marg_names)
            explicit_linear = {
                name: d
                for name, d in self.prior.linear_prior.items()
                if name not in marg_set
            }
            if explicit_linear:
                key, lin_key = jr.split(key)
                lin_keys = jr.split(lin_key, len(explicit_linear))
                lp_units = lik.linear_param_units
                for (name, d), k in zip(explicit_linear.items(), lin_keys, strict=True):
                    raw = _unwrap_dist(d).sample(k, (n_total,))
                    # Convert to data units if the prior carries a unit.
                    target_u = lp_units.get(name, "")
                    if isinstance(d, QuantityDistribution) and target_u:
                        raw = ustrip(target_u, Q(raw, str(d.unit)))
                    prior_samples[name] = raw

        # Sample jitter parameters from jitter_priors (keyed by data type).
        # Each jitter sample is stored with a namespaced key (e.g.
        # "_jitter_rv") that the strategy maps to the "jitter" param field.
        _jitter_keys: list[str] = []
        if self.prior.jitter_priors is not None:
            key, jit_key = jr.split(key)
            jit_keys = jr.split(jit_key, len(self.prior.jitter_priors))
            for (dt_label, d), k in zip(
                self.prior.jitter_priors.items(), jit_keys, strict=True
            ):
                values_key = f"_jitter_{dt_label}"
                _jitter_keys.append(values_key)
                # Sample from the bare distribution (unitless).  The unit from
                # the QuantityDistribution is re-attached downstream by
                # build_marginalized_params via _jitter_units_from_prior().
                prior_samples[values_key] = _unwrap_dist(d).sample(k, (n_total,))

        # Reshape all parameter arrays into (n_batches, batch_size).
        _zeros = jnp.zeros(n_total)
        _required_keys = list(strategy.required_prior_params(self.prior))
        _required_keys.extend(_jitter_keys)
        batched: dict[str, jax.Array] = {
            k: prior_samples.get(k, _zeros).reshape(n_batches, self.batch_size)
            for k in _required_keys
        }

        # Static list of keys for dict reconstruction inside the fori_loop.
        _keys = tuple(batched.keys())
        _marg_names = self.prior.marginalize_names
        _lp_units = lik.linear_param_units
        _ju = _jitter_units_from_prior(self.prior)

        def body_fn(i: int, acc: jax.Array) -> jax.Array:
            values = {k: batched[k][i] for k in _keys}
            params = strategy.build_marginalized_params(
                values, time_unit, _marg_names, _lp_units, _ju
            )
            return acc.at[i].set(jax.vmap(lik.log_prob)(params))

        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )

        # Trim oversampled entries to match the requested count.
        trimmed = {k: v[:n_prior_samples] for k, v in prior_samples.items()}
        return trimmed, log_liks_batched.flatten()[:n_prior_samples]

    @staticmethod
    @jax.jit
    def _rejection_step(key: jax.Array, log_likelihoods: jax.Array) -> jax.Array:
        """Compute rejection mask."""
        weights = jnp.exp(log_likelihoods - jnp.max(log_likelihoods))
        uniform_draws = jr.uniform(key, shape=log_likelihoods.shape)
        return uniform_draws < weights

    def _sample_linear_parameters(
        self,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
        _: dict[str, Any],
        strategy: DataTypeStrategy,
        data: InputData,
        lik: Any,
    ) -> dict[str, AbstractQuantity]:
        """Sample linear parameters from conditional posterior using vmap.

        For each accepted nonlinear sample, draws from the conditional posterior
        of the linear parameters given the nonlinear parameters and data.
        """
        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            names = strategy.all_linear_names(self.prior, data)
            return {name: Q(jnp.zeros(0), "") for name in names}

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        time_unit = _ref.time.unit

        keys = jr.split(key, n_samples)

        def _sample_one(key: jax.Array, sample: dict[str, jax.Array]) -> dict[str, Q]:
            return strategy.sample_linear_one(key, sample, self.prior, time_unit, lik)

        return jax.vmap(_sample_one)(keys, nonlinear_samples)
