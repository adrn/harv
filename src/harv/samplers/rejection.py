"""Rejection sampler for orbital parameter inference.

This module implements rejection sampling with analytical marginalization over
linear parameters. The sampler draws samples from the prior distribution over
nonlinear parameters, evaluates the marginalized likelihood, and performs
rejection sampling to obtain posterior samples.

Data-type-specific logic (param struct construction, likelihood building,
linear sampling) is encapsulated in ``_DataTypeStrategy`` descriptors in
``_strategies.py``; numpyro model builder helpers live in ``_numpyro.py``.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpyro import infer as _numpyro_infer
from unxt import Quantity, ustrip

from harv.data import (
    AbstractAstrometryData,
    InputData,
    RadialVelocityData,
    SourceData,
)
from harv.priors.rejection import RejectionPrior
from harv.samplers.numpyro import (
    _build_extra_numpyro_model,
    _build_full_numpyro_model,
    _build_marginalized_numpyro_model,
)
from harv.samplers.samples import Samples, _WarmStartMCMC
from harv.samplers.strategies import (
    _STRATEGIES,
    _DataTypeStrategy,
)

__all__ = ["RejectionSampler"]


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
    >>> prior = RejectionPrior.default_astrometry()
    >>> sampler = RejectionSampler(prior)
    >>> samples = sampler.run(data, n_prior_samples=100_000)
    """

    prior: RejectionPrior
    batch_size: int = eqx.field(static=True, default=100_000)

    def _infer_strategy(self, data: InputData) -> _DataTypeStrategy:
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
            has_rv = data.n_rv() > 0
            has_astro = data.n_astrometry() > 0
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
        elif isinstance(data, RadialVelocityData):
            data_type = "rv"
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        strategy = _STRATEGIES[data_type]

        # Validate prior — required params derived from orbit param class fields
        missing = [
            p
            for p in strategy.required_prior_params
            if p not in self.prior.nonlinear_priors
        ]
        if missing:
            msg = (
                f"Prior missing required parameters for {data_type} data: {missing}. "
                f"Use RejectionPrior.default_{data_type}() or provide these parameters."
            )
            raise ValueError(msg)

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
        """
        strategy = self._infer_strategy(data)
        datasets = strategy.extract_data(data)
        lik = strategy.build_likelihood(datasets, self.prior, data)

        key = jr.PRNGKey(seed)
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
        if isinstance(t_ref, Quantity):
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
        _nl_units: dict[str, str] = {
            "period": time_unit,
            "eccentricity": "",
            "phase_peri": "",
            "arg_peri": "rad",
            "cos_i": "",
            "lon_asc_node": "rad",
        }
        nonlinear_q: dict[str, AbstractQuantity] = {
            k: Quantity(v, _nl_units.get(k, "")) for k, v in accepted_nonlinear.items()
        }

        return Samples(
            nonlinear=nonlinear_q,
            linear=linear_samples,
            orbit_cls=strategy.nonlinear_cls,
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
    ) -> _WarmStartMCMC:
        """Construct a numpyro MCMC object warm-started from rejection-sampler output.

        Builds a numpyro model from this sampler's prior and the observed data,
        draws one starting position per chain from ``samples``, and returns a
        :class:`~harv.samplers.samples._WarmStartMCMC` whose
        :meth:`~harv.samplers.samples._WarmStartMCMC.run` injects those positions
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
            data's time unit, ``pars["eccentricity"]``, …).  The function may
            call ``numpyro.sample`` for any number of new sites (e.g. stellar
            masses, inclination) and must return a dict mapping linear
            parameter names (e.g. ``"K"``) to their computed values.  Any
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
                    return {"K": K}

        extra_init_params : dict, optional
            Initial values for the parameters introduced by ``extra_model``,
            one entry per chain.  Required when ``extra_model`` is provided,
            since harv cannot automatically invert K → (m1, m2, inc).
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
        mcmc : _WarmStartMCMC
            Configured MCMC wrapper.  Call ``mcmc.run(jr.PRNGKey(seed))`` to
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
        **Marginalized (default)** — MCMC over nonlinear parameters only,
        ``K`` and ``v0`` analytically marginalized:

        >>> import jax.random as jr
        >>> prior = RejectionPrior.default_rv(period_min=50, period_max=200)
        >>> sampler = RejectionSampler(prior)
        >>> samples = sampler.run(rv_data, n_prior_samples=500_000)
        >>> mcmc = sampler.init_mcmc(samples, rv_data,
        ...                          num_chains=4, num_warmup=500,
        ...                          num_samples=2000)
        >>> mcmc.run(jr.PRNGKey(0))
        >>> posterior = mcmc.get_samples()
        >>> # Keys: period, eccentricity, phase_peri, arg_peri

        **Full model** — all parameters sampled jointly:

        >>> mcmc = sampler.init_mcmc(samples, rv_data, marginalized=False,
        ...                          num_chains=4, num_warmup=500,
        ...                          num_samples=2000)
        >>> mcmc.run(jr.PRNGKey(0))
        >>> posterior = mcmc.get_samples()
        >>> # Adds K and v0 (as deterministic sites) to the above

        **Physical reparameterization** — replace ``K`` with stellar masses
        and inclination; ``v0`` is analytically marginalized:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>>
        >>> _K_FACTOR = 28.4329  # km/s · day^(1/3) · M_sun^(-1/3)
        >>>
        >>> def K_from_masses(m1, m2, inc, period_days, ecc):
        ...     return (
        ...         _K_FACTOR
        ...         * (m2 * jnp.sin(inc))
        ...         * (m1 + m2) ** (-2.0 / 3.0)
        ...         * (period_days / 365.25) ** (-1.0 / 3.0)
        ...         / jnp.sqrt(1.0 - ecc**2)
        ...     )
        >>>
        >>> def mass_model(pars):
        ...     # pars["period"] is in the data's time unit (days here)
        ...     m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
        ...     m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
        ...     inc = numpyro.sample("inc", dist.Uniform(0.0, jnp.pi / 2))
        ...     K   = K_from_masses(m1, m2, inc,
        ...                         pars["period"], pars["eccentricity"])
        ...     return {"K": K}
        >>>
        >>> mcmc = sampler.init_mcmc(
        ...     samples, rv_data,
        ...     extra_model=mass_model,
        ...     extra_init_params={
        ...         "m1":  jnp.full(4, 1.0),   # shape (num_chains,)
        ...         "m2":  jnp.full(4, 0.5),
        ...         "inc": jnp.full(4, 1.0),
        ...     },
        ...     num_chains=4, num_warmup=500, num_samples=2000,
        ... )
        >>> mcmc.run(jr.PRNGKey(0))
        >>> posterior = mcmc.get_samples()
        >>> # Sampled sites:      period, eccentricity, …, m1, m2, inc
        >>> # Deterministic site: K  (computed from m1, m2, inc, P, e)
        >>> # Marginalized:       v0 (analytically integrated out)
        """
        if samples.n_samples == 0:
            msg = "Cannot initialise MCMC: no posterior samples available."
            raise ValueError(msg)
        if samples.n_samples < num_chains:
            msg = (
                f"Fewer posterior samples ({samples.n_samples}) than requested "
                f"chains ({num_chains}). Reduce num_chains or increase "
                "n_prior_samples in RejectionSampler.run()."
            )
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

        # Take the first num_chains posterior samples as starting positions.
        # Strip units from each nonlinear Quantity to get the raw array that
        # matches what numpyro's prior model sampled (unit-free floats).
        indices = list(range(num_chains))
        init_params: dict[str, Any] = {
            key_name: jnp.stack([ustrip(str(qty.unit), qty)[i] for i in indices])
            for key_name, qty in samples.nonlinear.items()
        }

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
                init_params["_linear"] = jnp.stack(
                    [jnp.asarray(lin_arr[i]) for i in indices]
                )

        kernel_instance = kernel(model)
        return _WarmStartMCMC(
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
        strategy: _DataTypeStrategy,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        The pre-built ``lik`` (a single marginalized likelihood or a
        ``CompositeLikelihood``) is evaluated with ``jax.vmap`` inside a
        ``fori_loop`` over batches of ``batch_size`` samples.  ``strategy`` is
        a static value (hashed by class identity) so ``build_orbit_params``
        dispatches to the correct param type at trace time.
        """
        prior_samples = self.prior.sample_nonlinear(key, n_prior_samples)

        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        total_size = n_batches * self.batch_size
        pad_size = total_size - n_prior_samples

        def pad_batch(arr: jax.Array) -> jax.Array:
            return jnp.pad(arr, (0, pad_size)).reshape(n_batches, self.batch_size)

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        time_unit = _ref.time.unit

        period_batched = pad_batch(prior_samples["period"])
        ecc_batched = pad_batch(prior_samples["eccentricity"])
        phase_batched = pad_batch(prior_samples["phase_peri"])
        arg_peri_batched = pad_batch(prior_samples["arg_peri"])

        # Pad optional params with zeros (unused values are ignored by the builder).
        _zeros = jnp.zeros(n_prior_samples)
        cos_i_batched = pad_batch(prior_samples.get("cos_i", _zeros))
        lon_asc_batched = pad_batch(prior_samples.get("lon_asc_node", _zeros))

        def body_fn(i: int, acc: jax.Array) -> jax.Array:
            params = strategy.build_orbit_params(
                period_batched[i],
                ecc_batched[i],
                phase_batched[i],
                arg_peri_batched[i],
                cos_i_batched[i],
                lon_asc_batched[i],
                time_unit,
            )
            return acc.at[i].set(jax.vmap(lik.log_prob)(params))

        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )
        return prior_samples, log_liks_batched.flatten()[:n_prior_samples]

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
        datasets: dict[str, Any],
        strategy: _DataTypeStrategy,
        data: InputData,
        lik: Any,
    ) -> dict[str, Quantity]:
        """Sample linear parameters from conditional posterior using vmap.

        For each accepted nonlinear sample, draws from the conditional posterior
        of the linear parameters given the nonlinear parameters and data.

        Parameters
        ----------
        key
            Random key.
        nonlinear_samples
            Accepted nonlinear parameter samples.
        datasets
            Extracted data objects keyed by component name.
        strategy
            Data-type strategy for building params and design matrices.
        data
            Original input data (needed for multi-survey instrument ordering).
        lik
            Pre-built likelihood (or CompositeLikelihood) for DM construction.

        Returns
        -------
        dict[str, Quantity]
            One Quantity per linear parameter, each with shape ``(n_samples,)``.
        """
        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            names = strategy.all_linear_names(self.prior, data)
            units = strategy.linear_param_units(datasets, self.prior)
            return {
                name: Quantity(jnp.zeros(0), unit)
                for name, unit in zip(names, units, strict=False)
            }

        _ref = next(iter(datasets.values()))
        time_unit = _ref.time.unit

        keys = jr.split(key, n_samples)

        def _sample_one(
            key: jax.Array, sample: dict[str, jax.Array]
        ) -> dict[str, Quantity]:
            return strategy.sample_linear_one(
                key, sample, datasets, self.prior, time_unit, data, lik
            )

        return jax.vmap(_sample_one)(keys, nonlinear_samples)
