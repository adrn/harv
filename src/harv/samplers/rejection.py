"""Rejection sampler for orbital parameter inference.

This module implements rejection sampling with analytical marginalization over
linear parameters. The sampler draws samples from the prior distribution over
nonlinear parameters, evaluates the marginalized likelihood, and performs
rejection sampling to obtain posterior samples.

Data-type-specific logic (param struct construction, likelihood building,
linear sampling) is encapsulated in ``_DataTypeStrategy`` descriptors in
``_strategies.py``; numpyro model builder helpers live in ``_numpyro.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from unxt import Quantity, ustrip

from harv.data import (
    AbstractAstrometryData,
    GaiaAstrometryData,
    InputData,
    RadialVelocityData,
    SourceData,
)
from harv.samplers._numpyro import (
    _build_extra_numpyro_model,
    _build_full_numpyro_model,
    _build_marginalized_numpyro_model,
)
from harv.samplers._strategies import (
    _STRATEGIES,
    _DataTypeStrategy,
)
from harv.samplers.samples import Samples, _WarmStartMCMC

if TYPE_CHECKING:
    from harv.priors.rejection import RejectionPrior

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
        astro_data, rv_data = strategy.extract_data(data)
        lik = strategy.build_likelihood(astro_data, rv_data, self.prior, data)

        key = jr.PRNGKey(seed)
        sample_key, rej_key = jr.split(key)

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            sample_key, data, n_prior_samples, lik, strategy
        )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)
        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}

        linear_key = jr.fold_in(key, 2)
        linear_samples = self._sample_linear_parameters(
            linear_key, accepted_nonlinear, astro_data, rv_data, strategy, data, lik
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
                linear_samples = linear_samples[idx]

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        t_ref = _ref.t_ref
        time_unit = str(_ref.time.unit)

        # Convert t_ref to a plain Python float (in time_unit) so it can be stored
        # safely in the static _metadata dict without "JAX array set as static"
        # warnings. Samples.__getitem__ reads _metadata["t_ref"] as a scalar in
        # _time_unit when computing t_peri.
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

        return Samples(
            _nonlinear=accepted_nonlinear,
            _linear=linear_samples,
            _orbit_cls=strategy.nonlinear_cls,
            _full_cls=strategy.full_cls,
            _linear_param_units=strategy.linear_param_units(
                astro_data, rv_data, self.prior
            ),
            _time_unit=time_unit,
            _data_type=strategy.data_type,
            _metadata={"t_ref": t_ref_stored},
            _extra_linear_names=extra_linear_names,
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
        try:
            from numpyro import infer as _infer
        except ImportError as e:
            msg = "numpyro is required. Install with: pip install numpyro"
            raise ImportError(msg) from e

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
            kernel = _infer.NUTS

        # Take the first num_chains posterior samples as starting positions.
        indices = list(range(num_chains))
        init_params: dict[str, Any] = {
            key_name: jnp.stack([arr[i] for i in indices])
            for key_name, arr in samples._nonlinear.items()
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
            linear = np.asarray(samples._linear)
            if linear.ndim == 2 and linear.shape[-1] > 0:
                init_params["_linear"] = jnp.stack(
                    [jnp.asarray(linear[i]) for i in indices]
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
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        strategy: _DataTypeStrategy,
        data: InputData,
        lik: Any,
    ) -> jax.Array:
        """Sample linear parameters from conditional posterior using vmap.

        For each accepted nonlinear sample, draws from the conditional posterior
        of the linear parameters given the nonlinear parameters and data.

        Parameters
        ----------
        key
            Random key.
        nonlinear_samples
            Accepted nonlinear parameter samples.
        astro_data
            Gaia astrometry data, or None.
        rv_data
            Radial velocity data, or None.
        strategy
            Data-type strategy for building params and design matrices.
        data
            Original input data (needed for multi-survey instrument ordering).
        lik
            Pre-built likelihood (or CompositeLikelihood) for DM construction.

        Returns
        -------
        linear_samples
            Shape ``(n_samples, n_linear)``.
        """
        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            n_offsets = sum(
                1 for v in (self.prior.offsets or {}).values() if v is not None
            )
            return jnp.zeros((0, strategy.n_linear + n_offsets))

        _ref = rv_data if rv_data is not None else astro_data
        time_unit = _ref.time.unit  # type: ignore[union-attr]

        keys = jr.split(key, n_samples)

        def _sample_one(key: jax.Array, sample: dict[str, jax.Array]) -> jax.Array:
            return strategy.sample_linear_one(
                key, sample, astro_data, rv_data, self.prior, time_unit, data, lik
            )

        return jax.vmap(_sample_one)(keys, nonlinear_samples)
