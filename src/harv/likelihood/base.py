"""Abstract base class for likelihood components."""

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.likelihood.helpers import LinearPriorCallable, _resolve_linear_prior
from harv.likelihood.params import AbstractParameters, MarginalizedParameters


class AbstractLikelihood[ParamT: eqx.Module](eqx.Module):
    """Abstract base class for likelihood components.

    Generic over the parameter struct type ``ParamT``. Subclasses declare
    their expected parameter type explicitly, for example::

        class RVLikelihood(AbstractLikelihood[MarginalizedParameters | RVParameters]):
            ...

    """

    def log_prob(self, params: ParamT) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample."""
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def _marginalize_partial(
        params: MarginalizedParameters,
        X: jax.Array,
        obs: jax.Array,
        obs_err: jax.Array,
        obs_unit: Any,
        linear_prior: dist.MultivariateNormal | LinearPriorCallable,
        indicator_matrix: jax.Array | None = None,
    ) -> jax.Array:
        """Partially or fully marginalize linear parameters.

        Generic helper shared by all likelihood subclasses.  Subtracts the
        contribution of any fixed (non-marginalized) named linear parameters
        from the observations, then analytically marginalizes the remaining
        named parameters plus any indicator (multi-survey offset) columns via
        ``MarginalizedLinear``.

        Named parameters come from ``params.source_cls.linear_param_names``
        and may be partially or fully marginalized.  Indicator columns (when
        present) are always appended to the marginalized design matrix after
        the named columns -- they are always marginalized, never held fixed.

        The prior must be dimensioned to match the number of marginalized
        columns: ``len(marg_names) + n_indicator_cols``.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear params; ``source_cls`` must not be ``None``.
        X : jax.Array
            Base design matrix of shape ``(n_obs, n_named_linear)``.
        obs : jax.Array
            Observed data in ``obs_unit``, shape ``(n_obs,)``.
        obs_err : jax.Array
            Per-observation standard deviations in ``obs_unit``.
        obs_unit : Any
            Unit used to strip fixed linear parameter values to bare numbers.
        linear_prior : dist.MultivariateNormal or LinearPriorCallable
            Prior over the marginalized parameters.
        indicator_matrix : jax.Array or None
            Float indicator matrix of shape ``(n_obs, n_non_ref)`` for
            multi-survey data; always marginalized.  ``None`` for
            single-survey.

        Returns
        -------
        jax.Array
            Scalar log-marginal-likelihood.
        """
        if params.source_cls is None:
            msg = "_marginalize_partial requires params.source_cls to be set"
            raise ValueError(msg)
        src_cls = cast("type[AbstractParameters]", params.source_cls)
        all_names = src_cls.linear_param_names
        marg_names = tuple(n for n in all_names if n in params.marginalized_names)
        fixed_names = tuple(n for n in all_names if n not in params.marginalized_names)

        if fixed_names:
            fixed_vals = jnp.array(
                [ustrip(AllowValue, obs_unit, getattr(params, n)) for n in fixed_names]
            )
            fixed_idx = jnp.array([all_names.index(n) for n in fixed_names])
            obs = obs - X[:, fixed_idx] @ fixed_vals

        marg_idx = jnp.array([all_names.index(n) for n in marg_names])
        X_marg = X[:, marg_idx]
        if indicator_matrix is not None:
            X_marg = jnp.concatenate([X_marg, indicator_matrix], axis=-1)

        lp = _resolve_linear_prior(linear_prior, params)
        return MarginalizedLinear(
            design_matrix=X_marg,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, obs_err),
        ).log_prob(obs)
