"""Abstract base class for likelihood components."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.likelihood.helpers import LinearPriorCallable, _resolve_linear_prior
from harv.likelihood.params import MarginalizedParameters


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
        linear_names: tuple[str, ...],
        linear_units: tuple[str, ...],
        linear_prior: dist.MultivariateNormal | LinearPriorCallable,
    ) -> jax.Array:
        """Partially or fully marginalize linear parameters.

        Generic helper shared by all likelihood subclasses.  Subtracts the
        contribution of any fixed (non-marginalized) linear parameters
        from the observations, then analytically marginalizes the remaining
        parameters via ``MarginalizedLinear``.

        A linear parameter is considered **fixed** when its name appears as a
        key in ``params.values`` (i.e. the user provided an explicit value).
        All other names in ``linear_names`` are marginalized.  This allows
        data-dependent columns (such as multi-survey offset indicators) to be
        marginalized automatically without requiring the parameter struct to
        know their names at class-definition time.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear params plus any fixed linear values.
        X : jax.Array
            Full design matrix of shape ``(n_obs, len(linear_names))``,
            including any indicator columns for multi-survey offsets.
        obs : jax.Array
            Observed data (unitless), shape ``(n_obs,)``.
        obs_err : jax.Array
            Per-observation uncertainties (unitless), shape ``(n_obs,)``.
        linear_names : tuple[str, ...]
            Names of all linear parameters, in design-matrix column order.
        linear_units : tuple[str, ...]
            Unit strings for each linear parameter, used to strip fixed
            values to bare numbers.  Must have the same length as
            ``linear_names``.
        linear_prior : dist.MultivariateNormal or LinearPriorCallable
            Prior over the marginalized parameters.

        Returns
        -------
        jax.Array
            Scalar log-marginal-likelihood.
        """
        fixed_names = tuple(n for n in linear_names if n in params.values)
        marg_names = tuple(n for n in linear_names if n not in fixed_names)

        if fixed_names:
            fixed_vals = jnp.array(
                [
                    ustrip(
                        AllowValue,
                        linear_units[linear_names.index(n)],
                        getattr(params, n),
                    )
                    for n in fixed_names
                ]
            )
            fixed_idx = jnp.array([linear_names.index(n) for n in fixed_names])
            obs = obs - X[:, fixed_idx] @ fixed_vals

        marg_idx = jnp.array([linear_names.index(n) for n in marg_names])
        X_marg = X[:, marg_idx]

        lp = _resolve_linear_prior(linear_prior, params)
        return MarginalizedLinear(
            design_matrix=X_marg,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, obs_err),
        ).log_prob(obs)
