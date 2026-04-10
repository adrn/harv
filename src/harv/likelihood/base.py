"""Abstract base class for likelihood components."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import NamedTuple, cast

import equinox as eqx
import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro_ext.distributions import MarginalizedLinear
from unxt import AbstractQuantity, Quantity, ustrip

from harv.data import AbstractData
from harv.likelihood.helpers import (
    LinearPriorCallable,
    LinearPriorDist,
    PriorDist,
    _resolve_linear_prior_mvn,
)
from harv.likelihood.params import AbstractParameters, MarginalizedParameters
from harv.quantity_distribution import QuantityDistribution


class _MargLinearComponents(NamedTuple):
    """Components assembled by _build_marginalized_linear.

    Only used (and meaningful) internally.
    """

    dist: MarginalizedLinear
    obs: jax.Array
    marg_names: tuple[str, ...]
    explicit_names: tuple[str, ...]
    linear_params: dict[str, jax.Array]


class AbstractLikelihood[DataT: AbstractData, ParamT: AbstractParameters](eqx.Module):
    """Abstract base class for likelihood components.

    Generic over the parameter struct type ``ParamT``. Subclasses declare
    their expected parameter type explicitly, for example::

        class RVLikelihood(AbstractLikelihood[RadialVelocityData, RVParameters]):
            ...

    """

    data: DataT

    linear_marginalized_prior: LinearPriorDist | None = None
    offsets_marginalized_prior: Mapping[str, PriorDist | LinearPriorCallable] | None = (
        None
    )
    indicator_matrix: jax.Array | None = None
    instrument_names: tuple[str, ...] | None = None
    # TODO: important to note that the instrument names here should match the order of
    # the columns in the indicator matrix, and should _not_ include the reference
    # instrument (the one that has no parameters).

    def __check_init__(self) -> None:
        """Check that the indicator matrix has the right shape if provided."""
        # TODO: do we need to eqx.error_if() instead?
        if self.indicator_matrix is not None:
            if self.instrument_names is None:
                raise ValueError("indicator_matrix provided without instrument_names")

            n_obs, n_inst_indi = self.indicator_matrix.shape
            n_instruments = len(self.instrument_names)

            if n_instruments != n_inst_indi:
                raise ValueError(
                    f"indicator_matrix has {n_inst_indi} columns but "
                    f"{n_instruments} instrument names provided. Only provide the "
                    "non-reference instrument names."
                )

            if n_obs != self.data.n_times:
                raise ValueError(
                    f"indicator_matrix has {n_obs} rows but data has "
                    f"{self.data.n_times} observations"
                )

    @property
    def _has_marginalized(self) -> bool:
        """Whether any linear parameters have marginalized priors."""
        return (
            len(self.linear_marginalized_prior or {})
            + len(self.offsets_marginalized_prior or {})
        ) > 0

    # Abstract methods:

    @property
    @abstractmethod
    def linear_param_units(self) -> dict[str, str]:
        """Map from linear parameter name to its physical unit string."""

    @abstractmethod
    def design_matrix(self, params: MarginalizedParameters | ParamT) -> jax.Array:
        """Build the _full_ design matrix for the given parameters.

        Importantly, this includes columns for all linear parameters, including those
        that are not marginalized! The functions that call this internally must remove
        columns corresponding to fixed parameters before constructing the
        ``MarginalizedLinear`` instance.
        """

    # Other methods:

    def linear_unmarginalized_param_values(
        self,
        params: MarginalizedParameters | ParamT,
        offsets: dict[str, AbstractQuantity],
    ) -> dict[str, jax.Array]:
        """Get the linear parameter values with units stripped.

        For single-instrument data, unmarginalized, returns an array of length
        num_linear_params. For multi-survey data, it appends the offsets to the end of
        the array.
        """
        marginalized_names = getattr(params, "marginalized_names", ())
        obs_unit = str(self.data._get_obs().unit)

        vals = {}

        linear_names = (
            params.source_cls.linear_param_names
            if isinstance(params, MarginalizedParameters)
            else params.linear_param_names
        )
        for name in linear_names:
            if name not in marginalized_names:
                vals[name] = jnp.array(
                    ustrip(self.linear_param_units[name], getattr(params, name))
                )

        # Within this function, we don't know which offsets are marginalized vs. sampled
        # over, so we trust that the outer function only passes in the relevant offsets.
        # TODO: I suppose it might be safer to draw from self.instrument_names here and
        # only pull out those values that appear in the specified input offsets?
        offset_vals = {
            name: jnp.array(ustrip(obs_unit, offsets[name])) for name in offsets
        }
        vals.update(offset_vals)

        return vals

    def _build_marginalized_linear(  # noqa: C901
        self,
        params: MarginalizedParameters,
        offsets: dict[str, AbstractQuantity],
    ) -> _MargLinearComponents:
        """Build the MarginalizedLinear distribution and residual observations.

        Shared by `_log_prob_marginalized` and `sample_conditional_linear`.
        Handles column classification, explicit-parameter subtraction, prior
        assembly, and construction of the ``MarginalizedLinear`` instance.
        """
        X = self.design_matrix(params)

        # Unit-stripped values for all non-marginalized linear parameters
        # (including any explicitly-provided offsets).
        linear_params = self.linear_unmarginalized_param_values(params, offsets)

        # Full column ordering of the design matrix: base linear params first,
        # then one column per non-reference instrument (if multi-survey).
        cols = (*params.source_cls.linear_param_names, *(self.instrument_names or ()))
        base_names = params.source_cls.linear_param_names

        # Classify each column as explicit (value known) or marginalized:
        #  - A base param is explicit if it is NOT in the marginalized set.
        #  - An instrument offset is explicit if an offset value was provided.
        explicit_names = tuple(
            name
            for name in cols
            if (name in base_names and name not in params.marginalized_names)
            or (name not in base_names and name in offsets)
        )
        marg_names = tuple(name for name in cols if name not in explicit_names)

        if len(marg_names) == 0:
            raise ValueError(
                "No marginalized parameters remain after classification — "
                "cannot build MarginalizedLinear"
            )

        # Strip units from observed data and errors.
        obs = self.data._get_obs()
        obs_unit = str(obs.unit)
        arr_obs = jnp.array(ustrip(obs_unit, obs))
        arr_obs_err = jnp.array(ustrip(obs_unit, self.data._get_obs_err()))

        # Now, assemble the linear prior for marginalized parameters:

        # Gather base linear priors for marginalized params.
        if self.linear_marginalized_prior is not None:
            linear_prior: dict[str, PriorDist | LinearPriorCallable] = {
                name: self.linear_marginalized_prior[name]
                for name in base_names
                if name in marg_names
            }
        else:
            linear_prior = {}

        # Append offset priors for marginalized instrument offsets.
        if self.offsets_marginalized_prior is not None:
            for name in self.offsets_marginalized_prior:
                if name in marg_names:
                    linear_prior[name] = self.offsets_marginalized_prior[name]

        # Collect physical units for each marginalized column so
        # _resolve_linear_prior_mvn can convert QuantityDistributions.
        marg_units = {
            n: self.linear_param_units[n]
            for n in marg_names
            if n in self.linear_param_units
        }

        # offset columns share the obs unit — add those too:
        for n in marg_names:
            if self.instrument_names is not None and n in self.instrument_names:
                marg_units[n] = obs_unit

        # If a prior entry is dist.Delta, the parameter is effectively fixed at a known
        # value. Reclassify it as explicit so its contribution is subtracted from the
        # observations rather than marginalized over.
        # NOTE: this only catches Delta entries specified directly in the prior dicts.
        # A LinearPriorCallable that *returns* a Delta is not caught here and will raise
        # TypeError in _resolve_linear_prior_mvn.
        delta_fixed: dict[str, jax.Array] = {}
        for name in list(linear_prior.keys()):
            # NOTE: we modify the linear_prior dict in place, so we need to list() the
            # keys to avoid "dictionary changed size during iteration" errors.
            prior_dist = linear_prior[name]
            if isinstance(prior_dist, QuantityDistribution) and isinstance(
                prior_dist.distribution, dist.Delta
            ):
                unit = marg_units.get(name, obs_unit)
                delta_fixed[name] = jnp.array(
                    ustrip(
                        unit,
                        Quantity(
                            prior_dist.distribution.v, cast("str", prior_dist.unit)
                        ),
                    )
                )
                del linear_prior[name]
            elif isinstance(prior_dist, dist.Delta):
                delta_fixed[name] = jnp.array(prior_dist.v)
                del linear_prior[name]

        if delta_fixed:
            # Reclassify Delta columns from "marginalized" to "explicit"
            explicit_names = (*explicit_names, *delta_fixed)
            marg_names = tuple(n for n in marg_names if n not in delta_fixed)
            linear_params.update(delta_fixed)
            marg_units = {n: v for n, v in marg_units.items() if n not in delta_fixed}

        if len(marg_names) == 0:
            raise ValueError(
                "No marginalized parameters remain after classification — "
                "cannot build MarginalizedLinear"
            )

        # Subtract the contribution of all explicit linear parameters (including any
        # dist.Delta params) from the observations so the MarginalizedLinear only sees
        # the residual.
        if explicit_names:
            idx = jnp.array([cols.index(n) for n in explicit_names])
            y = jnp.array([linear_params[n] for n in explicit_names])
            arr_obs = arr_obs - X[:, idx] @ y

        # Slice design matrix down to marginalized columns only.
        marg_idx = jnp.array([cols.index(n) for n in marg_names])
        X_marg = X[:, marg_idx]

        # Resolve per-param Normal / QuantityDistribution / callable entries
        # into a single diagonal MultivariateNormal prior.
        lp = _resolve_linear_prior_mvn(linear_prior, params, marg_units)

        marg_dist = MarginalizedLinear(
            design_matrix=X_marg,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, arr_obs_err),
        )
        return _MargLinearComponents(
            marg_dist, arr_obs, marg_names, explicit_names, linear_params
        )

    # Log-probability evaluation:

    def log_prob(
        self,
        params: MarginalizedParameters | ParamT,
        offsets: dict[str, AbstractQuantity] | None = None,
    ) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample.

        Dispatches to marginalized or explicit evaluation based on the
        presence of ``linear_prior``.
        """
        if self._has_marginalized:
            return self._log_prob_marginalized(
                cast("MarginalizedParameters", params), offsets or {}
            )
        return self._log_prob_explicit(cast("ParamT", params), offsets or {})

    def _log_prob_marginalized(
        self, params: MarginalizedParameters, offsets: dict[str, AbstractQuantity]
    ) -> jax.Array:
        """Marginalized log-likelihood."""
        c = self._build_marginalized_linear(params, offsets)
        return c.dist.log_prob(c.obs)

    def _log_prob_explicit(
        self, params: ParamT, offsets: dict[str, AbstractQuantity]
    ) -> jax.Array:
        """Explicit (non-marginalized) Gaussian log-likelihood.

        Computes ``Normal(X @ y, obs_err).log_prob(obs)`` where ``X`` is the
        full design matrix and ``y`` is the vector of all linear parameter
        values (base parameters + any per-instrument offsets).
        """
        X = self.design_matrix(params)
        linear_params = self.linear_unmarginalized_param_values(params, offsets)
        y = jnp.array(list(linear_params.values()))

        y_pred = X @ y

        obs = self.data._get_obs()
        obs_unit = str(obs.unit)

        arr_obs = ustrip(obs_unit, obs)
        arr_obs_err = ustrip(obs_unit, self.data._get_obs_err())

        return dist.Normal(y_pred, arr_obs_err).log_prob(arr_obs)

    # Sampling from the conditional posterior over linear parameters:

    def sample_conditional_linear(
        self,
        params: MarginalizedParameters,
        key: jax.Array,
        offsets: dict[str, AbstractQuantity] | None = None,
    ) -> dict[str, AbstractQuantity]:
        """Sample linear parameters from the conditional posterior.

        Draws one sample from the posterior over marginalized linear
        parameters, conditioned on the observed data.  Explicit (non-marginalized)
        parameters — including any whose priors are ``dist.Delta`` — are
        returned unchanged.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear orbital parameters plus any fixed linear parameter values.
        key : jax.Array
            PRNG key for sampling.
        offsets : dict[str, AbstractQuantity] | None
            Explicitly-sampled per-instrument offsets, if any.

        Returns
        -------
        dict[str, AbstractQuantity]
            All linear parameter values (both sampled and explicit) with units.
        """
        c = self._build_marginalized_linear(params, offsets or {})
        sample = c.dist.conditional(c.obs).sample(key)

        obs_unit = str(self.data._get_obs().unit)
        result: dict[str, AbstractQuantity] = {}
        for i, name in enumerate(c.marg_names):
            result[name] = Quantity(
                sample[i], self.linear_param_units.get(name, obs_unit)
            )
        for name in c.explicit_names:
            result[name] = Quantity(
                c.linear_params[name], self.linear_param_units.get(name, obs_unit)
            )
        return result
