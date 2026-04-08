"""Likelihood functions for radial velocity data.

This module implements the unified :class:`RVLikelihood` for radial velocity
observations.  The class supports three evaluation modes via the same
:meth:`RVLikelihood.log_prob` interface, determined by the type of input ``params``
and the presence of ``linear_prior`` / ``indicator_matrix``:

1. **Marginalized** (``linear_prior`` provided, ``params`` is
   :class:`MarginalizedParameters`): analytically integrates over the linear RV
   parameters (K, v0) given a Gaussian prior. The prior can depend on the nonlinear
   parameters. This supports partial marginalization (e.g., only marginalizing over K
   and not v0) via ``params.marginalized_names``.

2. **Multi-survey marginalized** (``linear_prior`` provided, ``indicator_matrix``
   provided): appends per-instrument-offset columns to the design matrix and
   marginalizes ``[K, v0, d1, ..., dk]`` jointly, where the d1, ..., dk are the
   per-instrument offsets (for k+1 instruments - one is chosen as the reference).
   Partial marginalization is not yet supported in the multi-survey case. TODO: support
   partial marginalization for multi-survey marginalization.

3. **Explicit** (``linear_prior`` is ``None``, ``params`` is :class:`RVParameters`):
   evaluates the Gaussian data log-likelihood directly at the provided K and v₀.
   For multi-survey data (``indicator_matrix`` present), per-instrument offsets are
   included when ``params.offsets`` is provided (shape ``(n_non_ref,)``), giving the
   full model ``K·rv_shape(t) + v₀ + δⱼ·I(j)``.  If ``offsets`` is ``None`` with
   multi-survey data, offset corrections are omitted — pre-correct the data
   externally or use mode 2 to marginalize offsets analytically.

For the SB1 model the RV model is:

    RV(t) = K · [cos(ω + f(t)) + e·cos(ω)] + v0
           = K · rv_shape(t) + v0

Note: SB2 support (two semi-amplitudes K₁, K₂) is not yet available.  It
requires a dedicated ``SystemData`` container — see §Planned in docs/spec.md.
"""

from typing import Any, cast

import equinox as eqx
import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro_ext.distributions import MarginalizedLinear
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.data import RadialVelocityData
from harv.kepler.orbits import rv_shape as _rv_shape
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import (
    LinearPriorCallable,
    _resolve_linear_prior,
    _solve_kepler,
)
from harv.likelihood.params import (
    AbstractParameters,
    MarginalizedParameters,
    RVParameters,
)

__all__ = ("RVLikelihood",)

_RV_LINEAR_NAMES: tuple[str, ...] = RVParameters.linear_param_names


def _get_design_matrix_sb1(
    params: AbstractParameters | MarginalizedParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build (n_obs, 2) design matrix for SB1: columns [rv_amplitude, 1]."""
    arg_peri = ustrip(AllowValue, "", params.arg_peri)
    rv_shape = _rv_shape(sin_f, cos_f, params.eccentricity, arg_peri)
    return jnp.column_stack([rv_shape, jnp.ones_like(rv_shape)])


def _get_design_matrix_sb2(
    params: AbstractParameters | MarginalizedParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
    primary: bool,
) -> jax.Array:
    """Build (n_obs, 3) design matrix for SB2: columns [K1, K2, v0].

    For primary: [X(t), 0, 1].  For secondary: [0, -X(t), 1].

    TODO: SB2 support requires ``SystemData`` (not yet implemented).
    """
    arg_peri = ustrip(AllowValue, "", params.arg_peri)
    rv_shape = _rv_shape(sin_f, cos_f, params.eccentricity, arg_peri)

    if primary:
        return jnp.column_stack(
            [rv_shape, jnp.zeros_like(rv_shape), jnp.ones_like(rv_shape)]
        )
    return jnp.column_stack(
        [jnp.zeros_like(rv_shape), -rv_shape, jnp.ones_like(rv_shape)]
    )


class RVLikelihood(AbstractLikelihood[MarginalizedParameters | RVParameters]):
    """Unified RV likelihood supporting marginalized and explicit evaluation.

    When ``linear_prior`` is provided and ``params`` is a
    :class:`MarginalizedParameters` instance, the likelihood analytically marginalizes
    over the linear parameters (K, v0), and optionally per-instrument offsets when
    ``indicator_matrix`` is supplied.

    When ``linear_prior`` is ``None``, ``params`` must be a full :class:`RVParameters`
    and the likelihood is evaluated explicitly.

    Parameters
    ----------
    data : RadialVelocityData
        Radial velocity observations.
    linear_prior : dist.MultivariateNormal or LinearPriorCallable or None
        Gaussian prior over the marginalized linear parameters.  ``None``
        for explicit evaluation.
    indicator_matrix : jax.Array or None
        For multi-survey RV: float indicator matrix of shape
        ``(n_obs_total, n_non_ref)`` where ``indicator_matrix[i, j] = 1``
        when observation ``i`` belongs to non-reference instrument ``j``.
        Column order must match ``instrument_names`` when per-instrument
        offsets are used.  The columns are appended to the design matrix
        so that ``[K, v₀, δ₁, …, δₖ]`` are marginalized jointly.
        ``None`` for single-instrument data.
    instrument_names : tuple[str, ...] or None
        Names of the non-reference instruments in ``indicator_matrix``
        column order.  Required when ``params.offsets`` is a dict (explicit
        multi-survey evaluation); unused for marginalized evaluation.
        ``None`` for single-instrument data or when explicit offsets are
        not used.

    Examples
    --------
    **Single-instrument, fully marginalized linear parameters:**

    >>> import numpyro.distributions as dist
    >>> from unxt import Quantity
    >>> from harv.likelihood.rv import RVLikelihood
    >>> from harv.likelihood.params import RVParameters
    >>> linear_prior = dist.MultivariateNormal(
    ...     loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * 100.0
    ... )
    >>> lik = RVLikelihood(data=rv_data, linear_prior=linear_prior)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    **Partial marginalization** — fix K, marginalize v0 only:

    >>> params = RVParameters.marginalized(
    ...     period=Quantity(200.0, "day"),
    ...     eccentricity=0.3,
    ...     phase_peri=0.1,
    ...     arg_peri=Quantity(1.2, "rad"),
    ...     marginalized_names=("v0",),  # only v0 is integrated out
    ...     K=Quantity(10.0, "km/s"),    # K is held fixed
    ... )
    >>> linear_prior_1d = dist.Normal(0.0, 100.0)  # prior on v0 alone
    >>> lik = RVLikelihood(data=rv_data, linear_prior=linear_prior_1d)
    >>> lp = lik.log_prob(params)

    **Multi-survey marginalized** — two instruments, one reference:

    >>> # Stack all observations; indicator_matrix marks non-reference rows.
    >>> # If instrument B has observations at rows [3, 7, 11]:
    >>> ind = jnp.zeros((n_obs_total, 1))
    >>> ind = ind.at[[3, 7, 11], 0].set(1.0)
    >>> lik = RVLikelihood(
    ...     data=stacked_rv, linear_prior=prior_3d, indicator_matrix=ind,
    ...     instrument_names=("ESPRESSO",),
    ... )
    >>> # Marginalizes [K, v0, delta_ESPRESSO] jointly.
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    **Explicit multi-survey** — named offsets per instrument:

    >>> params = RVParameters(
    ...     period=Quantity(200.0, "day"), eccentricity=0.3,
    ...     phase_peri=0.0, arg_peri=1.0,
    ...     K=Quantity(30.0, "km/s"), v0=Quantity(0.0, "km/s"),
    ...     offsets={"ESPRESSO": Quantity(5.0, "km/s")},
    ... )
    >>> lik = RVLikelihood(
    ...     data=stacked_rv, indicator_matrix=ind,
    ...     instrument_names=("ESPRESSO",),
    ... )
    >>> log_lik = lik.log_prob(params)
    """

    data: RadialVelocityData
    linear_prior: dist.MultivariateNormal | LinearPriorCallable | None = None
    indicator_matrix: jax.Array | None = None
    instrument_names: tuple[str, ...] | None = eqx.field(static=True, default=None)

    def design_matrix(self, params: MarginalizedParameters | RVParameters) -> jax.Array:
        """Build the full design matrix for the given parameters.

        Returns shape ``(n_obs, n_cols)`` where ``n_cols`` is 2 for single-
        instrument data or ``2 + n_non_ref`` for multi-survey data.
        """
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_sb1(params, sin_f, cos_f)
        if self.indicator_matrix is not None:
            X = jnp.concatenate([X, self.indicator_matrix], axis=-1)
        return X

    def log_prob(self, params: MarginalizedParameters | RVParameters) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample.

        Dispatches to marginalized or explicit evaluation based on the
        presence of ``linear_prior``.
        """
        if self.linear_prior is None:
            return self._log_prob_explicit(cast("RVParameters", params))
        return self._log_prob_marginalized(cast("MarginalizedParameters", params))

    def sample_conditional_linear(
        self, params: MarginalizedParameters, key: jax.Array
    ) -> jax.Array:
        """Sample linear parameters from the conditional posterior.

        Builds a ``MarginalizedLinear`` from the design matrix, the resolved
        linear prior, and the data errors, then draws one sample from the
        posterior conditioned on the observed data.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear orbital parameters (period, eccentricity, etc.).
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        jax.Array
            Sampled linear parameter vector of length ``n_cols``, ordered
            ``[K, v₀]`` for single-instrument or ``[K, v₀, δ₁, …]`` for
            multi-survey.  The vector is unit-free (values in data units).
        """
        X = self.design_matrix(params)
        lp = _resolve_linear_prior(self.linear_prior, params)
        rv_unit = self.data.rv.unit
        marg = MarginalizedLinear(
            design_matrix=X,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, ustrip(rv_unit, self.data.rv_err)),
        )
        return marg.conditional(ustrip(rv_unit, self.data.rv)).sample(key)

    # -- private helpers ----------------------------------------------------

    def _log_prob_marginalized(self, params: MarginalizedParameters) -> jax.Array:
        """Marginalized log-likelihood — dispatches to single or multi-survey."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_sb1(params, sin_f, cos_f)  # (n_obs, 2)
        rv_unit = self.data.rv.unit
        rv_obs = jnp.asarray(ustrip(rv_unit, self.data.rv))
        rv_err = jnp.asarray(ustrip(rv_unit, self.data.rv_err))

        if self.indicator_matrix is not None:
            return self._marg_multi_survey(params, X, rv_obs, rv_err)
        return self._marg_single_survey(params, X, rv_obs, rv_err, rv_unit)

    def _marg_single_survey(
        self,
        params: MarginalizedParameters,
        X: jax.Array,
        rv_obs: jax.Array,
        rv_err: jax.Array,
        rv_unit: Any,
    ) -> jax.Array:
        """Single-instrument: partial or full marginalization over [K, v0].

        When all linear params are marginalized (the common case), the
        subtraction step is skipped and the full (n_obs, 2) design matrix is
        passed to ``MarginalizedLinear``.  When some params are held fixed
        (partial marginalization), their contribution is subtracted from the
        data before marginalizing the rest.
        """
        marg_names = tuple(
            n for n in _RV_LINEAR_NAMES if n in params.marginalized_names
        )
        fixed_names = tuple(
            n for n in _RV_LINEAR_NAMES if n not in params.marginalized_names
        )

        if fixed_names:
            fixed_vals = jnp.array(
                [ustrip(AllowValue, rv_unit, getattr(params, n)) for n in fixed_names]
            )
            fixed_idx = jnp.array([_RV_LINEAR_NAMES.index(n) for n in fixed_names])
            rv_obs = rv_obs - X[:, fixed_idx] @ fixed_vals

        marg_idx = jnp.array([_RV_LINEAR_NAMES.index(n) for n in marg_names])
        lp = _resolve_linear_prior(self.linear_prior, params)
        return MarginalizedLinear(
            design_matrix=X[:, marg_idx],
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, rv_err),
        ).log_prob(rv_obs)

    def _marg_multi_survey(
        self,
        params: MarginalizedParameters,
        X: jax.Array,
        rv_obs: jax.Array,
        rv_err: jax.Array,
    ) -> jax.Array:
        """Multi-survey: marginalize [K, v0, d1, ..., dk] jointly.

        The indicator columns are appended to the base (n_obs, 2) design
        matrix and all linear parameters are marginalized in one shot.
        Partial marginalization is not yet supported in the multi-survey case.
        """
        ind = cast("jax.Array", self.indicator_matrix)  # checked by caller
        X_full = jnp.concatenate([X, ind], axis=-1)
        lp = _resolve_linear_prior(self.linear_prior, params)
        return MarginalizedLinear(
            design_matrix=X_full,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, rv_err),
        ).log_prob(rv_obs)

    def _log_prob_explicit(self, params: RVParameters) -> jax.Array:
        """Explicit log-likelihood with all parameters specified.

        Evaluates the Gaussian data log-likelihood at the provided K, v₀, and
        (optionally) per-instrument offsets.  When ``params.offsets`` is not
        ``None`` and ``indicator_matrix`` is present on the likelihood, the
        full multi-survey model ``K·rv_shape(t) + v₀ + δⱼ·I(j)`` is
        evaluated.  Otherwise only the base SB1 model ``K·X(t) + v₀`` is used.
        """
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_sb1(params, sin_f, cos_f)

        rv_unit = self.data.rv.unit
        rv_obs = ustrip(rv_unit, self.data.rv)
        rv_err = ustrip(rv_unit, self.data.rv_err)
        K = ustrip(AllowValue, rv_unit, params.K)
        v0 = ustrip(AllowValue, rv_unit, params.v0)

        if (
            params.offsets is not None
            and self.indicator_matrix is not None
            and self.instrument_names is not None
        ):
            offset_vals = jnp.array(
                [ustrip(AllowValue, rv_unit, params.offsets[name])
                 for name in self.instrument_names]
            )
            X = jnp.concatenate([X, self.indicator_matrix], axis=-1)
            linear_params = jnp.concatenate([jnp.array([K, v0]), offset_vals])
        else:
            linear_params = jnp.array([K, v0])

        rv_pred = X @ linear_params
        return dist.Normal(rv_pred, rv_err).log_prob(rv_obs).sum()
