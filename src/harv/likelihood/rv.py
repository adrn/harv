"""Likelihood functions for radial velocity data.

This module implements the unified :class:`RVLikelihood` for radial velocity
observations.  The class supports three evaluation modes via the same
:meth:`RVLikelihood.log_prob` interface, determined by the type of input ``params``
and the presence of ``linear_prior`` / ``indicator_matrix``:

1. **Marginalized** (``linear_prior`` provided, ``params`` is
   :class:`MarginalizedParameters`): analytically integrates over the linear RV
   parameters (K, v0) given a Gaussian prior. The prior can depend on the nonlinear
   parameters. Supports partial marginalization (e.g., only marginalizing over K and not
   v0) via ``params.marginalized_names``.  For multi-survey data (``indicator_matrix``
   provided), the per-instrument offset columns are always appended to the marginalized
   design matrix -- partial marginalization of the named parameters (K, v0) works
   simultaneously with multi-survey offset columns.

2. **Explicit** (``linear_prior`` is ``None``, ``params`` is :class:`RVParameters`):
   evaluates the Gaussian data log-likelihood directly at the provided K and v0.
   For multi-survey data (``indicator_matrix`` present), per-instrument offsets are
   included when ``params.offsets`` is provided, giving the full model
   ``K * rv_shape(t) + v0 + dj * I(j)``.  If ``offsets`` is ``None`` with multi-survey
   data, offset corrections are omitted -- pre-correct the data externally or use
   mode 1 to marginalize offsets analytically.

For the SB1 model the RV model is:

    RV(t) = K * [cos(w + f(t)) + e * cos(w)] + v0
           = K * rv_shape(t) + v0

Note on SB2: :func:`_get_design_matrix_sb2` provides the (n_obs, 3) design matrix
for double-lined spectroscopic binaries (columns [K1, K2, v0]), but no likelihood
class uses it yet.  SB2 support requires a dedicated ``SystemData`` container that
does not yet exist -- see the Planned section in docs/spec.md.
"""

from typing import cast

import equinox as eqx
import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Quantity, ustrip
from unxt.quantity import AllowValue

from harv.custom_types import ScalarQSpeed
from harv.data import RadialVelocityData, SourceData
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

__all__ = ("RVLikelihood", "build_rv_indicator_matrix", "stack_rv_datasets")


# ---------------------------------------------------------------------------
# Multi-survey RV helpers
# ---------------------------------------------------------------------------


def stack_rv_datasets(
    rv_datasets: dict[str, RadialVelocityData],
) -> RadialVelocityData:
    """Concatenate multiple RV datasets in dict order into a single one.

    Parameters
    ----------
    rv_datasets : dict[str, RadialVelocityData]
        Ordered mapping of instrument name -> dataset.  Dict order determines
        the row order in the stacked output; it must match the order used when
        building the indicator matrix (see :func:`build_rv_indicator_matrix`).

    Returns
    -------
    RadialVelocityData
        Single dataset containing all observations stacked in dict order.
    """
    ref = next(iter(rv_datasets.values()))
    time_unit = str(ref.time.unit)
    rv_unit = str(ref.rv.unit)
    all_time = jnp.concatenate(
        [ustrip(time_unit, ds.time) for ds in rv_datasets.values()]
    )
    all_rv = jnp.concatenate([ustrip(rv_unit, ds.rv) for ds in rv_datasets.values()])
    all_err = jnp.concatenate(
        [ustrip(rv_unit, ds.rv_err) for ds in rv_datasets.values()]
    )
    return RadialVelocityData(
        time=Quantity(all_time, time_unit),
        rv=Quantity(all_rv, rv_unit),
        rv_err=Quantity(all_err, rv_unit),
    )


def build_rv_indicator_matrix(
    rv_datasets: dict[str, RadialVelocityData],
    reference: str | None = None,
) -> tuple[jax.Array, tuple[str, ...]]:
    """Build indicator matrix for multi-survey RV data.

    Parameters
    ----------
    rv_datasets : dict[str, RadialVelocityData]
        Ordered mapping of instrument name -> dataset.  Dict order must match
        the order used when stacking (see :func:`stack_rv_datasets`).
    reference : str or None
        Name of the reference instrument (its observations get no offset
        column).  Defaults to the first key in ``rv_datasets``.

    Returns
    -------
    indicator_matrix : jax.Array
        Shape ``(n_obs_total, n_non_ref)``.  ``indicator[i, j] = 1`` when
        observation ``i`` belongs to non-reference instrument ``j``.
    instrument_names : tuple[str, ...]
        Names of the non-reference instruments, in column order.

    Raises
    ------
    ValueError
        If ``reference`` is not found in ``rv_datasets``.
    """
    ref_name = reference if reference is not None else next(iter(rv_datasets))
    if ref_name not in rv_datasets:
        msg = f"Reference instrument {ref_name!r} not in {list(rv_datasets)}"
        raise ValueError(msg)
    non_ref_names = [k for k in rv_datasets if k != ref_name]
    n_non_ref = len(non_ref_names)
    rows = []
    for name, ds in rv_datasets.items():
        n_obs = len(ds.time)
        row = jnp.zeros((n_obs, n_non_ref))
        if name != ref_name:
            j = non_ref_names.index(name)
            row = row.at[:, j].set(1.0)
        rows.append(row)
    return jnp.concatenate(rows, axis=0), tuple(non_ref_names)


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

    Not yet called by any likelihood class -- SB2 support requires
    ``SystemData`` (not yet implemented).  See the Planned section in docs/spec.md.
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

    If you have multi-survey data, you probably want to use
    :func:`RVLikelihood.from_source_data` to construct the likelihood instance.

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
        Constructed by stacking all observations and marking which rows belong
        to each non-reference instrument::

            # Instruments A (reference) and B; B observations at rows [3, 7, 11]:
            ind = jnp.zeros((n_obs_total, 1))
            ind = ind.at[[3, 7, 11], 0].set(1.0)

        Column order must match ``instrument_names``.  ``None`` for
        single-instrument data.
    instrument_names : tuple[str, ...] or None
        Names of the non-reference instruments in ``indicator_matrix``
        column order.  Required when ``params.offsets`` is a dict (explicit
        multi-survey evaluation).  ``None`` for single-instrument data or
        when explicit offsets are not used.

    Examples
    --------
    **Single-instrument, fully marginalized:**

    >>> linear_prior = dist.MultivariateNormal(
    ...     loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * 100.0
    ... )
    >>> lik = RVLikelihood(data=rv_data, linear_prior=linear_prior)
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    **Partial marginalization** -- fix K, marginalize v0 only:

    >>> params = RVParameters.marginalized(
    ...     "v0",                              # only v0 is integrated out
    ...     period=Quantity(200.0, "day"),
    ...     eccentricity=0.3,
    ...     phase_peri=0.1,
    ...     arg_peri=1.2,
    ...     K=Quantity(10.0, "km/s"),          # K held fixed
    ... )
    >>> linear_prior_1d = dist.Normal(0.0, 100.0)  # 1-D prior on v0
    >>> lik = RVLikelihood(data=rv_data, linear_prior=linear_prior_1d)
    >>> lp = lik.log_prob(params)

    **Multi-survey marginalized** -- two instruments, joint prior on [K, v0, d]:

    >>> ind = jnp.zeros((n_obs_total, 1))
    >>> ind = ind.at[[3, 7, 11], 0].set(1.0)   # B observations at these rows
    >>> prior_3d = dist.MultivariateNormal(
    ...     loc=jnp.zeros(3), covariance_matrix=jnp.eye(3) * 100.0
    ... )
    >>> lik = RVLikelihood(
    ...     data=stacked_rv, linear_prior=prior_3d, indicator_matrix=ind,
    ...     instrument_names=("B",),
    ... )
    >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    **Multi-survey, fix K, marginalize v0 and d** -- (1+k)-D prior:

    >>> params = RVParameters.marginalized(
    ...     "v0",
    ...     period=Quantity(200.0, "day"), eccentricity=0.3,
    ...     phase_peri=0.1, arg_peri=1.2,
    ...     K=Quantity(10.0, "km/s"),
    ... )
    >>> prior_2d = dist.MultivariateNormal(
    ...     loc=jnp.zeros(2), covariance_matrix=jnp.eye(2) * 100.0
    ... )
    >>> lik = RVLikelihood(
    ...     data=stacked_rv, linear_prior=prior_2d, indicator_matrix=ind,
    ...     instrument_names=("B",),
    ... )
    >>> lp = lik.log_prob(params)

    **Explicit multi-survey** -- named offsets per instrument:

    >>> params = RVParameters(
    ...     period=Quantity(200.0, "day"), eccentricity=0.3,
    ...     phase_peri=0.0, arg_peri=1.0,
    ...     K=Quantity(30.0, "km/s"), v0=Quantity(0.0, "km/s"),
    ...     offsets={"B": Quantity(5.0, "km/s")},
    ... )
    >>> lik = RVLikelihood(
    ...     data=stacked_rv, indicator_matrix=ind, instrument_names=("B",)
    ... )
    >>> log_lik = lik.log_prob(params)
    """

    data: RadialVelocityData
    linear_prior: dist.MultivariateNormal | LinearPriorCallable | None = None
    indicator_matrix: jax.Array | None = None
    instrument_names: tuple[str, ...] | None = eqx.field(static=True, default=None)

    @property
    def linear_names(self) -> tuple[str, ...]:
        """All linear parameter names in design-matrix column order.

        Includes the base SB1 parameters (K, v0) and, for multi-survey data,
        one ``offset_<name>`` entry per non-reference instrument.
        """
        base = RVParameters.linear_param_names
        if self.instrument_names is not None:
            return base + tuple(f"offset_{n}" for n in self.instrument_names)
        return base

    @property
    def linear_units(self) -> tuple[str, ...]:
        """Unit string for each linear parameter (all share the RV data unit)."""
        rv_unit = str(self.data.rv.unit)
        return tuple(rv_unit for _ in self.linear_names)

    @classmethod
    def from_source_data(
        cls,
        data: SourceData | RadialVelocityData,
        linear_prior: dist.MultivariateNormal | LinearPriorCallable | None = None,
        *,
        reference: str | None = None,
    ) -> "RVLikelihood":
        """Construct an ``RVLikelihood`` from a ``SourceData``.

        Handles multi-survey stacking automatically.

        For single-instrument data this is equivalent to
        ``RVLikelihood(data=data, linear_prior=linear_prior)``.  For
        ``SourceData`` with multiple RV datasets the observations are stacked
        in dict order and an indicator matrix is built automatically -- the
        caller never needs to touch :func:`stack_rv_datasets` or
        :func:`build_rv_indicator_matrix` directly.

        Parameters
        ----------
        data : SourceData or RadialVelocityData
            Input data.  If a ``SourceData`` with more than one
            ``RadialVelocityData`` dataset, observations are stacked and an
            indicator matrix is built.
        linear_prior : MultivariateNormal or LinearPriorCallable or None
            Gaussian prior over the marginalized linear parameters.  Must be
            dimensioned for ``2 + n_non_ref`` parameters when
            ``indicator_matrix`` is present.  ``None`` for explicit
            evaluation.
        reference : str or None
            Name of the reference instrument (receives no offset column).
            Defaults to the first RV dataset in ``data``.

        Returns
        -------
        RVLikelihood

        Examples
        --------
        >>> source = SourceData(harps=harps_data, espresso=espresso_data)
        >>> prior = dist.MultivariateNormal(
        ...     loc=jnp.zeros(3), covariance_matrix=jnp.eye(3) * 100.0
        ... )
        >>> lik = RVLikelihood.from_source_data(
        ...     source, prior, reference="harps"
        ... )
        >>> # Marginalizes [K, v0, delta_espresso] jointly.
        >>> log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)
        """
        if isinstance(data, RadialVelocityData):
            return cls(data=data, linear_prior=linear_prior)

        rv_datasets = data.get_datasets_by_type(RadialVelocityData)
        if not rv_datasets:
            msg = "SourceData contains no RadialVelocityData datasets"
            raise ValueError(msg)
        if len(rv_datasets) == 1:
            return cls(data=next(iter(rv_datasets.values())), linear_prior=linear_prior)

        # Multi-survey: put the reference instrument first so stacking order
        # is deterministic, then build the indicator matrix.
        ref_name = reference if reference is not None else next(iter(rv_datasets))
        if ref_name not in rv_datasets:
            msg = f"Reference instrument {ref_name!r} not in {list(rv_datasets)}"
            raise ValueError(msg)
        ordered = {ref_name: rv_datasets[ref_name]} | {
            k: v for k, v in rv_datasets.items() if k != ref_name
        }
        stacked = stack_rv_datasets(ordered)
        indicator, names = build_rv_indicator_matrix(ordered, reference=ref_name)
        return cls(
            data=stacked,
            linear_prior=linear_prior,
            indicator_matrix=indicator,
            instrument_names=names,
        )

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
    ) -> dict[str, ScalarQSpeed]:
        """Sample linear parameters from the conditional posterior.

        Builds a ``MarginalizedLinear`` from the full design matrix (including
        any indicator columns), the resolved linear prior, and the data errors,
        then draws one sample from the posterior conditioned on the observed
        data.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear orbital parameters (period, eccentricity, etc.).
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        dict[str, ScalarQSpeed]
            Sampled linear parameters as a dict.  Keys follow
            ``RVParameters.linear_param_names`` (``"K"``, ``"v0"``) plus any
            instrument names from ``self.instrument_names`` for multi-survey
            data.  Values are :class:`~unxt.Quantity` in the RV data unit.
        """
        X = self.design_matrix(params)
        lp = _resolve_linear_prior(
            cast("dist.MultivariateNormal", self.linear_prior), params
        )
        rv_unit = self.data.rv.unit
        marg = MarginalizedLinear(
            design_matrix=X,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, ustrip(rv_unit, self.data.rv_err)),
        )
        sample = marg.conditional(ustrip(rv_unit, self.data.rv)).sample(key)

        names: tuple[str, ...] = RVParameters.linear_param_names
        if self.indicator_matrix is not None and self.instrument_names is not None:
            names = names + self.instrument_names
        return {name: Quantity(sample[i], rv_unit) for i, name in enumerate(names)}

    # -- private helpers ----------------------------------------------------

    def _log_prob_marginalized(self, params: MarginalizedParameters) -> jax.Array:
        """Marginalized log-likelihood.

        Delegates to ``_marginalize_partial`` (defined on ``AbstractLikelihood``)
        which handles both partial/full marginalization of named linear
        parameters and any multi-survey indicator columns.
        """
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_sb1(params, sin_f, cos_f)
        if self.indicator_matrix is not None:
            X = jnp.concatenate([X, self.indicator_matrix], axis=-1)
        rv_unit = self.data.rv.unit
        rv_obs = jnp.asarray(ustrip(rv_unit, self.data.rv))
        rv_err = jnp.asarray(ustrip(rv_unit, self.data.rv_err))
        return self._marginalize_partial(
            params,
            X,
            rv_obs,
            rv_err,
            self.linear_names,
            self.linear_units,
            cast("dist.MultivariateNormal", self.linear_prior),
        )

    def _log_prob_explicit(self, params: RVParameters) -> jax.Array:
        """Explicit log-likelihood with all parameters specified.

        Evaluates the Gaussian data log-likelihood at the provided K, v0, and
        (optionally) per-instrument offsets.  When ``params.offsets`` is not
        ``None`` and ``indicator_matrix`` is present on the likelihood, the
        full multi-survey model ``K * rv_shape(t) + v0 + dj * I(j)`` is
        evaluated.  Otherwise only the base SB1 model ``K * X(t) + v0`` is used.
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
                [
                    ustrip(AllowValue, rv_unit, params.offsets[name])
                    for name in self.instrument_names
                ]
            )
            X = jnp.concatenate([X, self.indicator_matrix], axis=-1)
            linear_params = jnp.concatenate([jnp.array([K, v0]), offset_vals])
        else:
            linear_params = jnp.array([K, v0])

        rv_pred = X @ linear_params
        return dist.Normal(rv_pred, rv_err).log_prob(rv_obs).sum()
