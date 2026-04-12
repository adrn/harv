"""Likelihood functions for radial velocity data.

This module implements :class:`RVLikelihood` for single-lined spectroscopic
binaries (SB1) and :class:`SB2RVLikelihood` for double-lined systems (SB2).

Both classes support three evaluation modes via the inherited
:meth:`~AbstractLikelihood.log_prob` interface:

1. **Marginalized** (``linear_marginalized_prior`` provided, ``params`` is
   :class:`MarginalizedParameters`): analytically integrates over the linear RV
   parameters given a Gaussian prior.
2. **Explicit** (``linear_marginalized_prior`` is ``None``): evaluates at
   the provided linear parameter values.

For the SB1 model the RV model is:

    RV(t) = rv_semiamp * [cos(w + f(t)) + e * cos(w)] + v_sys
           = rv_semiamp * rv_shape(t) + v_sys

Polynomial trends may be appended to the design matrix via the ``trend_order``
field, adding columns ``(t - t_ref)^k`` for ``k = 1 .. trend_order``.  Trend
coefficients are always analytically marginalized and require priors in
``trend_marginalized_prior``.

For the SB2 model, primary and secondary RV observations are stacked and the
design matrix has three linear columns [K1, K2, v_sys].  The secondary's
amplitude column is negated (anti-phase motion).
"""

from typing import final

import jax
import numpy as np
import quaxed.numpy as jnp
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.data import RVData, SystemData
from harv.kepler.orbits import rv_shape as _rv_shape
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import (
    _solve_kepler,
)
from harv.likelihood.params import (
    AbstractParameters,
    MarginalizedParameters,
    RVParameters,
    SB2RVParameters,
)

__all__ = ("RVLikelihood", "SB2RVLikelihood")


def _get_design_matrix_sb1(
    params: AbstractParameters | MarginalizedParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build (n_obs, 2) design matrix for SB1: columns [rv_amplitude, 1]."""
    arg_peri = ustrip(AllowValue, "rad", params.arg_peri)
    rv_shape = _rv_shape(sin_f, cos_f, params.eccentricity, arg_peri)

    # NOTE: the order here should match the order of the linear parameters in
    # RVParameters.linear_param_names
    return jnp.column_stack([rv_shape, jnp.ones_like(rv_shape)])


def _get_design_matrix_sb2(
    params: AbstractParameters | MarginalizedParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
    primary: bool,
) -> jax.Array:
    """Build (n_obs, 3) design matrix for SB2: columns [K1, K2, v_sys].

    For primary: [X(t), 0, 1].  For secondary: [0, -X(t), 1].
    """
    arg_peri = ustrip(AllowValue, "rad", params.arg_peri)
    rv_shape = _rv_shape(sin_f, cos_f, params.eccentricity, arg_peri)

    if primary:
        return jnp.column_stack(
            [rv_shape, jnp.zeros_like(rv_shape), jnp.ones_like(rv_shape)]
        )
    return jnp.column_stack(
        [jnp.zeros_like(rv_shape), -rv_shape, jnp.ones_like(rv_shape)]
    )


def _build_trend_columns(
    times: jax.Array | np.ndarray,
    t_ref: jax.Array | float,
    order: int,
) -> jax.Array:
    """Monomial trend basis: columns [(t-t_ref)^1, (t-t_ref)^2, ...].

    The constant (order-0) term is NOT included -- it is already captured by
    ``v_sys`` (RV) or the 5-parameter astrometric solution (astrometry).

    Returns shape ``(n_obs, order)``.

    See "Pluggable trend basis" in ``docs/spec.md`` for the planned
    ``TrendBasis`` protocol that will generalize this helper.
    """
    dt = times - t_ref
    return jnp.column_stack([dt**k for k in range(1, order + 1)])


@final
class RVLikelihood(AbstractLikelihood[RVData, RVParameters]):
    """Unified RV likelihood supporting marginalized and explicit evaluation.

    This is an internal class that implements the core RV likelihood logic.

    When ``linear_marginalized_prior`` is provided and ``params`` is a
    :class:`MarginalizedParameters` instance, the likelihood analytically marginalizes
    over the linear parameters (rv_semiamp, v_sys), and optionally per-instrument offsets
    and polynomial trend coefficients.

    When ``linear_marginalized_prior`` is ``None``, ``params`` must be a full
    :class:`RVParameters` and the likelihood is evaluated explicitly.

    Parameters
    ----------
    data : RVData
        Radial velocity observations.
    linear_marginalized_prior : dict or None
        Per-parameter Gaussian priors for linear parameters (rv_semiamp, v_sys).
    offsets_marginalized_prior : Mapping or None
        Per-instrument Gaussian priors for offset parameters.
    trend_marginalized_prior : Mapping or None
        Per-trend-column Gaussian priors, keyed by ``"trend_1"``, ``"trend_2"``, etc.
    trend_order : int
        Polynomial trend order.  0 = no trend (default), 1 = linear drift,
        2 = quadratic, etc.  The constant term is already captured by ``v_sys``.
    indicator_matrix : jax.Array or None
        Multi-survey indicator matrix.
    instrument_names : tuple[str, ...] or None
        Non-reference instrument names matching indicator_matrix columns.
    """

    trend_order: int = 0

    @property
    def trend_column_names(self) -> tuple[str, ...]:
        """Names of polynomial trend columns in the design matrix."""
        return tuple(f"trend_{k}" for k in range(1, self.trend_order + 1))

    def design_matrix(self, params: MarginalizedParameters | RVParameters) -> jax.Array:
        """Build the full design matrix including trend and offset columns."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_sb1(params, sin_f, cos_f)

        if self.trend_order > 0:
            times = ustrip(str(self.data.time.unit), self.data.time)
            t_ref = ustrip(str(self.data.time.unit), self.data.t_ref)
            trend_cols = _build_trend_columns(times, t_ref, self.trend_order)
            X = jnp.concatenate([X, trend_cols], axis=-1)

        if self.indicator_matrix is not None:
            X = jnp.concatenate([X, self.indicator_matrix], axis=-1)
        return X

    @property
    def linear_param_units(self) -> dict[str, str]:
        """Units of the linear parameters."""
        u = str(self.data.rv.unit)
        return {"rv_semiamp": u, "v_sys": u}


# ---------------------------------------------------------------------------
# SB2 likelihood
# ---------------------------------------------------------------------------


@final
class SB2RVLikelihood(AbstractLikelihood[SystemData, SB2RVParameters]):
    """RV likelihood for double-lined spectroscopic binaries (SB2).

    Stacks primary and secondary RV observations and builds a design matrix
    with three linear columns: [K1, K2, v_sys].  Primary rows have
    ``[rv_shape, 0, 1]``; secondary rows ``[0, -rv_shape, 1]``.

    Supports the same marginalized/explicit modes as :class:`RVLikelihood`.
    Polynomial trends via ``trend_order`` are appended *after* the 3 base
    columns (trend columns span the full stacked observation vector).

    The ``SystemData`` must contain exactly two components. The first component
    (in key order) is treated as the primary (positive K column), the second
    as the secondary (negative K column, anti-phase motion).

    Parameters
    ----------
    data : SystemData
        Container holding two named :class:`RVData` components (e.g.
        ``SystemData(primary=RVData(...), secondary=RVData(...))``).
    linear_marginalized_prior : dict or None
        Priors for ``"rv_semiamp_1"``, ``"rv_semiamp_2"``, ``"v_sys"``.
    trend_marginalized_prior : Mapping or None
        Priors for trend columns, keyed ``"trend_1"``, ``"trend_2"``, etc.
    trend_order : int
        Polynomial trend order (default 0 = no trend).
    """

    trend_order: int = 0

    @property
    def _components(self) -> tuple[RVData, RVData]:
        """Primary and secondary RVData in key order."""
        vals = list(self.data.values())
        return vals[0], vals[1]

    @property
    def trend_column_names(self) -> tuple[str, ...]:
        """Names of polynomial trend columns in the design matrix."""
        return tuple(f"trend_{k}" for k in range(1, self.trend_order + 1))

    def design_matrix(
        self, params: MarginalizedParameters | SB2RVParameters
    ) -> jax.Array:
        """Build the stacked (n_prim + n_sec, 3 + trend_order) design matrix."""
        comp_primary, comp_secondary = self._components

        # Primary component
        sin_f_p, cos_f_p = _solve_kepler(comp_primary, params)
        X_p = _get_design_matrix_sb2(params, sin_f_p, cos_f_p, primary=True)

        # Secondary component (same orbital elements, anti-phase RV)
        sin_f_s, cos_f_s = _solve_kepler(comp_secondary, params)
        X_s = _get_design_matrix_sb2(params, sin_f_s, cos_f_s, primary=False)

        X = jnp.concatenate([X_p, X_s], axis=0)

        if self.trend_order > 0:
            time_unit = str(comp_primary.time.unit)
            t_ref = ustrip(time_unit, comp_primary.t_ref)
            all_times = jnp.concatenate(
                [
                    ustrip(time_unit, comp_primary.time),
                    ustrip(time_unit, comp_secondary.time),
                ]
            )
            trend_cols = _build_trend_columns(all_times, t_ref, self.trend_order)
            X = jnp.concatenate([X, trend_cols], axis=-1)

        return X

    @property
    def linear_param_units(self) -> dict[str, str]:
        """Units of the linear parameters (all in the first component's RV unit)."""
        comp_primary, _ = self._components
        u = str(comp_primary.rv.unit)
        return {"rv_semiamp_1": u, "rv_semiamp_2": u, "v_sys": u}
