"""Likelihood functions for radial velocity data.

This module implements the unified :class:`RVLikelihood` for radial velocity
observations.  The class supports three evaluation modes via the same
:meth:`RVLikelihood.log_prob` interface, determined by the type of input ``params``
and the presence of ``linear_prior`` / ``indicator_matrix``:

1. **Marginalized** (``linear_prior`` provided, ``params`` is
   :class:`MarginalizedParameters`): analytically integrates over the linear RV
   parameters (rv_semiamp, v_sys) given a Gaussian prior. The prior can depend on the
   nonlinear parameters. Supports partial marginalization (e.g., only marginalizing over
   rv_semiamp and not v_sys) via ``params.marginalized_names``.  For multi-survey data
   (``indicator_matrix`` provided), the per-instrument offset columns are always appended
   to the marginalized design matrix -- partial marginalization of the named parameters
   (rv_semiamp, v_sys) works simultaneously with multi-survey offset columns.

2. **Explicit** (``linear_prior`` is ``None``, ``params`` is :class:`RVParameters`):
   evaluates the Gaussian data log-likelihood directly at the provided rv_semiamp
   and v_sys. For multi-survey data (``indicator_matrix`` present), per-instrument
   offsets are included when ``params.offsets`` is provided, giving the full model
   ``rv_semiamp * rv_shape(t) + v_sys + dj * I(j)``.  If ``offsets`` is ``None`` with multi-survey
   data, offset corrections are omitted -- pre-correct the data externally or use
   mode 1 to marginalize offsets analytically.

For the SB1 model the RV model is:

    RV(t) = rv_semiamp * [cos(w + f(t)) + e * cos(w)] + v_sys
           = rv_semiamp * rv_shape(t) + v_sys

Note on SB2: :func:`_get_design_matrix_sb2` provides the (n_obs, 3) design matrix
for double-lined spectroscopic binaries (columns [K1, K2, v_sys]), but no likelihood
class uses it yet.  SB2 support requires a dedicated ``SystemData`` container that
does not yet exist -- see the Planned section in docs/spec.md.
"""

from typing import final

import jax
import quaxed.numpy as jnp
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.data import RVData
from harv.kepler.orbits import rv_shape as _rv_shape
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import (
    _solve_kepler,
)
from harv.likelihood.params import (
    AbstractParameters,
    MarginalizedParameters,
    RVParameters,
)

__all__ = ("RVLikelihood",)


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

    Not yet called by any likelihood class -- SB2 support requires
    ``SystemData`` (not yet implemented).  See the Planned section in docs/spec.md.
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


@final
class RVLikelihood(AbstractLikelihood[RVData, RVParameters]):
    """Unified RV likelihood supporting marginalized and explicit evaluation.

    This is an internal class that implements the core RV likelihood logic.

    When ``linear_marginalized_prior`` is provided and ``params`` is a
    :class:`MarginalizedParameters` instance, the likelihood analytically marginalizes
    over the linear parameters (rv_semiamp, v_sys), and optionally per-instrument offsets when
    ``indicator_matrix`` is supplied.

    When ``linear_marginalized_prior`` is ``None``, ``params`` must be a full
    :class:`RVParameters` and the likelihood is evaluated explicitly.

    Parameters
    ----------
    data : RVData
        Radial velocity observations.
    linear_marginalized_prior : dict[str, PriorDist | LinearPriorCallable] or None
        Per-parameter Gaussian priors for linear parameters to be analytically
        marginalized.  Keys are parameter names (``"rv_semiamp"``, ``"v_sys"``).  Values
        are ``dist.Normal``, ``QuantityDistribution(dist.Normal(...), unit)``,
        or a callable ``(params) -> dist.Normal``.  ``None`` for explicit
        evaluation.
    offsets_marginalized_prior : Mapping[str, PriorDist | LinearPriorCallable] or None
        Per-instrument Gaussian priors for offset parameters to be analytically
        marginalized.  Keys are instrument names matching ``instrument_names``.
        Same value types as ``linear_marginalized_prior``.  ``None`` when
        offsets are explicitly sampled or single-instrument data.
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

    Examples
    --------
    Single-instrument, marginalized over rv_semiamp and v_sys::

        >>> lik = RVLikelihood(
        ...     data=rv_data,
        ...     linear_marginalized_prior={
        ...         "rv_semiamp": QuantityDistribution(dist.Normal(0., 100.), "km/s"),
        ...         "v_sys": QuantityDistribution(dist.Normal(0., 100.), "km/s"),
        ...     },
        ... )
        >>> ll = lik.log_prob(marg_params)

    Single-instrument, explicit evaluation::

        >>> lik = RVLikelihood(data=rv_data)
        >>> ll = lik.log_prob(full_rv_params)
    """

    def design_matrix(self, params: MarginalizedParameters | RVParameters) -> jax.Array:
        """Build the _full_ design matrix for the given parameters.

        Importantly, this includes columns for all linear parameters, including those
        that are not marginalized! The functions that call this internally must remove
        columns corresponding to fixed parameters before constructing the
        ``MarginalizedLinear`` instance.
        """
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_sb1(params, sin_f, cos_f)

        if self.indicator_matrix is not None:
            X = jnp.concatenate([X, self.indicator_matrix], axis=-1)
        return X

    @property
    def linear_param_units(self) -> dict[str, str]:
        """Units of the linear parameters."""
        u = str(self.data.rv.unit)
        return {"rv_semiamp": u, "v_sys": u}
