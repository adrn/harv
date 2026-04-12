r"""Likelihood functions for Gaia epoch astrometry data.

This module implements the unified :class:`GaiaAstrometryLikelihood` for Gaia
along-scan astrometry.  The class supports two evaluation modes via the same
``log_prob`` interface (inherited from :class:`AbstractLikelihood`):

1. **Marginalized** (``linear_marginalized_prior`` provided, ``params`` is
   :class:`MarginalizedParameters`): analytically marginalizes over some or all
   of the 6 linear astrometric parameters (ra0, dec0, pmra, pmdec, parallax,
   semi_major_axis) given a Gaussian prior.  Supports partial marginalization
   via ``params.marginalized_names``.

2. **Explicit** (``linear_marginalized_prior`` is ``None``, ``params`` is
   :class:`GaiaAstrometryParameters`): evaluates the Gaussian data
   log-likelihood directly at the provided linear parameter values.

The astrometric model is:

.. math::

    y_\mathrm{AL} &= \alpha_0 \cos\psi + \delta_0 \sin\psi \\
        &+ (\mu_\alpha \cos\psi + \mu_\delta \sin\psi) \, dt \\
        &+ \varpi \, H_\varpi(t) \\
        &+ a \, [(A \sin\psi + B \cos\psi) \cos f
        + (F \sin\psi + G \cos\psi) \sin f]

where :math:`A, B, F, G` are Thiele-Innes constants and :math:`f` is the
true anomaly.
"""

from typing import final

import jax
import numpy as np
import quaxed.numpy as jnp
from unxt import ustrip
from unxt.quantity import AllowValue

from harv.data import GaiaAstrometryData
from harv.kepler.orbits import thiele_innes_ABFG
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import (
    _solve_kepler,
)
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    MarginalizedParameters,
)

__all__ = ("GaiaAstrometryLikelihood",)

# NOTE: we need to adopt an internal time unit for the design matrix columns. In most
# astrometry settings, it is reasonable to use years, but we should consider whether
# this should be a customizable value
_AST_TIME_UNIT = "yr"  # for proper motion columns in design matrix


# TODO: here and in RV, we seem to be missing the t_peri or phase_peri parameter in the
# design matrix construction
def _get_design_matrix_gaia_ast(
    data: GaiaAstrometryData,
    params: MarginalizedParameters | GaiaAstrometryParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build the (n_obs, 6) Gaia along-scan design matrix.

    Columns: [ra0, dec0, pmra, pmdec, parallax, semi_major_axis].

    The projection follows the Gaia local plane coordinate (LPC) convention
    from Lindegren & Bastian (GAIA-C3-TN-LU-LL-061-08, Eqs. 4, 6, 8):
    RA direction (a) uses sin(theta) and Thiele-Innes B, G;
    Dec direction (d) uses cos(theta) and Thiele-Innes A, F.
    """
    dt = ustrip(_AST_TIME_UNIT, data.time - data.t_ref)
    scan_angle_rad = ustrip("rad", data.scan_angle)
    cos_psi = jnp.cos(scan_angle_rad)
    sin_psi = jnp.sin(scan_angle_rad)

    _parallax_factor = ustrip(AllowValue, "", data.parallax_factor)
    _cos_i = ustrip(AllowValue, "", params.cos_i)
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    _lon_asc_node = ustrip(AllowValue, "", params.lon_asc_node)

    # Thiele-Innes constants (unit, i.e. semi-major axis = 1)
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(_arg_peri),
        jnp.sin(_arg_peri),
        jnp.cos(_lon_asc_node),
        jnp.sin(_lon_asc_node),
        _cos_i,
    )

    # Along-scan orbital term: w_orbit = (B*cos f + G*sin f)*sin theta
    #                                    + (A*cos f + F*sin f)*cos theta
    semimaj_term = (B * cos_f + G * sin_f) * sin_psi + (A * cos_f + F * sin_f) * cos_psi

    # NOTE: the order here should match the order of the linear parameters in
    # GaiaAstrometryParameters.linear_param_names
    return jnp.stack(
        [
            sin_psi,  # ra0: a = Deltaalpha* -> sin theta
            cos_psi,  # dec0: d = Deltadelta -> cos theta
            sin_psi * dt,  # pmra: mu_alpha* * dt -> sin theta * dt
            cos_psi * dt,  # pmdec: mu_delta * dt -> cos theta * dt
            _parallax_factor,
            semimaj_term,
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Unified Gaia astrometry likelihood
# ---------------------------------------------------------------------------


@final
class GaiaAstrometryLikelihood(
    AbstractLikelihood[GaiaAstrometryData, GaiaAstrometryParameters],
):
    """Unified Gaia astrometry likelihood.

    Supports marginalized and explicit evaluation via the inherited
    :meth:`~AbstractLikelihood.log_prob` interface.

    When ``linear_marginalized_prior`` is provided and ``params`` is a
    :class:`MarginalizedParameters` instance, the likelihood analytically
    marginalizes over the linear astrometric parameters.

    When ``linear_marginalized_prior`` is ``None``, ``params`` must be a full
    :class:`GaiaAstrometryParameters` and the likelihood is evaluated
    explicitly.

    Polynomial trends via ``trend_order`` append higher-order proper-motion
    acceleration columns.  Each order *k* adds two columns (RA and Dec
    components): ``cos(psi)*dt^(k+1)`` and ``sin(psi)*dt^(k+1)``, where
    ``dt = (t - t_ref)`` in the internal astrometry time unit. The ``+1``
    offset is because order-0 proper motion (dt^1) is already in the base
    5-parameter solution.
    """

    trend_order: int = 0

    @property
    def trend_column_names(self) -> tuple[str, ...]:
        """Names of trend columns: two per order (RA + Dec component)."""
        names: list[str] = []
        for k in range(1, self.trend_order + 1):
            names.append(f"trend_ra_{k}")
            names.append(f"trend_dec_{k}")
        return tuple(names)

    def design_matrix(
        self, params: MarginalizedParameters | GaiaAstrometryParameters
    ) -> jax.Array:
        """Build the design matrix, optionally including trend columns."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_gaia_ast(self.data, params, sin_f, cos_f)

        if self.trend_order > 0:
            dt = np.asarray(ustrip(_AST_TIME_UNIT, self.data.time - self.data.t_ref))
            scan_angle_rad = np.asarray(ustrip("rad", self.data.scan_angle))
            cos_psi = jnp.cos(scan_angle_rad)
            sin_psi = jnp.sin(scan_angle_rad)
            # Each order k adds columns sin(theta)*dt^(k+1) (RA) and
            # cos(theta)*dt^(k+1) (Dec), following the LPC convention.
            # k+1 because dt^1 proper motion is already in the base matrix.
            trend_cols = []
            for k in range(1, self.trend_order + 1):
                dt_power = dt ** (k + 1)
                trend_cols.append(sin_psi * dt_power)
                trend_cols.append(cos_psi * dt_power)
            X = jnp.concatenate([X, jnp.stack(trend_cols, axis=-1)], axis=-1)

        return X

    @property
    def linear_param_units(self) -> dict[str, str]:
        """Units of the linear astrometric parameters (incl. trend columns)."""
        u = str(self.data.al_position.unit)
        units: dict[str, str] = {
            "ra0": u,
            "dec0": u,
            "pmra": f"{u}/{_AST_TIME_UNIT}",
            "pmdec": f"{u}/{_AST_TIME_UNIT}",
            "parallax": u,
            "semi_major_axis": u,
        }
        for name in self.trend_column_names:
            units[name] = u
        return units
