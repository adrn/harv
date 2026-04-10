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
    See Appendix A of https://arxiv.org/abs/2206.05726.
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

    semimaj_term = (A * sin_psi + B * cos_psi) * cos_f + (
        F * sin_psi + G * cos_psi
    ) * sin_f

    # NOTE: the order here should match the order of the linear parameters in
    # GaiaAstrometryParameters.linear_param_names
    return jnp.stack(
        [
            cos_psi,
            sin_psi,
            cos_psi * dt,
            sin_psi * dt,
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

    Parameters
    ----------
    data : GaiaAstrometryData
        Gaia epoch astrometry observations.
    linear_marginalized_prior : dict[str, PriorDist | LinearPriorCallable] or None
        Per-parameter Gaussian priors for linear parameters to be analytically
        marginalized.  Keys are parameter names (``"ra0"``, ``"dec0"``,
        ``"pmra"``, ``"pmdec"``, ``"parallax"``, ``"semi_major_axis"``).
        Values are ``dist.Normal``,
        ``QuantityDistribution(dist.Normal(...), unit)``, or a callable
        ``(params) -> dist.Normal``.  ``None`` for explicit evaluation.

    Examples
    --------
    Marginalized over all 6 linear parameters::

        lik = GaiaAstrometryLikelihood(
            data=gaia_data,
            linear_marginalized_prior={
                "ra0": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
                "dec0": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
                "pmra": QuantityDistribution(dist.Normal(0., 1e3), "mas/yr"),
                "pmdec": QuantityDistribution(dist.Normal(0., 1e3), "mas/yr"),
                "parallax": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
                "semi_major_axis": QuantityDistribution(dist.Normal(0., 1e3), "mas"),
            },
        )
        ll = lik.log_prob(marg_params)

    Explicit::

        lik = GaiaAstrometryLikelihood(data=gaia_data)
        ll = lik.log_prob(full_astro_params)
    """

    def design_matrix(
        self, params: MarginalizedParameters | GaiaAstrometryParameters
    ) -> jax.Array:
        """Build the (n_obs, 6) design matrix for the given parameters."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        return _get_design_matrix_gaia_ast(self.data, params, sin_f, cos_f)

    @property
    def linear_param_units(self) -> dict[str, str]:
        """Units of the linear astrometric parameters."""
        u = str(self.data.al_position.unit)
        return {
            "ra0": u,
            "dec0": u,
            "pmra": f"{u}/{_AST_TIME_UNIT}",
            "pmdec": f"{u}/{_AST_TIME_UNIT}",
            "parallax": u,
            "semi_major_axis": u,
        }
