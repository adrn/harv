"""Concrete Gaia epoch-astrometry component model.

:class:`GaiaAstrometryModel` wraps :class:`~harv.data.GaiaAstrometryData`,
a parameterization (default
:class:`~harv.models.parametrizations.gaia.StandardGaiaAstrometry`),
and optional extensions to provide ``log_prob`` and
``sample_conditional_linear``.
"""

__all__ = ("GaiaAstrometryModel",)

from typing import Any, final

import jax
import quaxed.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.data import GaiaAstrometryData
from harv.extensions.base import AbstractExtension, ParamInfo
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.models.component import AbstractComponentModel
from harv.models.parametrizations.gaia import StandardGaiaAstrometry

# Internal time unit for proper-motion columns in the design matrix (must
# match the convention in the existing GaiaAstrometryLikelihood).
_AST_TIME_UNIT = "yr"


@final
class GaiaAstrometryModel(AbstractComponentModel):
    """Gaia epoch astrometry component model.

    Supports marginalized and explicit likelihood evaluation for along-scan astrometric
    data.

    Parameters
    ----------
    data : GaiaAstrometryData
        Along-scan epoch astrometry.
    parameterization : StandardGaiaAstrometry
        Declares parameter names/roles and builds the base design matrix.
    extensions : tuple of Extension
        Model extensions (jitter, trends, ...).
    linear_prior : dict or None
        Per-parameter priors for analytic marginalization.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from unxt import Q
    >>> from harv.data import GaiaAstrometryData
    >>> from harv.models.astrometry import GaiaAstrometryModel
    >>> data = GaiaAstrometryData(
    ...     time=Q([0.0, 100.0, 200.0], "day"),
    ...     al_position=Q([0.1, -0.2, 0.05], "mas"),
    ...     al_position_err=Q([0.01, 0.01, 0.01], "mas"),
    ...     scan_angle=Q([0.5, 1.2, 2.8], "rad"),
    ...     parallax_factor=jnp.array([0.3, -0.1, 0.4]),
    ... )
    >>> model = GaiaAstrometryModel(data=data)
    >>> sorted(model._all_nonlinear_names())
    ['arg_peri', 'cos_i', 'eccentricity', 'lon_asc_node', 'period', 'phase_peri']
    """

    data: GaiaAstrometryData
    parameterization: StandardGaiaAstrometry = StandardGaiaAstrometry()
    linear_prior: dict[str, Any] | None = None
    extensions: tuple[AbstractExtension, ...] = ()

    def _param_infos(self) -> tuple[ParamInfo, ...]:
        base = self.parameterization.params()
        ext_params: tuple[ParamInfo, ...] = ()
        for ext in self.extensions:
            ext_params = (*ext_params, *ext.extra_params())
        return (*base, *ext_params)

    def _obs_unit(self) -> str:
        return str(self.data.al_position.unit)

    def _linear_param_units(self) -> dict[str, str]:
        """Astrometric linear params have different units (mas vs mas/yr)."""
        u = self._obs_unit()
        units: dict[str, str] = {
            "ra0": u,
            "dec0": u,
            "pmra": f"{u}/{_AST_TIME_UNIT}",
            "pmdec": f"{u}/{_AST_TIME_UNIT}",
            "parallax": u,
            "semi_major_axis": u,
        }
        # Extension-added linear params default to obs_unit
        for pi in self._param_infos():
            if pi.linear and pi.name not in units:
                units[pi.name] = u
        return units

    def _strip_obs(self) -> tuple[jax.Array, jax.Array]:
        obs_unit = self._obs_unit()
        arr_obs = jnp.array(ustrip(obs_unit, self.data.al_position))
        arr_obs_err = jnp.array(ustrip(obs_unit, self.data.al_position_err))
        return arr_obs, arr_obs_err

    def _solve_kepler(self, nl_values: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        """Solve Kepler's equation from nonlinear parameter values."""
        period = nl_values["period"]
        phase_peri = nl_values["phase_peri"]
        eccentricity = nl_values["eccentricity"]

        t_peri = phase_peri * period
        dt = (self.data.time - self.data.t_ref) - t_peri
        M = mean_anomaly(dt, period)
        sin_f, cos_f = true_anomaly_from_mean(M, eccentricity)
        return ustrip(AllowValue, "", sin_f), ustrip(AllowValue, "", cos_f)

    def _base_design_matrix(self, nl_values: dict[str, Any]) -> jax.Array:
        sin_f, cos_f = self._solve_kepler(nl_values)

        # Prepare auxiliary data arrays
        dt = jnp.array(ustrip(_AST_TIME_UNIT, self.data.time - self.data.t_ref))
        scan_angle_rad = ustrip("rad", self.data.scan_angle)
        sin_psi = jnp.sin(scan_angle_rad)
        cos_psi = jnp.cos(scan_angle_rad)
        parallax_factor = ustrip(AllowValue, "", self.data.parallax_factor)

        # Strip nonlinear values for parameterization
        nl_stripped = {
            "eccentricity": ustrip(AllowValue, "", nl_values["eccentricity"]),
            "arg_peri": ustrip(AllowValue, "rad", nl_values["arg_peri"]),
            "lon_asc_node": ustrip(AllowValue, "rad", nl_values["lon_asc_node"]),
            "cos_i": ustrip(AllowValue, "", nl_values["cos_i"]),
        }

        X = self.parameterization.design_matrix(
            sin_f, cos_f, dt, sin_psi, cos_psi, parallax_factor, nl_stripped
        )
        # Ensure plain JAX array (quax dispatch may return Quantity)
        return jnp.asarray(ustrip(AllowValue, "", X))
