"""Concrete Gaia epoch-astrometry component model.

:class:`GaiaAstrometryModel` is a *template* carrying a parameterization
(default :class:`~harv.models.parameterizations.gaia.StandardGaiaAstrometry`)
and extensions. Data and linear priors are passed at evaluation time.
"""

__all__ = ("GaiaAstrometryModel",)

from dataclasses import KW_ONLY
from typing import Any, final

import jax
import quaxed.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.data import GaiaAstrometryData
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.models.component import AbstractComponentModel
from harv.models.extensions.base import AbstractExtension, ParamInfo
from harv.models.parameterizations._base import AbstractParameterization
from harv.models.parameterizations.gaia import StandardGaiaAstrometry


@final
class GaiaAstrometryModel(AbstractComponentModel):
    """Gaia epoch astrometry component model (template).

    Carries a parameterization and extensions. Data and the linear prior are
    passed at evaluation time so the same model instance can be reused across
    multiple datasets.

    Parameters
    ----------
    parameterization
        Declares parameter names/roles and builds the base design matrix.
    extensions
        Model extensions (jitter, trends, ...).
    pm_time_unit
        If not None, override the proper motion units in the design matrix to
        use this unit instead of the default (obs_unit / yr).

    Examples
    --------
    >>> from harv.models.astrometry import GaiaAstrometryModel
    >>> model = GaiaAstrometryModel()
    >>> sorted(model._all_nonlinear_names())
    ['arg_peri', 'cos_i', 'eccentricity', 'lon_asc_node', 'period', 'phase_peri']
    """

    parameterization: AbstractParameterization = StandardGaiaAstrometry()
    extensions: tuple[AbstractExtension, ...] = ()
    _: KW_ONLY
    pm_time_unit: str = "yr"

    def _param_infos(self) -> tuple[ParamInfo, ...]:
        base = self.parameterization.params()
        ext_params: tuple[ParamInfo, ...] = ()
        for ext in self.extensions:
            ext_params = (*ext_params, *ext.extra_params())
        return (*base, *ext_params)

    def _obs_unit(self, data: GaiaAstrometryData) -> str:
        return str(data.al_position.unit)

    def _linear_param_units(self, data: GaiaAstrometryData) -> dict[str, str]:
        """Astrometric linear params have different units (mas vs mas/yr)."""
        u = self._obs_unit(data)
        pm_unit = f"{u}/{self.pm_time_unit}"
        unit_kind_map: dict[str, str] = {
            "angle": u,
            "angular_speed": pm_unit,
            "": "",
        }
        units: dict[str, str] = {}
        for pi in self.parameterization.linear_params():
            units[pi.name] = unit_kind_map.get(pi.unit, u)
        # Extension-added linear params default to obs_unit
        for pi in self._param_infos():
            if pi.linear and pi.name not in units:
                units[pi.name] = u
        return units

    def _strip_obs(self, data: GaiaAstrometryData) -> tuple[jax.Array, jax.Array]:
        obs_unit = self._obs_unit(data)
        arr_obs = jnp.array(ustrip(obs_unit, data.al_position))
        arr_obs_err = jnp.array(ustrip(obs_unit, data.al_position_err))
        return arr_obs, arr_obs_err

    def _solve_kepler(
        self, nl_values: dict[str, Any], data: GaiaAstrometryData
    ) -> tuple[jax.Array, jax.Array]:
        """Solve Kepler's equation from nonlinear parameter values."""
        period = nl_values["period"]
        phase_peri = nl_values["phase_peri"]
        eccentricity = nl_values["eccentricity"]

        t_peri = phase_peri * period
        dt = (data.time - data.t_ref) - t_peri
        M = mean_anomaly(dt, period)
        sin_f, cos_f = true_anomaly_from_mean(M, eccentricity)
        return ustrip(AllowValue, "", sin_f), ustrip(AllowValue, "", cos_f)

    def _base_design_matrix(
        self, nl_values: dict[str, Any], data: GaiaAstrometryData
    ) -> jax.Array:
        sin_f, cos_f = self._solve_kepler(nl_values, data)

        # Prepare auxiliary data arrays
        dt = jnp.array(ustrip(self.pm_time_unit, data.time - data.t_ref))
        scan_angle_rad = ustrip("rad", data.scan_angle)
        sin_psi = jnp.sin(scan_angle_rad)
        cos_psi = jnp.cos(scan_angle_rad)
        parallax_factor = ustrip(AllowValue, "", data.parallax_factor)

        # Strip nonlinear values for parameterization.
        # Derive strip targets from the parameterization's declared parameter units:
        #   "angle" → radians, "time" → pm_time_unit, "" → dimensionless.
        # period and phase_peri are consumed by _solve_kepler above; they are passed
        # through if the parameterization also requests them (no-op for the design
        # matrix, but harmless).
        _strip_target: dict[str, str] = {"angle": "rad", "time": self.pm_time_unit}
        nl_stripped: dict[str, Any] = {}
        for pi in self.parameterization.nonlinear_params():
            name = pi.name
            if name not in nl_values:
                continue
            target = _strip_target.get(pi.unit, "")
            if target:
                nl_stripped[name] = ustrip(AllowValue, target, nl_values[name])
            else:
                nl_stripped[name] = ustrip(AllowValue, "", nl_values[name])

        # TODO: we need to implement an abstract design_matrix method
        X = self.parameterization.design_matrix(
            sin_f, cos_f, dt, sin_psi, cos_psi, parallax_factor, nl_stripped
        )
        # Ensure plain JAX array (quax dispatch may return Quantity)
        return jnp.asarray(ustrip(AllowValue, "", X))
