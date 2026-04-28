"""Concrete RV component model.

:class:`RVModel` wraps an :class:`~harv.data.RVData`, a parameterization
(default :class:`~harv.models.parameterizations.rv.StandardRV`), and optional
extensions to provide ``log_prob`` and ``sample_conditional_linear``.
"""

__all__ = ("RVModel",)

from typing import Any, final

import jax
import quaxed.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.data import RVData
from harv.extensions.base import AbstractExtension, ParamInfo
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.models.component import AbstractComponentModel
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV

# Type alias for any RV parameterization
RVParameterizationType = StandardRV | EcoswEsinwRV


@final
class RVModel(AbstractComponentModel):
    """Radial-velocity component model.

    Supports marginalized and explicit likelihood evaluation. Extensions (jitter,
    trends, offsets) modify the design matrix and/or covariance via "extensions"
    (:class:`~harv.extensions.base.AbstractExtension` subclass instances).

    Parameters
    ----------
    data : RVData
        Observed radial velocities.
    parameterization : StandardRV
        Declares parameter names/roles and builds the base design matrix.
    extensions : tuple of Extension
        Model extensions (jitter, trends, offsets, GP, ...).
    linear_prior : dict or None
        Per-parameter priors for analytic marginalization.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.data import RVData
    >>> from harv.models.rv import RVModel
    >>> data = RVData(
    ...     time=Q([0.0, 50.0, 100.0], "day"),
    ...     rv=Q([1.0, -2.0, 0.5], "km/s"),
    ...     rv_err=Q([0.5, 0.5, 0.5], "km/s"),
    ... )
    >>> model = RVModel(data=data)
    >>> sorted(model._all_nonlinear_names())
    ['arg_peri', 'eccentricity', 'period', 'phase_peri']
    """

    data: RVData
    parameterization: StandardRV | EcoswEsinwRV = StandardRV()
    linear_prior: dict[str, Any] | None = None
    extensions: tuple[AbstractExtension, ...] = ()

    def _param_infos(self) -> tuple[ParamInfo, ...]:
        base = self.parameterization.params()
        ext_params: tuple[ParamInfo, ...] = ()
        for ext in self.extensions:
            ext_params = (*ext_params, *ext.extra_params())
        return (*base, *ext_params)

    def _obs_unit(self) -> str:
        return str(self.data.rv.unit)

    def _strip_obs(self) -> tuple[jax.Array, jax.Array]:
        obs_unit = self._obs_unit()
        arr_obs = jnp.array(ustrip(obs_unit, self.data.rv))
        arr_obs_err = jnp.array(ustrip(obs_unit, self.data.rv_err))
        return arr_obs, arr_obs_err

    def _solve_kepler(self, nl_values: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
        """Solve Kepler's equation from nonlinear parameter values.

        Returns (sin_f, cos_f) as unit-stripped arrays.
        """
        period = nl_values["period"]
        phase_peri = nl_values["phase_peri"]
        eccentricity = self.parameterization.eccentricity(nl_values)

        t_peri = phase_peri * period
        dt = (self.data.time - self.data.t_ref) - t_peri
        M = mean_anomaly(dt, period)
        sin_f, cos_f = true_anomaly_from_mean(M, eccentricity)
        return ustrip(AllowValue, "", sin_f), ustrip(AllowValue, "", cos_f)

    def _base_design_matrix(self, nl_values: dict[str, Any]) -> jax.Array:
        sin_f, cos_f = self._solve_kepler(nl_values)
        nl_stripped = self.parameterization.strip_nl_for_design(nl_values)
        X = self.parameterization.design_matrix(sin_f, cos_f, nl_stripped)
        # Ensure the design matrix is a plain JAX array (rv_shape may return
        # dimensionless Quantity via quax dispatch)
        return jnp.asarray(ustrip(AllowValue, "", X))
