"""Concrete RV component model.

:class:`RVModel` is a *template* carrying a parameterization (default
:class:`~harv.models.parameterizations.rv.StandardRV`) and extensions. Data and
linear priors are passed at evaluation time to :meth:`.RVModel.log_prob` /
:meth:`.RVModel.sample_conditional_linear` / :meth:`.RVModel.numpyro_model`.
"""

__all__ = ("RVModel",)

from typing import Any, final

import jax
import quaxed.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.data import RVData
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.models.component import AbstractComponentModel
from harv.models.extensions.base import AbstractExtension, ParamInfo
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV

# Type alias for any RV parameterization
RVParameterizationType = StandardRV | EcoswEsinwRV


@final
class RVModel(AbstractComponentModel):
    """Radial-velocity component model (template).

    Carries a parameterization and extensions. Data and the linear prior are
    passed at evaluation time so the same model instance can be reused across
    multiple datasets.

    Parameters
    ----------
    parameterization
        Declares parameter names/roles and builds the base design matrix.
    extensions
        Model extensions (jitter, trends, offsets, GP, ...).

    Examples
    --------
    >>> from harv.models.rv import RVModel
    >>> model = RVModel()
    >>> sorted(model._all_nonlinear_names())
    ['arg_peri', 'eccentricity', 'period', 'phase_peri']
    """

    parameterization: StandardRV | EcoswEsinwRV = StandardRV()
    extensions: tuple[AbstractExtension, ...] = ()

    def _param_infos(self) -> tuple[ParamInfo, ...]:
        base = self.parameterization.params()
        ext_params: tuple[ParamInfo, ...] = ()
        for ext in self.extensions:
            ext_params = (*ext_params, *ext.extra_params())
        return (*base, *ext_params)

    def _obs_unit(self, data: RVData) -> str:
        return str(data.rv.unit)

    def _strip_obs(self, data: RVData) -> tuple[jax.Array, jax.Array]:
        obs_unit = self._obs_unit(data)
        arr_obs = jnp.array(ustrip(obs_unit, data.rv))
        arr_obs_err = jnp.array(ustrip(obs_unit, data.rv_err))
        return arr_obs, arr_obs_err

    def _solve_kepler(
        self, nl_values: dict[str, Any], data: RVData
    ) -> tuple[jax.Array, jax.Array]:
        """Solve Kepler's equation from nonlinear parameter values.

        Returns (sin_f, cos_f) as unit-stripped arrays.
        """
        period = nl_values["period"]
        phase_peri = nl_values["phase_peri"]
        eccentricity = self.parameterization.eccentricity(nl_values)

        t_peri = phase_peri * period
        dt = (data.time - data.t_ref) - t_peri
        M = mean_anomaly(dt, period)
        sin_f, cos_f = true_anomaly_from_mean(M, eccentricity)
        return ustrip(AllowValue, "", sin_f), ustrip(AllowValue, "", cos_f)

    def _base_design_matrix(self, nl_values: dict[str, Any], data: RVData) -> jax.Array:
        sin_f, cos_f = self._solve_kepler(nl_values, data)
        nl_stripped = self.parameterization.strip_nl_for_design(nl_values)
        X = self.parameterization.design_matrix(sin_f, cos_f, nl_stripped)
        # Ensure the design matrix is a plain JAX array (rv_shape may return
        # dimensionless Quantity via quax dispatch)
        return jnp.asarray(ustrip(AllowValue, "", X))
