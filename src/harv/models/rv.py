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
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.custom_types import BatchQTime, ScalarQTime
from harv.data import RVData
from harv.kepler.orbits import mean_anomaly, true_anomaly_from_mean
from harv.models.component import AbstractComponentModel
from harv.models.extensions.base import AbstractExtension, ParamInfo
from harv.models.extensions.multi_survey import MultiSurveyOffset
from harv.models.parameterizations.fourier import FourierRV
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV

# Type alias for any RV parameterization
RVParameterizationType = StandardRV | EcoswEsinwRV | FourierRV


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

    parameterization: StandardRV | EcoswEsinwRV | FourierRV = StandardRV()
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
        # Fourier parameterizations never reach here (dispatched in
        # _base_design_matrix), so they need no eccentricity():
        eccentricity = self.parameterization.eccentricity(  # ty: ignore[unresolved-attribute]
            nl_values
        )

        t_peri = phase_peri * period
        dt = (data.time - data.t_ref) - t_peri
        M = mean_anomaly(dt, period)
        sin_f, cos_f = true_anomaly_from_mean(M, eccentricity)
        return ustrip(AllowValue, "", sin_f), ustrip(AllowValue, "", cos_f)

    def _mean_longitude(
        self, nl_values: dict[str, Any], data: RVData
    ) -> tuple[jax.Array, jax.Array]:
        """(sin M, cos M) of the mean longitude ``M = 2*pi*(t - t_ref)/P``.

        Kepler-free path used by Fourier parameterizations: no periastron
        phase (absorbed into the linear amplitude pairs) and no Kepler solve.
        """
        M = mean_anomaly(data.time - data.t_ref, nl_values["period"])
        m_rad = ustrip(AllowValue, "rad", M)
        return jnp.sin(m_rad), jnp.cos(m_rad)

    def _base_design_matrix(self, nl_values: dict[str, Any], data: RVData) -> jax.Array:
        # Fourier parameterizations are Kepler-free: their basis is the mean
        # longitude, not the true anomaly (trace-time dispatch, no runtime cost).
        if isinstance(self.parameterization, FourierRV):
            sin_f, cos_f = self._mean_longitude(nl_values, data)
        else:
            sin_f, cos_f = self._solve_kepler(nl_values, data)
        nl_stripped = self.parameterization.strip_nl_for_design(nl_values)
        X = self.parameterization.design_matrix(sin_f, cos_f, nl_stripped)
        # Ensure the design matrix is a plain JAX array (rv_shape may return
        # dimensionless Quantity via quax dispatch)
        return jnp.asarray(ustrip(AllowValue, "", X))

    def predict_at_times(
        self,
        times: BatchQTime,
        nl_values: dict[str, Any],
        linear_values: dict[str, jax.Array],
        *,
        t_ref: ScalarQTime,
        obs_unit: str = "km/s",
    ) -> jax.Array:
        """Predicted RV at arbitrary *times*, no observed-data object required.

        Internally constructs an :class:`~harv.data.RVData` shim at ``times``
        with dummy ``rv`` / ``rv_err`` (zeros / ones) and delegates to
        :meth:`predict`.  The dummy obs are never read by the prediction path
        (``_full_design_matrix`` only consumes ``data.time`` and ``data.t_ref``;
        extensions read at most those fields too).  The returned array is in
        the same units the model's linear parameters are expressed in.

        Raises ``TypeError`` if any :class:`MultiSurveyOffset` is in
        ``self.extensions`` — its ``indicator_matrix`` is fixed to the original
        data's row count and cannot apply on an arbitrary time grid.  Callers
        building a smooth-curve overlay should filter it out before calling.
        """
        for ext in self.extensions:
            if isinstance(ext, MultiSurveyOffset):
                msg = (
                    "predict_at_times cannot evaluate a MultiSurveyOffset on an "
                    "arbitrary time grid (its indicator_matrix is data-row-bound). "
                    "Filter it out of model.extensions before calling."
                )
                raise TypeError(msg)
        n = times.shape[0]
        dummy = RVData(
            time=times,
            rv=Q(jnp.zeros(n), obs_unit),
            rv_err=Q(jnp.ones(n), obs_unit),
            t_ref=t_ref,
        )
        return self.predict(nl_values, linear_values, dummy)
