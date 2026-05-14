"""Gaussian Processes (GP)."""

__all__ = ("GP",)

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.models.extensions.base import AbstractExtension, ParamInfo


@final
class GP(AbstractExtension):
    """Gaussian Process covariance extension.

    Adds the kernel matrix ``K(t, t')`` to the observation covariance via the
    ``modify_covariance`` hook, allowing analytic marginalization of linear
    parameters in the presence of correlated noise.

    Plotting code may choose to visualize the GP contribution, but that support
    is handled privately on the plotting side rather than through the
    extension base API.

    Parameters
    ----------
    kernel_builder
        Receives the full nonlinear-parameter dict (unit-stripped) and returns a
        callable kernel object that can be called with ``(X, Xp)`` to produce the kernel
        matrix. For example, from ``tinygp``.
    hyperparams
        Nonlinear hyperparameters declared by this extension (e.g. ``gp_amp``,
        ``gp_length_scale``). These are sampled alongside other nonlinear model
        parameters.
    time_unit
        Unit to strip times to before building the coordinate array. Default ``""``
        (dimensionless / already stripped).

    Examples
    --------
    >>> from harv.models.extensions.base import ParamInfo
    >>> from harv.models.extensions.gp import GP; GP(
    ...     kernel_builder=lambda hp: hp["gp_amp"] ** 2,
    ...     hyperparams=(ParamInfo("gp_amp", "km/s"),),
    ...     time_unit="day",
    ... ).extra_params()[0].name
    'gp_amp'
    """

    kernel_builder: Callable[[dict[str, Any]], Any] = eqx.field(static=True)
    hyperparams: tuple[ParamInfo, ...] = eqx.field(static=True)
    time_unit: str = eqx.field(static=True, default="")

    def extra_params(self) -> tuple[ParamInfo, ...]:
        """Parameters introduced by this extension."""
        return self.hyperparams

    def modify_covariance(
        self,
        cov: jax.Array,
        data: Any,
        nl_values: dict[str, Any],
    ) -> jax.Array:
        """Add the GP kernel matrix ``K(t, t')`` to the covariance.

        If the incoming covariance is diagonal (1-d), it is promoted to a full
        (n_obs, n_obs) matrix first.
        """
        # Extract time coordinates
        if self.time_unit:
            t = jnp.asarray(ustrip(self.time_unit, data.time))
        else:
            t = jnp.asarray(ustrip(AllowValue, "", data.time))

        # Build kernel and evaluate
        kernel = self.kernel_builder(nl_values)
        K = kernel(t, t)

        # Promote diagonal to full if needed
        if cov.ndim == 1:
            cov = jnp.diag(cov)

        return cov + K
