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

    def conditional_mean(
        self,
        residuals: jax.Array,
        data_times: jax.Array,
        prediction_times: jax.Array,
        data_err: jax.Array,
        hp: dict[str, Any],
    ) -> jax.Array:
        """Conditional-mean GP prediction at ``prediction_times``.

        Conditions on observed *residuals* (already mean-subtracted by the
        deterministic model prediction) at ``data_times`` with measurement
        noise ``data_err``, then returns the GP posterior mean evaluated at
        ``prediction_times``.  Used by the plotting layer to overlay structured
        noise; the same kernel and hyperparameter conventions as
        :meth:`modify_covariance` apply.

        All time arrays are assumed to be unit-stripped to the same unit (the
        kernel's coordinate unit).  ``hp`` is the per-sample hyperparameter
        dict (e.g. from a single posterior sample).
        """
        import tinygp  # noqa: PLC0415

        # Quasisep kernels require sorted training coordinates. Keep the
        # prediction grid order unchanged and reorder the training data to
        # match.
        sort_idx = jnp.argsort(data_times)
        data_times = data_times[sort_idx]
        residuals = residuals[sort_idx]
        data_err = data_err[sort_idx]

        kernel = self.kernel_builder(hp)
        gp = tinygp.GaussianProcess(kernel, data_times, diag=data_err**2)

        # Predict in chunks: quasisep training is scalable, but conditioning
        # on a large prediction grid can still fall back to dense test-time
        # covariance matrices.
        prediction_times = jnp.asarray(prediction_times)
        chunk_size = 2048
        if prediction_times.shape[0] <= chunk_size:
            _, cond = gp.condition(residuals, prediction_times)
            return cond.loc

        pred_chunks: list[jax.Array] = []
        for start in range(0, int(prediction_times.shape[0]), chunk_size):
            stop = start + chunk_size
            _, cond = gp.condition(residuals, prediction_times[start:stop])
            pred_chunks.append(cond.loc)
        return jnp.concatenate(pred_chunks)
