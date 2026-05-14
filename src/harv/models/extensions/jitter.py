"""Jitter (excess variance).

Examples
--------
>>> from harv.models.extensions import Jitter; [p.name for p in Jitter().extra_params()]
['jitter']
"""

__all__ = ("Jitter",)

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp

from harv.models.extensions.base import AbstractExtension, ParamInfo


@final
class Jitter(AbstractExtension):
    """Add excess variance (jitter / white noise) to the observation covariance.

    The extension declares one nonlinear parameter, ``jitter``, and modifies the
    covariance by adding ``jitter**2`` in quadrature to the diagonal variances. The
    ``jitter`` unit must match the observation unit (e.g. ``"km/s"`` for RV, ``"mas"``
    for astrometry). Unit stripping is the model's responsibility.

    Parameters
    ----------
    param_unit
        Physical unit string for the jitter parameter metadata. Default ``""``
        (dimensionless / observation units).

    Examples
    --------
    >>> from harv.models.extensions import Jitter
    >>> j = Jitter(param_unit="km/s")
    >>> j.extra_params()[0].unit
    'km/s'
    """

    param_unit: str = eqx.field(static=True, default="")

    def extra_params(self) -> tuple[ParamInfo, ...]:
        """Parameters introduced by this extension."""
        return (ParamInfo("jitter", self.param_unit),)

    def modify_covariance(
        self,
        cov: jax.Array,
        data: Any,  # noqa: ARG002
        nl_values: dict[str, Any],
    ) -> jax.Array:
        """Add jitter**2 in quadrature to the diagonal variances.

        Works on both 1-d (diagonal) and 2-d (full) covariance representations.
        """
        s2 = nl_values["jitter"] ** 2
        if cov.ndim == 1:
            return cov + s2

        # Add to diagonal without allocating a full n_obs x n_obs eye matrix.
        idx = jnp.diag_indices(cov.shape[0])
        return cov.at[idx].add(s2)
