"""Monomial polynomial trends."""

__all__ = ("MonomialTrend",)

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.models.extensions.base import AbstractExtension, ParamInfo


@final
class MonomialTrend(AbstractExtension):
    """Append monomial trend columns to the design matrix.

    Uses a standard monomial basis: ``dt^1, dt^2, ..., dt^order`` (RV) or
    scan-angle-projected monomials (astrometry).

    Parameters
    ----------
    order
        Polynomial order (number of trend terms). Must be >= 1.
    time_unit
        Unit string for time -- used in ``ParamInfo`` metadata.
    obs_unit
        Unit string for the observations -- used in ``ParamInfo`` metadata.
    astrometry
        If ``True``, add *two* columns per order (RA + Dec, projected by scan angle)
        with exponents ``k + 1`` to avoid degeneracy with the base proper-motion
        columns. Default ``False`` (single column per order).

    Examples
    --------
    >>> from harv.models.extensions import MonomialTrend
    >>> trend = MonomialTrend(order=2)
    >>> [p.name for p in trend.extra_params()]
    ['trend_1', 'trend_2']
    """

    order: int = eqx.field(static=True, default=1)
    time_unit: str = eqx.field(static=True, default="")
    obs_unit: str = eqx.field(static=True, default="")
    astrometry: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        if self.order < 1:
            msg = f"order must be >= 1, got {self.order}"
            raise ValueError(msg)

    def extra_params(self) -> tuple[ParamInfo, ...]:
        """Parameters introduced by this extension."""
        if self.astrometry:
            params: list[ParamInfo] = []
            for k in range(1, self.order + 1):
                params.append(ParamInfo(f"trend_ra_{k}", self.obs_unit, linear=True))
                params.append(ParamInfo(f"trend_dec_{k}", self.obs_unit, linear=True))
            return tuple(params)

        return tuple(
            ParamInfo(f"trend_{k}", self.obs_unit, linear=True)
            for k in range(1, self.order + 1)
        )

    def modify_design_matrix(
        self,
        X: jax.Array,
        data: Any,
        nl_values: dict[str, Any],  # noqa: ARG002
    ) -> jax.Array:
        """Append trend columns to the design matrix.

        For RV (``astrometry=False``):
            Columns are ``(t - t_ref)^k`` for ``k = 1..order``.

        For Gaia astrometry (``astrometry=True``):
            Two columns per order:
            ``sin(psi) * dt^(k+1)`` and ``cos(psi) * dt^(k+1)``
            where ``dt = (t - t_ref)`` and ``psi`` is the scan angle.
            The exponent ``k + 1`` avoids degeneracy with the base
            proper-motion (``dt^1``) columns.

        The data object must have ``time`` and ``t_ref`` attributes (and
        ``scan_angle`` for astrometry mode).
        """
        if self.time_unit:
            dt = ustrip(self.time_unit, data.time - data.t_ref)
        else:
            dt = jnp.asarray(ustrip(AllowValue, "", data.time - data.t_ref))

        if self.astrometry:
            scan_angle = jnp.asarray(ustrip("rad", data.scan_angle))
            sin_psi = jnp.sin(scan_angle)
            cos_psi = jnp.cos(scan_angle)

            cols: list[jax.Array] = []
            for k in range(1, self.order + 1):
                dt_power = dt ** (k + 1)
                cols.append(sin_psi * dt_power)
                cols.append(cos_psi * dt_power)
            return jnp.concatenate([X, jnp.stack(cols, axis=-1)], axis=-1)

        # Standard (RV) trend columns
        cols = [dt**k for k in range(1, self.order + 1)]
        return jnp.concatenate([X, jnp.stack(cols, axis=-1)], axis=-1)
