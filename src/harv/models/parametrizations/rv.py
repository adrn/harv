"""Parameterizations for radial-velocity (RV) models.

A *parameterization* is an ``eqx.Module`` subclass that declares:

- The names, units, and roles (linear/nonlinear) of parameters.
- How to build the design matrix from data and nonlinear parameter values.

Multiple parameterizations exist for the same physical observable when users
want alternative coordinate choices (e.g. ``(K, omega)`` vs
``(e*cos(omega), e*sin(omega))``).
"""

__all__ = ("EcoswEsinwRV", "StandardRV")

from typing import Any, final

import jax
import quaxed.numpy as jnp
from unxt.quantity import AllowValue, ustrip

from harv.extensions.base import ParamInfo
from harv.kepler.orbits import rv_shape as _rv_shape
from harv.models.parametrizations._base import AbstractParameterization


def _build_rv_design_col(
    sin_f: jax.Array,
    cos_f: jax.Array,
    eccentricity: float | jax.Array,
    arg_peri: float | jax.Array,
) -> jax.Array:
    """Single column: the RV shape function (dimensionless)."""
    return _rv_shape(sin_f, cos_f, eccentricity, arg_peri)


@final
class StandardRV(AbstractParameterization):
    """Standard RV parameterization: (period, ecc, phase_peri, omega, K, v_sys).

    Nonlinear: period, eccentricity, phase_peri, arg_peri.
    Linear: rv_semiamp, v_sys.

    The design matrix has shape ``(n_obs, 2)`` with columns ``[X(t), 1]``
    where ``X(t) = cos(omega + f(t)) + e * cos(omega)`` is the RV shape.

    Examples
    --------
    >>> from harv.models.parametrizations.rv import StandardRV
    >>> p = StandardRV()
    >>> [pi.name for pi in p.params()]
    ['period', 'eccentricity', 'phase_peri', 'arg_peri', 'rv_semiamp', 'v_sys']
    """

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        return (
            ParamInfo("period", "time"),
            ParamInfo("eccentricity", ""),
            ParamInfo("phase_peri", ""),
            ParamInfo("arg_peri", "angle"),
            ParamInfo("rv_semiamp", "speed", linear=True),
            ParamInfo("v_sys", "speed", linear=True),
        )

    def eccentricity(self, nl_values: dict[str, Any]) -> Any:
        """Return the orbital eccentricity from nonlinear values."""
        return nl_values["eccentricity"]

    def strip_nl_for_design(self, nl_values: dict[str, Any]) -> dict[str, Any]:
        """Return nl_values with units stripped for ``design_matrix``."""
        d = dict(nl_values)
        d["eccentricity"] = ustrip(AllowValue, "", nl_values["eccentricity"])
        d["arg_peri"] = ustrip(AllowValue, "rad", nl_values["arg_peri"])
        return d

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        nl_values: dict[str, Any],
    ) -> jax.Array:
        """Build (n_obs, 2) design matrix: columns [rv_shape, 1].

        Parameters
        ----------
        sin_f : jax.Array, shape (n_obs,)
            Sine of true anomaly (unit-stripped).
        cos_f : jax.Array, shape (n_obs,)
            Cosine of true anomaly (unit-stripped).
        nl_values : dict
            Must contain ``"eccentricity"`` and ``"arg_peri"`` (both
            unit-stripped scalars).

        Returns
        -------
        jax.Array, shape (n_obs, 2)
        """
        rv_col = _build_rv_design_col(
            sin_f, cos_f, nl_values["eccentricity"], nl_values["arg_peri"]
        )
        return jnp.column_stack([rv_col, jnp.ones_like(rv_col)])


@final
class EcoswEsinwRV(AbstractParameterization):
    """Alternative RV parameterization using ``e*cos(w)`` and ``e*sin(w)``.

    Replaces the ``(eccentricity, arg_peri)`` pair with
    ``(ecosw, esinw)`` = ``(e*cos(omega), e*sin(omega))``, which has better
    sampling geometry for low eccentricities.

    Nonlinear: period, ecosw, esinw, phase_peri.
    Linear: rv_semiamp, v_sys.

    Examples
    --------
    >>> from harv.models.parametrizations.rv import EcoswEsinwRV
    >>> p = EcoswEsinwRV()
    >>> [pi.name for pi in p.params()]
    ['period', 'ecosw', 'esinw', 'phase_peri', 'rv_semiamp', 'v_sys']
    """

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        return (
            ParamInfo("period", "time"),
            ParamInfo("ecosw", ""),
            ParamInfo("esinw", ""),
            ParamInfo("phase_peri", ""),
            ParamInfo("rv_semiamp", "speed", linear=True),
            ParamInfo("v_sys", "speed", linear=True),
        )

    def eccentricity(self, nl_values: dict[str, Any]) -> Any:
        """Return the orbital eccentricity derived from ecosw/esinw."""
        ecosw = nl_values["ecosw"]
        esinw = nl_values["esinw"]
        return jnp.sqrt(ecosw**2 + esinw**2)

    def strip_nl_for_design(self, nl_values: dict[str, Any]) -> dict[str, Any]:
        """Return nl_values with units stripped for ``design_matrix``."""
        d = dict(nl_values)
        d["ecosw"] = ustrip(AllowValue, "", nl_values["ecosw"])
        d["esinw"] = ustrip(AllowValue, "", nl_values["esinw"])
        return d

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        nl_values: dict[str, Any],
    ) -> jax.Array:
        """Build (n_obs, 2) design matrix from ecosw/esinw.

        Parameters
        ----------
        sin_f : jax.Array, shape (n_obs,)
            Sine of true anomaly (unit-stripped).
        cos_f : jax.Array, shape (n_obs,)
            Cosine of true anomaly (unit-stripped).
        nl_values : dict
            Must contain ``"ecosw"`` and ``"esinw"`` (dimensionless scalars).
            The eccentricity and arg_peri are derived internally.

        Returns
        -------
        jax.Array, shape (n_obs, 2)
        """
        ecosw = nl_values["ecosw"]
        esinw = nl_values["esinw"]
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        arg_peri = jnp.arctan2(esinw, ecosw)
        rv_col = _build_rv_design_col(sin_f, cos_f, ecc, arg_peri)
        return jnp.column_stack([rv_col, jnp.ones_like(rv_col)])
