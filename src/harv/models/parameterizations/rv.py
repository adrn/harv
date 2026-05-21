"""Parameterizations for radial-velocity (RV) models.

A *parameterization* is an ``eqx.Module`` subclass that declares:

- The names, units, and roles (linear/nonlinear) of parameters.
- How to build the design matrix from data and nonlinear parameter values.

Multiple parameterizations exist for the same physical observable when users
want alternative coordinate choices (e.g. ``(K, omega)`` vs
``(e*cos(omega), e*sin(omega))``).
"""

__all__ = ("EcoswEsinwRV", "StandardRV")

from typing import TYPE_CHECKING, Any, final

import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.custom_types import ScalarQSpeed, ScalarQTime
from harv.distributions import QuantityDistribution
from harv.kepler.orbits import rv_shape
from harv.models._helpers import LinearPriorDist, PriorDist
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations._base import AbstractParameterization

if TYPE_CHECKING:
    from harv.samplers.rejection_prior import RejectionPrior


@final
class StandardRV(AbstractParameterization):
    """Standard RV parameterization.

    The default harv parameterization for radial velocity modeling uses the following
    Keplerian parameters:

        - Nonlinear:
            - ``period`` - orbital period
            - ``eccentricity`` - orbital eccentricity
            - ``phase_peri`` - phase at which the mean anomaly is zero (i.e.
              periastron passage), using a time system relative to the data's reference
              time
            - ``arg_peri`` - argument of periastron
        - Linear:
            - ``rv_semiamp`` - sometimes called "K", the RV semi-amplitude
            - ``v_sys`` - systemic velocity

    The design matrix has shape ``(n_obs, 2)`` with columns ``[zd(t), 1]`` where
    ``zd(t) = cos(omega + f(t)) + e * cos(omega)`` is the RV shape function.

    Examples
    --------
    >>> from harv.models.parameterizations.rv import StandardRV
    >>> p = StandardRV()
    >>> [pp.name for pp in p.params()]
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
        sin_f
            Sine of true anomaly (unit-stripped).
        cos_f
            Cosine of true anomaly (unit-stripped).
        nl_values
            Must contain ``"eccentricity"`` and ``"arg_peri"`` (both unit-stripped
            scalars).

        Returns
        -------
            Design matrix block, shape ``(n_obs, 2)``.
        """
        rv_col = rv_shape(
            sin_f, cos_f, nl_values["eccentricity"], nl_values["arg_peri"]
        )
        return jnp.column_stack([rv_col, jnp.ones_like(rv_col)])

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "RejectionPrior":
        """Build a :class:`~harv.samplers.RejectionPrior` with sensible defaults.

        Same defaults as :meth:`harv.samplers.RejectionPrior.default_rv` (and
        ``default_rv`` is a thin wrapper around this method).

        Parameters
        ----------
        period_min, period_max
            Log-uniform period bounds.
        sigma_K0
            RV semi-amplitude scale at reference period ``P0``.
        sigma_v0
            Systemic-velocity prior scale.
        P0
            Reference period for the period-dependent ``rv_semiamp`` prior.
        **kwargs
            Per-parameter prior overrides or extension priors.
        """
        from harv.samplers.rejection_prior import (  # noqa: PLC0415
            RejectionPrior,
            _apply_overrides,
            _make_period_prior,
            _make_rv_semiamp_prior,
            _make_vsys_prior,
            kipping_2013_ecc_prior,
        )

        nonlinear: dict[str, PriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
            "eccentricity": kipping_2013_ecc_prior,
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QuantityDistribution(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        }
        linear_prior: dict[str, LinearPriorDist] = {
            "rv_semiamp": _make_rv_semiamp_prior(
                rv_semiamp=kwargs.pop("rv_semiamp", None),
                sigma_K0=sigma_K0,
                P0=P0,
            ),
            "v_sys": _make_vsys_prior(
                v_sys=kwargs.pop("v_sys", None),
                sigma_v0=sigma_v0,
            ),
        }
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_prior, extension_priors)
        return RejectionPrior(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )


@final
class EcoswEsinwRV(AbstractParameterization):
    """Alternative RV parameterization using ``e*cos(w)`` and ``e*sin(w)``.

    Replaces the ``(eccentricity, arg_peri)`` pair with ``(ecosw, esinw)`` =
    ``(e*cos(omega), e*sin(omega))``, which often has better sampling geometry for low
    eccentricities:

        - Nonlinear:
            - ``period`` - the orbital period
            - ``ecosw`` - the eccentricity times cosine of argument of periastron
            - ``esinw`` - the eccentricity times sine of argument of periastron
            - ``phase_peri`` - the phase at which the mean anomaly is zero (i.e.
              periastron passage), using a time system relative to the data's reference
              time
        - Linear:
            - ``rv_semiamp`` - sometimes called "K", the RV semi-amplitude
            - ``v_sys`` - the systemic velocity

    Examples
    --------
    >>> from harv.models.parameterizations.rv import EcoswEsinwRV
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
        sin_f
            Sine of true anomaly (unit-stripped).
        cos_f
            Cosine of true anomaly (unit-stripped).
        nl_values
            Must contain ``"ecosw"`` and ``"esinw"`` (dimensionless scalars). The
            eccentricity and arg_peri are derived internally.

        Returns
        -------
            Design matrix block, shape ``(n_obs, 2)``.
        """
        ecosw = nl_values["ecosw"]
        esinw = nl_values["esinw"]
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        arg_peri = jnp.arctan2(esinw, ecosw)
        rv_col = rv_shape(sin_f, cos_f, ecc, arg_peri)
        return jnp.column_stack([rv_col, jnp.ones_like(rv_col)])

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "RejectionPrior":
        """Build a :class:`~harv.samplers.RejectionPrior` with sensible defaults.

        Nonlinear priors:

        - ``period``: log-uniform on ``[period_min, period_max]``.
        - ``ecosw``: ``Uniform(-1, 1)``.
        - ``esinw``: ``Uniform(-1, 1)``.
        - ``phase_peri``: ``Uniform(0, 1)``.

        Linear priors are the same as :meth:`StandardRV.default_prior`.

        Independent ``Uniform(-1, 1)`` priors on ``ecosw`` and ``esinw`` do *not*
        match the implicit prior under ``e ~ Beta(0.867, 3.03)`` with
        ``omega ~ Uniform(0, 2*pi)``.  This is the simplest sensible default
        for this parameterization; for a matched prior, sample with
        ``StandardRV`` and convert.
        """
        from harv.samplers.rejection_prior import (  # noqa: PLC0415
            RejectionPrior,
            _apply_overrides,
            _make_period_prior,
            _make_rv_semiamp_prior,
            _make_vsys_prior,
        )

        nonlinear: dict[str, PriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
            "ecosw": dist.Uniform(-1.0, 1.0),
            "esinw": dist.Uniform(-1.0, 1.0),
            "phase_peri": dist.Uniform(0.0, 1.0),
        }
        linear_prior: dict[str, LinearPriorDist] = {
            "rv_semiamp": _make_rv_semiamp_prior(
                rv_semiamp=kwargs.pop("rv_semiamp", None),
                sigma_K0=sigma_K0,
                P0=P0,
            ),
            "v_sys": _make_vsys_prior(
                v_sys=kwargs.pop("v_sys", None),
                sigma_v0=sigma_v0,
            ),
        }
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_prior, extension_priors)
        return RejectionPrior(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )
