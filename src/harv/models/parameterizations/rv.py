"""Parameterizations for radial-velocity (RV) models.

A *parameterization* is an ``eqx.Module`` subclass that declares:

- The names, units, and roles (linear/nonlinear) of parameters.
- How to build the design matrix from data and nonlinear parameter values.

Multiple parameterizations exist for the same physical observable when users
want alternative coordinate choices (e.g. ``(K, omega)`` vs
``(sqrt(e)*cos(omega), sqrt(e)*sin(omega))``).
"""

__all__ = ("SqrtEccRV", "StandardRV")

from typing import Any, final

import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.custom_types import ScalarQSpeed, ScalarQTime
from harv.distributions import QuantityDistribution, joint
from harv.kepler.orbits import ecc_omega_from_sqrte_cosw_sinw, rv_shape
from harv.models._helpers import LinearPriorDist, NonlinearPriorDist, PriorDist
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations.base import AbstractParameterization
from harv.models.priors import HarvPrior
from harv.models.priors._helpers import (
    _apply_overrides,
    _make_period_prior,
    _make_rv_semiamp_prior,
    _make_vsys_prior,
)
from harv.stats import KippingDisk
from harv.stats.kipping_disk import KIPPING_2013_ECC


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

    def eccentricity(self, nonlinear_values: dict[str, Any]) -> Any:
        """Return the orbital eccentricity from nonlinear values."""
        return nonlinear_values["eccentricity"]

    def phase_peri(self, nonlinear_values: dict[str, Any]) -> Any:
        """Return the periastron phase from nonlinear values."""
        return nonlinear_values["phase_peri"]

    def strip_nonlinear_for_design(
        self, nonlinear_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Return nonlinear_values with units stripped for ``design_matrix``."""
        d = dict(nonlinear_values)
        d["eccentricity"] = ustrip(AllowValue, "", nonlinear_values["eccentricity"])
        d["arg_peri"] = ustrip(AllowValue, "rad", nonlinear_values["arg_peri"])
        return d

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        nonlinear_values: dict[str, Any],
    ) -> jax.Array:
        """Build (n_obs, 2) design matrix: columns [rv_shape, 1].

        Parameters
        ----------
        sin_f
            Sine of true anomaly (unit-stripped).
        cos_f
            Cosine of true anomaly (unit-stripped).
        nonlinear_values
            Must contain ``"eccentricity"`` and ``"arg_peri"`` (both unit-stripped
            scalars).

        Returns
        -------
            Design matrix block, shape ``(n_obs, 2)``.
        """
        rv_col = rv_shape(
            sin_f, cos_f, nonlinear_values["eccentricity"], nonlinear_values["arg_peri"]
        )
        return jnp.column_stack([rv_col, jnp.ones_like(rv_col)])

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        period_ref: ScalarQTime = Q(1.0, "yr"),
        **kwargs: NonlinearPriorDist | LinearPriorDist,
    ) -> "HarvPrior":
        """Build a :class:`~harv.models.priors.HarvPrior` with sensible defaults.

        Parameters
        ----------
        period_min, period_max
            Log-uniform period bounds.
        sigma_K0
            RV semi-amplitude scale at reference period ``period_ref``.
        sigma_v0
            Systemic-velocity prior scale.
        period_ref
            Reference period for the period-dependent ``rv_semiamp`` prior.
        **kwargs
            Per-parameter prior overrides or extension priors.
        """
        nonlinear: dict[str, NonlinearPriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
            "eccentricity": KIPPING_2013_ECC,
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QuantityDistribution(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        }
        linear_priors: dict[str, LinearPriorDist] = {
            "rv_semiamp": _make_rv_semiamp_prior(
                rv_semiamp=kwargs.pop("rv_semiamp", None),
                sigma_K0=sigma_K0,
                period_ref=period_ref,
            ),
            "v_sys": _make_vsys_prior(
                v_sys=kwargs.pop("v_sys", None),
                sigma_v0=sigma_v0,
            ),
        }
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_priors, extension_priors)
        return HarvPrior(
            nonlinear_priors=nonlinear,
            linear_priors=linear_priors,
            extension_priors=extension_priors,
        )


@final
class SqrtEccRV(AbstractParameterization):
    """Alternative RV parameterization using ``sqrt(e)*cos(w)`` and ``sqrt(e)*sin(w)``.

    Replaces ``(eccentricity, arg_peri)`` with ``(sqrte_cosw, sqrte_sinw)`` =
    ``(sqrt(e)*cos(omega), sqrt(e)*sin(omega))`` and ``phase_peri`` with the mean
    longitude at the reference time, ``lambda = omega - 2*pi*phase_peri``. Near
    ``e = 0`` the data pin ``lambda`` but not ``omega`` or ``phase_peri`` separately,
    so this has much better sampling geometry for low eccentricities. The ``sqrt(e)``
    makes common eccentricity priors nearly flat on the disk (``docs/spec.md``
    §SqrtEccRV):

        - Nonlinear:
            - ``period`` - the orbital period
            - ``sqrte_cosw`` - ``sqrt(e)`` times cosine of argument of periastron
            - ``sqrte_sinw`` - ``sqrt(e)`` times sine of argument of periastron
            - ``mean_longitude`` - the mean longitude ``omega + M`` at the data's
              reference time (an angle)
        - Linear:
            - ``rv_semiamp`` - sometimes called "K", the RV semi-amplitude
            - ``v_sys`` - the systemic velocity

    Examples
    --------
    >>> from harv.models.parameterizations.rv import SqrtEccRV
    >>> p = SqrtEccRV()
    >>> [pi.name for pi in p.params()]
    ['period', 'sqrte_cosw', 'sqrte_sinw', 'mean_longitude', 'rv_semiamp', 'v_sys']
    """

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        return (
            ParamInfo("period", "time"),
            ParamInfo("sqrte_cosw", ""),
            ParamInfo("sqrte_sinw", ""),
            ParamInfo("mean_longitude", "angle"),
            ParamInfo("rv_semiamp", "speed", linear=True),
            ParamInfo("v_sys", "speed", linear=True),
        )

    def _ecc_omega(self, nonlinear_values: dict[str, Any]) -> tuple[Any, Any]:
        """Return unit-stripped ``(e, omega)`` from ``(sqrte_cosw, sqrte_sinw)``."""
        ecc, arg_peri = ecc_omega_from_sqrte_cosw_sinw(
            Q(ustrip(AllowValue, "", nonlinear_values["sqrte_cosw"]), ""),
            Q(ustrip(AllowValue, "", nonlinear_values["sqrte_sinw"]), ""),
        )
        return ustrip("", ecc), ustrip("rad", arg_peri)

    def eccentricity(self, nonlinear_values: dict[str, Any]) -> Any:
        """Return the orbital eccentricity, ``sqrte_cosw**2 + sqrte_sinw**2``."""
        return self._ecc_omega(nonlinear_values)[0]

    def phase_peri(self, nonlinear_values: dict[str, Any]) -> Any:
        """Return the periastron phase, ``(omega - mean_longitude) / (2*pi)``."""
        arg_peri = self._ecc_omega(nonlinear_values)[1]
        mean_longitude = ustrip(AllowValue, "rad", nonlinear_values["mean_longitude"])
        return (arg_peri - mean_longitude) / (2 * jnp.pi)

    def derived_eccentricity(self, nonlinear_values: dict[str, Any]) -> Any:
        """Recover ``eccentricity`` for priors that expect the standard names.

        This parameterization carries ``(sqrte_cosw, sqrte_sinw)`` rather than
        ``(eccentricity, arg_peri)``, so callable linear priors that depend on
        eccentricity -- the default ``rv_semiamp`` prior among them -- have no
        ``eccentricity`` key to read. See
        :meth:`~harv.models.parameterizations.base.AbstractParameterization.derived_eccentricity`.
        """
        return self.eccentricity(nonlinear_values)

    def strip_nonlinear_for_design(
        self, nonlinear_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Return nonlinear_values with units stripped for ``design_matrix``."""
        d = dict(nonlinear_values)
        d["sqrte_cosw"] = ustrip(AllowValue, "", nonlinear_values["sqrte_cosw"])
        d["sqrte_sinw"] = ustrip(AllowValue, "", nonlinear_values["sqrte_sinw"])
        return d

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        nonlinear_values: dict[str, Any],
    ) -> jax.Array:
        """Build (n_obs, 2) design matrix from ``sqrte_cosw`` / ``sqrte_sinw``.

        Parameters
        ----------
        sin_f
            Sine of true anomaly (unit-stripped).
        cos_f
            Cosine of true anomaly (unit-stripped).
        nonlinear_values
            Must contain ``"sqrte_cosw"`` and ``"sqrte_sinw"`` (dimensionless
            scalars). The eccentricity and arg_peri are derived internally.

        Returns
        -------
            Design matrix block, shape ``(n_obs, 2)``.
        """
        ecc, arg_peri = self._ecc_omega(nonlinear_values)
        rv_col = rv_shape(sin_f, cos_f, ecc, arg_peri)
        return jnp.column_stack([rv_col, jnp.ones_like(rv_col)])

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        period_ref: ScalarQTime = Q(1.0, "yr"),
        **kwargs: NonlinearPriorDist | LinearPriorDist,
    ) -> "HarvPrior":
        """Build a :class:`~harv.models.priors.HarvPrior` with sensible defaults.

        Nonlinear priors:

        - ``period``: log-uniform on ``[period_min, period_max]``.
        - ``sqrte_cosw``, ``sqrte_sinw``:
          ``joint(("sqrte_cosw", "sqrte_sinw"), KippingDisk())``, the Kipping (2013)
          ``e ~ Beta(0.867, 3.03)`` with uniform ``omega``, so every draw is a bound
          orbit (``e < 1``).
        - ``mean_longitude``: ``Uniform(0, 2*pi)`` in ``rad``.

        Linear priors are the same as :meth:`StandardRV.default_prior`.

        This matches :meth:`StandardRV.default_prior`, so both parameterizations
        target the same posterior.  Override ``sqrte_cosw`` and ``sqrte_sinw``
        together (e.g. ``**joint(("sqrte_cosw", "sqrte_sinw"), UnitDisk())`` for a
        uniform ``e``): replacing only one raises ``ValueError``.

        Examples
        --------
        >>> import jax
        >>> import quaxed.numpy as jnp
        >>> from unxt import Q
        >>> from harv import SqrtEccRV
        >>> prior = SqrtEccRV().default_prior(
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> s = prior.sample_nonlinear(1000, key=jax.random.key(0))
        >>> bool(jnp.all(s["sqrte_cosw"] ** 2 + s["sqrte_sinw"] ** 2 < 1))
        True
        """
        if "phase_peri" in kwargs:
            # Would otherwise land in extension_priors and be silently ignored.
            msg = (
                "SqrtEccRV has no phase_peri: it samples mean_longitude "
                "(omega - 2*pi*phase_peri). Override mean_longitude instead."
            )
            raise TypeError(msg)
        nonlinear: dict[str, NonlinearPriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
            **joint(("sqrte_cosw", "sqrte_sinw"), KippingDisk()),
            "mean_longitude": QuantityDistribution(
                dist.Uniform(0.0, 2.0 * jnp.pi), "rad"
            ),
        }
        linear_priors: dict[str, LinearPriorDist] = {
            "rv_semiamp": _make_rv_semiamp_prior(
                rv_semiamp=kwargs.pop("rv_semiamp", None),
                sigma_K0=sigma_K0,
                period_ref=period_ref,
            ),
            "v_sys": _make_vsys_prior(
                v_sys=kwargs.pop("v_sys", None),
                sigma_v0=sigma_v0,
            ),
        }
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_priors, extension_priors)
        return HarvPrior(
            nonlinear_priors=nonlinear,
            linear_priors=linear_priors,
            extension_priors=extension_priors,
        )
