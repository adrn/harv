"""Parameterizations for Gaia epoch-astrometry models.

A *parameterization* is an ``eqx.Module`` subclass that declares the
names, units, and roles (linear / nonlinear) of parameters and knows how to
build the corresponding design matrix.
"""

__all__ = ("StandardGaiaAstrometry", "ThieleInnesGaiaAstrometry")

from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt import Q
from unxt.quantity import ustrip

from harv.custom_types import (
    ScalarQAngle,
    ScalarQLength,
    ScalarQSpeed,
    ScalarQTime,
)
from harv.distributions import QuantityDistribution
from harv.kepler.orbits import thiele_innes_ABFG
from harv.models._helpers import LinearPriorDist, PriorDist
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations._base import AbstractParameterization
from harv.models.priors import HarvPrior
from harv.models.priors.helpers import (
    _apply_overrides,
    _make_parallax_prior,
    _make_period_prior,
    _make_pm_prior,
    _make_pos_prior,
    _make_semi_major_axis_prior,
    kipping_2013_ecc_prior,
)

if TYPE_CHECKING:
    from harv.data.datasets import GaiaAstrometryData


@final
class StandardGaiaAstrometry(AbstractParameterization):
    """Standard Gaia astrometry parameterization.

    The default harv parameterization for Gaia epoch astrometry modeling uses the
    following parameters:

        - Nonlinear:
            - ``period`` - orbital period
            - ``eccentricity`` - orbital eccentricity
            - ``phase_peri`` - phase at which the mean anomaly is zero (i.e.
              periastron passage), using a time system relative to the data's reference
              time
            - ``arg_peri`` - argument of periastron
            - ``lon_asc_node`` - longitude of the ascending node
            - ``cos_i`` - cosine of the inclination
        - Linear:
            - ``ra0`` - right ascension at the reference epoch
            - ``dec0`` - declination at the reference epoch
            - ``pmra`` - proper motion in right ascension
            - ``pmdec`` - proper motion in declination
            - ``parallax`` - parallax
            - ``semi_major_axis`` - semi-major axis

    The design matrix has shape ``(n_obs, 6)`` with columns
    ``[ra0, dec0, pmra, pmdec, parallax, a_orbit]``.

    Parameters are defined in the Gaia local-plane coordinate (LPC) convention
    from Lindegren & Bastian (GAIA-C3-TN-LU-LL-061-08).

    Examples
    --------
    >>> from harv.models.parameterizations.gaia import StandardGaiaAstrometry
    >>> p = StandardGaiaAstrometry()
    >>> [pp.name for pp in p.nonlinear_params()]
    ['period', 'eccentricity', 'phase_peri', 'arg_peri', 'lon_asc_node', 'cos_i']
    """

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        return (
            ParamInfo("period", "time"),
            ParamInfo("eccentricity", ""),
            ParamInfo("phase_peri", ""),
            ParamInfo("arg_peri", "angle"),
            ParamInfo("lon_asc_node", "angle"),
            ParamInfo("cos_i", ""),
            ParamInfo("ra0", "angle", linear=True),
            ParamInfo("dec0", "angle", linear=True),
            ParamInfo("pmra", "angular_speed", linear=True),
            ParamInfo("pmdec", "angular_speed", linear=True),
            ParamInfo("parallax", "angle", linear=True),
            ParamInfo("semi_major_axis", "angle", linear=True),
        )

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        dt: jax.Array,
        sin_psi: jax.Array,
        cos_psi: jax.Array,
        parallax_factor: jax.Array,
        nl_values: dict[str, Any],
    ) -> jax.Array:
        """Build (n_obs, 6) along-scan design matrix.

        Columns: [ra0, dec0, pmra, pmdec, parallax, semi_major_axis].

        Parameters
        ----------
        sin_f
            Sine of true anomaly (unit-stripped).
        cos_f
            Cosine of true anomaly (unit-stripped).
        dt
            Time elapsed since reference epoch (unit-stripped, in internal
            time unit, typically years).
        sin_psi
            Sine of scan angle.
        cos_psi
            Cosine of scan angle.
        parallax_factor
            Parallax factor (unit-stripped).
        nl_values
            Must contain ``"eccentricity"``, ``"arg_peri"``,
            ``"lon_asc_node"``, ``"cos_i"`` (unit-stripped scalars).

        Returns
        -------
            Design matrix block, shape ``(n_obs, 6)``.
        """
        ecc = nl_values["eccentricity"]
        arg_peri = nl_values["arg_peri"]
        lon_asc_node = nl_values["lon_asc_node"]
        cos_i = nl_values["cos_i"]

        A, B, F, G = thiele_innes_ABFG(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )

        # Thiele-Innes orbital coordinates
        r_over_a = (1 - ecc**2) / (1 + ecc * cos_f)
        X = r_over_a * cos_f
        Y = r_over_a * sin_f

        # Along-scan orbital element
        semimaj_term = (B * X + G * Y) * sin_psi + (A * X + F * Y) * cos_psi

        return jnp.stack(
            [
                sin_psi,  # ra0
                cos_psi,  # dec0
                sin_psi * dt,  # pmra
                cos_psi * dt,  # pmdec
                parallax_factor,  # parallax
                semimaj_term,  # semi_major_axis
            ],
            axis=-1,
        )

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_a0: ScalarQLength | None = None,
        sigma_parallax: ScalarQAngle | None = None,
        sigma_pos: ScalarQAngle | None = None,
        sigma_vtan: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "HarvPrior":
        """Build a :class:`~harv.samplers.HarvPrior` with sensible defaults.

        Same defaults as :meth:`harv.samplers.HarvPrior.default_gaia_astrometry`
        (and ``default_gaia_astrometry`` is a thin wrapper around this method).
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
            "eccentricity": kipping_2013_ecc_prior,
            "phase_peri": dist.Uniform(0.0, 1.0),
            "cos_i": dist.Uniform(-1.0, 1.0),
            "arg_peri": QuantityDistribution(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
            "lon_asc_node": QuantityDistribution(
                dist.Uniform(0.0, 2.0 * jnp.pi), "rad"
            ),
        }
        linear_prior: dict[str, LinearPriorDist] = {
            "ra0": _make_pos_prior(
                pos=kwargs.pop("ra0", None), sigma_pos=sigma_pos, name="ra0"
            ),
            "dec0": _make_pos_prior(
                pos=kwargs.pop("dec0", None), sigma_pos=sigma_pos, name="dec0"
            ),
            "pmra": _make_pm_prior(
                pm=kwargs.pop("pmra", None), sigma_vtan=sigma_vtan, name="pmra"
            ),
            "pmdec": _make_pm_prior(
                pm=kwargs.pop("pmdec", None), sigma_vtan=sigma_vtan, name="pmdec"
            ),
            "parallax": _make_parallax_prior(
                parallax=kwargs.pop("parallax", None),
                sigma_parallax=sigma_parallax,
            ),
            "semi_major_axis": _make_semi_major_axis_prior(
                semi_major_axis=kwargs.pop("semi_major_axis", None),
                sigma_a0=sigma_a0,
                P0=P0,
            ),
        }
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_prior, extension_priors)
        return HarvPrior(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )


@final
class ThieleInnesGaiaAstrometry(AbstractParameterization):
    r"""Thiele-Innes parameterization for Gaia epoch astrometry.

    Replaces the four Campbell orientation parameters ``(arg_peri, lon_asc_node, cos_i,
    semi_major_axis)`` with the four Thiele-Innes constants ``(ti_A, ti_B, ti_F,
    ti_G)``, which enter the along-scan model *linearly*.  This reduces the nonlinear
    parameter space from 6-D to 3-D.

    The along-scan measurement model is:

    .. math::

        w = (\alpha_{*,0} + \mu_\alpha \Delta t)\sin\psi
            + (\delta_0 + \mu_\delta \Delta t)\cos\psi
            + \varpi \cdot pf
            + (B X + G Y)\sin\psi
            + (A X + F Y)\cos\psi

    - Nonlinear: ``period``, ``eccentricity``, ``phase_peri`` (3 params)
    - Linear: ``ra0``, ``dec0``, ``pmra``, ``pmdec``, ``parallax``,
      ``ti_A``, ``ti_B``, ``ti_F``, ``ti_G`` (9 params)

    A flat prior on the Thiele-Innes constants is not the same as a flat prior on
    the physical Campbell elements :math:`(a_0, \omega, \Omega, \cos i)`, so a
    Jacobian correction is needed to recover the correct posterior. The
    zeroth-order correction (evaluated at the conditional-mean TI constants)
    multiplies the marginal likelihood by :math:`(a_0 + \delta_a)^{-m}(\sin^2 i +
    \delta_{s})^{-1}`, where :math:`m = 3` (uniform prior in :math:`a_0`) or
    :math:`m = 4` (log-uniform), and :math:`\delta_a`, :math:`\delta_s` are
    numerical floors that prevent singularities near face-on orbits or zero
    semi-major axis.

    **The default is ``apply_jacobian_correction=False``** because the floors
    require sensible data-driven values that the bare constructor cannot infer.
    The recommended way to construct this class with the correction enabled is via
    :meth:`from_data`, which sets ``a_floor = med(sigma_AL) / sqrt(N)``
    automatically. Without the correction, the marginal likelihood can be
    dominated by spurious long-period configurations where the orbital signal is
    absorbed into proper motion -- this is especially severe when the data
    baseline is shorter than the orbital period. **For sub-orbit data baselines,
    prefer ``ThieleInnesGaiaAstrometry.from_data(data)`` (or the Standard
    parameterization).**

    Parameters
    ----------
    a_floor : float or None, optional
        Floor on :math:`a_0` (in the same angular units as the astrometric data, e.g.
        mas) used to regularize the Jacobian correction near zero semi-major axis.
        Required (non-``None``) when ``apply_jacobian_correction=True``; must be
        ``None`` when it is ``False``.
    sin2i_floor : float or None, optional
        Floor on :math:`\sin^2 i` for the Jacobian denominator.  Falls back to
        ``0.01`` when ``None``.  Must be ``None`` when
        ``apply_jacobian_correction=False``.
    log_uniform_in_a : bool or None, optional
        If ``True``, assume a log-uniform (Jeffreys) prior on :math:`a_0` (uses
        :math:`m = 4`).  Falls back to ``False`` (uniform in :math:`a_0`,
        :math:`m = 3`) when ``None``.  Must be ``None`` when
        ``apply_jacobian_correction=False``.
    apply_jacobian_correction : bool, optional
        Whether to apply the Jacobian correction.  Default ``False``. To enable
        with sensible floors, use :meth:`from_data`.

    Raises
    ------
    ValueError
        If ``apply_jacobian_correction=True`` but ``a_floor`` is ``None``, or if
        ``apply_jacobian_correction=False`` but any of ``a_floor``, ``sin2i_floor``,
        or ``log_uniform_in_a`` is supplied.

    Examples
    --------
    >>> from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
    >>> p = ThieleInnesGaiaAstrometry(a_floor=0.01)
    >>> [pp.name for pp in p.nonlinear_params()]
    ['period', 'eccentricity', 'phase_peri']
    >>> [pp.name for pp in p.linear_params()]
    ['ra0', 'dec0', 'pmra', 'pmdec', 'parallax', 'ti_A', 'ti_B', 'ti_F', 'ti_G']

    Disable the Jacobian correction (no floor parameters needed):

    >>> p = ThieleInnesGaiaAstrometry(apply_jacobian_correction=False)
    >>> p.linear_log_prior_correction({}) is None
    True
    """

    a_floor: float | None = None
    sin2i_floor: float | None = None
    log_uniform_in_a: bool | None = eqx.field(static=True, default=None)
    apply_jacobian_correction: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        """Validate that floor parameters match ``apply_jacobian_correction``."""
        if self.apply_jacobian_correction:
            if self.a_floor is None:
                msg = (
                    "a_floor is required when apply_jacobian_correction=True (it "
                    "floors a_0 in the Jacobian correction). Pass a_floor=..., use "
                    "ThieleInnesGaiaAstrometry.from_data(data), or set "
                    "apply_jacobian_correction=False to disable the correction."
                )
                raise ValueError(msg)
        else:
            supplied = sorted(
                name
                for name, value in (
                    ("a_floor", self.a_floor),
                    ("sin2i_floor", self.sin2i_floor),
                    ("log_uniform_in_a", self.log_uniform_in_a),
                )
                if value is not None
            )
            if supplied:
                msg = (
                    "Jacobian-correction parameters must be left unset (None) when "
                    f"apply_jacobian_correction=False, but got: {supplied}."
                )
                raise ValueError(msg)

    @classmethod
    def from_data(
        cls,
        data: "GaiaAstrometryData",
        sin2i_floor: float | None = None,
        log_uniform_in_a: bool | None = None,
        *,
        apply_jacobian_correction: bool = True,
    ) -> "ThieleInnesGaiaAstrometry":
        r"""Construct with ``a_floor = med(sigma_AL) / sqrt(N)`` from the data.

        Parameters
        ----------
        data : GaiaAstrometryData
            Along-scan epoch astrometry data.
        sin2i_floor : float or None, optional
            Floor on :math:`\sin^2 i`.  Falls back to ``0.01`` when ``None``.
        log_uniform_in_a : bool or None, optional
            Use log-uniform prior on :math:`a_0`.  Falls back to ``False`` when
            ``None``.
        apply_jacobian_correction : bool, optional
            Whether to apply the Jacobian correction.  Default ``True``.  When
            ``False``, ``a_floor`` is not derived and ``sin2i_floor`` /
            ``log_uniform_in_a`` must be left as ``None``.

        Returns
        -------
        ThieleInnesGaiaAstrometry

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from unxt import Q
        >>> from harv.data import GaiaAstrometryData
        >>> from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
        >>> data = GaiaAstrometryData(
        ...     time=Q([0.0, 100.0, 200.0], "day"),
        ...     al_position=Q([0.1, -0.2, 0.05], "mas"),
        ...     al_position_err=Q([0.05, 0.06, 0.04], "mas"),
        ...     scan_angle=Q([0.5, 1.2, 2.8], "rad"),
        ...     parallax_factor=jnp.array([0.3, -0.1, 0.4]),
        ... )
        >>> p = ThieleInnesGaiaAstrometry.from_data(data)
        >>> p.a_floor > 0
        True
        """
        if not apply_jacobian_correction:
            return cls(apply_jacobian_correction=False)
        errs = ustrip(str(data.al_position_err.unit), data.al_position_err)
        a_floor = float(jnp.median(errs) / jnp.sqrt(jnp.asarray(errs).size))
        return cls(
            a_floor=a_floor,
            sin2i_floor=sin2i_floor,
            log_uniform_in_a=log_uniform_in_a,
            apply_jacobian_correction=True,
        )

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        return (
            ParamInfo("period", "time"),
            ParamInfo("eccentricity", ""),
            ParamInfo("phase_peri", ""),
            ParamInfo("ra0", "angle", linear=True),
            ParamInfo("dec0", "angle", linear=True),
            ParamInfo("pmra", "angular_speed", linear=True),
            ParamInfo("pmdec", "angular_speed", linear=True),
            ParamInfo("parallax", "angle", linear=True),
            ParamInfo("ti_A", "angle", linear=True),
            ParamInfo("ti_B", "angle", linear=True),
            ParamInfo("ti_F", "angle", linear=True),
            ParamInfo("ti_G", "angle", linear=True),
        )

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        dt: jax.Array,
        sin_psi: jax.Array,
        cos_psi: jax.Array,
        parallax_factor: jax.Array,
        nl_values: dict[str, Any],
    ) -> jax.Array:
        """Build (n_obs, 9) along-scan design matrix.

        Columns: [ra0, dec0, pmra, pmdec, parallax, ti_A, ti_B, ti_F, ti_G].

        Parameters
        ----------
        sin_f : jax.Array, shape (n_obs,)
            Sine of true anomaly (unit-stripped).
        cos_f : jax.Array, shape (n_obs,)
            Cosine of true anomaly (unit-stripped).
        dt : jax.Array, shape (n_obs,)
            Time elapsed since reference epoch (unit-stripped).
        sin_psi : jax.Array, shape (n_obs,)
            Sine of scan angle.
        cos_psi : jax.Array, shape (n_obs,)
            Cosine of scan angle.
        parallax_factor : jax.Array, shape (n_obs,)
            Parallax factor (unit-stripped).
        nl_values : dict
            Must contain ``"eccentricity"`` (unit-stripped scalar).

        Returns
        -------
        jax.Array, shape (n_obs, 9)
        """
        ecc = nl_values["eccentricity"]

        # Orbital coordinates (dimensionless, in units of semi-major axis)
        r_over_a = (1 - ecc**2) / (1 + ecc * cos_f)
        X = r_over_a * cos_f  # (n_obs,)
        Y = r_over_a * sin_f  # (n_obs,)

        # Each TI constant multiplies a specific combination of (X,Y) and scan:
        #   dra = (B*X + G*Y) * sin_psi  -> columns for ti_B and ti_G
        #   ddec = (A*X + F*Y) * cos_psi  -> columns for ti_A and ti_F
        return jnp.stack(
            [
                sin_psi,  # ra0
                cos_psi,  # dec0
                sin_psi * dt,  # pmra
                cos_psi * dt,  # pmdec
                parallax_factor,  # parallax
                X * cos_psi,  # ti_A  (coefficient of A in ddec projection)
                X * sin_psi,  # ti_B  (coefficient of B in dra projection)
                Y * cos_psi,  # ti_F  (coefficient of F in ddec projection)
                Y * sin_psi,  # ti_G  (coefficient of G in dra projection)
            ],
            axis=-1,
        )

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_a0: ScalarQLength | None = None,
        sigma_parallax: ScalarQAngle | None = None,
        sigma_pos: ScalarQAngle | None = None,
        sigma_vtan: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "HarvPrior":
        """Build a :class:`~harv.samplers.HarvPrior` with sensible defaults.

        Nonlinear priors:

        - ``period``: log-uniform on ``[period_min, period_max]``.
        - ``eccentricity``: Kipping (2013) Beta prior.
        - ``phase_peri``: ``Uniform(0, 1)``.

        Linear priors:

        - ``ra0``, ``dec0``, ``pmra``, ``pmdec``, ``parallax``: same defaults as
          :meth:`StandardGaiaAstrometry.default_prior`.
        - ``ti_A``, ``ti_B``, ``ti_F``, ``ti_G``: each uses
          :class:`~harv.models.priors.custom_priors.PeriodDependentSemiMajorAxisPrior`,
          the same scaling as ``semi_major_axis`` in
          :class:`StandardGaiaAstrometry`.

        The Jacobian correction
        (:meth:`linear_log_prior_correction`, active when
        ``apply_jacobian_correction=True``) restores the correct posterior
        under a flat-Campbell-elements prior.
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
            "eccentricity": kipping_2013_ecc_prior,
            "phase_peri": dist.Uniform(0.0, 1.0),
        }
        linear_prior: dict[str, LinearPriorDist] = {
            "ra0": _make_pos_prior(
                pos=kwargs.pop("ra0", None), sigma_pos=sigma_pos, name="ra0"
            ),
            "dec0": _make_pos_prior(
                pos=kwargs.pop("dec0", None), sigma_pos=sigma_pos, name="dec0"
            ),
            "pmra": _make_pm_prior(
                pm=kwargs.pop("pmra", None), sigma_vtan=sigma_vtan, name="pmra"
            ),
            "pmdec": _make_pm_prior(
                pm=kwargs.pop("pmdec", None), sigma_vtan=sigma_vtan, name="pmdec"
            ),
            "parallax": _make_parallax_prior(
                parallax=kwargs.pop("parallax", None),
                sigma_parallax=sigma_parallax,
            ),
        }
        # All four TI constants share the same period/parallax-dependent scale.
        # Per-constant overrides flow through kwargs ("ti_A", ...).
        for name in ("ti_A", "ti_B", "ti_F", "ti_G"):
            override = kwargs.pop(name, None)
            linear_prior[name] = _make_semi_major_axis_prior(
                semi_major_axis=override,
                sigma_a0=None if override is not None else sigma_a0,
                P0=P0,
            )
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_prior, extension_priors)
        return HarvPrior(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )

    def linear_log_prior_correction(
        self, linear_map: dict[str, jax.Array]
    ) -> jax.Array | None:
        r"""Zeroth-order Jacobian correction for the Thiele-Innes change of variables.

        Returns ``None`` (no correction) when ``apply_jacobian_correction=False``.
        Otherwise evaluates :math:`-m \ln(a_0 + \delta_a) - \ln(\sin^2 i + \delta_s)`
        at the conditional-mean Thiele-Innes constants, where :math:`a_0` and
        :math:`\sin^2 i` are derived from the standard identities:

        .. math::

            u &= \tfrac{1}{2}(A^2 + B^2 + F^2 + G^2) \\
            v &= AG - BF \\
            a_0 &= \sqrt{u + \sqrt{\max(u^2 - v^2, 0)}} \\
            \sin^2 i &= 1 - v^2 / a_0^4

        Parameters
        ----------
        linear_map : dict[str, jax.Array]
            Conditional-mean values of the marginalized linear parameters.
            Must contain keys ``"ti_A"``, ``"ti_B"``, ``"ti_F"``, ``"ti_G"`` when
            the correction is enabled (ignored when it is disabled).

        Returns
        -------
        jax.Array or None
            Scalar log-correction, or ``None`` when the correction is disabled.
        """
        if not self.apply_jacobian_correction:
            return None

        # __check_init__ guarantees a_floor is not None here; re-narrow for
        # type-checkers and fall back to the documented defaults otherwise.
        a_floor = self.a_floor
        if a_floor is None:  # pragma: no cover - guarded by __check_init__
            msg = "a_floor must be set when apply_jacobian_correction=True"
            raise ValueError(msg)
        sin2i_floor = 0.01 if self.sin2i_floor is None else self.sin2i_floor

        A = linear_map["ti_A"]
        B = linear_map["ti_B"]
        F = linear_map["ti_F"]
        G = linear_map["ti_G"]

        u = 0.5 * (A**2 + B**2 + F**2 + G**2)
        v = A * G - B * F
        # a0 = sqrt(u + sqrt(max(u² - v², 0)))  [Halbwachs & Pourbaix identity]
        a0 = jnp.sqrt(u + jnp.sqrt(jnp.maximum(u * u - v * v, 0.0)))
        # sin²i = 1 - cos²i = 1 - (v/a0²)²  = 1 - v²/a0⁴
        sin2i = jnp.clip(1.0 - v**2 / jnp.maximum(a0**4, 1e-30), 0.0, None)

        m = 4 if self.log_uniform_in_a else 3
        return -m * jnp.log(a0 + a_floor) - jnp.log(sin2i + sin2i_floor)
