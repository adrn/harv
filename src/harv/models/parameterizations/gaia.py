"""Parameterizations for Gaia epoch-astrometry models.

A *parameterization* is an ``eqx.Module`` subclass that declares the
names, units, and roles (linear / nonlinear) of parameters and knows how to
build the corresponding design matrix.
"""

__all__ = ("StandardGaiaAstrometry", "ThieleInnesGaiaAstrometry")

from typing import TYPE_CHECKING, Any, final

import equinox as eqx
import jax
import quaxed.numpy as jnp

from harv.kepler.orbits import thiele_innes_ABFG
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations._base import AbstractParameterization

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


@final
class ThieleInnesGaiaAstrometry(AbstractParameterization):
    """Thiele-Innes parameterization for Gaia epoch astrometry.

    Replaces the four Campbell orientation parameters ``(arg_peri,
    lon_asc_node, cos_i, semi_major_axis)`` with the four Thiele-Innes
    constants ``(ti_A, ti_B, ti_F, ti_G)``, which enter the along-scan model
    *linearly*.  This reduces the nonlinear parameter space from 6-D to 3-D,
    making rejection and MCMC sampling significantly cheaper.

    The along-scan measurement model is:

    .. math::

        w = (\\alpha_{*,0} + \\mu_\\alpha \\Delta t)\\sin\\psi
            + (\\delta_0 + \\mu_\\delta \\Delta t)\\cos\\psi
            + \\varpi \\cdot pf
            + (B X + G Y)\\sin\\psi
            + (A X + F Y)\\cos\\psi

    where :math:`X = (r/a)\\cos f`, :math:`Y = (r/a)\\sin f`.

    - Nonlinear: ``period``, ``eccentricity``, ``phase_peri`` (3 params)
    - Linear: ``ra0``, ``dec0``, ``pmra``, ``pmdec``, ``parallax``,
      ``ti_A``, ``ti_B``, ``ti_F``, ``ti_G`` (9 params)

    The Jacobian correction is **always applied**: a flat prior on the
    Thiele-Innes constants is not the same as a flat prior on the physical
    Campbell elements :math:`(a_0, \\omega, \\Omega, \\cos i)`.  The zeroth-order
    correction (evaluated at the conditional-mean TI constants) multiplies the
    marginal likelihood by :math:`(a_0 + \\delta_a)^{-m}(\\sin^2 i + \\delta_{s})^{-1}`,
    where :math:`m = 3` (uniform prior in :math:`a_0`) or :math:`m = 4`
    (log-uniform), and :math:`\\delta_a`, :math:`\\delta_s` are numerical
    floors that prevent singularities near face-on orbits or zero semi-major
    axis.

    The recommended way to construct this class is via :meth:`from_data`,
    which sets ``a_floor = Med(σ_AL) / sqrt(N)`` automatically.

    Parameters
    ----------
    a_floor : float
        Floor on :math:`a_0` (in the same angular units as the astrometric
        data, e.g. mas) used to regularize the Jacobian correction near zero
        semi-major axis.  Required.
    sin2i_floor : float, optional
        Floor on :math:`\\sin^2 i` for the Jacobian denominator.
        Default 0.01 (following Hsieh et al.).
    log_uniform_in_a : bool, optional
        If ``True``, assume a log-uniform (Jeffreys) prior on :math:`a_0`
        (uses :math:`m = 4`).  Default ``False`` (uniform in :math:`a_0`,
        :math:`m = 3`).

    Examples
    --------
    >>> from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
    >>> p = ThieleInnesGaiaAstrometry(a_floor=0.01)
    >>> [pp.name for pp in p.nonlinear_params()]
    ['period', 'eccentricity', 'phase_peri']
    >>> [pp.name for pp in p.linear_params()]
    ['ra0', 'dec0', 'pmra', 'pmdec', 'parallax', 'ti_A', 'ti_B', 'ti_F', 'ti_G']
    """

    a_floor: float
    sin2i_floor: float = 0.01
    log_uniform_in_a: bool = eqx.field(static=True, default=False)

    @classmethod
    def from_data(
        cls,
        data: "GaiaAstrometryData",
        sin2i_floor: float = 0.01,
        log_uniform_in_a: bool = False,
    ) -> "ThieleInnesGaiaAstrometry":
        """Construct with ``a_floor = Med(σ_AL) / sqrt(N)`` from the data.

        Parameters
        ----------
        data : GaiaAstrometryData
            Along-scan epoch astrometry data.
        sin2i_floor : float, optional
            Floor on :math:`\\sin^2 i`.  Default 0.01.
        log_uniform_in_a : bool, optional
            Use log-uniform prior on :math:`a_0`.  Default ``False``.

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
        from unxt.quantity import ustrip

        errs = ustrip(str(data.al_position_err.unit), data.al_position_err)
        a_floor = float(jnp.median(errs) / jnp.sqrt(jnp.asarray(errs).size))
        return cls(
            a_floor=a_floor,
            sin2i_floor=sin2i_floor,
            log_uniform_in_a=log_uniform_in_a,
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

        Columns: [ra0, dec0, pmra, pmdec, parallax,
        ti_A, ti_B, ti_F, ti_G].

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
        #   Δα = (B*X + G*Y) * sin_psi  → columns for ti_B and ti_G
        #   Δδ = (A*X + F*Y) * cos_psi  → columns for ti_A and ti_F
        return jnp.stack(
            [
                sin_psi,  # ra0
                cos_psi,  # dec0
                sin_psi * dt,  # pmra
                cos_psi * dt,  # pmdec
                parallax_factor,  # parallax
                X * cos_psi,  # ti_A  (coefficient of A in Δδ projection)
                X * sin_psi,  # ti_B  (coefficient of B in Δα projection)
                Y * cos_psi,  # ti_F  (coefficient of F in Δδ projection)
                Y * sin_psi,  # ti_G  (coefficient of G in Δα projection)
            ],
            axis=-1,
        )

    def linear_log_prior_correction(
        self, linear_map: dict[str, jax.Array]
    ) -> jax.Array:
        """Zeroth-order Jacobian correction for the Campbell ↔ Thiele-Innes change of variables.

        Evaluates :math:`-m \\ln(a_0 + \\delta_a) - \\ln(\\sin^2 i + \\delta_s)`
        at the conditional-mean Thiele-Innes constants, where :math:`a_0` and
        :math:`\\sin^2 i` are derived from the standard identities:

        .. math::

            u &= \\tfrac{1}{2}(A^2 + B^2 + F^2 + G^2) \\\\
            v &= AG - BF \\\\
            a_0 &= \\sqrt{u + \\sqrt{\\max(u^2 - v^2, 0)}} \\\\
            \\sin^2 i &= 1 - v^2 / a_0^4

        Parameters
        ----------
        linear_map : dict[str, jax.Array]
            Conditional-mean values of the marginalized linear parameters.
            Must contain keys ``"ti_A"``, ``"ti_B"``, ``"ti_F"``, ``"ti_G"``.

        Returns
        -------
        jax.Array
            Scalar log-correction.
        """
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
        return -m * jnp.log(a0 + self.a_floor) - jnp.log(sin2i + self.sin2i_floor)
