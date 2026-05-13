"""Parameterizations for Gaia epoch-astrometry models.

A *parameterization* is an ``eqx.Module`` subclass that declares the
names, units, and roles (linear / nonlinear) of parameters and knows how to
build the corresponding design matrix.
"""

__all__ = ("StandardGaiaAstrometry",)

from typing import Any, final

import jax
import quaxed.numpy as jnp

from harv.kepler.orbits import thiele_innes_ABFG
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations._base import AbstractParameterization


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
        jax.Array, shape (n_obs, 6)
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
