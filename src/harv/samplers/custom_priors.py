"""Custom prior distributions and callables that produce numpyro distributions."""

from typing import Any

import equinox as eqx
import numpyro.distributions as dist
from unxt import ustrip

from harv.custom_types import ScalarQLength, ScalarQSpeed, ScalarQTime
from harv.distributions import QuantityDistribution

__all__ = (
    "ParallaxDependentProperMotionPrior",
    "PeriodDependentKPrior",
    "PeriodDependentSemiMajorAxisPrior",
)


class PeriodDependentKPrior(eqx.Module):
    r"""Period-dependent RV semi-amplitude prior matching The Joker's default.

    The std dev of the Gaussian prior on the RV semi-amplitude scales with orbital
    period and eccentricity so that it remains approximately constant in companion mass
    at fixed primary mass:

    .. math::

        \sigma_K(P, e) = \sigma_{K,0}
            \left(\frac{P}{P_0}\right)^{-1/3}
            \left(1 - e^2\right)^{-1/2}

    Parameters
    ----------
    sigma_K0
        RV semi-amplitude scale (km/s) at the reference period ``P0``.
        Default: 30 km/s -- appropriate for stellar binary searches.
    P0
        Numeric value of the reference period in units of ``P0_unit``.

    Notes
    -----
    This class implements :class:`~harv.likelihood.helpers.LinearPriorCallable` and is
    the default ``linear_prior`` returned by :meth:`RejectionPrior.default_rv`.

    See Also
    --------
    Price-Whelan et al. (2017) -- *The Joker: A Custom Monte Carlo Sampler for
    Binary-star and Exoplanet Radial Velocity Data*.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers.custom_priors import PeriodDependentKPrior
    >>> prior = PeriodDependentKPrior(sigma_K0=Q(30.0, "km/s"), P0=Q(1.0, "yr"))
    >>> prior.sigma_K0.unit
    Unit("km / s")

    Used as the default ``linear_prior`` for ``rv_semiamp`` in
    :meth:`RejectionPrior.default_rv`.  Called with a param struct to
    condition on nonlinear parameters:

    >>> qd = prior(params)  # doctest: +SKIP
    """

    sigma_K0: ScalarQSpeed
    P0: ScalarQTime

    def __call__(self, params: Any) -> QuantityDistribution:
        r"""Return the linear prior conditioned on nonlinear parameters.

        Parameters
        ----------
        params
            A parameter struct whose ``.period`` and ``.eccentricity`` fields
            are accessible.  ``.period`` must be a ``Quantity`` compatible
            with ``P0_unit``.

        Returns
        -------
        QuantityDistribution
            Prior over the RV semi-amplitude ``rv_semiamp``.
        """
        P_ratio = ustrip("", params.period / self.P0)
        sigma_K = (
            self.sigma_K0
            * P_ratio ** (-1.0 / 3.0)
            * (1.0 - params.eccentricity**2) ** (-0.5)
        )
        return QuantityDistribution(
            dist.Normal(loc=0.0, scale=ustrip(self.sigma_K0.unit, sigma_K)),
            str(self.sigma_K0.unit),
        )


class PeriodDependentSemiMajorAxisPrior(eqx.Module):
    r"""Period- and parallax-dependent semi-major axis prior for astrometry.

    The std dev of the Gaussian prior on the astrometric semi-major axis scales
    with orbital period and parallax so that it remains approximately constant
    in companion mass at fixed primary mass:

    .. math::

        \sigma_a(P, \varpi) = \sigma_{a,0}
            \left(\frac{P}{P_0}\right)^{2/3}
            \varpi

    where :math:`\sigma_{a,0}` is in physical length units (e.g. AU) and
    :math:`\varpi` is the parallax in mas.  Since 1 AU at 1 mas parallax
    subtends 1 mas, the product gives the angular semi-major axis in mas.

    Parameters
    ----------
    sigma_a0
        Semi-major axis scale in physical units (e.g. AU) at the reference
        period ``P0``.  Converted to angular size via the parallax.
    P0
        Reference period.

    Notes
    -----
    This class implements :class:`~harv.likelihood.helpers.LinearPriorCallable`
    and is the default ``linear_prior`` for ``semi_major_axis`` returned by
    :meth:`RejectionPrior.default_gaia_astrometry`.

    The ``params`` struct must have ``.period``, ``.eccentricity``, and
    ``.parallax`` fields.  ``parallax`` is available because it is classified
    as an explicit (non-marginalized) linear parameter by default.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers.custom_priors import PeriodDependentSemiMajorAxisPrior
    >>> prior = PeriodDependentSemiMajorAxisPrior(
    ...     sigma_a0=Q(5.0, "AU"), P0=Q(1.0, "yr"),
    ... )
    >>> prior.sigma_a0.unit
    Unit("AU")
    """

    sigma_a0: ScalarQLength
    P0: ScalarQTime

    def __call__(self, params: Any) -> QuantityDistribution:
        r"""Return the linear prior conditioned on nonlinear parameters.

        Parameters
        ----------
        params
            A parameter struct with ``.period`` (``Quantity``),
            ``.eccentricity`` (float), and ``.parallax`` (``Quantity``)
            fields.

        Returns
        -------
        QuantityDistribution
            Prior over the astrometric semi-major axis, in the same
            angular unit as ``params.parallax``.
        """
        P_ratio = ustrip("", params.period / self.P0)
        # By the definition of parallax (varpi == 1 AU / d), the angular
        # semi-major axis is a_angular = a_physical [AU] * varpi [angle].
        sigma_a0_au = ustrip("AU", self.sigma_a0)
        sigma_angular = params.parallax * (sigma_a0_au * P_ratio ** (2.0 / 3.0))
        out_unit = str(params.parallax.unit)
        return QuantityDistribution(
            dist.Normal(loc=0.0, scale=ustrip(out_unit, sigma_angular)),
            out_unit,
        )


class ParallaxDependentProperMotionPrior(eqx.Module):
    r"""Parallax-dependent proper-motion prior for astrometry.

    The std dev of the Gaussian prior on proper motion scales with parallax so
    that the prior is fixed in transverse velocity rather than in angular
    proper motion:

    .. math::

        \sigma_\mu(\varpi) = \sigma_{v,0}\;\varpi \;/\; (1\,\text{AU})

    where :math:`\sigma_{v,0}` is a velocity dispersion scale (e.g. km/s)
    and the division by 1 AU converts `velocity x parallax` to angular speed.
    Numerically, :math:`\sigma_{v,0}` is first converted to AU/yr, and then
    :math:`\sigma_\mu = \sigma_{v,0}[\text{AU/yr}] \times \varpi`.

    Parameters
    ----------
    sigma_v0
        Transverse-velocity dispersion scale (e.g. km/s).

    Notes
    -----
    This class implements :class:`~harv.likelihood.helpers.LinearPriorCallable`
    and is the default ``linear_prior`` for ``pmra`` and ``pmdec`` returned by
    :meth:`RejectionPrior.default_gaia_astrometry`.

    The ``params`` struct must have a ``.parallax`` field (``Quantity`` with
    angular units).  ``parallax`` is available because it is classified as an
    explicit (non-marginalized) linear parameter by default.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers.custom_priors import ParallaxDependentProperMotionPrior
    >>> prior = ParallaxDependentProperMotionPrior(sigma_v0=Q(50.0, "km/s"))
    >>> prior.sigma_v0.unit
    Unit("km / s")
    """

    sigma_v0: ScalarQSpeed

    def __call__(self, params: Any) -> QuantityDistribution:
        r"""Return the linear prior conditioned on nonlinear parameters.

        Parameters
        ----------
        params
            A parameter struct with a ``.parallax`` (``Quantity``) field.

        Returns
        -------
        QuantityDistribution
            Prior over the proper-motion component, in units of
            ``params.parallax.unit + "/yr"``.
        """
        # Convert velocity to AU/yr so that (AU/yr) x parallax [angle]
        # gives angular speed [angle/yr].
        sigma_v_au_yr = ustrip("AU/yr", self.sigma_v0)
        plx_unit = str(params.parallax.unit)
        plx_val = ustrip(plx_unit, params.parallax)
        scale = sigma_v_au_yr * plx_val
        return QuantityDistribution(
            dist.Normal(loc=0.0, scale=scale),
            plx_unit + "/yr",
        )
