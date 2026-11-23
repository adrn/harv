"""Custom prior distributions and callables that produce numpyro distributions."""

from typing import Any, ClassVar, Literal

import equinox as eqx
import numpyro.distributions as dist
from unxt import ustrip

from harv.custom_types import ScalarQAngle, ScalarQLength, ScalarQSpeed, ScalarQTime
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
        RV semi-amplitude scale (km/s) at the reference period ``period_ref``.
        Default: 30 km/s -- appropriate for stellar binary searches.
    period_ref
        Numeric value of the reference period in units of ``P0_unit``.
    support
        Sign constraint on the semi-amplitude: ``"real"`` (the default, an
        ordinary zero-mean Gaussian), ``"positive"``, or ``"negative"``.

        The constrained forms exist for SB2 systems.  Because
        ``rv_shape`` is exactly antisymmetric in the argument of pericenter
        (``S(omega + pi) == -S(omega)``) and
        :meth:`~harv.models.JointModel.for_sb2` *shares* ``arg_peri`` between the
        two components, a negative semi-amplitude is the only way harv's SB2
        model expresses the secondary's antiphase motion -- ``phase_peri`` is
        shared too, and is not the mechanism.  Leaving both semi-amplitudes
        sign-free therefore admits an exact two-fold degeneracy,
        ``(K_1, K_2, omega)`` versus ``(-K_1, -K_2, omega + pi)``, which under
        symmetric priors makes the ``arg_peri`` posterior bimodal.  Pinning
        ``K_1 > 0`` and ``K_2 < 0`` selects one branch and also excludes the
        unphysical same-sign region, where both stars sit on the same side of
        the barycenter.

    Notes
    -----
    This class implements the ``LinearPriorCallable`` protocol and is the default
    ``linear_priors`` returned by
    :meth:`~harv.models.priors.HarvPrior.default_rv`.

    References
    ----------
    Price-Whelan et al. (2017) — *The Joker: A Custom Monte Carlo Sampler for
    Binary-star and Exoplanet Radial Velocity Data*.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.models.priors.callables import PeriodDependentKPrior
    >>> prior = PeriodDependentKPrior(sigma_K0=Q(30.0, "km/s"), period_ref=Q(1.0, "yr"))
    >>> prior.sigma_K0.unit
    Unit("km / s")

    Used as the default ``linear_priors`` for ``rv_semiamp`` in
    :meth:`~harv.models.priors.HarvPrior.default_rv`.  Called with a dict of
    parameter values to condition on the nonlinear parameters:

    >>> qd = prior({"period": Q(100.0, "day"), "eccentricity": 0.3})
    >>> qd.unit
    'km / s'
    """

    sigma_K0: ScalarQSpeed
    period_ref: ScalarQTime
    support: Literal["real", "positive", "negative"] = eqx.field(
        static=True, default="real"
    )

    requires: ClassVar[tuple[str, ...]] = ("period", "eccentricity")
    """Parameter values this prior reads; see ``pinned_linear_names``."""

    def __call__(self, params: dict[str, Any]) -> QuantityDistribution:
        r"""Return the linear prior conditioned on nonlinear parameters.

        Parameters
        ----------
        params
            Parameter values keyed by bare parameter name.  Must contain
            ``"period"`` (a ``Quantity`` compatible with ``period_ref``) and
            ``"eccentricity"``.

        Returns
        -------
        QuantityDistribution
            Prior over the RV semi-amplitude ``rv_semiamp``.
        """
        P_ratio = ustrip("", params["period"] / self.period_ref)
        sigma_K = (
            self.sigma_K0
            * P_ratio ** (-1.0 / 3.0)
            * (1.0 - params["eccentricity"] ** 2) ** (-0.5)
        )
        scale = ustrip(self.sigma_K0.unit, sigma_K)
        unit = str(self.sigma_K0.unit)
        if self.support == "real":
            return QuantityDistribution(dist.Normal(loc=0.0, scale=scale), unit)
        # HalfNormal is positive-only, so the negative branch needs the general
        # truncated form; both are marginalized analytically by
        # ``harv.stats.marginalized``.
        if self.support == "positive":
            truncated = dist.TruncatedNormal(0.0, scale, low=0.0)
        else:
            truncated = dist.TruncatedNormal(0.0, scale, high=0.0)
        return QuantityDistribution(truncated, unit)


def _check_scale_parallax_unit(scale_parallax: Any) -> None:
    """Static unit check; the sign is checked where it is used (``_width_parallax``)."""
    if scale_parallax is None:
        return
    if not (
        hasattr(scale_parallax, "unit") and scale_parallax.unit.is_equivalent("mas")
    ):
        msg = (
            "scale_parallax must be an angle, e.g. Q(2.16, 'mas'); "
            f"got {scale_parallax!r}"
        )
        raise TypeError(msg)


def _width_parallax(prior: Any, params: dict[str, Any]) -> Any:
    """The parallax that sets the width: ``scale_parallax``, else the sampled one."""
    plx = prior.scale_parallax
    if plx is not None:
        # On the data path, so the check survives jit (a discarded error_if is
        # dropped); `~(x > 0)` also catches NaN, e.g. a 2-parameter Gaia solution.
        return eqx.error_if(
            plx,
            ~((ustrip("mas", plx) > 0) & (ustrip("mas", plx) < float("inf"))),
            "scale_parallax must be positive and finite. For a parallax that is "
            "not measured to about 10% or better, leave scale_parallax unset and "
            "sample the parallax.",
        )
    if "parallax" not in params:
        raise KeyError(
            f"{type(prior).__name__} requires 'parallax' in the parameter dict, "
            "but it is missing (it was not sampled explicitly). Set scale_parallax "
            "to compute the width from a fixed parallax instead."
        )
    return params["parallax"]


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
        period ``period_ref``.  Converted to angular size via the parallax.
    period_ref
        Reference period.
    scale_parallax
        A fixed parallax (e.g. the catalog value) for :math:`\varpi` in the width,
        used instead of the sampled ``parallax``.  ``None`` (the default) reads the
        sampled value.  When set, nothing reads ``parallax``, so it is analytically
        marginalized; see the Notes.

    Notes
    -----
    This class implements the ``LinearPriorCallable`` protocol
    and is the default ``linear_priors`` for ``semi_major_axis`` returned by
    :meth:`~harv.models.StandardGaiaAstrometry.default_prior`.

    The ``params`` dict must contain ``"period"`` and, unless ``scale_parallax``
    is set, ``"parallax"``.  Reading the sampled parallax pins it to explicit
    sampling (see ``requires``), which costs the rejection sampler a factor of
    about :math:`\sigma_{\rm prior} / (\sqrt{2}\,\sigma_\varpi)` in efficiency for a
    parallax measured to :math:`\sigma_\varpi`.

    With ``scale_parallax`` set the parallax is marginalized exactly, including its
    covariance with the semi-major axis; the only approximation is that the width
    is off by the fractional error of ``scale_parallax``.  Use it when
    :math:`\sigma_\varpi / \varpi \lesssim 0.1`, keep the broad prior on
    ``parallax`` itself, and afterwards compare the posterior parallax with
    ``scale_parallax``.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.models.priors.callables import PeriodDependentSemiMajorAxisPrior
    >>> prior = PeriodDependentSemiMajorAxisPrior(
    ...     sigma_a0=Q(5.0, "AU"), period_ref=Q(1.0, "yr"),
    ... )
    >>> prior.sigma_a0.unit
    Unit("AU")

    >>> qd = prior(
    ...     {"period": Q(2.0, "yr"), "eccentricity": 0.1, "parallax": Q(10.0, "mas")}
    ... )
    >>> qd.unit
    'mas'

    A fixed parallax for the width drops ``parallax`` from ``requires``:

    >>> prior.requires
    ('period', 'eccentricity', 'parallax')
    >>> fixed = PeriodDependentSemiMajorAxisPrior(
    ...     sigma_a0=Q(5.0, "AU"), period_ref=Q(1.0, "yr"),
    ...     scale_parallax=Q(10.0, "mas"),
    ... )
    >>> fixed.requires
    ('period', 'eccentricity')
    """

    sigma_a0: ScalarQLength
    period_ref: ScalarQTime
    scale_parallax: ScalarQAngle | None = None

    def __check_init__(self) -> None:
        _check_scale_parallax_unit(self.scale_parallax)

    @property
    def requires(self) -> tuple[str, ...]:
        """Parameter values this prior reads; see ``pinned_linear_names``.

        Naming ``parallax`` here is what keeps it explicitly sampled: the
        auto-classification pins any linear parameter another prior reads. With
        ``scale_parallax`` set the width does not read it, so it is marginalized.
        """
        if self.scale_parallax is None:
            return ("period", "eccentricity", "parallax")
        return ("period", "eccentricity")

    def __call__(self, params: dict[str, Any]) -> QuantityDistribution:
        r"""Return the linear prior conditioned on nonlinear parameters.

        Parameters
        ----------
        params
            Parameter values keyed by bare parameter name.  Must contain
            ``"period"`` (``Quantity``) and, unless ``scale_parallax`` is set,
            ``"parallax"`` (``Quantity``).

        Returns
        -------
        QuantityDistribution
            Prior over the astrometric semi-major axis, in the same angular unit
            as the parallax that sets the width.
        """
        parallax = _width_parallax(self, params)
        P_ratio = ustrip("", params["period"] / self.period_ref)
        # By the definition of parallax (varpi == 1 AU / d), the angular
        # semi-major axis is a_angular = a_physical [AU] * varpi [angle].
        sigma_a0_au = ustrip("AU", self.sigma_a0)
        sigma_angular = parallax * (sigma_a0_au * P_ratio ** (2.0 / 3.0))
        out_unit = str(parallax.unit)
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
    scale_parallax
        A fixed parallax (e.g. the catalog value) for :math:`\varpi` in the width,
        used instead of the sampled ``parallax``.  ``None`` (the default) reads the
        sampled value.  When set, nothing reads ``parallax``, so it is analytically
        marginalized; see
        :class:`~harv.models.priors.callables.PeriodDependentSemiMajorAxisPrior`
        for when to use it.

    Notes
    -----
    This class implements the ``LinearPriorCallable`` protocol
    and is the default ``linear_priors`` for ``pmra`` and ``pmdec`` returned by
    :meth:`~harv.models.StandardGaiaAstrometry.default_prior`.

    The ``params`` dict must contain ``"parallax"`` (``Quantity`` with angular
    units) unless ``scale_parallax`` is set.  Reading the sampled parallax pins it
    to explicit sampling (see ``requires``).

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.models.priors.callables import ParallaxDependentProperMotionPrior
    >>> prior = ParallaxDependentProperMotionPrior(sigma_v0=Q(50.0, "km/s"))
    >>> prior.sigma_v0.unit
    Unit("km / s")

    >>> qd = prior({"parallax": Q(10.0, "mas")})
    >>> qd.unit
    'mas/yr'

    >>> prior.requires
    ('parallax',)
    >>> ParallaxDependentProperMotionPrior(
    ...     sigma_v0=Q(50.0, "km/s"), scale_parallax=Q(10.0, "mas")
    ... ).requires
    ()
    """

    sigma_v0: ScalarQSpeed
    scale_parallax: ScalarQAngle | None = None

    def __check_init__(self) -> None:
        _check_scale_parallax_unit(self.scale_parallax)

    @property
    def requires(self) -> tuple[str, ...]:
        """Parameter values this prior reads; see ``pinned_linear_names``."""
        return ("parallax",) if self.scale_parallax is None else ()

    def __call__(self, params: dict[str, Any]) -> QuantityDistribution:
        r"""Return the linear prior conditioned on nonlinear parameters.

        Parameters
        ----------
        params
            Parameter values keyed by bare parameter name.  Must contain
            ``"parallax"`` (``Quantity``) unless ``scale_parallax`` is set.

        Returns
        -------
        QuantityDistribution
            Prior over the proper-motion component, in the angular unit of the
            parallax that sets the width, per year.
        """
        parallax = _width_parallax(self, params)
        # Convert velocity to AU/yr so that (AU/yr) x parallax [angle]
        # gives angular speed [angle/yr].
        sigma_v_au_yr = ustrip("AU/yr", self.sigma_v0)
        plx_unit = str(parallax.unit)
        plx_val = ustrip(plx_unit, parallax)
        scale = sigma_v_au_yr * plx_val
        return QuantityDistribution(
            dist.Normal(loc=0.0, scale=scale),
            plx_unit + "/yr",
        )
