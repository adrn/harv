"""Prior distributions for rejection sampling of Keplerian orbits.

This module implements the RejectionPrior class which manages prior distributions
for both nonlinear and linear parameters in the rejection sampling algorithm.

The prior is agnostic to data type - it simply holds distributions for any/all
parameters. The sampler validates which parameters are needed based on the data.
"""

from typing import Any

import equinox as eqx
import jax
import jax.random as jr
import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt import Q, ustrip

from harv.custom_types import ScalarQAngle, ScalarQLength, ScalarQSpeed, ScalarQTime
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    LinearPriorDict,
    LinearPriorDist,
    PriorDist,
    _unwrap_dist,
)
from harv.samplers.custom_priors import (
    ParallaxDependentProperMotionPrior,
    PeriodDependentKPrior,
    PeriodDependentSemiMajorAxisPrior,
)

__all__ = ("RejectionPrior",)

kipping_2013_ecc_prior = dist.Beta(0.867, 3.03)  # Kipping 2013 eccentricity prior


def _apply_overrides(
    kwargs: dict[str, Any],
    nonlinear: dict[str, PriorDist],
    linear: dict[str, Any],
    extension_priors: dict[str, PriorDist],
) -> None:
    """Partition *kwargs* into nonlinear/linear/extension overrides *in place*.

    Known nonlinear and linear parameter names are added directly to their respective
    dicts - in place!  Anything else is accepted without validation and placed into
    *extension_priors* for later resolution at run-time when the sampler's extensions
    are known.
    """
    for name, value in kwargs.items():
        if name in nonlinear:
            nonlinear[name] = value
        elif name in linear:
            linear[name] = value
        else:
            extension_priors[name] = value


# Custom prior helper functions:


def _make_period_prior(
    *,
    period: Any | None = None,
    period_min: ScalarQTime | None = None,
    period_max: ScalarQTime | None = None,
) -> PriorDist:
    """Return a period prior from an explicit distribution or from bounds."""
    if period is not None:
        if period_min is not None or period_max is not None:
            raise TypeError(
                "Cannot specify both an explicit period prior and period_min/period_max"
            )
        return period

    if period_min is None or period_max is None:
        raise TypeError(
            "Must specify either an explicit period prior or both period_min and "
            "period_max"
        )

    return QuantityDistribution(
        dist.LogUniform(
            ustrip(str(period_min.unit), period_min),
            ustrip(str(period_min.unit), period_max),
        ),
        str(period_min.unit),
    )


def _make_rv_semiamp_prior(
    *,
    rv_semiamp: LinearPriorDist | None = None,
    sigma_K0: ScalarQSpeed | None = None,
    P0: ScalarQTime = Q(1.0, "yr"),
) -> LinearPriorDist:
    if rv_semiamp is not None:
        if sigma_K0 is not None:
            raise TypeError("Cannot specify both rv_semiamp and sigma_K0")
        return rv_semiamp
    if sigma_K0 is None:
        raise TypeError("Must specify either rv_semiamp or sigma_K0")
    return PeriodDependentKPrior(sigma_K0=sigma_K0, P0=P0)


def _make_vsys_prior(
    *,
    v_sys: LinearPriorDist | None = None,
    sigma_v0: ScalarQSpeed | None = None,
) -> LinearPriorDist:
    if v_sys is not None:
        if sigma_v0 is not None:
            raise TypeError("Cannot specify both v_sys and sigma_v0")
        return v_sys
    if sigma_v0 is None:
        raise TypeError("Must specify either v_sys or sigma_v0")
    return QuantityDistribution(
        dist.Normal(0.0, ustrip(str(sigma_v0.unit), sigma_v0)),
        str(sigma_v0.unit),
    )


def _make_pm_prior(
    *,
    pm: LinearPriorDist | None = None,
    sigma_vtan: ScalarQSpeed | None = None,
    name: str = "pmra/pmdec",
) -> LinearPriorDist:
    if pm is not None:
        if sigma_vtan is not None:
            raise TypeError(
                f"Cannot specify both an explicit {name} prior and sigma_vtan"
            )
        return pm
    if sigma_vtan is None:
        raise TypeError(f"Must specify either an explicit {name} prior or sigma_vtan")
    return ParallaxDependentProperMotionPrior(sigma_v0=sigma_vtan)


class RejectionPrior(eqx.Module):
    """Prior distribution for rejection sampling of Keplerian orbits.

    This class encapsulates the prior distributions for both nonlinear and linear
    parameters. It is agnostic to data type - the sampler determines which parameters
    are required based on the provided data.

    We recommend using the "default" factory constructors (e.g. ``default_rv()``,
    ``default_gaia_astrometry()``, etc.), which set up sensible priors for common use
    cases.

    **Nonlinear parameterization:**

    Parameter names in ``nonlinear_priors`` must match the field names of the
    parameterization, for example, ``period``, ``eccentricity``, ``phase_peri``, etc.
    These parameters are sampled explicitly.

    See the options available in `harv.models.parameterizations`.

    **Default parameterizations:**

    Radial Velocity:
        - Nonlinear: ``period``, ``eccentricity``, ``phase_peri``, ``arg_peri``
        - Linear: ``rv_semiamp``, ``v_sys``

    Astrometry:
        - Nonlinear: ``period``, ``eccentricity``, ``phase_peri``, ``cos_i``,
          ``arg_peri``, ``lon_asc_node``
        - Linear params: ``ra0``, ``dec0``, ``pmra``, ``pmdec``, ``parallax``,
          ``semi_major_axis``

    Parameters
    ----------
    nonlinear_priors
        Mapping from parameter name to its prior distribution (a bare
        ``dist.Distribution`` for dimensionless parameters, or a
        :class:`QuantityDistribution` wrapper for parameters with physical units).
    linear_prior
        Per-parameter priors for linear parameters. Each entry is classified:

        - ``dist.Normal`` or ``QD(Normal)`` -- Gaussian, can be analytically
          marginalized.
        - ``LinearPriorCallable`` -- called with nonlinear params to produce a Normal,
          can be marginalized.
        - ``dist.HalfNormal``, ``dist.Delta``, etc. -- non-Gaussian, sampled
          explicitly alongside nonlinear params.

        When using ``default_rv()`` with ``offsets``, the non-reference offset
        priors are automatically included as linear parameters.
    extension_priors
        Priors for extension parameters declared via ``extra_params()``.

    Examples
    --------
    >>> from harv.samplers import RejectionPrior  # doctest: +SKIP
    >>> from unxt import Q  # doctest: +SKIP
    >>> prior = RejectionPrior.default_rv(...)  # doctest: +SKIP
    >>> prior.n_nonlinear  # doctest: +SKIP
    4
    """

    nonlinear_priors: dict[str, PriorDist]
    linear_prior: LinearPriorDict

    # Priors for extension parameters (jitter, offsets, GP hyperparams, etc.).
    # Keys are the parameter names declared by the extension via extra_params().
    # Values are distributions (bare or QuantityDistribution).  These are not
    # validated here -- the sampler checks at run-time that every extension
    # parameter has a matching entry.
    extension_priors: dict[str, PriorDist] = eqx.field(default_factory=dict)

    @property
    def n_nonlinear(self) -> int:
        """Number of nonlinear parameters sampled by this prior.

        Returns the count of base orbital parameters in ``nonlinear_priors``.
        Extension parameters (e.g. jitter) are resolved at run time by the
        sampler and are not included in this count.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers import RejectionPrior
        >>> prior = RejectionPrior.default_rv(
        ...     period_min=Q(2.0, "day"), period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"), sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> prior.n_nonlinear
        4
        """
        return len(self.nonlinear_priors)

    def sample_nonlinear(self, key: jax.Array, n_samples: int) -> dict[str, Any]:
        """Sample nonlinear parameters from priors.

        Parameters
        ----------
        key
            Random key for sampling.
        n_samples
            Number of samples to draw.

        Returns
        -------
        samples
            Dictionary mapping each parameter name to an array of shape
            ``(n_samples,)``.  Values are bare JAX arrays regardless of
            whether the distribution is wrapped in ``QuantityDistribution``.

        Examples
        --------
        >>> import jax
        >>> from unxt import Q
        >>> from harv.samplers import RejectionPrior
        >>> sorted(
        ...     RejectionPrior.default_rv(
        ...         period_min=Q(2.0, "day"),
        ...         period_max=Q(1000.0, "day"),
        ...         sigma_K0=Q(30.0, "km/s"),
        ...         sigma_v0=Q(50.0, "km/s"),
        ...     ).sample_nonlinear(jax.random.key(0), 10).keys()
        ... )
        ['arg_peri', 'eccentricity', 'period', 'phase_peri']
        >>> RejectionPrior.default_rv(
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... ).sample_nonlinear(jax.random.key(0), 10)["period"].shape
        (10,)
        """
        keys = jr.split(key, len(self.nonlinear_priors))
        return {
            name: _unwrap_dist(d).sample(k, (n_samples,))
            for (name, d), k in zip(self.nonlinear_priors.items(), keys, strict=True)
        }

    # Default / factory constructors:

    @classmethod
    def default_rv(
        cls,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "RejectionPrior":
        r"""Create default prior for radial velocity data.

        The default linear prior for the RV semi-amplitude ,:math:`K`, has a standard
        deviation that scales with period and eccentricity to keep the prior
        approximately constant in companion mass at fixed primary mass:

        .. math::

            \sigma_K(P, e) = \sigma_{K,0}
                \left(\frac{P}{P_0}\right)^{-1/3}
                \left(1 - e^2\right)^{-1/2}

        The systemic velocity :math:`v_0` has a fixed Gaussian prior with specified
        scale ``sigma_v0``.

        Parameters
        ----------
        period_min
            Lower bound for the log-uniform period prior.  Pass a ``Quantity`` with time
            units (e.g. ``u.Q(50, "day")``) so the sampler can convert to whatever unit
            the data uses.
        period_max
            Upper bound for the log-uniform period prior (same unit as ``period_min``).
        sigma_K0
            RV semi-amplitude scale at the reference period ``P0``. For binary-star
            systems, a reasonable value is around 30 km/s. For exoplanets, something
            less than 1 km/s might be appropriate.
        sigma_v0
            Systemic velocity prior scale.
        P0
            Reference period for the K prior scaling.  Default: 1 yr.
        offsets
            Multi-instrument offset priors. Keys are instrument names, values are
            ``QuantityDistribution`` priors (or ``None`` for the reference instrument).
            Non-reference priors are merged into ``linear_prior`` automatically;
            reference entries (``None``) are ignored.
        **kwargs
            Override any default nonlinear or linear prior by name, or add extension
            parameter priors for unknown names (e.g. ``jitter=QD(...)``,
            ``espresso=QD(...)``).  Unknown names are not validated here -- the sampler
            checks them at run-time against the declared extension params.

        Returns
        -------
        prior
            Prior configured for RV data.

        Examples
        --------
        Basic RV prior with log-uniform period and Kipping eccentricity:

        >>> from unxt import Q; from harv.samplers import RejectionPrior; sorted(
        ...     RejectionPrior.default_rv(
        ...         period_min=Q(2.0, "day"),
        ...         period_max=Q(1000.0, "day"),
        ...         sigma_K0=Q(30.0, "km/s"),
        ...         sigma_v0=Q(50.0, "km/s"),
        ...     ).nonlinear_priors.keys()
        ... )
        ['arg_peri', 'eccentricity', 'period', 'phase_peri']

        With jitter (from a ``Jitter`` extension) and a multi-survey offset:

        >>> from unxt import Q
        >>> from harv.samplers import RejectionPrior
        >>> import numpyro.distributions as dist
        >>> from harv.distributions import QuantityDistribution as QD
        >>> sorted(
        ...     RejectionPrior.default_rv(
        ...         period_min=Q(2.0, "day"),
        ...         period_max=Q(1000.0, "day"),
        ...         sigma_K0=Q(30.0, "km/s"),
        ...         sigma_v0=Q(50.0, "km/s"),
        ...         jitter=QD(dist.HalfNormal(1.0), "km/s"),
        ...         espresso=QD(dist.Normal(0.0, 5.0), "km/s"),
        ...     ).extension_priors
        ... )
        ['espresso', 'jitter']
        """
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

        linear_prior: LinearPriorDict = {
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

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )

    @classmethod
    def default_gaia_astrometry(
        cls,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_a0: ScalarQLength,
        sigma_parallax: ScalarQAngle,
        sigma_pos: ScalarQAngle,
        sigma_vtan: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "RejectionPrior":
        r"""Create default prior for Gaia astrometry data.

        The default semi-major axis prior scales with period and parallax so that it is
        approximately constant in companion mass:

        .. math::

            \sigma_a(P, \varpi) = \sigma_{a,0}
                \left(\frac{P}{P_0}\right)^{2/3}
                \varpi

        where :math:`\sigma_{a,0}` is in physical length units (AU) and :math:`\varpi`
        is the parallax in mas.

        The proper motion priors are Gaussian with a standard deviation that also scales
        with the parallax to keep the prior approximately constant in transverse
        velocity.

        Parallax is explicitly sampled here (not analytically marginalized) by
        specifying it as a :class:`~numpyro.distributions.HalfNormal` distribution even
        though it is a linear parameter. This is needed for the semi-major axis and
        proper motion priors above.

        If the catalog parallax is trustworthy (e.g., for exoplanet cases), you can
        instead pass a tight Gaussian prior on parallax, which will then get
        marginalized by default in the sampler.

        Parameters
        ----------
        period_min
            Lower bound for the log-uniform period prior.
        period_max
            Upper bound for the log-uniform period prior.
        sigma_a0
            Semi-major axis scale in physical length units (e.g. AU) at reference period
            ``P0``.
        sigma_parallax
            Scale for the half-normal parallax prior (mas).
        sigma_pos
            Scale for the position (ra0, dec0) Gaussian priors (mas).
        sigma_vtan
            Transverse-velocity dispersion scale (e.g. km/s) for the proper-motion
            (pmra, pmdec) priors.  Converted to angular proper motion via the sampled
            parallax.
        P0
            Reference period for the semi-major axis scaling.  Default: 1 yr.
        **kwargs
            Override any default nonlinear or linear prior by name, or add extension
            parameter priors for unknown names.

        Returns
        -------
        prior
            Prior configured for Gaia astrometry data.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers import RejectionPrior
        >>> prior = RejectionPrior.default_gaia_astrometry(
        ...     period_min=Q(100.0, "day"),
        ...     period_max=Q(3000.0, "day"),
        ...     sigma_a0=Q(5.0, "AU"),
        ...     sigma_parallax=Q(10.0, "mas"),
        ...     sigma_pos=Q(100.0, "mas"),
        ...     sigma_vtan=Q(50.0, "km/s"),
        ... )
        >>> prior.n_nonlinear
        6
        >>> sorted(prior.nonlinear_priors.keys())
        ['arg_peri', 'cos_i', 'eccentricity', 'lon_asc_node', 'period', 'phase_peri']
        """
        # TODO: make sigma_pos, sigma_a0, sigma_parallax optional
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

        linear_prior: LinearPriorDict = {
            "ra0": QuantityDistribution(
                dist.Normal(0.0, ustrip("mas", sigma_pos)), "mas"
            ),
            "dec0": QuantityDistribution(
                dist.Normal(0.0, ustrip("mas", sigma_pos)), "mas"
            ),
            "pmra": _make_pm_prior(
                pm=kwargs.pop("pmra", None), sigma_vtan=sigma_vtan, name="pmra"
            ),
            "pmdec": _make_pm_prior(
                pm=kwargs.pop("pmdec", None), sigma_vtan=sigma_vtan, name="pmdec"
            ),
            "parallax": QuantityDistribution(
                dist.HalfNormal(ustrip("mas", sigma_parallax)), "mas"
            ),
            "semi_major_axis": PeriodDependentSemiMajorAxisPrior(
                sigma_a0=sigma_a0, P0=P0
            ),
        }

        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_prior, extension_priors)

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )

    @classmethod
    def default_sb2(
        cls,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        P0: ScalarQTime = Q(1.0, "yr"),
        component_names: tuple[str, str] = ("primary", "secondary"),
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "RejectionPrior":
        r"""Create default prior for SB2 (double-lined) radial velocity data.

        Both semi-amplitudes use the same period-dependent scaling as
        :meth:`default_rv`.  The systemic velocity prior is a fixed Gaussian.

        The default names for the two components are "primary" and "secondary", which
        means the linear priors for the semi-amplitudes must be keyed as
        "primary.rv_semiamp" and "secondary.rv_semiamp".  You can customize the
        component names via the ``component_names`` argument, but the linear prior keys
        must always be ``{component_name}.rv_semiamp``

        Parameters
        ----------
        period_min
            Lower bound for the log-uniform period prior.
        period_max
            Upper bound for the log-uniform period prior.
        sigma_K0
            RV semi-amplitude scale at the reference period ``P0``.
        sigma_v0
            Systemic velocity prior scale.
        P0
            Reference period for the K prior scaling.  Default: 1 yr.
        component_names
            Names of the two components.  These are used to construct the linear prior
            keys for the semi-amplitudes (e.g. "primary.rv_semiamp" and
            "secondary.rv_semiamp").
        **kwargs
            Override any default nonlinear or linear prior by name.

        Returns
        -------
        prior

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers import RejectionPrior
        >>> sorted(
        ...     RejectionPrior.default_sb2(
        ...         period_min=Q(2.0, "day"),
        ...         period_max=Q(1000.0, "day"),
        ...         sigma_K0=Q(30.0, "km/s"),
        ...         sigma_v0=Q(50.0, "km/s"),
        ...     ).nonlinear_priors.keys()
        ... )
        ['arg_peri', 'eccentricity', 'period', 'phase_peri']
        >>> sorted(
        ...     RejectionPrior.default_sb2(
        ...         period_min=Q(2.0, "day"),
        ...         period_max=Q(1000.0, "day"),
        ...         sigma_K0=Q(30.0, "km/s"),
        ...         sigma_v0=Q(50.0, "km/s"),
        ...     ).linear_prior
        ... )
        ['primary.rv_semiamp', 'secondary.rv_semiamp', 'v_sys']
        """
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

        linear_prior: LinearPriorDict = {
            f"{name}.rv_semiamp": _make_rv_semiamp_prior(
                rv_semiamp=kwargs.pop(f"{name}.rv_semiamp", None),
                sigma_K0=sigma_K0,
                P0=P0,
            )
            for name in component_names
        }
        linear_prior["v_sys"] = _make_vsys_prior(
            v_sys=kwargs.pop("v_sys", None),
            sigma_v0=sigma_v0,
        )

        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_prior, extension_priors)

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            extension_priors=extension_priors,
        )
