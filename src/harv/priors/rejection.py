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
from unxt import Quantity, ustrip

from harv.likelihood.helpers import (
    LinearPriorDist,
    PriorDist,
    _needs_explicit_sampling,
    _unwrap_dist,
)
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    RVParameters,
    SB2RVParameters,
)
from harv.priors.custom import (
    ParallaxDependentProperMotionPrior,
    PeriodDependentKPrior,
    PeriodDependentSemiMajorAxisPrior,
    _make_log_period_prior,
)
from harv.quantity_distribution import QuantityDistribution

__all__ = ("RejectionPrior",)

kipping_2013_ecc_prior = dist.Beta(0.867, 3.03)  # Kipping 2013 eccentricity prior


def _apply_overrides(
    kwargs: dict[str, Any],
    nonlinear: dict[str, PriorDist],
    linear: dict[str, Any],
    param_cls: type,
) -> None:
    """Partition *kwargs* into nonlinear/linear overrides and apply them.

    Raises ``TypeError`` for any key that is not a recognized parameter name
    in *param_cls*.
    """
    nl_names = set(param_cls.nonlinear_param_names)
    lin_names = set(param_cls.linear_param_names)

    for name, value in kwargs.items():
        if name in nl_names:
            nonlinear[name] = value
        elif name in lin_names:
            linear[name] = value
        else:
            msg = (
                f"default_{param_cls.__name__!s}() got an unexpected keyword "
                f"argument '{name}'. Valid parameter overrides: "
                f"{sorted(nl_names | lin_names)}"
            )
            raise TypeError(msg)


def _override_params_doc(param_cls: type) -> str:
    """Build a docstring fragment listing valid per-parameter overrides."""
    nl = ", ".join(f"``{n}``" for n in param_cls.nonlinear_param_names)
    lin = ", ".join(f"``{n}``" for n in param_cls.linear_param_names)
    return (
        f"**kwargs\n"
        f"            Per-parameter prior overrides.  Any nonlinear or linear\n"
        f"            parameter name from ``{param_cls.__name__}`` is accepted.\n"
        f"            Nonlinear: {nl}.\n"
        f"            Linear: {lin}."
    )


class RejectionPrior(eqx.Module):
    """Prior distribution for rejection sampling of Keplerian orbits.

    This class encapsulates the prior distributions for both nonlinear and linear
    parameters. It is agnostic to data type - the sampler determines which
    parameters are required based on the provided data.

    **Nonlinear parameterization:**

    Parameter names in ``nonlinear_priors`` match the field names of the orbit
    parameter structs (``period``, ``eccentricity``, ``phase_peri``, ``arg_peri``,
    ``cos_i``, ``lon_asc_node``). Distributions are sampled directly and the
    resulting values are used as-is when constructing param structs.

    **Common parameterizations:**

    **Radial Velocity:**
        - Nonlinear keys: ``period``, ``eccentricity``, ``phase_peri``, ``arg_peri``
        - Linear params: rv_semiamp, v_sys

    **Astrometry:**
        - Nonlinear keys: ``period``, ``eccentricity``, ``phase_peri``, ``cos_i``,
          ``arg_peri``, ``lon_asc_node``
        - Linear params: ra0, dec0, pmra, pmdec, parallax, semi_major_axis

    **Combined (astrometry + RV):**
        - Nonlinear keys: same as astrometry
        - Linear params: ra0, dec0, pmra, pmdec, parallax, semi_major_axis, rv_semiamp, v_sys

    Parameters
    ----------
    nonlinear_priors : dict[str, PriorDist]
        Mapping from parameter name to its prior distribution (a bare
        ``dist.Distribution`` for dimensionless parameters, or a
        :class:`QuantityDistribution` wrapper for parameters with physical
        units).  The sampler checks that this dict contains every field
        required by the chosen orbit param class.
    linear_prior: LinearPriorDist
        Prior over the linear parameters.  Accepts several forms:

        TODO: changed to only take a dictionary

        - ``dist.MultivariateNormal`` — joint Gaussian prior; all linear
          parameters are analytically marginalized.
        - ``QuantityDistribution`` wrapping a ``MultivariateNormal`` — same
          as above but with per-element unit tracking (tuple of unit strings).
        - ``LinearPriorCallable`` — callable returning a
          ``MultivariateNormal`` given the nonlinear parameters.
        - ``dict[str, PriorDist]`` — per-parameter priors.  Each entry is
          classified as Gaussian (``dist.Normal`` → marginalized), fixed
          (``dist.Delta`` → subtracted from residuals), or explicit
          (anything else → sampled from prior, treated as fixed for
          marginalization of the Gaussian subset).

        Dimension must match the number of linear parameters (2 for RV,
        6 for astrometry, 8 for combined).
    offsets : dict[str, dist.Normal | None], optional
        Multi-instrument offset priors. Keys are instrument names, values are
        ``dist.Normal`` priors (or ``None`` for the reference instrument).
        For RV data only.

    Examples
    --------
    >>> from harv.priors.rejection import RejectionPrior
    >>> from unxt import Quantity
    >>> prior = RejectionPrior.default_rv(...)
    >>> prior.n_nonlinear
    4

    TODO: support this
    RejectionPrior(
        {"period": ...},
        linear_prior={"rv_semiamp": ..., "v_sys": ...},
        offsets={"rv": {"survey1": None, "survey2": QuantityDistribution(...)}}
    )
    """

    nonlinear_priors: dict[str, PriorDist]
    linear_prior: LinearPriorDist

    # Which linear parameters to analytically marginalize.  ``None`` (default)
    # means "all linear params in ``linear_prior``".  An explicit tuple names
    # the subset to marginalize; the rest are sampled from ``linear_prior``
    # alongside the nonlinear params during rejection sampling.
    marginalize_names: tuple[str, ...] | None = None

    # Multi-instrument offsets (RV only, optional)
    # TODO: Changed type hint - this should be like offsets={"rv": {"ESPRESSO":
    # dist.Normal(...), "HARPS": None}} to be more generalizable to other data types
    # with linear params.
    offsets: dict[str, dict[str, QuantityDistribution | None]] | None = None

    # Polynomial trend support
    trend_order: int = 0
    trend_priors: dict[str, LinearPriorDist] | None = None

    # TODO: need to add something like this, to support, e.g., adding more complex model
    # extensions. For example, might want to add a Gaussian Process RV model with its
    # own hyperparameters to model a source with Keplerian + stellar activity.
    # {"rv": {"gp_mean": ..., etc.}} and then somewhere else need to know to pass
    # "gp_mean" to an extra_model function that constructs the GP model and incorporates
    # it into the likelihood. A simpler example would be adding a linear trend to the RV
    # model, which would add two linear parameters (slope and intercept of the trend)
    # with some prior, and then need to be incorporated into the RV likelihood's design
    # matrix and residuals. So we need a way to be able to add extra parameters that
    # could be linear or nonlinear
    # extra_priors: dict[str, dict[str, Any]] | None = None

    def __check_init__(self) -> None:
        # Validate marginalize_names against linear_prior keys.
        if self.marginalize_names is not None and isinstance(self.linear_prior, dict):
            unknown = set(self.marginalize_names) - set(self.linear_prior.keys())
            if unknown:
                msg = (
                    f"marginalize_names contains unknown linear parameter(s): "
                    f"{unknown}. Valid names: {tuple(self.linear_prior.keys())}"
                )
                raise ValueError(msg)

        # Auto-classify non-Gaussian linear prior entries as explicit.
        # Entries that cannot be analytically marginalized (e.g. HalfNormal,
        # Uniform) are excluded from marginalize_names so the sampler draws
        # from them alongside the nonlinear parameters.
        if isinstance(self.linear_prior, dict):
            explicit = {
                n for n, d in self.linear_prior.items() if _needs_explicit_sampling(d)
            }
            if explicit:
                if self.marginalize_names is None:
                    # Default "marginalize all" — exclude non-Gaussian entries.
                    new_marg = tuple(n for n in self.linear_prior if n not in explicit)
                else:
                    # User-specified set — also exclude non-Gaussian entries.
                    new_marg = tuple(
                        n for n in self.marginalize_names if n not in explicit
                    )
                object.__setattr__(self, "marginalize_names", new_marg)

        # TODO:
        # - Validate that within offsets["rv"], one value is None (reference instrument)
        #   and the rest are dist.Normal.  Raise ValueError if not.
        # - No "astrometry" key in offsets, NotImplementedError

    @property
    def n_nonlinear(self) -> int:
        """Number of nonlinear parameters."""
        return len(self.nonlinear_priors)

    def sample_nonlinear(self, key: jax.Array, n_samples: int) -> dict[str, Any]:
        """Sample nonlinear parameters from priors.

        Parameters
        ----------
        key : jax.Array
            Random key for sampling.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : dict[str, jax.Array]
            Dictionary mapping each parameter name to an array of shape
            ``(n_samples,)``.  Values are bare JAX arrays regardless of
            whether the distribution is wrapped in ``QuantityDistribution``.
        """
        keys = jr.split(key, len(self.nonlinear_priors))
        return {
            name: _unwrap_dist(d).sample(k, (n_samples,))
            for (name, d), k in zip(self.nonlinear_priors.items(), keys, strict=True)
        }

    # ------------------------------------------------------------------
    # Default constructors
    # ------------------------------------------------------------------

    @classmethod
    def default_rv(
        cls,
        *,
        period_min: Quantity["time"],
        period_max: Quantity["time"],
        sigma_K0: Quantity["speed"],
        sigma_v0: Quantity["speed"],
        P0: Quantity["time"] = Quantity(1.0, "yr"),
        offsets: dict[str, QuantityDistribution | None] | None = None,
        marginalize_names: tuple[str, ...] | None = None,
        trend_order: int = 0,
        trend_priors: dict[str, LinearPriorDist] | None = None,
        **kwargs: PriorDist,
    ) -> "RejectionPrior":
        r"""Create default prior for radial velocity data.

        The default linear prior follows thejoker's default: the RV semi-amplitude
        :math:`K` is assigned a zero-mean Gaussian whose width scales with period and
        eccentricity,

        .. math::

            \sigma_K(P, e) = \sigma_{K,0}
                \left(\frac{P}{P_0}\right)^{-1/3}
                \left(1 - e^2\right)^{-1/2}

        keeping the prior approximately constant in companion mass at fixed
        primary mass.  The systemic velocity :math:`v_0` has a fixed Gaussian
        prior with scale ``sigma_v0``.

        Parameters
        ----------
        period_min : Quantity["time"]
            Lower bound for the log-uniform period prior.  Pass a
            ``Quantity`` with time units (e.g. ``u.Q(50, "day")``) so
            the sampler can convert to whatever unit the data uses.
        period_max : Quantity["time"]
            Upper bound for the log-uniform period prior (same unit as
            ``period_min``).
        sigma_K0 : Quantity["speed"]
            RV semi-amplitude scale at the reference period ``P0``. For
            binary-star systems, a reasonable value is around 30 km/s.
        sigma_v0 : Quantity["speed"]
            Systemic velocity prior scale.
        P0 : Quantity["time"]
            Reference period for the K prior scaling.  Default: 1 yr.
        offsets : dict[str, QuantityDistribution | None], optional
            Multi-instrument offset priors. Keys are instrument names, values are
            ``QuantityDistribution`` priors (or ``None`` for the reference instrument).
        marginalize_names : tuple[str, ...] | None
            Subset of linear params to analytically marginalize.  ``None``
            (default) means "all that can be".
        {overrides}

        Returns
        -------
        prior : RejectionPrior
            Prior configured for RV data.
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_log_period_prior(period_min, period_max),
            "eccentricity": kipping_2013_ecc_prior,
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QuantityDistribution(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        }

        linear_prior: dict[str, Any] = {
            "rv_semiamp": PeriodDependentKPrior(sigma_K0=sigma_K0, P0=P0),
            "v_sys": QuantityDistribution(
                dist.Normal(0.0, sigma_v0.value), str(sigma_v0.unit)
            ),
        }

        _apply_overrides(kwargs, nonlinear, linear_prior, RVParameters)

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            marginalize_names=marginalize_names,
            offsets={"rv": offsets},
            trend_order=trend_order,
            trend_priors=trend_priors,
        )

    @classmethod
    def default_gaia_astrometry(
        cls,
        *,
        period_min: Quantity["time"],
        period_max: Quantity["time"],
        sigma_a0: Quantity["length"],
        sigma_parallax: Quantity["angle"],
        sigma_pos: Quantity["angle"],
        sigma_vtan: Quantity["speed"],
        P0: Quantity["time"] = Quantity(1.0, "yr"),
        marginalize_names: tuple[str, ...] | None = None,
        trend_order: int = 0,
        trend_priors: dict[str, LinearPriorDist] | None = None,
        **kwargs: PriorDist,
    ) -> "RejectionPrior":
        r"""Create default prior for Gaia astrometry data.

        The default semi-major axis prior scales with period and parallax so
        that it is approximately constant in companion mass:

        .. math::

            \sigma_a(P, \varpi) = \sigma_{a,0}
                \left(\frac{P}{P_0}\right)^{2/3}
                \varpi

        where :math:`\sigma_{a,0}` is in physical length units (AU) and
        :math:`\varpi` is the parallax in mas.

        Parallax is **explicitly sampled** (not analytically marginalized) by
        default because the Gaia catalog parallax is derived from the same
        epoch data being fitted — using it as a strong prior would double-count
        information.  For massive companions or black holes the catalog
        parallax can be biased.

        For **exoplanet** searches where the catalog parallax is trustworthy,
        pass a ``Normal`` prior on parallax (which will be auto-classified as
        marginalized) and a simple ``semi_major_axis`` prior that does not
        depend on parallax::

            prior = RejectionPrior.default_gaia_astrometry(
                ...,
                marginalize_names=("parallax", "ra0", "dec0", "pmra", "pmdec",
                                   "semi_major_axis"),
            )

        Parameters
        ----------
        period_min : Quantity["time"]
            Lower bound for the log-uniform period prior.
        period_max : Quantity["time"]
            Upper bound for the log-uniform period prior.
        sigma_a0 : Quantity["length"]
            Semi-major axis scale in physical length units (e.g. AU) at
            reference period ``P0``.
        sigma_parallax : Quantity["angle"]
            Scale for the half-normal parallax prior (mas).
        sigma_pos : Quantity["angle"]
            Scale for the position (ra0, dec0) Gaussian priors (mas).
        sigma_vtan : Quantity["speed"]
            Transverse-velocity dispersion scale (e.g. km/s) for the
            proper-motion (pmra, pmdec) priors.  Converted to angular
            proper motion via the sampled parallax.
        P0 : Quantity["time"]
            Reference period for the semi-major axis scaling.  Default: 1 yr.
        marginalize_names : tuple[str, ...] | None
            Subset of linear params to analytically marginalize.  ``None``
            (default) means "all that can be" — ``__check_init__`` will
            automatically classify ``HalfNormal`` entries as explicit.
        {overrides}

        Returns
        -------
        prior : RejectionPrior
            Prior configured for Gaia astrometry data.
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_log_period_prior(period_min, period_max),
            "eccentricity": kipping_2013_ecc_prior,
            "phase_peri": dist.Uniform(0.0, 1.0),
            "cos_i": dist.Uniform(-1.0, 1.0),
            "arg_peri": QuantityDistribution(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
            "lon_asc_node": QuantityDistribution(
                dist.Uniform(0.0, 2.0 * jnp.pi), "rad"
            ),
        }
        linear_prior: dict[str, Any] = {
            "ra0": QuantityDistribution(
                dist.Normal(0.0, ustrip("mas", sigma_pos)), "mas"
            ),
            "dec0": QuantityDistribution(
                dist.Normal(0.0, ustrip("mas", sigma_pos)), "mas"
            ),
            "pmra": ParallaxDependentProperMotionPrior(sigma_v0=sigma_vtan),
            "pmdec": ParallaxDependentProperMotionPrior(sigma_v0=sigma_vtan),
            "parallax": QuantityDistribution(
                dist.HalfNormal(ustrip("mas", sigma_parallax)), "mas"
            ),
            "semi_major_axis": PeriodDependentSemiMajorAxisPrior(
                sigma_a0=sigma_a0, P0=P0
            ),
        }

        _apply_overrides(kwargs, nonlinear, linear_prior, GaiaAstrometryParameters)

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            marginalize_names=marginalize_names,
            trend_order=trend_order,
            trend_priors=trend_priors,
        )

    @classmethod
    def default_sb2(
        cls,
        *,
        period_min: Quantity["time"],
        period_max: Quantity["time"],
        sigma_K0: Quantity["speed"],
        sigma_v0: Quantity["speed"],
        P0: Quantity["time"] = Quantity(1.0, "yr"),
        marginalize_names: tuple[str, ...] | None = None,
        trend_order: int = 0,
        trend_priors: dict[str, LinearPriorDist] | None = None,
        **kwargs: PriorDist,
    ) -> "RejectionPrior":
        r"""Create default prior for SB2 (double-lined) radial velocity data.

        Both semi-amplitudes use the same period-dependent scaling as
        :meth:`default_rv`.  The systemic velocity prior is a fixed Gaussian.

        Parameters
        ----------
        period_min : Quantity["time"]
            Lower bound for the log-uniform period prior.
        period_max : Quantity["time"]
            Upper bound for the log-uniform period prior.
        sigma_K0 : Quantity["speed"]
            RV semi-amplitude scale at the reference period ``P0``.
        sigma_v0 : Quantity["speed"]
            Systemic velocity prior scale.
        P0 : Quantity["time"]
            Reference period for the K prior scaling.  Default: 1 yr.
        marginalize_names : tuple[str, ...] | None
            Subset of linear params to analytically marginalize.
        trend_order : int
            Polynomial trend order (default 0).
        trend_priors : dict or None
            Per-trend-column priors.
        {overrides}

        Returns
        -------
        prior : RejectionPrior
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_log_period_prior(period_min, period_max),
            "eccentricity": kipping_2013_ecc_prior,
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QuantityDistribution(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        }

        linear_prior: dict[str, Any] = {
            "rv_semiamp_1": PeriodDependentKPrior(sigma_K0=sigma_K0, P0=P0),
            "rv_semiamp_2": PeriodDependentKPrior(sigma_K0=sigma_K0, P0=P0),
            "v_sys": QuantityDistribution(
                dist.Normal(0.0, sigma_v0.value), str(sigma_v0.unit)
            ),
        }

        _apply_overrides(kwargs, nonlinear, linear_prior, SB2RVParameters)

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            marginalize_names=marginalize_names,
            trend_order=trend_order,
            trend_priors=trend_priors,
        )


# Inject per-parameter override docs into the default_* method docstrings.
RejectionPrior.default_rv.__func__.__doc__ = (
    RejectionPrior.default_rv.__func__.__doc__.replace(
        "{overrides}", _override_params_doc(RVParameters)
    )
)
RejectionPrior.default_gaia_astrometry.__func__.__doc__ = (
    RejectionPrior.default_gaia_astrometry.__func__.__doc__.replace(
        "{overrides}", _override_params_doc(GaiaAstrometryParameters)
    )
)
RejectionPrior.default_sb2.__func__.__doc__ = (
    RejectionPrior.default_sb2.__func__.__doc__.replace(
        "{overrides}", _override_params_doc(SB2RVParameters)
    )
)
