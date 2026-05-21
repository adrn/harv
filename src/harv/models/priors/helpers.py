"""Helper functions for building the default priors in Harv."""

from typing import Any

import numpyro.distributions as dist
from unxt import Q, ustrip

from harv.custom_types import ScalarQAngle, ScalarQLength, ScalarQSpeed, ScalarQTime
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    LinearPriorDist,
    PriorDist,
)
from harv.models.priors.custom_priors import (
    ParallaxDependentProperMotionPrior,
    PeriodDependentKPrior,
    PeriodDependentSemiMajorAxisPrior,
)

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


def _make_parallax_prior(
    *,
    parallax: LinearPriorDist | None = None,
    sigma_parallax: ScalarQAngle | None = None,
) -> LinearPriorDist:
    if parallax is not None:
        if sigma_parallax is not None:
            raise TypeError("Cannot specify both parallax and sigma_parallax")
        return parallax
    if sigma_parallax is None:
        raise TypeError("Must specify either parallax or sigma_parallax")
    return QuantityDistribution(dist.HalfNormal(ustrip("mas", sigma_parallax)), "mas")


def _make_pos_prior(
    *,
    pos: LinearPriorDist | None = None,
    sigma_pos: ScalarQAngle | None = None,
    name: str = "ra0/dec0",
) -> LinearPriorDist:
    if pos is not None:
        if sigma_pos is not None:
            raise TypeError(
                f"Cannot specify both an explicit {name} prior and sigma_pos"
            )
        return pos
    if sigma_pos is None:
        raise TypeError(f"Must specify either an explicit {name} prior or sigma_pos")
    return QuantityDistribution(dist.Normal(0.0, ustrip("mas", sigma_pos)), "mas")


def _make_semi_major_axis_prior(
    *,
    semi_major_axis: LinearPriorDist | None = None,
    sigma_a0: ScalarQLength | None = None,
    P0: ScalarQTime = Q(1.0, "yr"),
) -> LinearPriorDist:
    if semi_major_axis is not None:
        if sigma_a0 is not None:
            raise TypeError("Cannot specify both semi_major_axis and sigma_a0")
        return semi_major_axis
    if sigma_a0 is None:
        raise TypeError("Must specify either semi_major_axis or sigma_a0")
    return PeriodDependentSemiMajorAxisPrior(sigma_a0=sigma_a0, P0=P0)
