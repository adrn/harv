"""Default priors without a parameterization analog."""

import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt import Q

from harv.custom_types import ScalarQSpeed, ScalarQTime
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    LinearPriorDict,
    LinearPriorDist,
    PriorDist,
)
from harv.models.priors.helpers import (
    _apply_overrides,
    _make_period_prior,
    _make_rv_semiamp_prior,
    _make_vsys_prior,
    kipping_2013_ecc_prior,
)
from harv.models.priors.prior import HarvPrior

__all__ = ("default_sb2_prior",)


def default_sb2_prior(
    *,
    period_min: ScalarQTime | None = None,
    period_max: ScalarQTime | None = None,
    sigma_K0: ScalarQSpeed | None = None,
    sigma_v0: ScalarQSpeed | None = None,
    P0: ScalarQTime = Q(1.0, "yr"),
    component_names: tuple[str, str] = ("primary", "secondary"),
    **kwargs: PriorDist | LinearPriorDist,
) -> HarvPrior:
    r"""Create default prior for SB2 (double-lined) radial velocity data.

    SB2 is a joint composition of two :class:`StandardRV` components, not a single
    parameterization, so this lives as a module-level factory rather than a
    classmethod on :class:`HarvPrior`.  It pairs naturally with
    :meth:`harv.models.JointModel.for_sb2` (``JointModel.for_sb2(prior=...)``).

    Both semi-amplitudes use the same period-dependent scaling as
    :meth:`HarvPrior.default_rv`.  The systemic velocity prior is a fixed
    Gaussian.

    The default names for the two components are "primary" and "secondary", which
    means the linear priors for the semi-amplitudes must be keyed as
    "primary.rv_semiamp" and "secondary.rv_semiamp".  You can customize the
    component names via the ``component_names`` argument, but the linear prior keys
    must always be ``{component_name}.rv_semiamp``.

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
    HarvPrior
        Prior configured for SB2 RV data.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers import default_sb2_prior
    >>> sorted(
    ...     default_sb2_prior(
    ...         period_min=Q(2.0, "day"),
    ...         period_max=Q(1000.0, "day"),
    ...         sigma_K0=Q(30.0, "km/s"),
    ...         sigma_v0=Q(50.0, "km/s"),
    ...     ).nonlinear_priors.keys()
    ... )
    ['arg_peri', 'eccentricity', 'period', 'phase_peri']
    >>> sorted(
    ...     default_sb2_prior(
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

    return HarvPrior(
        nonlinear_priors=nonlinear,
        linear_prior=linear_prior,
        extension_priors=extension_priors,
    )
