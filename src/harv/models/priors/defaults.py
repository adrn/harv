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
from harv.models.priors._helpers import (
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
    period_ref: ScalarQTime = Q(1.0, "yr"),
    component_names: tuple[str, str] = ("primary", "secondary"),
    signed_semiamp: bool = True,
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

    **The two semi-amplitudes have opposite signs by default.**  The secondary's
    antiphase motion is carried by a *negative* ``rv_semiamp``, not by
    ``phase_peri``: ``rv_shape`` is exactly antisymmetric in the argument of
    pericenter (``S(omega + pi) == -S(omega)``), and
    :meth:`~harv.models.JointModel.for_sb2` shares both ``arg_peri`` and
    ``phase_peri`` across the components, so flipping the sign of K is the only
    mechanism available.  Sign-free priors on both therefore admit an exact
    two-fold degeneracy -- ``(K_1, K_2, omega)`` and
    ``(-K_1, -K_2, omega + pi)`` predict identical RVs for both components --
    which makes the ``arg_peri`` posterior bimodal with modes 180 degrees apart,
    and additionally leave prior mass on the unphysical same-sign region.
    ``signed_semiamp=True`` pins ``K_1 > 0`` and ``K_2 < 0``, resolving both.
    The constrained priors are still marginalized analytically (see
    :mod:`harv.stats.marginalized`).

    :meth:`~harv.samplers.Samples.wrap_angles` also repairs the sign pattern
    after the fact -- the shared ``arg_peri`` shift flips both semi-amplitudes
    together -- so the sign prior is not the only route to a canonical answer.
    It is the route that avoids spending prior volume and acceptance on the
    mirror branch in the first place, and it additionally excludes the
    unphysical same-sign region, which the ``arg_peri`` symmetry cannot fix
    because it flips both semi-amplitudes at once. Under the signed prior
    ``wrap_angles`` is a no-op.

    Pass ``signed_semiamp=False`` for the older sign-free behaviour.

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
        RV semi-amplitude scale at the reference period ``period_ref``.
    sigma_v0
        Systemic velocity prior scale.
    period_ref
        Reference period for the K prior scaling.  Default: 1 yr.
    component_names
        Names of the two components.  These are used to construct the linear prior
        keys for the semi-amplitudes (e.g. "primary.rv_semiamp" and
        "secondary.rv_semiamp").
    signed_semiamp
        When ``True`` (the default), constrain the first component's
        semi-amplitude to be positive and the second's to be negative, resolving
        the sign/``omega + pi`` degeneracy described above.  When ``False``, both
        get sign-free zero-mean Gaussians.  Ignored for any component whose
        ``rv_semiamp`` prior is supplied explicitly via ``**kwargs``.
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
    ...     ).linear_priors
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

    # The sign convention is per component and only meaningful pairwise: the
    # first gets K > 0, the second K < 0.
    supports = ("positive", "negative") if signed_semiamp else ("real", "real")
    linear_priors: LinearPriorDict = {
        f"{name}.rv_semiamp": _make_rv_semiamp_prior(
            rv_semiamp=kwargs.pop(f"{name}.rv_semiamp", None),
            sigma_K0=sigma_K0,
            period_ref=period_ref,
            support=support,
        )
        for name, support in zip(component_names, supports, strict=True)
    }
    linear_priors["v_sys"] = _make_vsys_prior(
        v_sys=kwargs.pop("v_sys", None),
        sigma_v0=sigma_v0,
    )

    extension_priors: dict[str, PriorDist] = {}
    _apply_overrides(kwargs, nonlinear, linear_priors, extension_priors)

    return HarvPrior(
        nonlinear_priors=nonlinear,
        linear_priors=linear_priors,
        extension_priors=extension_priors,
    )
