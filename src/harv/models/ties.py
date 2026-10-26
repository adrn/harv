"""Tied linear parameters: a linear parameter set to ``scale(params) * source``.

A tie is a conditional delta prior, declared in ``linear_priors`` with
:func:`tied`. Because ``scale`` depends only on nonlinear or explicitly sampled
values, the tied parameter's design column folds into its source's column and the
analytic marginalization is unchanged. See ``docs/spec.md`` §Tied linear
parameters.
"""

__all__ = ("TiedPrior", "rv_semiamp_per_semi_major_axis", "tied")

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import quaxed.numpy as jnp
from unxt import Q
from unxt.quantity import AllowValue, ustrip


@final
class TiedPrior(eqx.Module):
    """A linear parameter equal to ``scale(params) * source``.

    Build with :func:`tied`. ``source`` is the component-qualified name of the
    linear parameter this one follows (e.g. ``"astrom.semi_major_axis"``), and
    ``scale`` maps the source component's parameter dict to a ``Q`` in
    ``[this parameter's unit] / [source unit]``.
    """

    source: str = eqx.field(static=True)
    # Not static: a scale may be an eqx.Module holding arrays (like the
    # LinearPriorCallable classes); a plain function is treated as static anyway.
    scale: Callable[[dict[str, Any]], Any]

    @property
    def requires(self) -> tuple[str, ...]:
        """Parameters ``scale`` reads; linear ones are pinned to explicit sampling."""
        return tuple(getattr(self.scale, "requires", ()))


def tied(source: str, scale: Callable[[dict[str, Any]], Any]) -> TiedPrior:
    """Tie a linear parameter to ``scale(params) * source``.

    Use it as the parameter's entry in ``linear_priors`` of a
    :class:`~harv.models.JointModel` (spec §Tied linear parameters).

    Parameters
    ----------
    source
        Component-qualified name of the linear parameter to follow, e.g.
        ``"astrom.semi_major_axis"``. It must be analytically marginalized.
    scale
        Function of the source component's parameter dict (the same values its
        callable priors receive) returning the ratio as a ``Q``. An optional
        ``requires`` attribute names the parameters it reads.

    Examples
    --------
    >>> from harv.models import rv_semiamp_per_semi_major_axis, tied
    >>> prior = tied("astrom.semi_major_axis", scale=rv_semiamp_per_semi_major_axis)
    >>> prior.requires
    ('period', 'eccentricity', 'cos_i', 'parallax')
    """
    return TiedPrior(source=source, scale=scale)


def rv_semiamp_per_semi_major_axis(params: dict[str, Any]) -> Q:
    """RV semi-amplitude per unit photocenter semi-major axis, for an unseen companion.

    ``K = c * a`` with ``c = 2 pi sin(i) / (P sqrt(1 - e^2) parallax)``: when the
    companion contributes no light, the photocenter traces the star's orbit, so
    ``a / parallax`` is the star's semi-major axis in AU. Uses the sky-frame
    convention of spec §``KeplerianOrientation`` (shared ``arg_peri``, ``K > 0``).

    Examples
    --------
    >>> from unxt import Q
    >>> c = rv_semiamp_per_semi_major_axis({
    ...     "period": Q(365.25, "day"), "eccentricity": 0.0, "cos_i": 0.0,
    ...     "parallax": Q(1.0, "mas"),
    ... })
    >>> round(float(c.to("km/(s mas)").value), 2)  # Earth's orbital speed per AU
    29.79
    """
    cos_i = jnp.asarray(ustrip(AllowValue, "", params["cos_i"]), dtype=float)
    # Floored so the gradient stays finite at |cos_i| = 1, which NUTS can reach.
    sin2_i = jnp.maximum((1 - cos_i) * (1 + cos_i), jnp.finfo(cos_i.dtype).tiny)
    sin_i = jnp.sqrt(sin2_i)
    eccentricity = ustrip(AllowValue, "", params["eccentricity"])
    return (
        2
        * jnp.pi
        * Q(1.0, "AU")
        * sin_i
        / (params["period"] * jnp.sqrt(1 - eccentricity**2) * params["parallax"])
    )


rv_semiamp_per_semi_major_axis.requires = (  # ty: ignore[unresolved-attribute]
    "period",
    "eccentricity",
    "cos_i",
    "parallax",
)
