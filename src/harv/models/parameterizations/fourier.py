"""Kepler-free Fourier-series parameterizations.

These parameterizations replace the Keplerian orbit with a truncated Fourier series in
the mean longitude ``M = 2*pi*(t - time_ref)/P``, and all coefficients are linear
(so they can be marginalized). The only nonlinear parameter is ``period``: the
periastron phase is absorbed into each ``(cos, sin)`` amplitude pair, and eccentricity
distortion of the orbit shape is absorbed by the higher harmonics. No Kepler solve
occurs here.

This parameterization drives the Kepler periodogram functionality (``harv.periodogram``)
through the standard model/likelihood machinery (one ``model.log_prob`` per trial period
with every amplitude analytically marginalized). But these are also first-class
parameterizations: extensions (survey offsets, trends), the rejection sampler, and joint
models work as usual.

Amplitude priors come in two forms, and the **period-dependent one is the primary
path**: ``sigma_K0``/``period_ref`` for RV, ``sigma_a0``/``period_ref`` for Gaia,
matching the priors harv's Keplerian parameterizations already default to. A flat
``sigma_amp`` remains one argument away. Both scales are always explicit — there is
deliberately no data-driven default.

``sigma_0`` is the amplitude expected for the companion being searched for **at
period_ref**, not a global width. Unlike a flat prior, where too wide is merely
wasteful, too large a ``sigma_0`` tilts the periodogram toward long periods and can
let a long-period alias outrank the true mode: the Occam factor only reaches
``d * ln sigma(P)`` once ``sigma^2 lambda(P) >> 1``, so the scale sets how much of
the grid feels the tilt.

TODO(parallax-marginalization): the Gaia form needs a parallax, supplied as a point
value via ``periodogram(..., prior_params={"parallax": ...})``. That restricts it to
sources with a well-measured parallax; where the parallax is uncertain the prior is
confidently the wrong width, and worst exactly where the data are weakest. Two routes,
neither in scope yet:

1. Moment-match, staying analytic:
   ``sigma_eff^2 = sigma_a0^2 (P/period_ref)^(4/3) (plx^2 + sigma_plx^2)``. Still
   Normal, so the analytic linear marginalization is untouched.
2. Marginalize numerically. ``p(a_0|P) = int N(a_0; 0, sigma_a(P, plx)) p(plx) d plx``
   is a scale mixture of Gaussians and therefore *not* Gaussian, which
   ``_resolve_prior_to_mvn`` rejects. Doing it properly means evaluating Delta on a
   quadrature grid in parallax and logsumexp-ing, multiplying the periodogram cost by
   the number of nodes.
"""

__all__ = ("FourierGaiaAstrometry", "FourierRV")

from typing import Any, final

import equinox as eqx
import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from unxt.quantity import ustrip

from harv.custom_types import (
    ScalarQAngle,
    ScalarQAngularSpeed,
    ScalarQLength,
    ScalarQSpeed,
    ScalarQTime,
)
from harv.distributions import QuantityDistribution
from harv.models._helpers import LinearPriorDist, PriorDist
from harv.models.extensions.base import ParamInfo
from harv.models.parameterizations._base import AbstractParameterization
from harv.models.priors import HarvPrior
from harv.models.priors.custom_priors import (
    PeriodDependentKPrior,
    PeriodDependentSemiMajorAxisPrior,
)
from harv.models.priors.helpers import (
    _apply_overrides,
    _make_period_prior,
    _make_pos_prior,
    _make_vsys_prior,
)


def _harmonic_columns(
    sin_m: jax.Array, cos_m: jax.Array, n_terms: int
) -> list[tuple[jax.Array, jax.Array]]:
    """``(cos(kM), sin(kM))`` for ``k = 1..n_terms`` via angle-addition recurrences."""
    out: list[tuple[jax.Array, jax.Array]] = []
    sin_k, cos_k = sin_m, cos_m
    for k in range(1, n_terms + 1):
        if k > 1:
            sin_k, cos_k = (
                sin_k * cos_m + cos_k * sin_m,
                cos_k * cos_m - sin_k * sin_m,
            )
        out.append((cos_k, sin_k))
    return out


def _validate_amp_scale(
    sigma_amp: Any, sigma_0: Any, period_ref: Any, scale_name: str
) -> None:
    """Reject conflicting or half-given amplitude scales.

    Checked unconditionally, even when every amplitude column is overridden by
    name, so a contradictory pair is never silently ignored.
    """
    tilted = sigma_0 is not None or period_ref is not None
    if tilted and sigma_amp is not None:
        raise TypeError(
            f"Cannot specify both sigma_amp and {scale_name}/period_ref: the "
            "first is a "
            "flat amplitude prior, the second a period-dependent one. Pass one."
        )
    if tilted and (sigma_0 is None or period_ref is None):
        raise TypeError(
            f"{scale_name} and period_ref must be given together: {scale_name} is the "
            "amplitude scale *at* period_ref, so neither means anything alone."
        )


def _make_amp_prior(
    override: LinearPriorDist | None,
    tilted: LinearPriorDist | None,
    sigma_amp: Any,
    scale_name: str,
    unit_error: str,
) -> LinearPriorDist:
    """Resolve one amplitude column's prior, in precedence order.

    An explicit *override* by name wins; then *tilted*, the **period-dependent**
    prior harv's Keplerian parameterizations already use and the primary path
    for the periodogram; then a flat ``Normal(0, sigma_amp)``. There is
    deliberately no data-driven default, so overriding every amplitude column by
    name is the only way to supply neither scale.
    """
    if override is not None:
        return override
    if tilted is not None:
        return tilted
    if sigma_amp is None:
        raise TypeError(
            f"Must specify {scale_name} and period_ref (period-dependent, "
            "recommended) or "
            "sigma_amp (flat); there is deliberately no data-driven default "
            f"amplitude scale. {unit_error}"
        )
    return QuantityDistribution(
        dist.Normal(0.0, ustrip(str(sigma_amp.unit), sigma_amp)),
        str(sigma_amp.unit),
    )


@final
class FourierRV(AbstractParameterization):
    """Kepler-free RV parameterization: an ``n_terms`` Fourier series.

    Declares the following parameters:

        - Nonlinear: ``period`` (the period of the fundamental).
        - Linear: ``cos_amp_k``, ``sin_amp_k`` for ``k = 1..n_terms`` —
          harmonic amplitudes (the periastron phase is absorbed into each
          pair; eccentricity distortion is absorbed by ``k > 1`` terms) —
          and ``v_sys`` (systemic velocity).

    The design matrix has shape ``(n_obs, 2*n_terms + 1)`` with columns
    ``[cos(k M), sin(k M)]`` for ``k = 1..n_terms`` plus a constant column,
    where ``M = 2*pi*(t - time_ref)/P`` is the mean longitude. ``n_terms = 0`` is
    the valid null (no-signal) model: just the constant column.

    Examples
    --------
    >>> from harv.models.parameterizations.fourier import FourierRV
    >>> p = FourierRV(n_terms=2)
    >>> [pp.name for pp in p.params()]
    ['period', 'cos_amp_1', 'sin_amp_1', 'cos_amp_2', 'sin_amp_2', 'v_sys']
    >>> len(FourierRV(n_terms=0).linear_params())
    1
    """

    n_terms: int = eqx.field(static=True, default=2)

    def __check_init__(self) -> None:
        if self.n_terms < 0:
            raise ValueError(f"n_terms must be >= 0, got {self.n_terms}")

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        amps: list[ParamInfo] = []
        for k in range(1, self.n_terms + 1):
            amps.append(ParamInfo(f"cos_amp_{k}", "speed", linear=True))
            amps.append(ParamInfo(f"sin_amp_{k}", "speed", linear=True))
        return (
            ParamInfo("period", "time"),
            *amps,
            ParamInfo("v_sys", "speed", linear=True),
        )

    def strip_nonlinear_for_design(
        self, nonlinear_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Return ``nonlinear_values`` unchanged (the design matrix needs none)."""
        return dict(nonlinear_values)

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        nonlinear_values: dict[str, Any],  # noqa: ARG002  (uniform signature with StandardRV)
    ) -> jax.Array:
        """Build the ``(n_obs, 2*n_terms + 1)`` Fourier design matrix.

        Parameters
        ----------
        sin_f
            Sine of the mean longitude ``M`` (unit-stripped). Named ``sin_f``
            for signature uniformity with the Keplerian parameterizations;
            for this class the model supplies the mean longitude, not the
            true anomaly.
        cos_f
            Cosine of the mean longitude ``M`` (unit-stripped).
        nonlinear_values
            Unused (present for signature uniformity).

        Returns
        -------
            Design matrix block, shape ``(n_obs, 2*n_terms + 1)``.
        """
        cols: list[jax.Array] = []
        for cos_k, sin_k in _harmonic_columns(sin_f, cos_f, self.n_terms):
            cols.extend([cos_k, sin_k])
        cols.append(jnp.ones_like(sin_f))
        return jnp.column_stack(cols)

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_K0: ScalarQSpeed | None = None,
        period_ref: ScalarQTime | None = None,
        sigma_amp: ScalarQSpeed | None = None,
        sigma_v0: ScalarQSpeed | None = None,
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "HarvPrior":
        """Build a :class:`~harv.models.priors.HarvPrior` for this parameterization.

        All scales are explicit — there is deliberately no data-driven default
        (see the module ``TODO`` on recommended amplitude scales).

        Parameters
        ----------
        period_min, period_max
            Log-uniform period bounds (or pass an explicit ``period=`` prior).
        sigma_K0, period_ref
            **The primary path.** Period-dependent amplitude prior
            ``sigma_K(P) = sigma_K0 (P/period_ref)^(-1/3) (1-e^2)^(-1/2)``
            (:class:`~harv.models.priors.PeriodDependentKPrior`), applied to
            every harmonic amplitude. ``sigma_K0`` is the semi-amplitude
            expected for the companion you are searching for **at period_ref**, not a
            global width: too large a value tilts the periodogram toward long
            periods. Mutually exclusive with ``sigma_amp``.
        sigma_amp
            Flat, period-independent alternative: a Gaussian scale applied to
            every harmonic amplitude (``cos_amp_k`` / ``sin_amp_k``). Mutually
            exclusive with ``sigma_K0``/``period_ref``. Individual amplitudes can be
            overridden by name; a scale is required when ``n_terms > 0`` unless
            every amplitude is overridden.
        sigma_v0
            Systemic-velocity prior scale (or pass an explicit ``v_sys=``
            prior). Note there is no data centering anywhere: the prior must
            be appropriate for the data's actual systemic velocity.
        **kwargs
            Per-parameter prior overrides or extension priors.
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
        }
        _validate_amp_scale(sigma_amp, sigma_K0, period_ref, "sigma_K0")
        amp_tilted = (
            None
            # `or period_ref is None` is unreachable after _validate_amp_scale; it is
            # here so the narrowing is visible to the type checker.
            if sigma_K0 is None or period_ref is None
            else PeriodDependentKPrior(sigma_K0=sigma_K0, period_ref=period_ref)
        )

        linear_priors: dict[str, LinearPriorDist] = {}
        for pi in self.linear_params():
            if pi.name == "v_sys":
                linear_priors["v_sys"] = _make_vsys_prior(
                    v_sys=kwargs.pop("v_sys", None), sigma_v0=sigma_v0
                )
            else:
                linear_priors[pi.name] = _make_amp_prior(
                    kwargs.pop(pi.name, None),  # type: ignore[arg-type]
                    amp_tilted,
                    sigma_amp,
                    "sigma_K0",
                    "For RV data both are speeds (e.g. Q(30, 'km/s')).",
                )
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_priors, extension_priors)
        return HarvPrior(
            nonlinear_priors=nonlinear,
            linear_priors=linear_priors,
            extension_priors=extension_priors,
        )


@final
class FourierGaiaAstrometry(AbstractParameterization):
    """Kepler-free Gaia along-scan parameterization: astrometric solution + Fourier.

    Declares the following parameters:

        - Nonlinear: ``period`` (the period of the fundamental).
        - Linear: ``ra0``, ``dec0`` (position offsets at the reference epoch),
          ``pmra``, ``pmdec`` (proper motion), ``parallax``, and per harmonic
          ``k = 1..n_terms`` the Thiele-Innes-like amplitudes ``ti_A_k``,
          ``ti_B_k``, ``ti_F_k``, ``ti_G_k``.

    Per harmonic ``k`` with mean longitude ``M = 2*pi*(t - time_ref)/P``, the four
    columns are ``[cos(kM)*cos_psi, cos(kM)*sin_psi, sin(kM)*cos_psi,
    sin(kM)*sin_psi]`` — the circular-orbit Thiele-Innes structure (compare
    :class:`~harv.models.parameterizations.gaia.ThieleInnesGaiaAstrometry` at
    ``e = 0``), with eccentricity distortion absorbed by ``k > 1`` terms. The
    five astrometric-solution columns match
    :class:`~harv.models.parameterizations.gaia.StandardGaiaAstrometry`.
    ``n_terms = 0`` is the valid null model: the 5-parameter astrometric
    solution alone.

    Examples
    --------
    >>> from harv.models.parameterizations.fourier import FourierGaiaAstrometry
    >>> p = FourierGaiaAstrometry(n_terms=1)
    >>> [pp.name for pp in p.params()][:6]
    ['period', 'ra0', 'dec0', 'pmra', 'pmdec', 'parallax']
    >>> [pp.name for pp in p.linear_params()][5:]
    ['ti_A_1', 'ti_B_1', 'ti_F_1', 'ti_G_1']
    """

    n_terms: int = eqx.field(static=True, default=2)

    def __check_init__(self) -> None:
        if self.n_terms < 0:
            raise ValueError(f"n_terms must be >= 0, got {self.n_terms}")

    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        amps: list[ParamInfo] = []
        for k in range(1, self.n_terms + 1):
            amps.extend(
                ParamInfo(f"ti_{c}_{k}", "angle", linear=True)
                for c in ("A", "B", "F", "G")
            )
        return (
            ParamInfo("period", "time"),
            ParamInfo("ra0", "angle", linear=True),
            ParamInfo("dec0", "angle", linear=True),
            ParamInfo("pmra", "angular_speed", linear=True),
            ParamInfo("pmdec", "angular_speed", linear=True),
            ParamInfo("parallax", "angle", linear=True),
            *amps,
        )

    def design_matrix(
        self,
        sin_f: jax.Array,
        cos_f: jax.Array,
        dt: jax.Array,
        sin_psi: jax.Array,
        cos_psi: jax.Array,
        parallax_factor: jax.Array,
        nonlinear_values: dict[str, Any],  # noqa: ARG002  (uniform signature)
    ) -> jax.Array:
        """Build the ``(n_obs, 5 + 4*n_terms)`` along-scan design matrix.

        Parameters
        ----------
        sin_f
            Sine of the mean longitude ``M`` (unit-stripped; named for
            signature uniformity with the Keplerian parameterizations).
        cos_f
            Cosine of the mean longitude ``M`` (unit-stripped).
        dt
            Time since the reference epoch (unit-stripped, in the model's
            ``pm_time_unit``).
        sin_psi
            Sine of the scan angle.
        cos_psi
            Cosine of the scan angle.
        parallax_factor
            Along-scan parallax factor (unit-stripped).
        nonlinear_values
            Unused (present for signature uniformity).

        Returns
        -------
            Design matrix block, shape ``(n_obs, 5 + 4*n_terms)``.
        """
        cols: list[jax.Array] = [
            sin_psi,  # ra0
            cos_psi,  # dec0
            sin_psi * dt,  # pmra
            cos_psi * dt,  # pmdec
            parallax_factor,  # parallax
        ]
        for cos_k, sin_k in _harmonic_columns(sin_f, cos_f, self.n_terms):
            cols.extend(
                [cos_k * cos_psi, cos_k * sin_psi, sin_k * cos_psi, sin_k * sin_psi]
            )
        return jnp.stack(cols, axis=-1)

    def default_prior(
        self,
        *,
        period_min: ScalarQTime | None = None,
        period_max: ScalarQTime | None = None,
        sigma_a0: ScalarQLength | None = None,
        period_ref: ScalarQTime | None = None,
        sigma_amp: ScalarQAngle | None = None,
        sigma_pos: ScalarQAngle | None = None,
        sigma_pm: ScalarQAngularSpeed | None = None,
        sigma_parallax: ScalarQAngle | None = None,
        **kwargs: PriorDist | LinearPriorDist,
    ) -> "HarvPrior":
        """Build a :class:`~harv.models.priors.HarvPrior` for this parameterization.

        All scales are explicit — there is deliberately no data-driven default
        (see the module ``TODO``). Every linear prior here is a plain Gaussian
        so the full model marginalizes analytically; in particular the
        ``parallax`` prior is a zero-mean Normal nuisance by default — override
        with e.g. ``parallax=QD(dist.Normal(plx, plx_err), "mas")`` when the
        catalog parallax is known.

        Parameters
        ----------
        period_min, period_max
            Log-uniform period bounds (or pass an explicit ``period=`` prior).
        sigma_a0, period_ref
            **The primary path.** Period-dependent amplitude prior
            ``sigma_a(P) = sigma_a0 (P/period_ref)^(2/3) * parallax``
            (:class:`~harv.models.priors.PeriodDependentSemiMajorAxisPrior`),
            applied to every ``ti_*_k``. ``sigma_a0`` is a *physical* length
            (e.g. ``Q(1.0, "AU")``) -- the orbit size expected for the companion
            you are searching for **at period_ref**.

            Because it needs a parallax to become an angle, and the periodogram
            marginalizes parallax rather than sampling it, the value must be
            supplied at scan time via
            ``periodogram(..., prior_params={"parallax": ...})``. That sets the
            prior's *scale* only: the parallax column stays in the design matrix
            and is still fitted and marginalized. This restricts the path to
            sources with a well-measured parallax -- see the module
            ``TODO(parallax-marginalization)``. Mutually exclusive with
            ``sigma_amp``.
        sigma_amp
            Flat, period-independent alternative: a Gaussian scale for every
            harmonic amplitude (``ti_*_k``), an angle (e.g. ``Q(1.0, "mas")``).
            Mutually exclusive with ``sigma_a0``/``period_ref``. Individual amplitudes
            can be overridden by name; a scale is required when ``n_terms > 0``
            unless every amplitude is overridden.
        sigma_pos
            Position-offset (``ra0``/``dec0``) prior scale. Must be generous
            enough to absorb the reference-position offset.
        sigma_pm
            Proper-motion (``pmra``/``pmdec``) prior scale, an angular speed
            (e.g. ``Q(50.0, "mas/yr")``).
        sigma_parallax
            Parallax-column prior scale (zero-mean Normal).
        **kwargs
            Per-parameter prior overrides or extension priors.
        """
        nonlinear: dict[str, PriorDist] = {
            "period": _make_period_prior(
                period_min=period_min,
                period_max=period_max,
                period=kwargs.pop("period", None),
            ),
        }

        def _normal(name: str, sigma: Any, kind: str) -> LinearPriorDist:
            override = kwargs.pop(name, None)
            if override is not None:
                return override  # type: ignore[return-value]
            if sigma is None:
                raise TypeError(
                    f"Must specify {kind} (or an explicit prior for {name!r})."
                )
            return QuantityDistribution(
                dist.Normal(0.0, ustrip(str(sigma.unit), sigma)), str(sigma.unit)
            )

        _validate_amp_scale(sigma_amp, sigma_a0, period_ref, "sigma_a0")
        amp_tilted = (
            None
            # `or period_ref is None` is unreachable after _validate_amp_scale; it is
            # here so the narrowing is visible to the type checker.
            if sigma_a0 is None or period_ref is None
            else PeriodDependentSemiMajorAxisPrior(
                sigma_a0=sigma_a0, period_ref=period_ref
            )
        )

        linear_priors: dict[str, LinearPriorDist] = {}
        for pi in self.linear_params():
            if pi.name in ("ra0", "dec0"):
                linear_priors[pi.name] = _make_pos_prior(
                    pos=kwargs.pop(pi.name, None),  # type: ignore[arg-type]
                    sigma_pos=sigma_pos,
                    name=pi.name,
                )
            elif pi.name in ("pmra", "pmdec"):
                linear_priors[pi.name] = _normal(pi.name, sigma_pm, "sigma_pm")
            elif pi.name == "parallax":
                linear_priors[pi.name] = _normal(
                    pi.name, sigma_parallax, "sigma_parallax"
                )
            else:
                linear_priors[pi.name] = _make_amp_prior(
                    kwargs.pop(pi.name, None),  # type: ignore[arg-type]
                    amp_tilted,
                    sigma_amp,
                    "sigma_a0",
                    "For Gaia data sigma_a0 is a length (e.g. Q(1.0, 'AU')) and "
                    "sigma_amp an angle (e.g. Q(1.0, 'mas')).",
                )
        extension_priors: dict[str, PriorDist] = {}
        _apply_overrides(kwargs, nonlinear, linear_priors, extension_priors)
        return HarvPrior(
            nonlinear_priors=nonlinear,
            linear_priors=linear_priors,
            extension_priors=extension_priors,
        )
