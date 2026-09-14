"""Linearized-model periodogram.

At each trial period, the data are modeled with a Kepler-free Fourier-series
parameterization (:class:`~harv.models.parameterizations.fourier.FourierRV` /
:class:`~harv.models.parameterizations.fourier.FourierGaiaAstrometry`) whose
amplitudes are all *linear* and analytically marginalized — so the periodogram
scans over period only. The statistic is

``delta_ln_likelihood(f) = lnL(f) - lnL_base``

where both terms are ordinary ``model.log_prob`` marginal likelihoods of the
standard model machinery: ``lnL(f)`` uses the ``n_terms``-harmonic model at
trial period ``1/f`` and ``lnL_base`` uses the same model with ``n_terms = 0``
(RV: constant offset only; Gaia: the 5-parameter astrometric solution — so
scan-law / parallax / proper-motion power cancels in Δ). Extensions that add
linear columns (e.g. survey offsets, trends) participate in both models and
work as usual. ``lnL_base`` carries no Fourier columns, so it is normally
period-independent and evaluated once; when one of its own linear priors is a
``LinearPriorCallable`` it is evaluated across the grid like ``lnL(f)``.

All priors are **explicit**: the required ``prior`` argument is a standard
:class:`~harv.models.priors.HarvPrior` built from the Fourier
parameterization's ``default_prior`` (or by hand). There is deliberately no
data-driven prior, no centering, and no hidden scale assumptions — Δ is a
per-frequency log Bayes factor under exactly the priors you supplied. Note the
Occam factors are constant across the grid only when the amplitude priors are
period-independent; a period-dependent amplitude prior (``LinearPriorCallable``)
intentionally tilts Δ.

See ``docs/spec.md``, "Periodogram and interim period priors".
"""

__all__ = ("PeriodogramResult", "periodogram")

import functools
import warnings
from collections.abc import Mapping
from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Any, Literal, cast, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from unxt import Q, ustrip

from harv.custom_types import NFloatArray, NFrequency, NTime, ScalarQTime
from harv.data.containers import AbstractDatasetContainer
from harv.data.datasets import AbstractData, GaiaAstrometryData, RVData
from harv.models._helpers import _is_callable_prior
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.extensions.base import AbstractExtension
from harv.models.parameterizations.fourier import FourierGaiaAstrometry, FourierRV
from harv.models.priors import HarvPrior
from harv.models.rv import RVModel
from harv.periodogram.grid import _data_t_span
from harv.periodogram.grid import frequency_grid as get_frequency_grid
from harv.samplers._prior_resolution import (
    effective_linear_prior_from_prior,
    validate_extension_priors,
)

if TYPE_CHECKING:
    from harv.models.component import AbstractComponentModel

# Dataset type -> (Fourier parameterization class, model class)
_FOURIER_DISPATCH: dict[type, tuple[type, type]] = {
    RVData: (FourierRV, RVModel),
    GaiaAstrometryData: (FourierGaiaAstrometry, GaiaAstrometryModel),
}


@final
class PeriodogramResult(eqx.Module):
    """Result of :func:`periodogram`.

    ``delta_ln_likelihood[i]`` is the marginal log-likelihood of the
    trial-period model at ``frequency[i]`` minus that of the base (no-signal)
    model, summed over datasets for container inputs. ``n_terms`` is the
    *effective* Fourier term count used (the maximum across datasets for
    container inputs). It equals the requested value except in profile mode,
    which reduces it per dataset to keep the trial model overdetermined.

    ``ln_likelihood_base`` is a scalar for the usual period-independent base
    model, and a per-frequency array when a base-column prior resolves against
    the trial period (see :func:`periodogram`).
    """

    frequency: NFrequency
    delta_ln_likelihood: NFloatArray
    ln_likelihood_base: Float[jax.Array, ""] | NFloatArray
    t_span: ScalarQTime
    t_ref: ScalarQTime
    _: KW_ONLY
    per_dataset: dict[str, NFloatArray] | None = None
    n_terms: int = eqx.field(static=True, default=1)
    statistic: str = eqx.field(static=True, default="marginal")

    @property
    def period(self) -> NTime:
        """Trial periods, ``1 / frequency`` (descending order)."""
        return cast("NTime", 1.0 / self.frequency)

    def max_period(self) -> ScalarQTime:
        """Trial period with the highest ``delta_ln_likelihood``."""
        return self.period[jnp.argmax(self.delta_ln_likelihood)]

    def plot(self, ax: Any = None, *, x: str = "period", **kwargs: Any) -> Any:
        """Plot ``delta_ln_likelihood`` against period (default) or frequency.

        Extra keyword arguments are forwarded to ``ax.plot``.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dependency)

        kwargs.setdefault("marker", "")

        if ax is None:
            _, ax = plt.subplots()
        xx = self.period if x == "period" else self.frequency
        ax.plot(ustrip(str(xx.unit), xx), self.delta_ln_likelihood, **kwargs)
        if x == "period":
            ax.set_xscale("log")
        ax.set_xlabel(f"{x} [{xx.unit}]")
        if self.statistic == "profile":
            ax.set_ylabel(r"$\Delta \ln \hat{\mathcal{L}}$")
        else:
            ax.set_ylabel(r"$\Delta \ln \mathcal{L}$")
        return ax


def _effective_n_terms(
    fourier_cls: type,
    n_requested: int,
    n_obs: int,
    n_ext_linear: int,
    *,
    profile: bool,
) -> int:
    """Warn when the trial model is not comfortably overdetermined.

    "Comfortably" means at least two observations per linear column (columns
    counted from the parameterization itself plus linear extension columns).
    That bar is a convention, not a rank condition: it sits well above the
    ``n_obs = n_cols`` point where the design matrix actually loses rank. It is
    placed where recovery of the true period empirically starts to fall off,
    which tracks the observations-per-column ratio rather than the absolute
    column count. Below it a weakly-constrained trial model fits almost any
    trial period, so spurious alias peaks come to dominate the periodogram.

    What happens past that threshold differs by statistic, because the two
    fail differently (see "Profile mode" in ``docs/spec.md``):

    - **Profile** (``profile=True``): the least-squares solve is unregularized,
      so once the columns outnumber the observations chi^2 hits zero at every
      trial period and the statistic is *identically flat* -- a hard,
      information-free failure. ``n_terms`` is reduced (floored at 1) to keep
      the model overdetermined, and the reduction is warned about.
    - **Marginal** (``profile=False``): the amplitude prior regularizes, so
      ``M = I + BᵀB`` stays invertible however rank-deficient the design matrix
      is and the statistic remains well-posed at any column count. There is no
      breakdown point to key a cap to, so ``n_terms`` is returned unchanged and
      the warning only reports that the scan is increasingly prior-driven.
    """
    n_base = len(fourier_cls(n_terms=0).linear_params()) + n_ext_linear
    n_per_term = len(fourier_cls(n_terms=1).linear_params()) - (n_base - n_ext_linear)
    h_max = int((n_obs / 2.0 - n_base) // n_per_term)
    if n_requested <= h_max:
        return n_requested

    eff = n_requested if not profile else max(1, h_max)
    n_cols = n_base + n_per_term * n_requested
    # State the criterion the check actually applies. The requested model is
    # usually still overdetermined here (n_cols < n_obs), so this is a
    # weak-constraint warning, not an overfitting one.
    head = (
        f"n_terms={n_requested} leaves {n_obs / n_cols:.1f} observations per "
        f"linear column ({n_cols} columns, {n_obs} observations), below the 2 "
        "per column this check expects"
    )
    # "Pass a smaller n_terms" only helps when some n_terms >= 1 clears the bar.
    silenceable = h_max >= 1 and n_requested > 1
    advice = " Pass a smaller n_terms to silence this." if silenceable else ""

    if not profile:
        why = (
            " The amplitude prior keeps the marginal statistic well-posed at "
            "any column count, so n_terms is not reduced, but the periodogram "
            "is increasingly prior-driven and spurious alias peaks may "
            "dominate."
        )
    elif eff < n_requested:
        head += f"; reducing to n_terms={eff} ({n_base + n_per_term * eff} columns)"
        why = (
            " In profile mode the least-squares solve is not regularized, so "
            "it degrades quickly here and goes identically flat once the "
            "columns reach the observation count."
        )
    else:
        why = (
            " n_terms is already at its minimum and cannot be reduced further. "
            "In profile mode the least-squares solve is not regularized, so "
            "chi^2 may hit zero at every trial period and leave the "
            "periodogram flat."
        )

    warnings.warn(head + "." + why + advice, UserWarning, stacklevel=4)
    return eff


def _nl(period: Any, period_unit: str) -> dict[str, Any]:
    """Nonlinear values passed to ``model.log_prob`` at one trial period.

    ``eccentricity = 0`` is adopted inside the periodogram: the Fourier trial
    model has no eccentricity (higher harmonics absorb the orbit-shape
    distortion), but carrying it lets eccentricity-dependent amplitude priors
    (e.g. :class:`~harv.models.priors.PeriodDependentKPrior`) resolve at
    ``e = 0`` through the standard prior machinery. It is ignored by the
    Fourier design matrix.
    """
    return {"period": Q(period, period_unit), "eccentricity": 0.0}


def _bind_prior_params(
    linear_priors: dict[str, Any], prior_params: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Bind concrete values into every ``LinearPriorCallable`` in *linear_priors*.

    Values are injected into the dict a callable prior is *resolved against*,
    never into ``nl_values``. That distinction is load-bearing: ``parallax`` is
    a linear parameter of :class:`~harv.models.FourierGaiaAstrometry`, and
    ``log_prob``'s auto mode pulls any linear name out of ``nl_values`` and
    reclassifies it as an explicit, *non-marginalized* column. Supplying a
    parallax that way would silently fix the parallax column in both the trial
    and base models -- a different model, not a resolved prior.

    Non-callable priors pass through untouched, and the scan's own values
    (``period``, ``eccentricity``) take precedence over *prior_params*.
    """
    if not prior_params:
        return linear_priors
    extra = dict(prior_params)

    def bind(prior: Any) -> Any:
        if not _is_callable_prior(prior):
            return prior
        # Still a plain callable, so _is_callable_prior and
        # _needs_explicit_sampling classify the wrapper exactly as they did the
        # prior: it stays analytically marginalized.
        return lambda params, _p=prior: _p({**extra, **params})

    return {name: bind(prior) for name, prior in linear_priors.items()}


def _check_callable_priors(
    linear_priors: dict[str, Any], probe: dict[str, Any]
) -> None:
    """Resolve callable priors once, eagerly, for a readable error.

    Without this a missing (or misspelled -- unresolved keys are silently
    ignored downstream) ``prior_params`` entry surfaces as a ``KeyError`` raised
    inside a ``jax.jit(jax.vmap(...))`` trace.
    """
    for name, prior in linear_priors.items():
        if not _is_callable_prior(prior):
            continue
        try:
            prior(probe)
        except KeyError as exc:
            msg = (
                f"Could not resolve the callable linear prior for {name!r}: "
                f"{exc.args[0]} Pass any value the periodogram does not scan "
                "over via prior_params, e.g. periodogram(..., "
                'prior_params={"parallax": Q(10.0, "mas")}).'
            )
            raise TypeError(msg) from exc


def _resolve_per_dataset(value: Any, name: str, ds_name: str) -> Any:
    """Resolve a per-dataset argument that may be a Mapping keyed by dataset name.

    Anything that is not such a Mapping -- including a single ``HarvPrior``
    shared across same-type datasets -- is passed through unchanged.
    """
    if isinstance(value, Mapping) and not isinstance(value, HarvPrior):
        try:
            return value[ds_name]
        except KeyError:
            raise TypeError(
                f"{name} mapping has no entry for dataset {ds_name!r}."
            ) from None
    return value


def _reject_unscannable(
    prior: HarvPrior | Literal[False],
    extensions: tuple[AbstractExtension, ...],
    fourier_cls: type,
) -> None:
    """Reject inputs the periodogram can neither scan nor marginalize."""
    nonlin_extra = set() if prior is False else set(prior.nonlinear_priors) - {"period"}
    if nonlin_extra:
        raise TypeError(
            f"The Fourier trial model has no nonlinear parameters besides 'period'; "
            f"prior.nonlinear_priors also contains {sorted(nonlin_extra)}. Build the "
            f"prior from {fourier_cls.__name__}(...).default_prior(...)."
        )
    for ext in extensions:
        nonlin_ext = [p.name for p in ext.extra_params() if not p.linear]
        if nonlin_ext:
            raise TypeError(
                f"Extension {type(ext).__name__} declares nonlinear parameter(s) "
                f"{nonlin_ext}, which the periodogram cannot scan or marginalize. "
                "Only linear-column extensions (e.g. MultiSurveyOffset, "
                "MonomialTrend) are supported."
            )


def _resolve_linear_priors(
    prior: HarvPrior,
    model: "AbstractComponentModel",
    fourier_cls: type,
    n_terms: int,
    prior_params: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the supplied prior against the trial model and bind *prior_params*.

    Only reached on the marginal path, which never reduces ``n_terms``, so the
    requested and effective term counts always agree here.
    """
    eff_lp = effective_linear_prior_from_prior(prior, model) or {}
    validate_extension_priors(prior, model, eff_lp)
    requested_names = {p.name for p in fourier_cls(n_terms=n_terms).linear_params()}
    allowed = requested_names | (set(eff_lp) - set(prior.linear_priors))
    unknown = set(prior.linear_priors) - allowed
    if unknown:
        raise TypeError(
            f"prior.linear_priors entries {sorted(unknown)} are not parameters of "
            f"{fourier_cls.__name__}(n_terms={n_terms}) (expected "
            f"{sorted(requested_names)})."
        )
    missing = [n for n in model._all_linear_names() if n not in eff_lp]
    if missing:
        raise TypeError(
            f"prior.linear_priors is missing entries for {missing} required by "
            f"{fourier_cls.__name__}(n_terms={n_terms})."
        )
    return _bind_prior_params(eff_lp, prior_params)


def _dataset_delta_lnl(
    dataset: AbstractData,
    prior: HarvPrior | Literal[False],
    extensions: tuple[AbstractExtension, ...],
    f_grid: NFrequency,
    n_terms: int,
    prior_params: Mapping[str, Any] | None = None,
) -> tuple[NFloatArray, Float[jax.Array, ""] | NFloatArray, int]:
    """Δ log-likelihood over the grid: marginal, or profile when ``prior is False``."""
    if type(dataset) not in _FOURIER_DISPATCH:
        raise NotImplementedError(
            f"No periodogram implementation for {type(dataset).__name__}; only "
            f"{', '.join(cls.__name__ for cls in _FOURIER_DISPATCH)} are "
            "currently supported."
        )
    fourier_cls, model_cls = _FOURIER_DISPATCH[type(dataset)]

    _reject_unscannable(prior, extensions, fourier_cls)

    n_obs = int(dataset.time.shape[0])
    n_ext_linear = sum(1 for ext in extensions for p in ext.extra_params() if p.linear)
    eff_terms = _effective_n_terms(
        fourier_cls, n_terms, n_obs, n_ext_linear, profile=prior is False
    )

    model = cast(
        "AbstractComponentModel",
        model_cls(
            parameterization=fourier_cls(n_terms=eff_terms), extensions=extensions
        ),
    )
    base_model = cast(
        "AbstractComponentModel",
        model_cls(parameterization=fourier_cls(n_terms=0), extensions=extensions),
    )

    period_grid = 1.0 / f_grid
    period_unit = str(period_grid.unit)
    p_vals = jnp.asarray(ustrip(period_unit, period_grid))

    if prior is False:
        # No priors at all, so every linear column is profiled and the base --
        # which carries no Fourier columns and no callable prior to resolve --
        # is period-independent by construction: one evaluation.
        def lnl_at(p: jax.Array) -> jax.Array:
            return model._log_prob_profile(_nl(p, period_unit), dataset)

        lnl0 = base_model._log_prob_profile(_nl(p_vals[0], period_unit), dataset)
    else:
        eff_lp = _resolve_linear_priors(
            prior, model, fourier_cls, n_terms, prior_params
        )
        full_lp = {n: eff_lp[n] for n in model._all_linear_names()}
        base_lp = {n: eff_lp[n] for n in base_model._all_linear_names()}

        # base_lp is a name-subset of full_lp sliced from the same dict, so
        # probing full_lp covers both models.
        _check_callable_priors(full_lp, _nl(p_vals[0], period_unit))

        def base_at(p: jax.Array) -> jax.Array:
            return base_model.log_prob(
                _nl(p, period_unit), dataset, linear_priors=base_lp
            )

        def lnl_at(p: jax.Array) -> jax.Array:
            return model.log_prob(_nl(p, period_unit), dataset, linear_priors=full_lp)

        # The base model carries no Fourier columns, so it is period-independent
        # and one evaluation suffices -- unless one of its own linear priors
        # resolves against the trial period (a LinearPriorCallable such as
        # PeriodDependentKPrior on v_sys). Then its baseline genuinely varies
        # across the grid and subtracting a single value would tilt every Delta.
        if any(_is_callable_prior(p) for p in base_lp.values()):
            lnl0 = jax.jit(jax.vmap(base_at))(p_vals)
        else:
            lnl0 = base_at(p_vals[0])

    lnl = jax.jit(jax.vmap(lnl_at))(p_vals)
    return lnl - lnl0, lnl0, eff_terms


def periodogram(
    data: AbstractData | AbstractDatasetContainer,
    frequency_grid: NFrequency | None = None,
    *,
    prior: HarvPrior | Mapping[str, HarvPrior | Literal[False]] | Literal[False],
    period_min: ScalarQTime | None = None,
    period_max: ScalarQTime | None = None,
    samples_per_peak: int | None = None,
    n_grid: int | None = None,
    n_terms: int = 2,
    extensions: tuple[AbstractExtension, ...]
    | Mapping[str, tuple[AbstractExtension, ...]] = (),
    prior_params: Mapping[str, Any] | None = None,
) -> PeriodogramResult:
    """Compute a periodogram of the data.

    At each trial frequency this evaluates the marginal log-likelihood of a Kepler-free
    ``n_terms``-harmonic Fourier model (every amplitude linear and analytically
    marginalized under the supplied priors) minus that of the ``n_terms = 0`` base
    model. Multiple harmonics capture non-sinusoidal periodicity (e.g. eccentric
    orbits); the base model carries the non-periodic structure (constant offset for RV;
    the 5-parameter astrometric solution for Gaia, so scan-law/parallax/proper-motion
    power cancels). For containers the per-dataset Δ are summed into one periodogram per
    source.

    Unlike a Lomb-Scargle periodogram, this is a (Bayesian) log-marginal-likelihood
    periodogram: the trial model is fully marginalized over its linear parameters under
    the supplied priors, and the base model is marginalized over its own linear
    parameters. The statistic is a log Bayes factor under the priors you supplied.
    Lomb-Scargle or other Keplerian periodograms are often instead computed from profile
    likelihoods at the maximum-likelihood linear amplitudes, which is a different
    statistic -- and not a limiting case of this one: as the amplitude priors widen the
    Occam factor grows without bound, so Delta diverges rather than approaching the
    profile statistic. Pass ``prior=False`` to compute the profile statistic directly.
    The recommended amplitude priors here scale with period the same way
    harv's Keplerian priors do — ``sigma_K0``/``P0`` for RV (semi-amplitude, falling as
    ``P^(-1/3)``) and ``sigma_a0``/``P0`` for astrometry (semi-major axis, rising as
    ``P^(2/3)``). Pass ``sigma_amp`` instead for the constant-amplitude case, which is
    the one comparable *in shape* to a profile-likelihood periodogram.

    Parameters
    ----------
    data
        `~harv.data.RVData`, `~harv.data.GaiaAstrometryData`, or a dataset
        container holding them.
    frequency_grid
        Explicit frequency grid. Mutually exclusive with the grid keywords
        (``period_min``, ``period_max``, ``n_grid``).
    prior
        REQUIRED. ``False`` selects **profile mode**: no priors at all, every
        linear column fitted by generalized least squares, and
        ``delta_ln_likelihood`` becomes ``0.5 * (chi2_base - chi2_trial)`` --
        the statistic Lomb-Scargle and kepmodel report, provided for
        comparison. It cannot be combined with ``prior_params`` and cannot
        appear inside a per-dataset mapping (the two statistics are not
        commensurable, so summing them across datasets is meaningless).
        ``PeriodogramResult.statistic`` records which one was computed.
        Otherwise a :class:`~harv.models.priors.HarvPrior` for the Fourier
        trial model — build it with
        ``FourierRV(n_terms=...).default_prior(...)`` /
        ``FourierGaiaAstrometry(n_terms=...).default_prior(...)`` — or, for
        containers, a mapping from dataset name to per-dataset priors (a
        single prior may be shared when all datasets have the same type).
        There is deliberately no data-driven default: Δ is a log Bayes factor
        under exactly these priors. Period-dependent amplitude priors
        (``LinearPriorCallable``) are resolved per trial period.
    period_min, period_max, samples_per_peak, n_grid
        Grid construction keywords, forwarded to :func:`frequency_grid`
        (``period_min`` is required when ``frequency_grid`` is not given).
    n_terms
        Number of Fourier terms (harmonics of the trial frequency).
        ``n_terms >= 2`` absorbs eccentricity distortion of the orbit shape.
        Must be at least 1. Default: 2. A ``UserWarning`` is emitted per
        dataset when the trial model is not comfortably overdetermined (fewer
        than two observations per linear column, including extension columns).
        In profile mode ``n_terms`` is also *reduced* to restore that, since an
        unregularized least-squares solve goes identically flat once the
        columns outnumber the observations; in the default marginal mode the
        amplitude prior keeps the statistic well-posed, so the requested value
        is kept and only the warning fires.
        ``PeriodogramResult.n_terms`` reports the effective value.
    prior_params
        Concrete values for parameters a ``LinearPriorCallable`` needs but the
        periodogram does not scan over -- in practice the ``parallax`` that
        :class:`~harv.models.priors.PeriodDependentSemiMajorAxisPrior` requires.
        Bound into the callable priors themselves, so the corresponding *column*
        (parallax included) stays in the design matrix and is still fitted and
        marginalized; only the prior's *scale* uses the supplied value. May not
        contain ``period`` or ``eccentricity``, which the scan owns.
    extensions
        Model extensions adding *linear* columns (e.g.
        :class:`~harv.models.MultiSurveyOffset`,
        :class:`~harv.models.MonomialTrend`), applied to both the trial and
        base models; their priors come from ``prior.extension_priors`` as
        usual. For containers, a mapping from dataset name to per-dataset
        extension tuples. Extensions with nonlinear parameters (jitter, GP)
        raise ``TypeError``.

    Examples
    --------
    RV, period-dependent semi-amplitude prior (recommended). ``sigma_K0`` is the
    semi-amplitude expected *at* ``P0`` for the companion being searched for, not a
    global width — see :class:`~harv.models.priors.PeriodDependentKPrior`:

    >>> from unxt import Q
    >>> import harv.models as hm
    >>> import harv.periodogram as hp
    >>> from harv.simulate import simulate_rv_sb1_data
    >>> data, _ = simulate_rv_sb1_data(seed=1, n_obs=40, period=Q(30.0, "day"))
    >>> rv_grid = dict(period_min=Q(5.0, "day"), period_max=Q(1000.0, "day"))
    >>> prior = hm.FourierRV(n_terms=2).default_prior(
    ...     **rv_grid,
    ...     sigma_K0=Q(1.0, "km/s"),
    ...     P0=Q(1.0, "yr"),
    ...     sigma_v0=Q(10.0, "km/s"),
    ... )
    >>> result = hp.periodogram(data, prior=prior, period_min=Q(5.0, "day"))
    >>> result.delta_ln_likelihood.shape == result.frequency.shape
    True

    RV, flat amplitude prior — swap ``sigma_K0``/``P0`` for a single ``sigma_amp``:

    >>> flat = hm.FourierRV(n_terms=2).default_prior(
    ...     **rv_grid, sigma_amp=Q(30.0, "km/s"), sigma_v0=Q(10.0, "km/s")
    ... )
    >>> flat_result = hp.periodogram(data, prior=flat, period_min=Q(5.0, "day"))
    >>> bool(flat_result.delta_ln_likelihood.max() > 0)
    True

    Gaia astrometry, flat amplitude prior. Here ``sigma_amp`` is an *angle*, since the
    Fourier amplitudes are angular:

    >>> from harv.simulate import simulate_gaia_epoch_astrometry
    >>> gaia, _ = simulate_gaia_epoch_astrometry(
    ...     seed=3, n_obs=80, period=Q(100.0, "day"),
    ...     semi_major_axis=Q(2.0, "mas"), parallax=Q(20.0, "mas"),
    ...     al_error=Q(0.05, "mas"),
    ... )
    >>> gaia_grid = dict(
    ...     period_min=Q(20.0, "day"), period_max=Q(2000.0, "day"),
    ...     sigma_pos=Q(500.0, "mas"), sigma_pm=Q(500.0, "mas/yr"),
    ...     sigma_parallax=Q(500.0, "mas"),
    ... )
    >>> gaia_flat = hm.FourierGaiaAstrometry(n_terms=2).default_prior(
    ...     **gaia_grid, sigma_amp=Q(20.0, "mas")
    ... )
    >>> res = hp.periodogram(gaia, prior=gaia_flat, period_min=Q(20.0, "day"))
    >>> res.delta_ln_likelihood.shape == res.frequency.shape
    True

    Gaia astrometry, period-dependent prior. ``sigma_a0`` is a physical *length*, so the
    prior needs a parallax to convert it to an angle — supply one via ``prior_params``:

    >>> gaia_tilted = hm.FourierGaiaAstrometry(n_terms=2).default_prior(
    ...     **gaia_grid, sigma_a0=Q(0.1, "AU"), P0=Q(1.0, "yr")
    ... )
    >>> res = hp.periodogram(
    ...     gaia, prior=gaia_tilted, period_min=Q(20.0, "day"),
    ...     prior_params={"parallax": Q(20.0, "mas")},
    ... )
    >>> res.delta_ln_likelihood.shape == res.frequency.shape
    True

    Profile mode, for comparison against a classical periodogram. Delta is
    ``0.5 * dchi2``, hence non-negative everywhere: the trial model nests the base
    one and there is no Occam factor to pay for the extra columns.

    >>> z0 = hp.periodogram(data, prior=False, period_min=Q(5.0, "day"))
    >>> z0.statistic
    'profile'
    >>> bool((z0.delta_ln_likelihood >= -1e-4).all())
    True

    Many sources at once. ``periodogram`` is safe under ``jax.jit`` and
    ``jax.vmap`` provided the frequency grid is *shape-fixed* -- an explicit
    ``frequency_grid``, or ``period_min``/``period_max``/``n_grid`` all given.
    A grid whose size is derived from each source's own baseline cannot be
    traced, since ``n_grid`` is then an output shape. Batching also requires
    one observation count across the stacked sources; differing counts retrace
    (see ``docs/spec.md``, "Batch inference over many datasets"):

    >>> import jax
    >>> import jax.numpy as jnp
    >>> sources = [
    ...     simulate_rv_sb1_data(seed=s, n_obs=40, period=Q(30.0, "day"))[0]
    ...     for s in range(3)
    ... ]
    >>> batched = jax.tree.map(lambda *xs: jnp.stack(xs), *sources)
    >>> grid = hp.frequency_grid(
    ...     t_span=Q(1000.0, "day"), period_min=Q(5.0, "day"), n_grid=128
    ... )
    >>> run = jax.jit(jax.vmap(lambda d: hp.periodogram(d, grid, prior=prior)))
    >>> run(batched).delta_ln_likelihood.shape
    (3, 128)
    """
    if prior is False and prior_params:
        raise TypeError(
            "prior_params cannot be used with prior=False: profile mode has no "
            "priors to resolve, so the values would be silently ignored."
        )
    # The annotation admits False per dataset only so this guard, rather than an
    # opaque type-check failure, is what reports the mistake.
    if isinstance(prior, Mapping) and any(v is False for v in prior.values()):
        raise TypeError(
            "prior=False cannot appear inside a per-dataset mapping: the profile "
            "statistic and the log Bayes factor are not commensurable, so summing "
            "them across datasets is meaningless. Pass prior=False for the whole "
            "periodogram instead."
        )
    reserved = {"period", "eccentricity"}.intersection(prior_params or ())
    if reserved:
        raise TypeError(
            f"prior_params may not contain {sorted(reserved)}: the periodogram "
            "supplies 'period' from the trial grid and adopts eccentricity = 0 "
            '(see docs/spec.md, "The Delta log-marginal-likelihood statistic").'
        )
    if n_terms < 1:
        raise ValueError(
            f"n_terms must be at least 1, got {n_terms}. A periodogram needs at "
            "least one harmonic of the trial frequency; with none, the trial "
            "model is the base model and every Delta would be zero."
        )
    if frequency_grid is not None:
        conflicting = period_min, period_max, samples_per_peak, n_grid
        if any(arg is not None for arg in conflicting):
            raise TypeError(
                "Cannot specify both an explicit frequency grid and "
                "period_min/period_max/samples_per_peak/n_grid"
            )
    else:
        if period_min is None:
            raise TypeError("Must specify either a frequency grid or period_min")
        # Forward samples_per_peak only when set, so frequency_grid owns its
        # default rather than this signature carrying a second copy of it.
        grid_kwargs: dict[str, Any] = {}
        if samples_per_peak is not None:
            grid_kwargs["samples_per_peak"] = samples_per_peak
        frequency_grid = get_frequency_grid(
            data,
            period_min=period_min,
            period_max=period_max,
            n_grid=n_grid,
            **grid_kwargs,
        )

    is_container = isinstance(data, AbstractDatasetContainer)
    datasets = dict(data.items()) if is_container else {"data": data}

    per_dataset: dict[str, NFloatArray] = {}
    base_lnls: list[Float[jax.Array, ""] | NFloatArray] = []
    eff_terms = 0
    for name, d in datasets.items():
        ds_prior = _resolve_per_dataset(prior, "prior", name)
        ds_ext = _resolve_per_dataset(extensions, "extensions", name)
        delta, lnl0, eff = _dataset_delta_lnl(
            d, ds_prior, tuple(ds_ext), frequency_grid, n_terms, prior_params
        )
        per_dataset[name] = delta
        base_lnls.append(lnl0)
        eff_terms = max(eff_terms, eff)

    total_delta = jnp.sum(jnp.stack(list(per_dataset.values())), axis=0)
    total_lnl0 = functools.reduce(jnp.add, base_lnls)

    time_unit = str((1.0 / frequency_grid[:1]).unit)

    # t_ref is always set by AbstractData.__check_init__ / the containers:
    t_ref = cast("ScalarQTime", data.t_ref)
    return PeriodogramResult(
        frequency=frequency_grid,
        delta_ln_likelihood=total_delta,
        ln_likelihood_base=total_lnl0,
        t_span=Q(_data_t_span(data, time_unit), time_unit),
        t_ref=t_ref,
        per_dataset=per_dataset if is_container else None,
        n_terms=eff_terms,
        statistic="profile" if prior is False else "marginal",
    )
