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
from typing import TYPE_CHECKING, Any, cast, final

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
    *effective* Fourier term count used (after the per-dataset overfitting
    cap; the maximum across datasets for container inputs).

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
    n_terms: int = eqx.field(static=True, default=2)

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
        ax.set_ylabel(r"$\Delta \ln \mathcal{L}$")
        return ax


def _effective_n_terms(
    fourier_cls: type,
    n_requested: int,
    n_obs: int,
    n_ext_linear: int,
) -> int:
    """Cap the Fourier term count so the trial model stays overdetermined.

    Requires at least two observations per linear column (columns counted from
    the parameterization itself plus linear extension columns), floored at 1.
    On sparse data an overfit trial model fits almost any trial period, so
    spurious alias peaks would dominate the periodogram.
    """
    n_base = len(fourier_cls(n_terms=0).linear_params()) + n_ext_linear
    n_per_term = len(fourier_cls(n_terms=1).linear_params()) - (n_base - n_ext_linear)
    h_max = int((n_obs / 2.0 - n_base) // n_per_term)
    eff = max(1, min(n_requested, h_max))
    if eff < n_requested:
        warnings.warn(
            f"n_terms={n_requested} overfits data with {n_obs} observations "
            f"(trial model would have {n_base + n_per_term * n_requested} linear "
            f"columns); reducing to n_terms={eff}. Spurious alias peaks would "
            "dominate the periodogram. Pass a smaller n_terms to silence this.",
            UserWarning,
            stacklevel=4,
        )
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


def _resolve_per_dataset(
    value: Any, name: str, ds_name: str, is_container: bool
) -> Any:
    """Resolve a per-dataset argument that may be a Mapping keyed by dataset name."""
    if isinstance(value, Mapping) and not isinstance(value, HarvPrior):
        try:
            return value[ds_name]
        except KeyError:
            raise TypeError(
                f"{name} mapping has no entry for dataset {ds_name!r}."
            ) from None
    if is_container and name == "prior":
        return value  # a single HarvPrior shared across (same-type) datasets
    return value


def _dataset_delta_lnl(
    dataset: AbstractData,
    prior: HarvPrior,
    extensions: tuple[AbstractExtension, ...],
    f_grid: NFrequency,
    n_terms: int,
) -> tuple[NFloatArray, Float[jax.Array, ""] | NFloatArray, int]:
    """Δ marginal log-likelihood over the grid for one dataset."""
    if type(dataset) not in _FOURIER_DISPATCH:
        raise NotImplementedError(
            f"No periodogram implementation for {type(dataset).__name__}; only "
            f"{', '.join(cls.__name__ for cls in _FOURIER_DISPATCH)} are "
            "currently supported."
        )
    fourier_cls, model_cls = _FOURIER_DISPATCH[type(dataset)]

    nonlin_extra = set(prior.nonlinear_priors) - {"period"}
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

    n_obs = int(dataset.time.shape[0])
    n_ext_linear = sum(1 for ext in extensions for p in ext.extra_params() if p.linear)
    eff_terms = _effective_n_terms(fourier_cls, n_terms, n_obs, n_ext_linear)

    # Catch typos / unusable prior entries against the *requested* term count,
    # then subset to the (possibly capped) effective model.
    requested_names = {p.name for p in fourier_cls(n_terms=n_terms).linear_params()}
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

    eff_lp = effective_linear_prior_from_prior(prior, model) or {}
    validate_extension_priors(prior, model, eff_lp)
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
            f"{fourier_cls.__name__}(n_terms={eff_terms})."
        )
    full_lp = {n: eff_lp[n] for n in model._all_linear_names()}
    base_lp = {n: eff_lp[n] for n in base_model._all_linear_names()}

    period_grid = 1.0 / f_grid
    period_unit = str(period_grid.unit)
    p_vals = jnp.asarray(ustrip(period_unit, period_grid))

    def base_at(p: jax.Array) -> jax.Array:
        return base_model.log_prob(_nl(p, period_unit), dataset, linear_priors=base_lp)

    # The base model carries no Fourier columns, so it is period-independent and
    # one evaluation suffices -- unless one of its own linear priors resolves
    # against the trial period (a LinearPriorCallable such as
    # PeriodDependentKPrior on v_sys). Then its baseline genuinely varies across
    # the grid and subtracting a single value would tilt every Delta.
    if any(_is_callable_prior(p) for p in base_lp.values()):
        lnl0 = jax.jit(jax.vmap(base_at))(p_vals)
    else:
        lnl0 = base_at(p_vals[0])

    def lnl_at(p: jax.Array) -> jax.Array:
        return model.log_prob(_nl(p, period_unit), dataset, linear_priors=full_lp)

    lnl = jax.jit(jax.vmap(lnl_at))(p_vals)
    return lnl - lnl0, lnl0, eff_terms


def periodogram(
    data: AbstractData | AbstractDatasetContainer,
    frequency_grid: NFrequency | None = None,
    *,
    prior: HarvPrior | Mapping[str, HarvPrior],
    period_min: ScalarQTime | None = None,
    period_max: ScalarQTime | None = None,
    samples_per_peak: int = 8,
    n_grid: int | None = None,
    n_terms: int = 2,
    extensions: tuple[AbstractExtension, ...]
    | Mapping[str, tuple[AbstractExtension, ...]] = (),
) -> PeriodogramResult:
    """Compute a Lomb-Scargle-like periodogram of the data.

    At each trial frequency this evaluates the marginal log-likelihood of a
    Kepler-free ``n_terms``-harmonic Fourier model (every amplitude linear and
    analytically marginalized under the supplied priors) minus that of the
    ``n_terms = 0`` base model. Multiple harmonics capture non-sinusoidal
    periodicity (e.g. eccentric orbits); the base model carries the
    non-periodic structure (constant offset for RV; the 5-parameter
    astrometric solution for Gaia, so scan-law/parallax/proper-motion power
    cancels). For containers the per-dataset Δ are summed into one
    periodogram per source.

    Parameters
    ----------
    data
        `~harv.data.RVData`, `~harv.data.GaiaAstrometryData`, or a dataset
        container holding them.
    frequency_grid
        Explicit frequency grid. Mutually exclusive with the grid keywords
        (``period_min``, ``period_max``, ``n_grid``).
    prior
        REQUIRED. A :class:`~harv.models.priors.HarvPrior` for the Fourier
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
        Must be at least 1. Default: 2. Automatically capped per dataset to
        keep the trial model overdetermined (at least two observations per
        linear column,
        including extension columns); a ``UserWarning`` is emitted when
        reduced. ``PeriodogramResult.n_terms`` reports the effective value.
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
    >>> from unxt import Q
    >>> import harv.models as hm
    >>> import harv.periodogram as hp
    >>> from harv.simulate import simulate_rv_sb1_data
    >>> data, _ = simulate_rv_sb1_data(seed=1, n_obs=40, period=Q(30.0, "day"))
    >>> prior = hm.FourierRV(n_terms=2).default_prior(
    ...     period_min=Q(5.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_amp=Q(30.0, "km/s"),
    ...     sigma_v0=Q(10.0, "km/s"),
    ... )
    >>> result = hp.periodogram(data, prior=prior, period_min=Q(5.0, "day"))
    >>> result.delta_ln_likelihood.shape == result.frequency.shape
    True
    """
    if n_terms < 1:
        raise ValueError(
            f"n_terms must be at least 1, got {n_terms}. A periodogram needs at "
            "least one harmonic of the trial frequency; with none, the trial "
            "model is the base model and every Delta would be zero."
        )
    if frequency_grid is not None:
        if period_min is not None or period_max is not None or n_grid is not None:
            raise TypeError(
                "Cannot specify both an explicit frequency grid and "
                "period_min/period_max/n_grid"
            )
    else:
        if period_min is None:
            raise TypeError("Must specify either a frequency grid or period_min")
        frequency_grid = get_frequency_grid(
            data,
            period_min=period_min,
            period_max=period_max,
            samples_per_peak=samples_per_peak,
            n_grid=n_grid,
        )

    is_container = isinstance(data, AbstractDatasetContainer)
    datasets = dict(data.items()) if is_container else {"data": data}

    per_dataset: dict[str, NFloatArray] = {}
    base_lnls: list[Float[jax.Array, ""] | NFloatArray] = []
    eff_terms = 0
    for name, d in datasets.items():
        ds_prior = _resolve_per_dataset(prior, "prior", name, is_container)
        ds_ext = _resolve_per_dataset(extensions, "extensions", name, is_container)
        delta, lnl0, eff = _dataset_delta_lnl(
            d, ds_prior, tuple(ds_ext), frequency_grid, n_terms
        )
        per_dataset[name] = delta
        base_lnls.append(lnl0)
        eff_terms = max(eff_terms, eff)

    total_delta = jnp.sum(jnp.stack(list(per_dataset.values())), axis=0)
    total_lnl0 = functools.reduce(jnp.add, base_lnls)

    time_unit = str((1.0 / frequency_grid[:1]).unit)
    all_times = [ustrip(time_unit, d.time) for d in datasets.values()]
    t_min = min(float(jnp.min(t)) for t in all_times)
    t_max = max(float(jnp.max(t)) for t in all_times)

    # t_ref is always set by AbstractData.__check_init__ / the containers:
    t_ref = cast("ScalarQTime", data.t_ref)
    return PeriodogramResult(
        frequency=frequency_grid,
        delta_ln_likelihood=total_delta,
        ln_likelihood_base=total_lnl0,
        t_span=Q(t_max - t_min, time_unit),
        t_ref=t_ref,
        per_dataset=per_dataset if is_container else None,
        n_terms=eff_terms,
    )
