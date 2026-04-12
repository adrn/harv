"""Data-type strategy descriptors for the rejection sampler.

Each concrete subclass encapsulates all branching logic for a specific data
type (RV-only, astrometry-only, combined).  The sampler itself is kept
branch-free by dispatching to the appropriate strategy instance.

``CompositeStrategy`` composes single-component strategies rather than
hardcoding parameter classes — adding a new data type requires only a new
single-component strategy.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal, final

import jax
import jax.random as jr
from equinox import AbstractVar
from unxt import Quantity

from harv.data import (
    AbstractData,
    GaiaAstrometryData,
    InputData,
    RVData,
    SourceData,
    SystemData,
    build_indicator_matrix,
    stack_datasets,
)
from harv.likelihood.composite import CompositeLikelihood
from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.params import (
    AbstractParameters,
    GaiaAstrometryParameters,
    MarginalizedParameters,
    RVParameters,
    SB2RVParameters,
)
from harv.likelihood.rv import RVLikelihood, SB2RVLikelihood
from harv.priors.rejection import RejectionPrior

DataType = Literal["astrometry", "rv", "combined"]


# ---------------------------------------------------------------------------
# Data-type strategy descriptors (private)
# ---------------------------------------------------------------------------


class DataTypeStrategy(ABC):
    """Per-data-type strategy encapsulating all branching logic.

    Each concrete subclass provides data extraction, likelihood construction,
    orbit param building, and linear parameter sampling for one data type.
    """

    data_type: AbstractVar[DataType]

    # Stateless strategies: equality/hashing by class identity so that
    # eqx.filter_jit can hash them as static arguments.
    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def required_prior_params(self, prior: RejectionPrior) -> tuple[str, ...]:
        """Parameter names that must appear in the batched values dict.

        Includes nonlinear params (always sampled from prior) plus any
        linear params that are *not* in ``prior.marginalize_names``
        and therefore sampled explicitly.
        """
        seen: set[str] = set()
        result: list[str] = []
        for cls in self.full_cls:
            for name in cls.nonlinear_param_names:
                if name not in seen:
                    seen.add(name)
                    result.append(name)

        # Explicitly-sampled linear params.
        if isinstance(prior.linear_prior, dict):
            marg_names = prior.marginalize_names
            if marg_names is None:
                # All linear params marginalized → nothing extra to sample.
                pass
            else:
                marg_set = set(marg_names)
                for name in prior.linear_prior:
                    if name not in marg_set and name not in seen:
                        seen.add(name)
                        result.append(name)

        return tuple(result)

    @property
    @abstractmethod
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        """The full parameter class(es) including linear params.

        Used to derive ``all_linear_names`` and for ``Samples`` serialization.
        """
        ...

    def extract_data(
        self,
        data: InputData,
    ) -> dict[str, AbstractData]:
        """Extract concrete data objects from the input.

        Returns a dict keyed by component name (e.g. ``{"rv": rv_data}`` or
        ``{"astro": astro_data, "rv": rv_data}``).
        """
        if isinstance(data, RVData):
            return {"rv": data}
        if isinstance(data, GaiaAstrometryData):
            return {"astro": data}

        if isinstance(data, SourceData):
            _data: dict[str, Any] = {}

            rv_datasets = data.get_datasets_by_type(RVData)
            if len(rv_datasets) == 1:
                _data["rv"] = next(iter(rv_datasets.values()))
            elif len(rv_datasets) > 1:
                # Pass through the raw dict; build_likelihood handles
                # stacking + indicator matrix construction.
                _data["rv"] = rv_datasets

            astro_datasets = data.get_datasets_by_type(GaiaAstrometryData)
            if len(astro_datasets) == 1:
                _data["astro"] = next(iter(astro_datasets.values()))
            elif len(astro_datasets) > 1:
                msg = "Multiple astrometry datasets not supported yet"
                raise NotImplementedError(msg)

            return _data

        msg = f"Expected AbstractData subclass or SourceData, got {type(data)}"
        raise TypeError(msg)

    @abstractmethod
    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        """Build the likelihood for batched evaluation."""
        ...

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        prior: RejectionPrior,
        time_unit: Any,
        lik: Any,
    ) -> dict[str, Quantity]:
        """Sample linear parameters for one accepted nonlinear sample."""
        params = self.build_marginalized_params(
            sample, time_unit, prior.marginalize_names, lik.linear_param_units
        )
        return lik.sample_conditional_linear(params, key)

    def build_marginalized_params(
        self,
        values: dict[str, Any],
        time_unit: str,
        marginalize_names: tuple[str, ...] | None = None,
        linear_param_units: dict[str, str] | None = None,
    ) -> Any:
        """Build ``MarginalizedParameters`` from sampled values.

        ``values`` is a dict of raw scalars keyed by parameter name
        (period already converted to ``time_unit``).  May contain both
        nonlinear and explicitly-sampled linear parameter values.

        When ``marginalize_names`` is ``None`` (default), all linear
        parameters are analytically marginalized.

        Returns the form expected by ``lik.log_prob``: a single
        ``MarginalizedParameters`` for single-component strategies, or a
        ``dict[str, MarginalizedParameters]`` for :class:`CompositeStrategy`.
        """
        cls = self.full_cls[0]
        kw: dict[str, Any] = {name: values[name] for name in cls.nonlinear_param_names}
        # TODO: I don't understand why this is necessary! Shouldn't kw["period"] already
        # be a Quantity?
        kw["period"] = Quantity(kw["period"], time_unit)

        # Determine which linear params to marginalize.
        marg = (
            marginalize_names
            if marginalize_names is not None
            else cls.linear_param_names
        )

        # Include explicit linear values (those not being marginalized).
        units = linear_param_units or {}
        for name in cls.linear_param_names:
            if name not in marg and name in values:
                u = units.get(name, "")
                kw[name] = Quantity(values[name], u) if u else values[name]

        # cls.marginalized() with no positional args defaults to marginalizing
        # all, so guard the empty-tuple case explicitly.
        if marg:
            return cls.marginalized(*marg, **kw)
        return MarginalizedParameters(values=kw, marginalized_names=(), source_cls=cls)

    def build_params_with_fixed_linear(
        self,
        values: dict[str, Any],
        fixed_linear: dict[str, Any],
        linear_units: dict[str, str],
        time_unit: str,
    ) -> Any:
        """Build ``MarginalizedParameters`` with some linear params provided.

        Like ``build_marginalized_params``, but some linear parameters are
        given explicit values (e.g. computed by an ``extra_model``).  The
        provided values are wrapped in ``Quantity`` with the correct unit and
        stored as non-marginalized fields; the remaining linear parameters
        are still analytically marginalized.

        Parameters
        ----------
        values
            Raw nonlinear scalars (period already in ``time_unit``).
        fixed_linear
            Raw scalar values for the linear parameters to fix, keyed by
            parameter name (e.g. ``{"rv_semiamp": 5.0}``).
        linear_units
            Unit string for each linear parameter (from the likelihood's
            ``linear_param_units``).
        time_unit
            Unit string of the data's time axis.
        """
        base = self.build_marginalized_params(values, time_unit)
        cls = base.source_cls
        _lin = cls.linear_param_names
        free = tuple(n for n in _lin if n not in fixed_linear)

        kw = dict(base.values)
        for name in _lin:
            if name in fixed_linear:
                kw[name] = Quantity(fixed_linear[name], linear_units[name])

        if free:
            return cls.marginalized(*free, **kw)
        return MarginalizedParameters(values=kw, marginalized_names=(), source_cls=cls)

    def all_linear_names(
        self,
        prior: RejectionPrior,
        data: InputData,
    ) -> tuple[str, ...]:
        """All linear parameter names including trends and multi-survey offsets."""
        names: tuple[str, ...] = sum(
            (cls.linear_param_names for cls in self.full_cls),
            (),
        )
        # Trend columns come after base linear params, before instrument offsets.
        names = names + self._trend_column_names(prior)

        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data._n_rv() > 1
        ):
            names = names + tuple(k for k, v in prior.offsets.items() if v is not None)
        return names

    def _trend_column_names(self, prior: RejectionPrior) -> tuple[str, ...]:
        """Trend column names implied by the prior's trend_priors."""
        if prior.trend_priors is None:
            return ()
        return tuple(prior.trend_priors.keys())


@final
class RVStrategy(DataTypeStrategy):
    data_type = "rv"

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        return (RVParameters,)

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        rv_raw = datasets["rv"]
        rv_offsets = prior.offsets.get("rv") if prior.offsets is not None else None

        indicator = None
        instrument_names = None
        if isinstance(rv_raw, dict) and rv_offsets is not None:
            # Multi-survey: extract_data passed through the raw per-instrument
            # dict.  Stack + build indicator matrix in one step.
            reference = next(name for name, v in rv_offsets.items() if v is None)
            rv_data, indicator, instrument_names = build_indicator_matrix(
                rv_raw, reference=reference
            )
        elif isinstance(rv_raw, dict):
            # Multi-survey data but no offsets configured — just stack.
            rv_data = stack_datasets(rv_raw)
        else:
            rv_data = rv_raw

        linear_prior = None
        if isinstance(prior.linear_prior, dict):
            linear_prior = {
                name: prior.linear_prior[name]
                for name in RVParameters.linear_param_names
                if name in prior.linear_prior
            }

        offsets_prior = None
        if rv_offsets is not None:
            offsets_prior = {name: v for name, v in rv_offsets.items() if v is not None}

        return RVLikelihood(
            data=rv_data,
            linear_marginalized_prior=linear_prior or None,
            offsets_marginalized_prior=offsets_prior,
            trend_marginalized_prior=(
                dict(prior.trend_priors) if prior.trend_priors is not None else None
            ),
            trend_order=prior.trend_order,
            indicator_matrix=indicator,
            instrument_names=instrument_names,
        )


@final
class AstrometryStrategy(DataTypeStrategy):
    data_type = "astrometry"

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        return (GaiaAstrometryParameters,)

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        linear_prior = None
        if isinstance(prior.linear_prior, dict):
            linear_prior = {
                name: prior.linear_prior[name]
                for name in GaiaAstrometryParameters.linear_param_names
                if name in prior.linear_prior
            }

        return GaiaAstrometryLikelihood(
            data=datasets["astro"],
            linear_marginalized_prior=linear_prior or None,
            trend_marginalized_prior=(
                dict(prior.trend_priors) if prior.trend_priors is not None else None
            ),
            trend_order=prior.trend_order,
        )


@final
class CompositeStrategy(DataTypeStrategy):
    """Compose single-component strategies to handle multiple data types.

    Rather than hardcoding parameter classes, this strategy delegates to its
    children for param construction, likelihood building, and linear sampling.
    Adding a new data type requires only a new single-component strategy.
    """

    data_type = "combined"

    def __init__(self, **sub_strategies: DataTypeStrategy) -> None:
        object.__setattr__(self, "_sub_strategies", sub_strategies)

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        """Ordered tuple of full parameter classes from all sub-strategies."""
        return sum(
            (sub.full_cls for sub in self._sub_strategies.values()),
            (),
        )

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,
    ) -> CompositeLikelihood:
        """Build a CompositeLikelihood by delegating to each sub-strategy."""
        # Guard: combined astrometry + multi-survey RV is not yet fully
        # validated.  See docs/spec.md §'Combined astrometry + multi-survey
        # RV'.
        rv_data = datasets.get("rv")
        if "astro" in datasets and isinstance(rv_data, dict) and len(rv_data) > 1:
            msg = (
                "Combined astrometry + multi-survey RV is not yet implemented. "
                "See docs/spec.md §'Combined astrometry + multi-survey RV'."
            )
            raise NotImplementedError(msg)

        components: dict[str, Any] = {}
        for name, sub in self._sub_strategies.items():
            # Each sub-strategy gets only the datasets it needs.
            sub_datasets = {name: datasets[name]}
            components[name] = sub.build_likelihood(sub_datasets, prior, data)
        return CompositeLikelihood(**components)

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        prior: RejectionPrior,
        time_unit: Any,
        lik: Any,
    ) -> dict[str, Quantity]:
        keys = jr.split(key, len(self._sub_strategies))
        result: dict[str, Quantity] = {}
        for (name, sub), k in zip(self._sub_strategies.items(), keys, strict=True):
            sub_lik = lik[name]
            sub_sample = sub.sample_linear_one(k, sample, prior, time_unit, sub_lik)
            result.update(sub_sample)
        return result

    def build_marginalized_params(
        self,
        values: dict[str, Any],
        time_unit: str,
        marginalize_names: tuple[str, ...] | None = None,
        linear_param_units: dict[str, str] | None = None,
    ) -> dict[str, MarginalizedParameters]:
        return {
            name: sub.build_marginalized_params(
                values, time_unit, marginalize_names, linear_param_units
            )
            for name, sub in self._sub_strategies.items()
        }

    def build_params_with_fixed_linear(
        self,
        values: dict[str, Any],
        fixed_linear: dict[str, Any],
        linear_units: dict[str, str],
        time_unit: str,
    ) -> dict[str, MarginalizedParameters]:
        return {
            name: sub.build_params_with_fixed_linear(
                values, fixed_linear, linear_units, time_unit
            )
            for name, sub in self._sub_strategies.items()
        }


@final
class SB2Strategy(DataTypeStrategy):
    """Strategy for double-lined spectroscopic binary (SB2) RV data."""

    data_type = "sb2"

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        return (SB2RVParameters,)

    def extract_data(
        self,
        data: InputData,
    ) -> dict[str, AbstractData]:
        if isinstance(data, SystemData):
            return {"sb2": data}
        msg = f"SB2Strategy expects SystemData, got {type(data)}"
        raise TypeError(msg)

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        sb2_data = datasets["sb2"]

        linear_prior = None
        if isinstance(prior.linear_prior, dict):
            linear_prior = {
                name: prior.linear_prior[name]
                for name in SB2RVParameters.linear_param_names
                if name in prior.linear_prior
            }

        return SB2RVLikelihood(
            data=sb2_data,
            linear_marginalized_prior=linear_prior or None,
            trend_marginalized_prior=(
                dict(prior.trend_priors) if prior.trend_priors is not None else None
            ),
            trend_order=prior.trend_order,
        )


# SB2 strategy placeholder — requires SystemData (not yet implemented).
# See spec §Planned: SystemData for details.
# class SB2Strategy(DataTypeStrategy): ...

_STRATEGIES: dict[str, DataTypeStrategy] = {
    "rv": RVStrategy(),
    "astrometry": AstrometryStrategy(),
    "sb2": SB2Strategy(),
    "combined": CompositeStrategy(
        astro=AstrometryStrategy(),
        rv=RVStrategy(),
    ),
}
