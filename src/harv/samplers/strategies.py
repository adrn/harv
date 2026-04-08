"""Data-type strategy descriptors for the rejection sampler.

Each concrete subclass encapsulates all branching logic for a specific data
type (RV-only, astrometry-only, combined).  The sampler itself is kept
branch-free by dispatching to the appropriate strategy instance.
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from unxt import Quantity, ustrip

from harv.data import (
    GaiaAstrometryData,
    InputData,
    RadialVelocityData,
    SourceData,
)
from harv.likelihood.combined import CompositeLikelihood
from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.helpers import _IndexedCallable, _sub_mvn
from harv.likelihood.params import (
    AbstractParameters,
    GaiaAstrometryParameters,
    MarginalizedParameters,
    RVParameters,
)
from harv.likelihood.rv import (
    RVLikelihood,
    build_rv_indicator_matrix,
    stack_rv_datasets,
)
from harv.priors.rejection import RejectionPrior

DataType = Literal["astrometry", "rv", "combined"]


# ---------------------------------------------------------------------------
# Component slice — metadata for one likelihood component in the joint vector
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ComponentSlice:
    """One component's metadata within the joint linear parameter vector.

    Produced by ``_DataTypeStrategy.build_component_slices`` and consumed by
    the numpyro model builders in ``_numpyro.py``.  ``global_col_indices``
    replaces the old hardcoded ``n_astro`` boundary: for combined data the
    astro component might be ``(0, 1, 2, 3, 4, 5)`` and the RV component
    ``(6, 7)``, with no coupling between them.
    """

    name: str
    """Human-readable label (e.g. ``"astro"``, ``"rv"``)."""

    lik: Any
    """The per-component likelihood object (has ``design_matrix``)."""

    global_col_indices: tuple[int, ...]
    """Indices into the joint linear parameter vector."""

    obs: jax.Array
    """Unit-stripped observed data vector."""

    err: jax.Array
    """Unit-stripped uncertainties vector."""


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data-type strategy descriptors (private)
# ---------------------------------------------------------------------------


class _DataTypeStrategy(ABC):
    """Per-data-type strategy encapsulating all branching logic.

    Each concrete subclass provides data extraction, likelihood construction,
    orbit param building, and linear parameter sampling for one data type.
    Required prior params are derived from the full param class fields minus
    the linear parameter names.
    """

    # Stateless strategies: equality/hashing by class identity so that
    # eqx.filter_jit can hash them as static arguments.
    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    @property
    @abstractmethod
    def nonlinear_cls(self) -> type[AbstractParameters]:
        """The full parameter class used to create marginalized params.

        For single-data-type strategies this is the single full class.
        For combined strategies it is the full class with the superset of
        nonlinear fields (``GaiaAstrometryParameters``).
        """
        ...

    @property
    @abstractmethod
    def full_cls(self) -> tuple[type[AbstractParameters], ...]: ...

    @property
    @abstractmethod
    def data_type(self) -> DataType: ...

    @property
    def required_prior_params(self) -> tuple[str, ...]:
        """Prior parameter names (nonlinear fields of ``nonlinear_cls``)."""
        linear = set(self.nonlinear_cls.linear_param_names)
        return tuple(
            f.name
            for f in dataclasses.fields(self.nonlinear_cls)
            if f.name not in linear and f.default is dataclasses.MISSING
        )

    @property
    def n_linear(self) -> int:
        """Total linear parameters, summed from ``full_cls.linear_param_names``."""
        return sum(len(cls.linear_param_names) for cls in self.full_cls)

    @abstractmethod
    def extract_data(
        self,
        data: InputData,
    ) -> dict[str, Any]:
        """Extract concrete data objects from the input.

        Returns a dict keyed by component name (e.g. ``{"rv": rv_data}`` or
        ``{"astro": astro_data, "rv": rv_data}``).
        """
        ...

    @abstractmethod
    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        """Build the likelihood for batched evaluation."""
        ...

    @abstractmethod
    def build_component_slices(
        self,
        lik: Any,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,
    ) -> tuple[_ComponentSlice, ...]:
        """Build component metadata for the numpyro model builders."""
        ...

    @abstractmethod
    def linear_param_units(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
    ) -> tuple[str, ...]:
        """Derive linear parameter unit strings from the data."""
        ...

    @abstractmethod
    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        datasets: dict[str, Any],
        prior: RejectionPrior,
        time_unit: Any,
        data: InputData,
        lik: Any,
    ) -> dict[str, Quantity]:
        """Sample linear parameters for one accepted nonlinear sample."""
        ...

    def all_linear_names(
        self,
        prior: RejectionPrior,
        data: InputData,
    ) -> tuple[str, ...]:
        """All linear parameter names including multi-survey offsets."""
        names: tuple[str, ...] = sum(
            (cls.linear_param_names for cls in self.full_cls),
            (),
        )
        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data.n_rv() > 1
        ):
            names = names + tuple(k for k, v in prior.offsets.items() if v is not None)
        return names

    @abstractmethod
    def build_orbit_params(
        self,
        period: jax.Array,
        ecc: jax.Array,
        phase: jax.Array,
        arg_peri: jax.Array,
        cos_i: jax.Array,
        lon_asc: jax.Array,
        time_unit: Any,
    ) -> eqx.Module:
        """Build orbit param struct from batch-sliced scalar arrays.

        Called inside ``fori_loop``.  The strategy is closed over as a static
        value so this branch is resolved at trace time.
        """
        ...


@final
class _RVStrategy(_DataTypeStrategy):
    @property
    def data_type(self) -> DataType:
        return "rv"

    @property
    def nonlinear_cls(self) -> type[AbstractParameters]:
        return RVParameters

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        return (RVParameters,)

    def extract_data(
        self,
        data: InputData,
    ) -> dict[str, RadialVelocityData]:
        if isinstance(data, RadialVelocityData):
            return {"rv": data}
        if isinstance(data, SourceData):
            rv_datasets = data.get_datasets_by_type(RadialVelocityData)
            if len(rv_datasets) == 1:
                return {"rv": next(iter(rv_datasets.values()))}
            return {"rv": stack_rv_datasets(rv_datasets)}
        msg = f"Expected RadialVelocityData or SourceData, got {type(data)}"
        raise TypeError(msg)

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        rv_data = datasets["rv"]
        indicator = None
        instrument_names = None
        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data.n_rv() > 1
        ):
            rv_datasets = data.get_datasets_by_type(RadialVelocityData)
            non_ref = [k for k, v in prior.offsets.items() if v is not None]
            ref = next(k for k in rv_datasets if k not in non_ref)
            indicator, instrument_names = build_rv_indicator_matrix(
                rv_datasets, reference=ref
            )
        return RVLikelihood(
            data=rv_data,
            linear_prior=prior.linear_prior,
            indicator_matrix=indicator,
            instrument_names=instrument_names,
        )

    def build_component_slices(
        self,
        lik: Any,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,
    ) -> tuple[_ComponentSlice, ...]:
        rv_data = datasets["rv"]
        obs = ustrip(str(rv_data.rv.unit), rv_data.rv)
        err = ustrip(str(rv_data.rv.unit), rv_data.rv_err)
        n_base = len(RVParameters.linear_param_names)
        n_offsets = 0
        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data.n_rv() > 1
        ):
            n_offsets = sum(1 for v in prior.offsets.values() if v is not None)
        return (_ComponentSlice("rv", lik, tuple(range(n_base + n_offsets)), obs, err),)

    def linear_param_units(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
    ) -> tuple[str, ...]:
        rv_data = datasets["rv"]
        rv_unit = str(rv_data.rv.unit)
        n_offsets = sum(1 for v in (prior.offsets or {}).values() if v is not None)
        return (rv_unit,) * (self.n_linear + n_offsets)

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        datasets: dict[str, Any],  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
        time_unit: Any,
        data: InputData,  # noqa: ARG002
        lik: Any,
    ) -> dict[str, Quantity]:
        params = RVParameters.marginalized(
            period=Quantity(sample["period"], time_unit),
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            arg_peri=sample["arg_peri"],
        )
        return lik.sample_conditional_linear(params, key)

    def build_orbit_params(
        self,
        period: jax.Array,
        ecc: jax.Array,
        phase: jax.Array,
        arg_peri: jax.Array,
        cos_i: jax.Array,  # noqa: ARG002
        lon_asc: jax.Array,  # noqa: ARG002
        time_unit: Any,
    ) -> MarginalizedParameters:
        return RVParameters.marginalized(
            period=Quantity(period, time_unit),
            eccentricity=ecc,
            phase_peri=phase,
            arg_peri=arg_peri,
        )


@final
class _AstrometryStrategy(_DataTypeStrategy):
    @property
    def data_type(self) -> DataType:
        return "astrometry"

    @property
    def nonlinear_cls(self) -> type[AbstractParameters]:
        return GaiaAstrometryParameters

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        return (GaiaAstrometryParameters,)

    def extract_data(
        self,
        data: InputData,
    ) -> dict[str, GaiaAstrometryData]:
        if isinstance(data, GaiaAstrometryData):
            return {"astro": data}
        if isinstance(data, SourceData):
            astro = next(iter(data.get_datasets_by_type(GaiaAstrometryData).values()))
            return {"astro": astro}
        msg = f"Expected GaiaAstrometryData or SourceData, got {type(data)}"
        raise TypeError(msg)

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        return GaiaAstrometryLikelihood(
            data=datasets["astro"], linear_prior=prior.linear_prior
        )

    def build_component_slices(
        self,
        lik: Any,
        datasets: dict[str, Any],
        prior: RejectionPrior,  # noqa: ARG002
        data: InputData,  # noqa: ARG002
    ) -> tuple[_ComponentSlice, ...]:
        astro_data = datasets["astro"]
        obs = ustrip(str(astro_data.al_position.unit), astro_data.al_position)
        err = ustrip(str(astro_data.al_position.unit), astro_data.al_position_err)
        n = len(GaiaAstrometryParameters.linear_param_names)
        return (_ComponentSlice("astro", lik, tuple(range(n)), obs, err),)

    def linear_param_units(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,  # noqa: ARG002
    ) -> tuple[str, ...]:
        astro_data = datasets["astro"]
        pos_unit = str(astro_data.al_position.unit)
        pm_unit = f"{pos_unit}/yr"
        return (pos_unit, pos_unit, pm_unit, pm_unit, pos_unit, pos_unit)

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        datasets: dict[str, Any],  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
        time_unit: Any,
        data: InputData,  # noqa: ARG002
        lik: Any,
    ) -> dict[str, Quantity]:
        params = GaiaAstrometryParameters.marginalized(
            period=Quantity(sample["period"], time_unit),
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            cos_i=sample["cos_i"],
            arg_peri=sample["arg_peri"],
            lon_asc_node=sample["lon_asc_node"],
        )
        return lik.sample_conditional_linear(params, key)

    def build_orbit_params(
        self,
        period: jax.Array,
        ecc: jax.Array,
        phase: jax.Array,
        arg_peri: jax.Array,
        cos_i: jax.Array,
        lon_asc: jax.Array,
        time_unit: Any,
    ) -> MarginalizedParameters:
        return GaiaAstrometryParameters.marginalized(
            period=Quantity(period, time_unit),
            eccentricity=ecc,
            phase_peri=phase,
            arg_peri=arg_peri,
            cos_i=cos_i,
            lon_asc_node=lon_asc,
        )


@final
class _CombinedStrategy(_DataTypeStrategy):
    @property
    def data_type(self) -> DataType:
        return "combined"

    @property
    def nonlinear_cls(self) -> type[AbstractParameters]:
        return GaiaAstrometryParameters

    @property
    def full_cls(self) -> tuple[type[AbstractParameters], ...]:
        return (GaiaAstrometryParameters, RVParameters)

    def extract_data(
        self,
        data: InputData,
    ) -> dict[str, Any]:
        if not isinstance(data, SourceData):
            msg = "Combined data type requires SourceData"
            raise TypeError(msg)
        rv_datasets = data.get_datasets_by_type(RadialVelocityData)
        if len(rv_datasets) > 1:
            msg = (
                "Combined astrometry + multi-survey RV (with per-instrument offsets) "
                "is not yet implemented. SourceData contains multiple "
                f"RadialVelocityData datasets ({list(rv_datasets.keys())}), but "
                "_CombinedStrategy only supports a single RV dataset alongside "
                "astrometry. See docs/spec.md §'Combined astrometry + multi-survey RV' "
                "for the planned design."
            )
            raise NotImplementedError(msg)
        astro = next(iter(data.get_datasets_by_type(GaiaAstrometryData).values()))
        rv = next(iter(rv_datasets.values()))
        return {"astro": astro, "rv": rv}

    def build_likelihood(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        astro_data = datasets["astro"]
        rv_data = datasets["rv"]
        n = len(GaiaAstrometryParameters.linear_param_names)  # 6
        astro_idx = tuple(range(n))
        rv_idx = tuple(range(n, n + 2))  # K, v0
        lp = prior.linear_prior
        if isinstance(lp, dist.MultivariateNormal):
            astro_lp = _sub_mvn(lp, astro_idx)
            rv_lp = _sub_mvn(lp, rv_idx)
        else:
            astro_lp = _IndexedCallable(lp, astro_idx)
            rv_lp = _IndexedCallable(lp, rv_idx)
        return CompositeLikelihood(
            astro=GaiaAstrometryLikelihood(astro_data, astro_lp),
            rv=RVLikelihood(rv_data, rv_lp),
        )

    def build_component_slices(
        self,
        lik: Any,
        datasets: dict[str, Any],
        prior: RejectionPrior,  # noqa: ARG002
        data: InputData,  # noqa: ARG002
    ) -> tuple[_ComponentSlice, ...]:
        astro_data = datasets["astro"]
        rv_data = datasets["rv"]
        n_astro = len(GaiaAstrometryParameters.linear_param_names)
        n_rv = len(RVParameters.linear_param_names)
        astro_obs = ustrip(str(astro_data.al_position.unit), astro_data.al_position)
        astro_err = ustrip(str(astro_data.al_position.unit), astro_data.al_position_err)
        rv_obs = ustrip(str(rv_data.rv.unit), rv_data.rv)
        rv_err = ustrip(str(rv_data.rv.unit), rv_data.rv_err)
        return (
            _ComponentSlice(
                "astro", lik["astro"], tuple(range(n_astro)), astro_obs, astro_err
            ),
            _ComponentSlice(
                "rv",
                lik["rv"],
                tuple(range(n_astro, n_astro + n_rv)),
                rv_obs,
                rv_err,
            ),
        )

    def linear_param_units(
        self,
        datasets: dict[str, Any],
        prior: RejectionPrior,  # noqa: ARG002
    ) -> tuple[str, ...]:
        astro_data = datasets["astro"]
        rv_data = datasets["rv"]
        pos_unit = str(astro_data.al_position.unit)
        pm_unit = f"{pos_unit}/yr"
        rv_unit = str(rv_data.rv.unit)
        return (
            pos_unit,
            pos_unit,
            pm_unit,
            pm_unit,
            pos_unit,
            pos_unit,
            rv_unit,
            rv_unit,
        )

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        datasets: dict[str, Any],  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
        time_unit: Any,
        data: InputData,  # noqa: ARG002
        lik: Any,
    ) -> dict[str, Quantity]:
        k_astro, k_rv = jr.split(key)
        params = GaiaAstrometryParameters.marginalized(
            period=Quantity(sample["period"], time_unit),
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            cos_i=sample["cos_i"],
            arg_peri=sample["arg_peri"],
            lon_asc_node=sample["lon_asc_node"],
        )
        astro_sample = lik["astro"].sample_conditional_linear(params, k_astro)
        rv_sample = lik["rv"].sample_conditional_linear(params, k_rv)
        return {**astro_sample, **rv_sample}

    def build_orbit_params(
        self,
        period: jax.Array,
        ecc: jax.Array,
        phase: jax.Array,
        arg_peri: jax.Array,
        cos_i: jax.Array,
        lon_asc: jax.Array,
        time_unit: Any,
    ) -> MarginalizedParameters:
        return GaiaAstrometryParameters.marginalized(
            period=Quantity(period, time_unit),
            eccentricity=ecc,
            phase_peri=phase,
            arg_peri=arg_peri,
            cos_i=cos_i,
            lon_asc_node=lon_asc,
        )


# SB2 strategy placeholder — requires SystemData (not yet implemented).
# See spec §Planned: SystemData for details.
# class _SB2Strategy(_DataTypeStrategy): ...

_STRATEGIES: dict[str, _DataTypeStrategy] = {
    "rv": _RVStrategy(),
    "astrometry": _AstrometryStrategy(),
    "combined": _CombinedStrategy(),
}
