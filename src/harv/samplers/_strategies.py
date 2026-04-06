"""Data-type strategy descriptors for the rejection sampler.

Each concrete subclass encapsulates all branching logic for a specific data
type (RV-only, astrometry-only, combined).  The sampler itself is kept
branch-free by dispatching to the appropriate strategy instance.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, final

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
from harv.likelihood._params import (
    GaiaAstrometryParameters,
    MarginalizedParameters,
    RVParameters,
)
from harv.likelihood.combined import CompositeLikelihood
from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.helpers import _IndexedCallable, _sub_mvn
from harv.likelihood.rv import RVLikelihood

if TYPE_CHECKING:
    from harv.priors.rejection import RejectionPrior

DataType = Literal["astrometry", "rv", "combined"]


# ---------------------------------------------------------------------------
# Multi-survey RV helpers (private)
# ---------------------------------------------------------------------------


def _stack_rv_datasets(
    rv_datasets: dict[str, RadialVelocityData],
) -> RadialVelocityData:
    """Concatenate multiple RV datasets in dict order into a single one."""
    ref = next(iter(rv_datasets.values()))
    time_unit = str(ref.time.unit)
    rv_unit = str(ref.rv.unit)

    all_time = jnp.concatenate(
        [ustrip(time_unit, ds.time) for ds in rv_datasets.values()]
    )
    all_rv = jnp.concatenate([ustrip(rv_unit, ds.rv) for ds in rv_datasets.values()])
    all_err = jnp.concatenate(
        [ustrip(rv_unit, ds.rv_err) for ds in rv_datasets.values()]
    )

    return RadialVelocityData(
        time=Quantity(all_time, time_unit),
        rv=Quantity(all_rv, rv_unit),
        rv_err=Quantity(all_err, rv_unit),
    )


def _build_indicator_matrix(
    rv_datasets: dict[str, RadialVelocityData],
    offsets: dict[str, Any],
) -> jax.Array:
    """Build indicator matrix (n_obs_total, n_non_ref) for multi-survey RV.

    ``indicator[i, j] == 1`` when observation *i* belongs to non-reference
    instrument *j* (i.e. the j-th non-None entry in ``offsets``).
    Datasets are iterated in ``rv_datasets`` dict order, which must match the
    order used by ``_stack_rv_datasets``.
    """
    non_ref_names = [k for k, v in offsets.items() if v is not None]
    n_non_ref = len(non_ref_names)
    rows: list[jax.Array] = []
    for name, ds in rv_datasets.items():
        n_obs = len(ds.time)
        row = jnp.zeros((n_obs, n_non_ref))
        if name in non_ref_names:
            j = non_ref_names.index(name)
            row = row.at[:, j].set(1.0)
        rows.append(row)
    return jnp.concatenate(rows, axis=0)


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
    def nonlinear_cls(self) -> type:
        """The full parameter class used to create marginalized params.

        For single-data-type strategies this is the single full class.
        For combined strategies it is the full class with the superset of
        nonlinear fields (``GaiaAstrometryParameters``).
        """
        ...

    @property
    @abstractmethod
    def full_cls(self) -> tuple[type, ...]: ...

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
            if f.name not in linear
        )

    @property
    def n_linear(self) -> int:
        """Total linear parameters, summed from ``full_cls.linear_param_names``."""
        return sum(
            len(cls.linear_param_names)  # type: ignore[attr-defined]
            for cls in self.full_cls
        )

    @abstractmethod
    def extract_data(
        self,
        data: InputData,
    ) -> tuple[GaiaAstrometryData | None, RadialVelocityData | None]:
        """Extract concrete data objects from the input."""
        ...

    @abstractmethod
    def build_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        """Build the likelihood for batched evaluation."""
        ...

    @abstractmethod
    def linear_param_units(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
    ) -> tuple[str, ...]:
        """Derive linear parameter unit strings from the data."""
        ...

    @abstractmethod
    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        time_unit: Any,
        data: InputData,
        lik: Any,
    ) -> jax.Array:
        """Sample linear parameters for one accepted nonlinear sample."""
        ...

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
    def nonlinear_cls(self) -> type:
        return RVParameters

    @property
    def full_cls(self) -> tuple[type, ...]:
        return (RVParameters,)

    def extract_data(
        self,
        data: InputData,
    ) -> tuple[None, RadialVelocityData]:
        if isinstance(data, RadialVelocityData):
            return None, data
        if isinstance(data, SourceData):
            rv_datasets = data.get_datasets_by_type(RadialVelocityData)
            if len(rv_datasets) == 1:
                return None, next(iter(rv_datasets.values()))
            # Multi-survey: stack all datasets in dict order.
            return None, _stack_rv_datasets(rv_datasets)
        msg = f"Expected RadialVelocityData or SourceData, got {type(data)}"
        raise TypeError(msg)

    def build_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,  # noqa: ARG002
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        if rv_data is None:
            msg = "_RVStrategy requires rv_data"
            raise TypeError(msg)
        indicator = None
        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data.n_rv() > 1
        ):
            indicator = _build_indicator_matrix(
                data.get_datasets_by_type(RadialVelocityData), prior.offsets
            )
        return RVLikelihood(
            data=rv_data,
            linear_prior=prior.linear_prior,
            indicator_matrix=indicator,
        )

    def linear_param_units(
        self,
        astro_data: GaiaAstrometryData | None,  # noqa: ARG002
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
    ) -> tuple[str, ...]:
        if rv_data is None:
            msg = "_RVStrategy requires rv_data"
            raise TypeError(msg)
        rv_unit = str(rv_data.rv.unit)
        n_offsets = sum(1 for v in (prior.offsets or {}).values() if v is not None)
        return (rv_unit,) * (self.n_linear + n_offsets)

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        astro_data: GaiaAstrometryData | None,  # noqa: ARG002
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
        time_unit: Any,
        data: InputData,  # noqa: ARG002
        lik: Any,
    ) -> jax.Array:
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
    def nonlinear_cls(self) -> type:
        return GaiaAstrometryParameters

    @property
    def full_cls(self) -> tuple[type, ...]:
        return (GaiaAstrometryParameters,)

    def extract_data(
        self,
        data: InputData,
    ) -> tuple[GaiaAstrometryData, None]:
        if isinstance(data, GaiaAstrometryData):
            return data, None
        if isinstance(data, SourceData):
            astro = next(iter(data.get_datasets_by_type(GaiaAstrometryData).values()))
            return astro, None
        msg = f"Expected GaiaAstrometryData or SourceData, got {type(data)}"
        raise TypeError(msg)

    def build_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        if astro_data is None:
            msg = "_AstrometryStrategy requires astro_data"
            raise TypeError(msg)
        return GaiaAstrometryLikelihood(
            data=astro_data, linear_prior=prior.linear_prior
        )

    def linear_param_units(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
    ) -> tuple[str, ...]:
        if astro_data is None:
            msg = "_AstrometryStrategy requires astro_data"
            raise TypeError(msg)
        pos_unit = str(astro_data.al_position.unit)
        pm_unit = f"{pos_unit}/yr"
        return (pos_unit, pos_unit, pm_unit, pm_unit, pos_unit, pos_unit)

    def sample_linear_one(
        self,
        key: jax.Array,
        sample: dict[str, jax.Array],
        astro_data: GaiaAstrometryData | None,  # noqa: ARG002
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
        time_unit: Any,
        data: InputData,  # noqa: ARG002
        lik: Any,
    ) -> jax.Array:
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
    def nonlinear_cls(self) -> type:
        return GaiaAstrometryParameters

    @property
    def full_cls(self) -> tuple[type, ...]:
        return (GaiaAstrometryParameters, RVParameters)

    def extract_data(
        self,
        data: InputData,
    ) -> tuple[GaiaAstrometryData, RadialVelocityData]:
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
        return astro, rv

    def build_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        if astro_data is None or rv_data is None:
            msg = "_CombinedStrategy requires both astro_data and rv_data"
            raise TypeError(msg)
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

    def linear_param_units(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,  # noqa: ARG002
    ) -> tuple[str, ...]:
        if astro_data is None or rv_data is None:
            msg = "_CombinedStrategy requires both astro_data and rv_data"
            raise TypeError(msg)
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
        astro_data: GaiaAstrometryData | None,  # noqa: ARG002
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,  # noqa: ARG002
        time_unit: Any,
        data: InputData,  # noqa: ARG002
        lik: Any,
    ) -> jax.Array:
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
        return jnp.concatenate([astro_sample, rv_sample])

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
