"""Rejection sampler for orbital parameter inference.

This module implements rejection sampling with analytical marginalization over
linear parameters. The sampler draws samples from the prior distribution over
nonlinear parameters, evaluates the marginalized likelihood, and performs
rejection sampling to obtain posterior samples.

Data-type-specific logic (param struct construction, likelihood building,
linear sampling) is encapsulated in ``_DataTypeStrategy`` descriptors, keeping
the sampler methods themselves branch-free.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Quantity, ustrip

from harv.data import (
    AbstractAstrometryData,
    GaiaAstrometryData,
    InputData,
    RadialVelocityData,
    SourceData,
)
from harv.likelihood._params import (
    GaiaAstrometryMarginalizedParameters,
    GaiaAstrometryParameters,
    RVMarginalizedParameters,
    RVParameters,
)
from harv.likelihood.combined import CompositeLikelihood
from harv.likelihood.gaia_astrometry import (
    MarginalizedGaiaAstrometryLikelihood,
)
from harv.likelihood.gaia_astrometry import (
    _get_design_matrix as _get_gaia_design_matrix,
)
from harv.likelihood.helpers import (
    _IndexedCallable,
    _resolve_linear_prior,
    _solve_kepler,
)
from harv.likelihood.rv import (
    MarginalizedMultiSurveyRVLikelihood,
    MarginalizedRVLikelihood,
)
from harv.likelihood.rv import (
    _get_design_matrix as _get_rv_design_matrix,
)
from harv.samplers.samples import Samples, _WarmStartMCMC

if TYPE_CHECKING:
    from harv.custom_types import Time
    from harv.priors.rejection import RejectionPrior

__all__ = ["RejectionSampler"]

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
    Required prior params are derived from the orbit param class fields.
    """

    # Stateless strategies: equality/hashing by class identity so that
    # eqx.filter_jit can hash them as static arguments.
    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    @property
    @abstractmethod
    def orbit_cls(self) -> type: ...

    @property
    @abstractmethod
    def full_cls(self) -> tuple[type, ...]: ...

    @property
    @abstractmethod
    def data_type(self) -> DataType: ...

    @property
    def required_prior_params(self) -> tuple[str, ...]:
        """Prior parameter names, derived from ``orbit_cls`` fields."""
        return tuple(f.name for f in dataclasses.fields(self.orbit_cls))

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
    def build_marginalized_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        """Build the marginalized likelihood for batched evaluation."""
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
    def orbit_cls(self) -> type:
        return RVMarginalizedParameters

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

    def build_marginalized_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,  # noqa: ARG002
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        data: InputData,
    ) -> Any:
        if rv_data is None:
            msg = "_RVStrategy requires rv_data"
            raise TypeError(msg)
        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data.n_rv() > 1
        ):
            indicator = _build_indicator_matrix(
                data.get_datasets_by_type(RadialVelocityData), prior.offsets
            )
            return MarginalizedMultiSurveyRVLikelihood(
                data=rv_data,
                indicator_matrix=indicator,
                linear_prior=prior.linear_prior,
            )
        return MarginalizedRVLikelihood(data=rv_data, linear_prior=prior.linear_prior)

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
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        time_unit: Any,
        data: InputData,
    ) -> jax.Array:
        if rv_data is None:
            msg = "_RVStrategy requires rv_data"
            raise TypeError(msg)
        period: Quantity[Time] = Quantity(sample["period"], time_unit)
        params = RVMarginalizedParameters(
            period=period,
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            arg_peri=sample["arg_peri"],
        )
        sin_f, cos_f = _solve_kepler(rv_data, params)
        dm_base = _get_rv_design_matrix(params, sin_f, cos_f)
        if (
            prior.offsets is not None
            and isinstance(data, SourceData)
            and data.n_rv() > 1
        ):
            indicator = _build_indicator_matrix(
                data.get_datasets_by_type(RadialVelocityData), prior.offsets
            )
            dm = jnp.concatenate([dm_base, indicator], axis=-1)
        else:
            dm = dm_base
        rv_unit = str(rv_data.rv.unit)
        lp = _resolve_linear_prior(prior.linear_prior, params)
        marg = MarginalizedLinear(
            design_matrix=dm,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, ustrip(rv_unit, rv_data.rv_err)),
        )
        return marg.conditional(ustrip(rv_unit, rv_data.rv)).sample(key)

    def build_orbit_params(
        self,
        period: jax.Array,
        ecc: jax.Array,
        phase: jax.Array,
        arg_peri: jax.Array,
        cos_i: jax.Array,  # noqa: ARG002
        lon_asc: jax.Array,  # noqa: ARG002
        time_unit: Any,
    ) -> eqx.Module:
        return RVMarginalizedParameters(
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
    def orbit_cls(self) -> type:
        return GaiaAstrometryMarginalizedParameters

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

    def build_marginalized_likelihood(
        self,
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,
        data: InputData,  # noqa: ARG002
    ) -> Any:
        if astro_data is None:
            msg = "_AstrometryStrategy requires astro_data"
            raise TypeError(msg)
        return MarginalizedGaiaAstrometryLikelihood(
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
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,  # noqa: ARG002
        prior: RejectionPrior,
        time_unit: Any,
        data: InputData,  # noqa: ARG002
    ) -> jax.Array:
        if astro_data is None:
            msg = "_AstrometryStrategy requires astro_data"
            raise TypeError(msg)
        period: Quantity[Time] = Quantity(sample["period"], time_unit)
        params = GaiaAstrometryMarginalizedParameters(
            period=period,
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            cos_i=sample["cos_i"],
            arg_peri=sample["arg_peri"],
            lon_asc_node=sample["lon_asc_node"],
        )
        sin_f, cos_f = _solve_kepler(astro_data, params)
        dm = _get_gaia_design_matrix(astro_data, params, sin_f, cos_f)
        astro_unit = str(astro_data.al_position.unit)
        lp = _resolve_linear_prior(prior.linear_prior, params)
        marg = MarginalizedLinear(
            design_matrix=dm,
            prior_distribution=lp,
            data_distribution=dist.Normal(
                0.0, ustrip(astro_unit, astro_data.al_position_err)
            ),
        )
        return marg.conditional(ustrip(astro_unit, astro_data.al_position)).sample(key)

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
        return GaiaAstrometryMarginalizedParameters(
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
    def orbit_cls(self) -> type:
        return GaiaAstrometryMarginalizedParameters

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

    def build_marginalized_likelihood(
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
            astro_lp = dist.MultivariateNormal(
                loc=lp.loc[list(astro_idx)],
                scale_tril=lp.scale_tril[
                    jnp.ix_(jnp.array(astro_idx), jnp.array(astro_idx))
                ],
            )
            rv_lp = dist.MultivariateNormal(
                loc=lp.loc[list(rv_idx)],
                scale_tril=lp.scale_tril[jnp.ix_(jnp.array(rv_idx), jnp.array(rv_idx))],
            )
        else:
            astro_lp = _IndexedCallable(lp, astro_idx)
            rv_lp = _IndexedCallable(lp, rv_idx)
        return CompositeLikelihood(
            astro=MarginalizedGaiaAstrometryLikelihood(astro_data, astro_lp),
            rv=MarginalizedRVLikelihood(rv_data, rv_lp),
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
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        prior: RejectionPrior,
        time_unit: Any,
        data: InputData,  # noqa: ARG002
    ) -> jax.Array:
        if astro_data is None or rv_data is None:
            msg = "_CombinedStrategy requires both astro_data and rv_data"
            raise TypeError(msg)
        k_astro, k_rv = jr.split(key)
        period: Quantity[Time] = Quantity(sample["period"], time_unit)

        params = GaiaAstrometryMarginalizedParameters(
            period=period,
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            cos_i=sample["cos_i"],
            arg_peri=sample["arg_peri"],
            lon_asc_node=sample["lon_asc_node"],
        )

        # Resolve callable or fixed prior, then slice into astro + RV blocks
        full_lp = _resolve_linear_prior(prior.linear_prior, params)
        n_astro = len(GaiaAstrometryParameters.linear_param_names)
        astro_linear_prior = dist.MultivariateNormal(
            loc=full_lp.loc[:n_astro],
            scale_tril=full_lp.scale_tril[:n_astro, :n_astro],
        )
        rv_linear_prior = dist.MultivariateNormal(
            loc=full_lp.loc[n_astro:],
            scale_tril=full_lp.scale_tril[n_astro:, n_astro:],
        )

        # Astrometry linear params
        astro_sin_f, astro_cos_f = _solve_kepler(astro_data, params)
        astro_dm = _get_gaia_design_matrix(astro_data, params, astro_sin_f, astro_cos_f)
        astro_unit = str(astro_data.al_position.unit)
        astro_marg = MarginalizedLinear(
            design_matrix=astro_dm,
            prior_distribution=astro_linear_prior,
            data_distribution=dist.Normal(
                0.0, ustrip(astro_unit, astro_data.al_position_err)
            ),
        )
        astro_sample = astro_marg.conditional(
            ustrip(astro_unit, astro_data.al_position)
        ).sample(k_astro)

        # RV linear params
        rv_params = RVMarginalizedParameters(
            period=period,
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            arg_peri=sample["arg_peri"],
        )
        rv_sin_f, rv_cos_f = _solve_kepler(rv_data, rv_params)
        rv_dm = _get_rv_design_matrix(rv_params, rv_sin_f, rv_cos_f)
        rv_unit = str(rv_data.rv.unit)
        rv_marg = MarginalizedLinear(
            design_matrix=rv_dm,
            prior_distribution=rv_linear_prior,
            data_distribution=dist.Normal(0.0, ustrip(rv_unit, rv_data.rv_err)),
        )
        rv_sample = rv_marg.conditional(ustrip(rv_unit, rv_data.rv)).sample(k_rv)

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
    ) -> eqx.Module:
        return GaiaAstrometryMarginalizedParameters(
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


# ---------------------------------------------------------------------------
# Numpyro model builder
# ---------------------------------------------------------------------------


def _build_marginalized_numpyro_model(
    sampler: "RejectionSampler",
    data: InputData,
) -> Callable[[], None]:
    """Build a marginalized numpyro model for MCMC.

    The returned callable samples each nonlinear parameter from its prior and
    evaluates the analytically-marginalized log-likelihood via ``numpyro.factor``.
    Linear parameters (K, v0, astrometric solution, etc.) are integrated out
    analytically; MCMC explores only the nonlinear subspace.

    Parameters
    ----------
    sampler : RejectionSampler
        The rejection sampler whose prior is used for the model.
    data : AbstractData or SourceData
        Observed data.  The data type determines which marginalized likelihood
        class is instantiated.

    Returns
    -------
    model : callable
        A numpyro model with no required arguments.  Sample sites: the keys of
        ``sampler.prior.nonlinear_priors`` (e.g. ``"period"``, ``"eccentricity"``).
    """
    import numpyro

    prior = sampler.prior
    strategy = sampler._infer_strategy(data)
    astro_data, rv_data = strategy.extract_data(data)
    time_unit = str(
        astro_data.time.unit if astro_data is not None else rv_data.time.unit  # type: ignore[union-attr]
    )
    lik = strategy.build_marginalized_likelihood(astro_data, rv_data, prior, data)
    orbit_cls = strategy.orbit_cls
    nonlinear_priors = prior.nonlinear_priors  # snapshot at model-build time

    def model() -> None:
        values: dict[str, Any] = {}
        for name, d in nonlinear_priors.items():
            values[name] = numpyro.sample(name, d)

        orbit_kwargs = {k: v for k, v in values.items() if k != "period"}
        orbit_kwargs["period"] = Quantity(values["period"], time_unit)
        params = orbit_cls(**orbit_kwargs)

        numpyro.factor("log_lik", lik.log_prob(params))

    return model


def _build_full_numpyro_model(
    sampler: "RejectionSampler",
    data: InputData,
) -> Callable[[], None]:
    """Build a full (unmarginalized) numpyro model for MCMC.

    The returned callable samples both nonlinear and linear parameters explicitly.
    Linear parameters are sampled jointly as a single latent site ``"_linear"``
    from the prior's ``MultivariateNormal`` (so the correlation structure of the
    prior is preserved), then exposed as named ``deterministic`` sites (e.g.
    ``"K"``, ``"v0"``) for convenient access via ``get_samples()``.  The
    Gaussian data log-likelihood is evaluated directly at the sampled values.

    Parameters
    ----------
    sampler : RejectionSampler
        The rejection sampler whose prior is used for the model.
    data : AbstractData or SourceData
        Observed data.  The data type determines the design matrix and noise
        model used for the likelihood.

    Returns
    -------
    model : callable
        A numpyro model with no required arguments.  Sample sites: keys of
        ``sampler.prior.nonlinear_priors`` plus ``"_linear"`` (the joint linear
        vector).  Deterministic sites: individual linear parameter names (e.g.
        ``"K"``, ``"v0"``, ``"semi_major_axis"``).
    """
    import numpyro

    prior = sampler.prior
    strategy = sampler._infer_strategy(data)
    astro_data, rv_data = strategy.extract_data(data)
    time_unit = str(
        astro_data.time.unit if astro_data is not None else rv_data.time.unit  # type: ignore[union-attr]
    )
    orbit_cls = strategy.orbit_cls
    nonlinear_priors = prior.nonlinear_priors

    # Linear parameter names, in the same column order as samples._linear.
    linear_param_names: tuple[str, ...] = sum(
        (cls.linear_param_names for cls in strategy.full_cls),  # type: ignore[attr-defined]
        (),
    )

    # Multi-survey RV: build the indicator matrix at model-build time (it is
    # constant — it only depends on which instrument each observation belongs to).
    indicator: jax.Array | None = None
    offset_names: tuple[str, ...] = ()
    if prior.offsets is not None and isinstance(data, SourceData) and data.n_rv() > 1:
        indicator = _build_indicator_matrix(
            data.get_datasets_by_type(RadialVelocityData), prior.offsets
        )
        offset_names = tuple(k for k, v in prior.offsets.items() if v is not None)
    linear_param_names = linear_param_names + offset_names

    # Slice boundaries for combined data (astro columns come first).
    n_astro = (
        len(GaiaAstrometryParameters.linear_param_names)
        if astro_data is not None
        else 0
    )

    # Pre-strip units from data arrays (outside the model closure for efficiency).
    if astro_data is not None:
        astro_unit = str(astro_data.al_position.unit)
        astro_obs = ustrip(astro_unit, astro_data.al_position)
        astro_err = ustrip(astro_unit, astro_data.al_position_err)
    if rv_data is not None:
        rv_unit = str(rv_data.rv.unit)
        rv_obs = ustrip(rv_unit, rv_data.rv)
        rv_err = ustrip(rv_unit, rv_data.rv_err)

    def model() -> None:
        # --- nonlinear parameters ---
        values: dict[str, Any] = {}
        for name, d in nonlinear_priors.items():
            values[name] = numpyro.sample(name, d)

        orbit_kwargs = {k: v for k, v in values.items() if k != "period"}
        orbit_kwargs["period"] = Quantity(values["period"], time_unit)
        params = orbit_cls(**orbit_kwargs)

        # --- linear parameters ---
        # Sample the full vector jointly to preserve the prior's covariance.
        resolved_lp = _resolve_linear_prior(prior.linear_prior, params)
        linear_vec = numpyro.sample("_linear", resolved_lp)
        # Expose each column as a named deterministic site.
        for i, lname in enumerate(linear_param_names):
            numpyro.deterministic(lname, linear_vec[i])

        # --- data log-likelihood ---
        log_lik: jax.Array = jnp.zeros(())

        if astro_data is not None:
            sin_f, cos_f = _solve_kepler(astro_data, params)
            dm = _get_gaia_design_matrix(astro_data, params, sin_f, cos_f)
            prediction = dm @ linear_vec[:n_astro]
            log_lik = (
                log_lik + dist.Normal(prediction, astro_err).log_prob(astro_obs).sum()
            )

        if rv_data is not None:
            sin_f, cos_f = _solve_kepler(rv_data, params)
            dm = _get_rv_design_matrix(params, sin_f, cos_f)
            if indicator is not None:
                dm = jnp.concatenate([dm, indicator], axis=-1)
            prediction = dm @ linear_vec[n_astro:]
            log_lik = log_lik + dist.Normal(prediction, rv_err).log_prob(rv_obs).sum()

        numpyro.factor("log_lik", log_lik)

    return model


def _marginal_mvn(
    mvn: dist.MultivariateNormal,
    indices: list[int],
) -> dist.MultivariateNormal:
    """Extract the marginal ``MultivariateNormal`` for the given column indices.

    Parameters
    ----------
    mvn :
        Joint multivariate normal distribution.
    indices :
        Column indices of the parameters to retain.  Must be a Python list
        (static at JAX trace time) so the indexing is resolved at trace time.

    Returns
    -------
    dist.MultivariateNormal
        Marginal distribution over the selected parameters.
    """
    idx = jnp.array(indices)
    cov = mvn.scale_tril @ mvn.scale_tril.T
    return dist.MultivariateNormal(
        loc=mvn.loc[idx],
        covariance_matrix=cov[idx][:, idx],
    )


def _build_extra_numpyro_model(
    sampler: "RejectionSampler",
    data: InputData,
    extra_model_fn: Callable[[dict[str, Any]], dict[str, Any]],
    marginalized: bool,
) -> Callable[[], None]:
    """Build a numpyro model with an ``extra_model`` reparameterization.

    Allows users to replace specific linear parameters (e.g. ``K``) with
    deterministic functions of additional physically-motivated parameters
    (e.g. stellar masses and inclination).  ``extra_model_fn`` is called
    inside the numpyro model after the nonlinear parameters have been
    sampled; it may call ``numpyro.sample`` for any number of new sites and
    must return a dict mapping linear parameter names to their computed values.

    Linear parameters *not* returned by ``extra_model_fn`` are handled
    according to ``marginalized``:

    - ``True``: analytically marginalized over the residual observations
      ``y - D_fixed @ fixed_vals``, using the marginal prior extracted from
      ``sampler.prior.linear_prior``.
    - ``False``: sampled explicitly as a joint latent site
      ``"_linear_free"``; each component is also exposed as a named
      ``deterministic`` site.

    Parameters
    ----------
    sampler :
        Rejection sampler providing the prior and strategy.
    data :
        Observed data.
    extra_model_fn :
        Callable ``(pars: dict[str, scalar]) -> dict[str, scalar]``.
        ``pars`` contains the already-sampled nonlinear parameter values
        keyed by name (e.g. ``pars["period"]`` in the data's time unit,
        ``pars["eccentricity"]``, …).  The callable may call
        ``numpyro.sample`` internally.  It must return a dict whose keys
        are a subset of the linear parameter names for this data type
        (e.g. ``"K"`` or ``"v0"`` for RV data).
    marginalized :
        If ``True``, analytically marginalize the free linear parameters.
        If ``False``, sample them explicitly from their marginal prior.

    Returns
    -------
    model : callable
        Numpyro model with no required arguments.

    Notes
    -----
    The ``pars`` dict passed to ``extra_model_fn`` uses raw scalar values in
    the same units as the prior.  In particular, ``pars["period"]`` is in the
    time unit of the input data (e.g. days if ``data.time`` is in days).

    Example — replace ``K`` with a mass-function reparameterization::

        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        # Semi-amplitude constant: K [km/s] = K_FACTOR * f(masses, inc, P, e)
        # (Lovis & Fischer 2010, converted to km/s with period in days)
        _K_FACTOR = 28.4329  # km/s · day^(1/3) · M_sun^(-1/3)

        def K_from_masses(m1, m2, inc, period_days, ecc):
            return (
                _K_FACTOR
                * (m2 * jnp.sin(inc))
                * (m1 + m2) ** (-2.0 / 3.0)
                * (period_days / 365.25) ** (-1.0 / 3.0)
                / jnp.sqrt(1.0 - ecc**2)
            )

        def mass_model(pars):
            m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
            m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
            inc = numpyro.sample("inc", dist.Uniform(0.0, jnp.pi / 2))
            K   = K_from_masses(m1, m2, inc,
                                 pars["period"], pars["eccentricity"])
            return {"K": K}

    With this ``extra_model_fn``, ``K`` becomes a deterministic site in
    ``get_samples()``; ``v0`` is analytically marginalized (if
    ``marginalized=True``) or sampled from its marginal prior.
    """
    import numpyro

    prior = sampler.prior
    strategy = sampler._infer_strategy(data)
    astro_data, rv_data = strategy.extract_data(data)
    time_unit = str(
        astro_data.time.unit if astro_data is not None else rv_data.time.unit  # type: ignore[union-attr]
    )
    orbit_cls = strategy.orbit_cls
    nonlinear_priors = prior.nonlinear_priors

    # All linear parameter names, in the same column order as samples._linear.
    all_linear_names: tuple[str, ...] = sum(
        (cls.linear_param_names for cls in strategy.full_cls),  # type: ignore[attr-defined]
        (),
    )

    # Multi-survey RV offset columns (appended after the base linear params).
    indicator: jax.Array | None = None
    offset_names: tuple[str, ...] = ()
    if prior.offsets is not None and isinstance(data, SourceData) and data.n_rv() > 1:
        indicator = _build_indicator_matrix(
            data.get_datasets_by_type(RadialVelocityData), prior.offsets
        )
        offset_names = tuple(k for k, v in prior.offsets.items() if v is not None)
    all_linear_names = all_linear_names + offset_names

    # Index boundary separating astrometry columns from RV columns.
    n_astro = (
        len(GaiaAstrometryParameters.linear_param_names)
        if astro_data is not None
        else 0
    )

    # Pre-strip units from data arrays.
    if astro_data is not None:
        astro_unit = str(astro_data.al_position.unit)
        astro_obs = ustrip(astro_unit, astro_data.al_position)
        astro_err = ustrip(astro_unit, astro_data.al_position_err)
    if rv_data is not None:
        rv_unit = str(rv_data.rv.unit)
        rv_obs = ustrip(rv_unit, rv_data.rv)
        rv_err = ustrip(rv_unit, rv_data.rv_err)

    def model() -> None:
        # --- nonlinear parameters ---
        values: dict[str, Any] = {}
        for name, d in nonlinear_priors.items():
            values[name] = numpyro.sample(name, d)

        orbit_kwargs = {k: v for k, v in values.items() if k != "period"}
        orbit_kwargs["period"] = Quantity(values["period"], time_unit)
        params = orbit_cls(**orbit_kwargs)

        # --- extra model: sample physical params, get fixed linear values ---
        # ``values`` contains raw scalar nonlinear parameters; period is in
        # ``time_unit`` (the unit of data.time).
        fixed_linear: dict[str, Any] = extra_model_fn(values)

        # Validate returned keys at trace time (string comparison is static).
        unknown = set(fixed_linear.keys()) - set(all_linear_names)
        if unknown:
            msg = (
                f"extra_model returned unknown linear parameter name(s): {unknown}. "
                f"Valid names for this data type: {all_linear_names}"
            )
            raise ValueError(msg)

        for name, val in fixed_linear.items():
            numpyro.deterministic(name, val)

        # Determine fixed/free column split (static Python-level at trace time).
        fixed_idx = [i for i, n in enumerate(all_linear_names) if n in fixed_linear]
        free_idx = [i for i, n in enumerate(all_linear_names) if n not in fixed_linear]

        # Resolve the linear prior once (may depend on orbit params if callable).
        resolved_lp = _resolve_linear_prior(prior.linear_prior, params)

        log_lik: jax.Array = jnp.zeros(())

        # --- astrometry component ---
        if astro_data is not None:
            sin_f, cos_f = _solve_kepler(astro_data, params)
            dm_a = _get_gaia_design_matrix(astro_data, params, sin_f, cos_f)

            a_fixed = [i for i in fixed_idx if i < n_astro]
            a_free = [i for i in free_idx if i < n_astro]

            y_a = astro_obs
            if a_fixed:
                fv = jnp.stack([fixed_linear[all_linear_names[i]] for i in a_fixed])
                y_a = y_a - dm_a[:, jnp.array(a_fixed)] @ fv

            if a_free and marginalized:
                marg = MarginalizedLinear(
                    design_matrix=dm_a[:, jnp.array(a_free)],
                    prior_distribution=_marginal_mvn(resolved_lp, a_free),
                    data_distribution=dist.Normal(0.0, astro_err),
                )
                log_lik = log_lik + marg.log_prob(y_a)
            elif a_free:
                free_vals = numpyro.sample(
                    "_astro_linear_free", _marginal_mvn(resolved_lp, a_free)
                )
                for j, col in enumerate(a_free):
                    numpyro.deterministic(all_linear_names[col], free_vals[j])
                prediction = dm_a[:, jnp.array(a_free)] @ free_vals
                if a_fixed:
                    fv = jnp.stack([fixed_linear[all_linear_names[i]] for i in a_fixed])
                    prediction = prediction + dm_a[:, jnp.array(a_fixed)] @ fv
                log_lik = (
                    log_lik
                    + dist.Normal(prediction, astro_err).log_prob(astro_obs).sum()
                )
            else:
                log_lik = (
                    log_lik
                    + dist.Normal(jnp.zeros_like(astro_obs), astro_err)
                    .log_prob(y_a)
                    .sum()
                )

        # --- RV component ---
        if rv_data is not None:
            sin_f, cos_f = _solve_kepler(rv_data, params)
            dm_r = _get_rv_design_matrix(params, sin_f, cos_f)
            if indicator is not None:
                dm_r = jnp.concatenate([dm_r, indicator], axis=-1)

            # Shift column indices into the RV block (starts at n_astro in the
            # joint linear vector, but the RV design matrix is zero-indexed).
            r_fixed_global = [i for i in fixed_idx if i >= n_astro]
            r_free_global = [i for i in free_idx if i >= n_astro]
            r_fixed_local = [i - n_astro for i in r_fixed_global]
            r_free_local = [i - n_astro for i in r_free_global]

            y_r = rv_obs
            if r_fixed_local:
                fv = jnp.stack(
                    [fixed_linear[all_linear_names[i]] for i in r_fixed_global]
                )
                y_r = y_r - dm_r[:, jnp.array(r_fixed_local)] @ fv

            if r_free_local and marginalized:
                marg = MarginalizedLinear(
                    design_matrix=dm_r[:, jnp.array(r_free_local)],
                    prior_distribution=_marginal_mvn(resolved_lp, r_free_global),
                    data_distribution=dist.Normal(0.0, rv_err),
                )
                log_lik = log_lik + marg.log_prob(y_r)
            elif r_free_local:
                free_vals = numpyro.sample(
                    "_rv_linear_free", _marginal_mvn(resolved_lp, r_free_global)
                )
                for j, col in enumerate(r_free_global):
                    numpyro.deterministic(all_linear_names[col], free_vals[j])
                prediction = dm_r[:, jnp.array(r_free_local)] @ free_vals
                if r_fixed_local:
                    fv = jnp.stack(
                        [fixed_linear[all_linear_names[i]] for i in r_fixed_global]
                    )
                    prediction = prediction + dm_r[:, jnp.array(r_fixed_local)] @ fv
                log_lik = (
                    log_lik + dist.Normal(prediction, rv_err).log_prob(rv_obs).sum()
                )
            else:
                log_lik = (
                    log_lik
                    + dist.Normal(jnp.zeros_like(rv_obs), rv_err).log_prob(y_r).sum()
                )

        numpyro.factor("log_lik", log_lik)

    return model


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class RejectionSampler(eqx.Module):
    """Rejection sampler for Keplerian orbital parameters.

    This class implements rejection sampling with analytical marginalization
    over linear parameters. It supports both astrometric and radial velocity
    data.

    Parameters
    ----------
    prior : RejectionPrior
        Prior distribution for nonlinear and linear parameters.
    batch_size : int, optional
        Number of samples to process per batch. Smaller values use less memory
        but may be slower. Default: 100_000.

    Examples
    --------
    >>> prior = RejectionPrior.default_astrometry()
    >>> sampler = RejectionSampler(prior)
    >>> samples = sampler.run(data, n_prior_samples=100_000)
    """

    prior: RejectionPrior
    batch_size: int = eqx.field(static=True, default=100_000)

    def _infer_strategy(self, data: InputData) -> _DataTypeStrategy:
        """Infer data type from ``data`` and return the matching strategy.

        Also validates that the prior has all required parameters for the
        inferred data type (derived from the orbit param class fields).

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters for the data type.
        """
        if isinstance(data, SourceData):
            has_rv = data.n_rv() > 0
            has_astro = data.n_astrometry() > 0
            if has_astro and has_rv:
                data_type = "combined"
            elif has_astro:
                data_type = "astrometry"
            elif has_rv:
                data_type = "rv"
            else:
                msg = "SourceData must contain at least one dataset"
                raise ValueError(msg)
        elif isinstance(data, AbstractAstrometryData):
            data_type = "astrometry"
        elif isinstance(data, RadialVelocityData):
            data_type = "rv"
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        strategy = _STRATEGIES[data_type]

        # Validate prior — required params derived from orbit param class fields
        missing = [
            p
            for p in strategy.required_prior_params
            if p not in self.prior.nonlinear_priors
        ]
        if missing:
            msg = (
                f"Prior missing required parameters for {data_type} data: {missing}. "
                f"Use RejectionPrior.default_{data_type}() or provide these parameters."
            )
            raise ValueError(msg)

        return strategy

    def run(
        self,
        data: InputData,
        n_prior_samples: int,
        *,
        max_posterior_samples: int | None = None,
        seed: int = 0,
    ) -> Samples:
        """Run rejection sampling.

        Parameters
        ----------
        data
            Observational data.
        n_prior_samples
            Number of samples to draw from the prior.
        max_posterior_samples
            Maximum number of posterior samples to return. If None, returns all
            accepted samples.
        seed
            Random seed for reproducibility. Default: 0.

        Returns
        -------
        samples
            Posterior samples container.

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters.
        """
        strategy = self._infer_strategy(data)
        astro_data, rv_data = strategy.extract_data(data)
        lik = strategy.build_marginalized_likelihood(
            astro_data, rv_data, self.prior, data
        )

        key = jr.PRNGKey(seed)
        sample_key, rej_key = jr.split(key)

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            sample_key, data, n_prior_samples, lik, strategy
        )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)
        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}

        linear_key = jr.fold_in(key, 2)
        linear_samples = self._sample_linear_parameters(
            linear_key, accepted_nonlinear, astro_data, rv_data, strategy, data
        )

        if max_posterior_samples is not None:
            n_accepted = len(next(iter(accepted_nonlinear.values())))
            if n_accepted > max_posterior_samples:
                idx_key = jr.fold_in(key, 3)
                idx = jr.choice(
                    idx_key,
                    n_accepted,
                    shape=(max_posterior_samples,),
                    replace=False,
                )
                accepted_nonlinear = {k: v[idx] for k, v in accepted_nonlinear.items()}
                linear_samples = linear_samples[idx]

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        t_ref = _ref.t_ref
        time_unit = str(_ref.time.unit)

        # Convert t_ref to a plain Python float (in time_unit) so it can be stored
        # safely in the static _metadata dict without "JAX array set as static"
        # warnings. Samples.__getitem__ reads _metadata["t_ref"] as a scalar in
        # _time_unit when computing t_peri.
        if isinstance(t_ref, Quantity):
            t_ref_stored: float | None = float(ustrip(time_unit, t_ref))
        elif t_ref is not None:
            t_ref_stored = float(t_ref)
        else:
            t_ref_stored = None

        extra_linear_names: tuple[str, ...] = ()
        if self.prior.offsets is not None:
            extra_linear_names = tuple(
                k for k, v in self.prior.offsets.items() if v is not None
            )

        return Samples(
            _nonlinear=accepted_nonlinear,
            _linear=linear_samples,
            _orbit_cls=strategy.orbit_cls,
            _full_cls=strategy.full_cls,
            _linear_param_units=strategy.linear_param_units(
                astro_data, rv_data, self.prior
            ),
            _time_unit=time_unit,
            _data_type=strategy.data_type,
            _metadata={"t_ref": t_ref_stored},
            _extra_linear_names=extra_linear_names,
        )

    def init_mcmc(
        self,
        samples: Samples,
        data: InputData,
        *,
        marginalized: bool = True,
        extra_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        extra_init_params: dict[str, Any] | None = None,
        kernel: type | None = None,
        num_chains: int = 4,
        **mcmc_kwargs: Any,
    ) -> _WarmStartMCMC:
        """Construct a numpyro MCMC object warm-started from rejection-sampler output.

        Builds a numpyro model from this sampler's prior and the observed data,
        draws one starting position per chain from ``samples``, and returns a
        :class:`~harv.samplers.samples._WarmStartMCMC` whose
        :meth:`~harv.samplers.samples._WarmStartMCMC.run` injects those positions
        automatically.

        Three model variants are supported:

        - **Marginalized** (``marginalized=True``, default): MCMC explores only
          the nonlinear subspace; linear parameters are analytically marginalized.
        - **Full** (``marginalized=False``): all parameters sampled jointly.
        - **Extra model** (``extra_model`` provided): some linear parameters are
          replaced by deterministic functions of new physical parameters sampled
          inside ``extra_model``; the remaining linear parameters are either
          analytically marginalized (``marginalized=True``) or sampled from their
          marginal prior (``marginalized=False``).

        Parameters
        ----------
        samples : Samples
            Posterior samples produced by :meth:`run`.  One sample per chain
            is used as the MCMC warm-start position.
        data : AbstractData or SourceData
            The observed data passed to :meth:`run`.
        marginalized : bool, optional
            If ``True`` (default) use the analytically-marginalized likelihood
            for any linear parameters not provided by ``extra_model``.
            If ``False``, sample those parameters explicitly from their
            marginal prior.
        extra_model : callable, optional
            A function ``(pars: dict[str, scalar]) -> dict[str, scalar]`` that
            is called inside the numpyro model after the nonlinear parameters
            have been sampled.  ``pars`` contains the raw scalar nonlinear
            parameter values keyed by name (e.g. ``pars["period"]`` in the
            data's time unit, ``pars["eccentricity"]``, …).  The function may
            call ``numpyro.sample`` for any number of new sites (e.g. stellar
            masses, inclination) and must return a dict mapping linear
            parameter names (e.g. ``"K"``) to their computed values.  Any
            linear parameter not in the returned dict is handled by
            ``marginalized``.

            Minimal pattern::

                import numpyro
                import numpyro.distributions as dist

                def extra_model(pars):
                    # pars["period"] is in the data's time unit (e.g. days)
                    m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
                    m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
                    inc = numpyro.sample("inc", dist.Uniform(0, jnp.pi / 2))
                    K   = _K_FACTOR * (m2 * jnp.sin(inc)) * (m1 + m2) ** (-2/3) \
                          * (pars["period"] / 365.25) ** (-1/3) \
                          / jnp.sqrt(1 - pars["eccentricity"] ** 2)
                    return {"K": K}

        extra_init_params : dict, optional
            Initial values for the parameters introduced by ``extra_model``,
            one entry per chain.  Required when ``extra_model`` is provided,
            since harv cannot automatically invert K → (m1, m2, inc).
            Each value must be a 1-D array of length ``num_chains``::

                extra_init_params={
                    "m1":  jnp.full(4, 1.0),
                    "m2":  jnp.full(4, 0.5),
                    "inc": jnp.full(4, 1.0),
                }

        kernel : type, optional
            A numpyro MCMC kernel *class* (not an instance).
            Defaults to ``numpyro.infer.NUTS``.
        num_chains : int, optional
            Number of independent MCMC chains.  Default: 4.
        **mcmc_kwargs :
            Forwarded unchanged to ``numpyro.infer.MCMC``.

        Returns
        -------
        mcmc : _WarmStartMCMC
            Configured MCMC wrapper.  Call ``mcmc.run(jr.PRNGKey(seed))`` to
            begin sampling.

        Raises
        ------
        ValueError
            If there are no posterior samples, fewer samples than chains, or
            ``extra_model`` is provided without ``extra_init_params``.
        ImportError
            If numpyro is not installed.

        Examples
        --------
        **Marginalized (default)** — MCMC over nonlinear parameters only,
        ``K`` and ``v0`` analytically marginalized:

        >>> import jax.random as jr
        >>> prior = RejectionPrior.default_rv(period_min=50, period_max=200)
        >>> sampler = RejectionSampler(prior)
        >>> samples = sampler.run(rv_data, n_prior_samples=500_000)
        >>> mcmc = sampler.init_mcmc(samples, rv_data,
        ...                          num_chains=4, num_warmup=500,
        ...                          num_samples=2000)
        >>> mcmc.run(jr.PRNGKey(0))
        >>> posterior = mcmc.get_samples()
        >>> # Keys: period, eccentricity, phase_peri, arg_peri

        **Full model** — all parameters sampled jointly:

        >>> mcmc = sampler.init_mcmc(samples, rv_data, marginalized=False,
        ...                          num_chains=4, num_warmup=500,
        ...                          num_samples=2000)
        >>> mcmc.run(jr.PRNGKey(0))
        >>> posterior = mcmc.get_samples()
        >>> # Adds K and v0 (as deterministic sites) to the above

        **Physical reparameterization** — replace ``K`` with stellar masses
        and inclination; ``v0`` is analytically marginalized:

        >>> import jax.numpy as jnp
        >>> import numpyro
        >>> import numpyro.distributions as dist
        >>>
        >>> _K_FACTOR = 28.4329  # km/s · day^(1/3) · M_sun^(-1/3)
        >>>
        >>> def K_from_masses(m1, m2, inc, period_days, ecc):
        ...     return (
        ...         _K_FACTOR
        ...         * (m2 * jnp.sin(inc))
        ...         * (m1 + m2) ** (-2.0 / 3.0)
        ...         * (period_days / 365.25) ** (-1.0 / 3.0)
        ...         / jnp.sqrt(1.0 - ecc**2)
        ...     )
        >>>
        >>> def mass_model(pars):
        ...     # pars["period"] is in the data's time unit (days here)
        ...     m1  = numpyro.sample("m1",  dist.Normal(1.0, 0.2))
        ...     m2  = numpyro.sample("m2",  dist.HalfNormal(1.0))
        ...     inc = numpyro.sample("inc", dist.Uniform(0.0, jnp.pi / 2))
        ...     K   = K_from_masses(m1, m2, inc,
        ...                         pars["period"], pars["eccentricity"])
        ...     return {"K": K}
        >>>
        >>> mcmc = sampler.init_mcmc(
        ...     samples, rv_data,
        ...     extra_model=mass_model,
        ...     extra_init_params={
        ...         "m1":  jnp.full(4, 1.0),   # shape (num_chains,)
        ...         "m2":  jnp.full(4, 0.5),
        ...         "inc": jnp.full(4, 1.0),
        ...     },
        ...     num_chains=4, num_warmup=500, num_samples=2000,
        ... )
        >>> mcmc.run(jr.PRNGKey(0))
        >>> posterior = mcmc.get_samples()
        >>> # Sampled sites:      period, eccentricity, …, m1, m2, inc
        >>> # Deterministic site: K  (computed from m1, m2, inc, P, e)
        >>> # Marginalized:       v0 (analytically integrated out)
        """
        try:
            from numpyro import infer as _infer
        except ImportError as e:
            msg = "numpyro is required. Install with: pip install numpyro"
            raise ImportError(msg) from e

        if samples.n_samples == 0:
            msg = "Cannot initialise MCMC: no posterior samples available."
            raise ValueError(msg)
        if samples.n_samples < num_chains:
            msg = (
                f"Fewer posterior samples ({samples.n_samples}) than requested "
                f"chains ({num_chains}). Reduce num_chains or increase "
                "n_prior_samples in RejectionSampler.run()."
            )
            raise ValueError(msg)
        if extra_model is not None and extra_init_params is None:
            msg = (
                "extra_init_params is required when extra_model is provided. "
                "Provide initial values for each parameter introduced by extra_model "
                "(one entry per chain, shape (num_chains,))."
            )
            raise ValueError(msg)

        if kernel is None:
            kernel = _infer.NUTS

        # Take the first num_chains posterior samples as starting positions.
        indices = list(range(num_chains))
        init_params: dict[str, Any] = {
            key_name: jnp.stack([arr[i] for i in indices])
            for key_name, arr in samples._nonlinear.items()
        }

        if extra_model is not None:
            model = _build_extra_numpyro_model(self, data, extra_model, marginalized)
            # Warm-start the extra physical parameters from user-provided values.
            init_params.update(extra_init_params)  # type: ignore[arg-type]
        elif marginalized:
            model = _build_marginalized_numpyro_model(self, data)
        else:
            model = _build_full_numpyro_model(self, data)
            # Warm-start the joint linear site from the rejection-sampler draws.
            linear = np.asarray(samples._linear)
            if linear.ndim == 2 and linear.shape[-1] > 0:
                init_params["_linear"] = jnp.stack(
                    [jnp.asarray(linear[i]) for i in indices]
                )

        kernel_instance = kernel(model)
        return _WarmStartMCMC(
            kernel_instance,
            _init_params=init_params,
            num_chains=num_chains,
            **mcmc_kwargs,
        )

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(
        self,
        key: jax.Array,
        data: InputData,
        n_prior_samples: int,
        lik: Any,
        strategy: _DataTypeStrategy,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        The pre-built ``lik`` (a single marginalized likelihood or a
        ``CompositeLikelihood``) is evaluated with ``jax.vmap`` inside a
        ``fori_loop`` over batches of ``batch_size`` samples.  ``strategy`` is
        a static value (hashed by class identity) so ``build_orbit_params``
        dispatches to the correct param type at trace time.
        """
        prior_samples = self.prior.sample_nonlinear(key, n_prior_samples)

        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        total_size = n_batches * self.batch_size
        pad_size = total_size - n_prior_samples

        def pad_batch(arr: jax.Array) -> jax.Array:
            return jnp.pad(arr, (0, pad_size)).reshape(n_batches, self.batch_size)

        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        time_unit = _ref.time.unit

        period_batched = pad_batch(prior_samples["period"])
        ecc_batched = pad_batch(prior_samples["eccentricity"])
        phase_batched = pad_batch(prior_samples["phase_peri"])
        arg_peri_batched = pad_batch(prior_samples["arg_peri"])

        # Pad optional params with zeros (unused values are ignored by the builder).
        _zeros = jnp.zeros(n_prior_samples)
        cos_i_batched = pad_batch(prior_samples.get("cos_i", _zeros))
        lon_asc_batched = pad_batch(prior_samples.get("lon_asc_node", _zeros))

        def body_fn(i: int, acc: jax.Array) -> jax.Array:
            params = strategy.build_orbit_params(
                period_batched[i],
                ecc_batched[i],
                phase_batched[i],
                arg_peri_batched[i],
                cos_i_batched[i],
                lon_asc_batched[i],
                time_unit,
            )
            return acc.at[i].set(jax.vmap(lik.log_prob)(params))

        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )
        return prior_samples, log_liks_batched.flatten()[:n_prior_samples]

    @staticmethod
    @jax.jit
    def _rejection_step(key: jax.Array, log_likelihoods: jax.Array) -> jax.Array:
        """Compute rejection mask."""
        weights = jnp.exp(log_likelihoods - jnp.max(log_likelihoods))
        uniform_draws = jr.uniform(key, shape=log_likelihoods.shape)
        return uniform_draws < weights

    def _sample_linear_parameters(
        self,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
        astro_data: GaiaAstrometryData | None,
        rv_data: RadialVelocityData | None,
        strategy: _DataTypeStrategy,
        data: InputData,
    ) -> jax.Array:
        """Sample linear parameters from conditional posterior using vmap.

        For each accepted nonlinear sample, draws from the conditional posterior
        of the linear parameters given the nonlinear parameters and data.

        Parameters
        ----------
        key
            Random key.
        nonlinear_samples
            Accepted nonlinear parameter samples.
        astro_data
            Gaia astrometry data, or None.
        rv_data
            Radial velocity data, or None.
        strategy
            Data-type strategy for building params and design matrices.
        data
            Original input data (needed for multi-survey instrument ordering).

        Returns
        -------
        linear_samples
            Shape ``(n_samples, n_linear)``.
        """
        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            n_offsets = sum(
                1 for v in (self.prior.offsets or {}).values() if v is not None
            )
            return jnp.zeros((0, strategy.n_linear + n_offsets))

        _ref = rv_data if rv_data is not None else astro_data
        time_unit = _ref.time.unit  # type: ignore[union-attr]

        keys = jr.split(key, n_samples)

        def _sample_one(key: jax.Array, sample: dict[str, jax.Array]) -> jax.Array:
            return strategy.sample_linear_one(
                key, sample, astro_data, rv_data, self.prior, time_unit, data
            )

        return jax.vmap(_sample_one)(keys, nonlinear_samples)
