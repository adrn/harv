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
from typing import TYPE_CHECKING, Any, Literal, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
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
    CombinedOrbitParameters,
    GaiaAstrometryFullParameters,
    GaiaAstrometryOrbitParameters,
    RVFullParameters,
    RVOrbitParameters,
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
from harv.samplers.samples import Samples

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
    def data_type(self) -> DataType:
        """Data type string, derived from ``orbit_cls.data_type``."""
        return self.orbit_cls.data_type  # type: ignore[attr-defined]

    @property
    def required_prior_params(self) -> tuple[str, ...]:
        """Prior parameter names, derived from ``orbit_cls`` fields."""
        return tuple(f.name for f in dataclasses.fields(self.orbit_cls))

    @property
    def n_linear(self) -> int:
        """Total linear parameters, summed from ``full_cls.linear_param_names``."""
        return sum(
            len(cls.linear_param_names)  # type: ignore[attr-defined,misc]
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
    def orbit_cls(self) -> type:
        return RVOrbitParameters

    @property
    def full_cls(self) -> tuple[type, ...]:
        return (RVFullParameters,)

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
        params = RVOrbitParameters(
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
        return RVOrbitParameters(
            period=Quantity(period, time_unit),
            eccentricity=ecc,
            phase_peri=phase,
            arg_peri=arg_peri,
        )


@final
class _AstrometryStrategy(_DataTypeStrategy):
    @property
    def orbit_cls(self) -> type:
        return GaiaAstrometryOrbitParameters

    @property
    def full_cls(self) -> tuple[type, ...]:
        return (GaiaAstrometryFullParameters,)

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
        params = GaiaAstrometryOrbitParameters(
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
        return GaiaAstrometryOrbitParameters(
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
    def orbit_cls(self) -> type:
        return CombinedOrbitParameters

    @property
    def full_cls(self) -> tuple[type, ...]:
        return (GaiaAstrometryFullParameters, RVFullParameters)

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
        n = len(GaiaAstrometryFullParameters.linear_param_names)  # 6
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

        params = CombinedOrbitParameters(
            period=period,
            eccentricity=sample["eccentricity"],
            phase_peri=sample["phase_peri"],
            cos_i=sample["cos_i"],
            arg_peri=sample["arg_peri"],
            lon_asc_node=sample["lon_asc_node"],
        )

        # Resolve callable or fixed prior, then slice into astro + RV blocks
        full_lp = _resolve_linear_prior(prior.linear_prior, params)
        n_astro = len(GaiaAstrometryFullParameters.linear_param_names)
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
        rv_params = RVOrbitParameters(
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
        return CombinedOrbitParameters(
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
            _metadata={"t_ref": t_ref_stored},
            _extra_linear_names=extra_linear_names,
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
