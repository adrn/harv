"""Rejection sampler for orbital parameter inference.

This module implements rejection sampling with analytical marginalization over
linear parameters. The sampler draws samples from the prior distribution over
nonlinear parameters, evaluates the marginalized likelihood, and performs
rejection sampling to obtain posterior samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
from harv.likelihood.gaia_astrometry import (
    MarginalizedGaiaAstrometryLikelihood,
)
from harv.likelihood.gaia_astrometry import (
    _get_design_matrix as _get_gaia_design_matrix,
)
from harv.likelihood.helpers import _solve_kepler
from harv.likelihood.rv import (
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

    def _infer_and_validate_data_type(
        self,
        data: InputData,
    ) -> DataType:
        """Infer data type and validate prior has required parameters.

        Parameters
        ----------
        data
            Observational data.

        Returns
        -------
        data_type
            Inferred data type.

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters for the data type.
        """
        if isinstance(data, SourceData):
            rv_datasets = list(data.get_datasets_by_type(RadialVelocityData).values())
            astro_datasets = list(
                data.get_datasets_by_type(AbstractAstrometryData).values()
            )

            if len(astro_datasets) > 0 and len(rv_datasets) > 0:
                data_type = "combined"
            elif len(astro_datasets) > 0:
                data_type = "astrometry"
            elif len(rv_datasets) > 0:
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

        # Validate prior has required parameters (prior fields use log_period)
        # TODO: these should be taken from the Parameters classes, not duplicated here.
        if data_type in ["astrometry", "combined"]:
            required = [
                "log_period",
                "eccentricity",
                "phase_peri",
                "cos_i",
                "arg_peri",
                "lon_asc_node",
            ]
        elif data_type == "rv":
            required = ["log_period", "eccentricity", "phase_peri", "arg_peri"]
        else:
            msg = f"Unknown data type: {data_type}"
            raise ValueError(msg)

        missing = [p for p in required if getattr(self.prior, p, None) is None]
        if missing:
            msg = (
                f"Prior missing required parameters for {data_type} data: {missing}. "
                f"Use RejectionPrior.default_{data_type}() or provide these parameters."
            )
            raise ValueError(msg)

        return data_type  # type: ignore[return-value]

    def run(
        self,
        data: InputData,
        n_prior_samples: int,
        *,
        max_posterior_samples: int | None = None,
        seed: int = 0,
    ) -> "Samples":
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
            Random seed for reproducibility. Default: 42.

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
        data_type = self._infer_and_validate_data_type(data)

        key = jr.PRNGKey(seed)
        sample_key, rej_key = jr.split(key)

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            sample_key, data, n_prior_samples, data_type
        )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)

        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}

        linear_key = jr.fold_in(key, 2)
        linear_samples = self._sample_linear_parameters(
            linear_key, accepted_nonlinear, data, data_type
        )

        # Determine orbit/full parameter classes and derive linear param units
        # from the actual data so that samples["K"] returns the data's native unit.
        if isinstance(data, SourceData):
            _rv_data = next(
                iter(data.get_datasets_by_type(RadialVelocityData).values()),
                None,
            )
            _astro_data = next(
                iter(data.get_datasets_by_type(GaiaAstrometryData).values()),
                None,
            )
        else:
            _rv_data = data if isinstance(data, RadialVelocityData) else None
            _astro_data = data if isinstance(data, GaiaAstrometryData) else None

        orbit_cls: type
        full_cls: tuple[type, ...]
        linear_param_units: tuple[str, ...]

        if data_type == "astrometry":
            orbit_cls = GaiaAstrometryOrbitParameters
            full_cls = (GaiaAstrometryFullParameters,)
            if _astro_data is None:
                msg = "Expected GaiaAstrometryData for astrometry data_type"
                raise TypeError(msg)
            _pos_unit = str(_astro_data.al_position.unit)
            _pm_unit = f"{_pos_unit}/yr"
            linear_param_units = (
                _pos_unit,
                _pos_unit,
                _pm_unit,
                _pm_unit,
                _pos_unit,
                _pos_unit,
            )
        elif data_type == "rv":
            orbit_cls = RVOrbitParameters
            full_cls = (RVFullParameters,)
            if _rv_data is None:
                msg = "Expected RadialVelocityData for rv data_type"
                raise TypeError(msg)
            _rv_unit = str(_rv_data.rv.unit)
            linear_param_units = (_rv_unit, _rv_unit)
        else:  # combined
            orbit_cls = CombinedOrbitParameters
            full_cls = (GaiaAstrometryFullParameters, RVFullParameters)
            if _astro_data is None or _rv_data is None:
                msg = "Expected GaiaAstrometryData and RadialVelocityData for combined"
                raise TypeError(msg)
            _pos_unit = str(_astro_data.al_position.unit)
            _pm_unit = f"{_pos_unit}/yr"
            _rv_unit = str(_rv_data.rv.unit)
            linear_param_units = (
                _pos_unit,
                _pos_unit,
                _pm_unit,
                _pm_unit,
                _pos_unit,
                _pos_unit,
                _rv_unit,
                _rv_unit,
            )

        if max_posterior_samples is not None:
            n_accepted = len(accepted_nonlinear["log_period"])
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

        if isinstance(data, SourceData):
            t_ref = next(iter(data.values())).t_ref
        else:
            t_ref = data.t_ref

        return Samples(
            _nonlinear=accepted_nonlinear,
            _linear=linear_samples,
            _orbit_cls=orbit_cls,
            _full_cls=full_cls,
            _linear_param_units=linear_param_units,
            _metadata={"t_ref": t_ref},
        )

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(  # noqa: C901
        self,
        key: jax.Array,
        data: InputData,
        n_prior_samples: int,
        data_type: DataType,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        TODO: the if statements and logic flow here is horrendous. Let's redesign.
        """
        prior_samples = self.prior.sample_nonlinear(key, n_prior_samples)

        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        total_size = n_batches * self.batch_size
        pad_size = total_size - n_prior_samples

        def pad_batch(arr: jax.Array) -> jax.Array:
            return jnp.pad(arr, (0, pad_size)).reshape(n_batches, self.batch_size)

        # Prior samples log_period = log10(period / data_time_unit); derive time unit
        # from data so that Quantity(10**log_period, time_unit) is consistent.
        _ref = next(iter(data.values())) if isinstance(data, SourceData) else data
        time_unit = _ref.time.unit

        period_batched = pad_batch(10.0 ** prior_samples["log_period"])
        ecc_batched = pad_batch(prior_samples["eccentricity"])
        phase_batched = pad_batch(prior_samples["phase_peri"])
        arg_peri_batched = pad_batch(prior_samples["arg_peri"])

        if data_type in ["rv", "sb2"]:
            if isinstance(data, RadialVelocityData):
                rv_data = data
            elif isinstance(data, SourceData):
                rv_data = next(
                    iter(data.get_datasets_by_type(RadialVelocityData).values())
                )
            else:
                msg = f"Expected RadialVelocityData or SourceData, got {type(data)}"
                raise TypeError(msg)
            linear_prior = dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=self.prior.linear_prior_scale**2 * jnp.eye(2),
            )
            lik = MarginalizedRVLikelihood(data=rv_data, linear_prior=linear_prior)

            def body_fn(i: int, acc: jax.Array) -> jax.Array:
                params = RVOrbitParameters(
                    period=Quantity(period_batched[i], time_unit),
                    eccentricity=ecc_batched[i],
                    phase_peri=phase_batched[i],
                    arg_peri=arg_peri_batched[i],
                )
                return acc.at[i].set(jax.vmap(lik.log_prob)(params))

        else:
            cos_i_batched = pad_batch(prior_samples["cos_i"])
            lon_asc_batched = pad_batch(prior_samples["lon_asc_node"])

            if isinstance(data, GaiaAstrometryData):
                astro_data = data
            elif isinstance(data, SourceData):
                astro_data = next(
                    iter(data.get_datasets_by_type(GaiaAstrometryData).values())
                )
            else:
                msg = f"Expected AbstractAstrometryData or SourceData, got {type(data)}"
                raise TypeError(msg)
            astro_linear_prior = dist.MultivariateNormal(
                loc=jnp.zeros(6),
                covariance_matrix=self.prior.linear_prior_scale**2 * jnp.eye(6),
            )
            astro_lik = MarginalizedGaiaAstrometryLikelihood(
                data=astro_data, linear_prior=astro_linear_prior
            )

            if data_type == "combined":
                if isinstance(data, SourceData):
                    rv_data = next(
                        iter(data.get_datasets_by_type(RadialVelocityData).values())
                    )
                else:
                    msg = "Combined data_type requires SourceData"
                    raise TypeError(msg)
                rv_linear_prior = dist.MultivariateNormal(
                    loc=jnp.zeros(2),
                    covariance_matrix=self.prior.linear_prior_scale**2 * jnp.eye(2),
                )
                rv_lik = MarginalizedRVLikelihood(
                    data=rv_data, linear_prior=rv_linear_prior
                )

                def body_fn(i: int, acc: jax.Array) -> jax.Array:
                    params = GaiaAstrometryOrbitParameters(
                        period=Quantity(period_batched[i], time_unit),
                        eccentricity=ecc_batched[i],
                        phase_peri=phase_batched[i],
                        cos_i=cos_i_batched[i],
                        arg_peri=arg_peri_batched[i],
                        lon_asc_node=lon_asc_batched[i],
                    )
                    rv_params = RVOrbitParameters(
                        period=params.period,
                        eccentricity=params.eccentricity,
                        phase_peri=params.phase_peri,
                        arg_peri=params.arg_peri,
                    )
                    log_lik = jax.vmap(astro_lik.log_prob)(params) + jax.vmap(
                        rv_lik.log_prob
                    )(rv_params)
                    return acc.at[i].set(log_lik)

            else:

                def body_fn(i: int, acc: jax.Array) -> jax.Array:
                    params = GaiaAstrometryOrbitParameters(
                        period=Quantity(period_batched[i], time_unit),
                        eccentricity=ecc_batched[i],
                        phase_peri=phase_batched[i],
                        cos_i=cos_i_batched[i],
                        arg_peri=arg_peri_batched[i],
                        lon_asc_node=lon_asc_batched[i],
                    )
                    return acc.at[i].set(jax.vmap(astro_lik.log_prob)(params))

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

    def _sample_linear_parameters(  # noqa: C901
        self,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
        data: InputData,
        data_type: DataType,
        n_linear_per_nonlinear: int = 1,
    ) -> jax.Array:
        """Sample linear parameters from conditional posterior.

        For each accepted nonlinear sample, draws from the conditional posterior
        distribution of the linear parameters given the nonlinear parameters and data.

        Parameters
        ----------
        key : jax.Array
            Random key.
        nonlinear_samples : dict[str, jax.Array]
            Accepted nonlinear parameter samples (contains log_period).
        data : GaiaAstrometryData | RadialVelocityData | SourceData
            Observational data.
        data_type : DataType
            Type of data being processed.
        n_linear_per_nonlinear : int, optional
            Number of linear samples per nonlinear sample. Default: 1.

        Returns
        -------
        linear_samples : jax.Array
            Shape (n_samples * n_linear_per_nonlinear, n_linear).
        """
        n_samples = len(nonlinear_samples["log_period"])

        if data_type == "rv":
            n_linear = 2
        elif data_type == "astrometry":
            n_linear = 6
        else:  # combined
            n_linear = 8

        if n_samples == 0:
            return jnp.zeros((0, n_linear))

        # Extract concrete data objects and time unit
        if isinstance(data, SourceData):
            rv_list = list(data.get_datasets_by_type(RadialVelocityData).values())
            astro_list = list(data.get_datasets_by_type(GaiaAstrometryData).values())
            rv_data: RadialVelocityData | None = rv_list[0] if rv_list else None
            astro_data: GaiaAstrometryData | None = (
                astro_list[0] if astro_list else None
            )
        else:
            rv_data = data if isinstance(data, RadialVelocityData) else None
            astro_data = data if isinstance(data, GaiaAstrometryData) else None

        _time_ref = rv_data if rv_data is not None else astro_data
        time_unit = _time_ref.time.unit  # type: ignore[union-attr]

        # Validate required data is present and narrow optional types for loop use.
        if data_type == "rv":
            if rv_data is None:
                msg = "Expected RadialVelocityData for rv data_type"
                raise TypeError(msg)
            _rv_unit: str = str(rv_data.rv.unit)
            _rv = rv_data
        elif data_type == "astrometry":
            if astro_data is None:
                msg = "Expected GaiaAstrometryData for astrometry data_type"
                raise TypeError(msg)
            _astro_unit: str = str(astro_data.al_position.unit)
            _astro = astro_data
        else:  # combined
            if rv_data is None or astro_data is None:
                msg = "Expected both data types for combined data_type"
                raise TypeError(msg)
            _rv_unit = str(rv_data.rv.unit)
            _astro_unit = str(astro_data.al_position.unit)
            _rv = rv_data
            _astro = astro_data

        linear_samples = []
        keys = jr.split(key, n_samples)

        for i in range(n_samples):
            # Prior stores log_period = log10(period / data_time_unit)
            period: Quantity[Time] = Quantity(
                10.0 ** nonlinear_samples["log_period"][i], time_unit
            )

            if data_type == "rv":
                params = RVOrbitParameters(
                    period=period,
                    eccentricity=nonlinear_samples["eccentricity"][i],
                    phase_peri=nonlinear_samples["phase_peri"][i],
                    arg_peri=nonlinear_samples["arg_peri"][i],
                )
                sin_f, cos_f = _solve_kepler(_rv, params)
                design_matrix = _get_rv_design_matrix(params, sin_f, cos_f)

                marg_dist = MarginalizedLinear(
                    design_matrix=design_matrix,
                    prior_distribution=dist.Normal(0.0, self.prior.linear_prior_scale),
                    data_distribution=dist.Normal(
                        0.0,
                        ustrip(_rv_unit, _rv.rv_err),
                    ),
                )
                posterior = marg_dist.conditional(ustrip(_rv_unit, _rv.rv))
                sample = posterior.sample(
                    keys[i], sample_shape=(n_linear_per_nonlinear,)
                )

            elif data_type == "astrometry":
                astro_params = GaiaAstrometryOrbitParameters(
                    period=period,
                    eccentricity=nonlinear_samples["eccentricity"][i],
                    phase_peri=nonlinear_samples["phase_peri"][i],
                    cos_i=nonlinear_samples["cos_i"][i],
                    arg_peri=nonlinear_samples["arg_peri"][i],
                    lon_asc_node=nonlinear_samples["lon_asc_node"][i],
                )
                sin_f, cos_f = _solve_kepler(_astro, astro_params)
                design_matrix = _get_gaia_design_matrix(
                    _astro, astro_params, sin_f, cos_f
                )

                marg_dist = MarginalizedLinear(
                    design_matrix=design_matrix,
                    prior_distribution=dist.Normal(0.0, self.prior.linear_prior_scale),
                    data_distribution=dist.Normal(
                        0.0,
                        ustrip(_astro_unit, _astro.al_position_err),
                    ),
                )
                posterior = marg_dist.conditional(
                    ustrip(_astro_unit, _astro.al_position)
                )
                sample = posterior.sample(
                    keys[i], sample_shape=(n_linear_per_nonlinear,)
                )

            else:  # combined: sample astro and RV linear params separately
                astro_params = GaiaAstrometryOrbitParameters(
                    period=period,
                    eccentricity=nonlinear_samples["eccentricity"][i],
                    phase_peri=nonlinear_samples["phase_peri"][i],
                    cos_i=nonlinear_samples["cos_i"][i],
                    arg_peri=nonlinear_samples["arg_peri"][i],
                    lon_asc_node=nonlinear_samples["lon_asc_node"][i],
                )

                astro_sin_f, astro_cos_f = _solve_kepler(_astro, astro_params)
                astro_dm = _get_gaia_design_matrix(
                    _astro, astro_params, astro_sin_f, astro_cos_f
                )
                astro_marg = MarginalizedLinear(
                    design_matrix=astro_dm,
                    prior_distribution=dist.Normal(0.0, self.prior.linear_prior_scale),
                    data_distribution=dist.Normal(
                        0.0,
                        ustrip(_astro_unit, _astro.al_position_err),
                    ),
                )
                astro_sample = astro_marg.conditional(
                    ustrip(_astro_unit, _astro.al_position)
                ).sample(keys[i], sample_shape=(n_linear_per_nonlinear,))

                rv_params = RVOrbitParameters(
                    period=astro_params.period,
                    eccentricity=astro_params.eccentricity,
                    phase_peri=astro_params.phase_peri,
                    arg_peri=astro_params.arg_peri,
                )
                rv_sin_f, rv_cos_f = _solve_kepler(_rv, rv_params)
                rv_dm = _get_rv_design_matrix(rv_params, rv_sin_f, rv_cos_f)
                rv_marg = MarginalizedLinear(
                    design_matrix=rv_dm,
                    prior_distribution=dist.Normal(0.0, self.prior.linear_prior_scale),
                    data_distribution=dist.Normal(
                        0.0,
                        ustrip(_rv_unit, _rv.rv_err),
                    ),
                )
                rv_sample = rv_marg.conditional(ustrip(_rv_unit, _rv.rv)).sample(
                    keys[i], sample_shape=(n_linear_per_nonlinear,)
                )

                sample = jnp.concatenate([astro_sample, rv_sample], axis=-1)

            linear_samples.append(sample)

        return jnp.concatenate(linear_samples, axis=0)
