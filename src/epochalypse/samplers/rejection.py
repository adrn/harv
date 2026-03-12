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
import numpy as np

from epochalypse.data import (
    AbstractAstrometryData,
    AbstractRadialVelocityData,
    SourceData,
)
from epochalypse.likelihood.astrometry import (
    compute_marginal_log_likelihood_astrometry_batch,
)
from epochalypse.likelihood.rv import (
    compute_marginal_log_likelihood_rv_batch,
)

if TYPE_CHECKING:
    from epochalypse.data import GaiaAstrometryData, RadialVelocityData
    from epochalypse.priors.rejection import RejectionPrior
    from epochalypse.samplers.samples import Samples

__all__ = ["RejectionSampler"]

DataType = Literal["astrometry", "rv", "combined", "sb2"]


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
        but may be slower. Default: 500_000.

    Examples
    --------
    >>> prior = RejectionPrior.default_astrometry()
    >>> sampler = RejectionSampler(prior)
    >>> samples = sampler.run(data, n_prior_samples=100_000)
    """

    prior: "RejectionPrior"
    batch_size: int = eqx.field(static=True, default=500_000)

    def _infer_and_validate_data_type(
        self,
        data: "GaiaAstrometryData | RadialVelocityData | SourceData",
    ) -> DataType:
        """Infer data type and validate prior has required parameters.

        Parameters
        ----------
        data : GaiaAstrometryData | RadialVelocityData | SourceData
            Observational data.

        Returns
        -------
        data_type : DataType
            Inferred data type.

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters for the data type.
        """
        # Infer data type
        if isinstance(data, SourceData):
            rv_datasets = list(
                data.get_datasets_by_type(AbstractRadialVelocityData).values()
            )
            astro_datasets = list(
                data.get_datasets_by_type(AbstractAstrometryData).values()
            )

            if len(rv_datasets) > 1:
                # SB2 case
                data_type = "sb2"
            elif len(astro_datasets) > 0 and len(rv_datasets) > 0:
                # Combined
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
        elif isinstance(data, AbstractRadialVelocityData):
            data_type = "rv"
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        # Validate prior has required parameters
        if data_type in ["astrometry", "combined"]:
            required = [
                "log_period",
                "eccentricity",
                "phase_peri",
                "cos_i",
                "arg_peri",
                "lon_asc_node",
            ]
        elif data_type in ["rv", "sb2"]:
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

        return data_type

    def run(
        self,
        data: "GaiaAstrometryData",
        n_prior_samples: int,
        *,
        max_posterior_samples: int | None = None,
        seed: int = 42,
    ) -> "Samples":
        """Run rejection sampling.

        Parameters
        ----------
        data : GaiaAstrometryData
            Observational data.
        n_prior_samples : int
            Number of samples to draw from the prior.
        max_posterior_samples : int, optional
            Maximum number of posterior samples to return. If None, returns all
            accepted samples.
        seed : int, optional
            Random seed for reproducibility. Default: 42.

        Returns
        -------
        samples : Samples
            Posterior samples container.

        Raises
        ------
        TypeError
            If data type is not supported.
        ValueError
            If prior is missing required parameters.
        """
        # Validate prior and infer data type
        data_type = self._infer_and_validate_data_type(data)

        # Import here to avoid circular dependency
        from epochalypse.samplers.samples import Samples

        key = jr.PRNGKey(seed)
        sample_key, rej_key = jr.split(key)

        # Sample from prior and evaluate likelihoods
        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            sample_key, data, n_prior_samples, data_type
        )

        # Perform rejection step
        accepted_mask = self._rejection_step(rej_key, log_likelihoods)

        # Extract accepted samples
        accepted_nonlinear = {
            k: np.asarray(v[accepted_mask]) for k, v in prior_samples.items()
        }

        # Sample linear parameters from conditional posterior
        linear_key = jr.fold_in(key, 2)
        linear_samples = self._sample_linear_parameters(
            linear_key, accepted_nonlinear, data, data_type
        )

        # Limit to max_posterior_samples if requested
        if max_posterior_samples is not None:
            n_accepted = len(accepted_nonlinear["log_period"])
            if n_accepted > max_posterior_samples:
                idx = np.random.choice(
                    n_accepted, size=max_posterior_samples, replace=False
                )
                accepted_nonlinear = {k: v[idx] for k, v in accepted_nonlinear.items()}
                linear_samples = linear_samples[idx]

        # Create Samples container with appropriate linear param names
        if data_type == "astrometry":
            linear_param_names = (
                "alpha_0",
                "delta_0",
                "mu_alpha",
                "mu_delta",
                "parallax",
                "semimajor_axis",
            )
        elif data_type == "rv":
            linear_param_names = ("K", "v0")
        elif data_type == "sb2":
            linear_param_names = ("K1", "K2", "v0")
        else:  # combined
            linear_param_names = (
                "alpha_0",
                "delta_0",
                "mu_alpha",
                "mu_delta",
                "parallax",
                "semimajor_axis",
                "K",
                "v0",
            )

        # Get t_ref from data
        if isinstance(data, SourceData):
            # Use first dataset's t_ref
            first_dataset = next(iter(data.datasets.values()))
            t_ref = first_dataset.t_ref
        else:
            t_ref = data.t_ref

        return Samples(
            _nonlinear=accepted_nonlinear,
            _linear=linear_samples,
            _linear_param_names=linear_param_names,
            _data_type=data_type,
            _metadata={"t_ref": t_ref},
        )

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(
        self,
        key: jax.Array,
        data: "GaiaAstrometryData | RadialVelocityData",
        n_prior_samples: int,
        data_type: DataType,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        Parameters
        ----------
        key : jax.Array
            Random key.
        data : GaiaAstrometryData | RadialVelocityData
            Observational data.
        n_prior_samples : int
            Number of prior samples.
        data_type : DataType
            Type of data being processed.

        Returns
        -------
        prior_samples : dict[str, jax.Array]
            Sampled nonlinear parameters.
        log_likelihoods : jax.Array
            Log-likelihood for each prior sample.
        """
        # Sample nonlinear parameters from prior
        prior_samples = self.prior.sample_nonlinear(key, n_prior_samples)

        # Compute number of batches, pad to exact multiple
        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        total_size = n_batches * self.batch_size
        pad_size = total_size - n_prior_samples

        # Pad arrays to exact multiple of batch_size (common parameters)
        log_period_padded = jnp.pad(prior_samples["log_period"], (0, pad_size))
        ecc_padded = jnp.pad(prior_samples["eccentricity"], (0, pad_size))
        phase_padded = jnp.pad(prior_samples["phase_peri"], (0, pad_size))
        arg_peri_padded = jnp.pad(prior_samples["arg_peri"], (0, pad_size))

        # Reshape into (n_batches, batch_size)
        log_period_batched = log_period_padded.reshape(n_batches, self.batch_size)
        ecc_batched = ecc_padded.reshape(n_batches, self.batch_size)
        phase_batched = phase_padded.reshape(n_batches, self.batch_size)
        arg_peri_batched = arg_peri_padded.reshape(n_batches, self.batch_size)

        # Create linear prior distribution
        import numpyro.distributions as dist

        linear_prior = dist.Normal(0.0, self.prior.linear_prior_scale)

        if data_type == "rv":
            # RV-only: 4 nonlinear parameters
            def body_fn(i, log_liks_batched):
                batch_log_lik = compute_marginal_log_likelihood_rv_batch(
                    log_period_batched[i],
                    ecc_batched[i],
                    phase_batched[i],
                    arg_peri_batched[i],
                    data.time,
                    data.rv,
                    data.rv_err,
                    data.t_ref,
                    linear_prior,
                )
                return log_liks_batched.at[i].set(batch_log_lik)
        else:
            # Astrometry or combined: need cos_i and lon_asc_node
            cos_i_padded = jnp.pad(prior_samples["cos_i"], (0, pad_size))
            lon_asc_padded = jnp.pad(prior_samples["lon_asc_node"], (0, pad_size))
            cos_i_batched = cos_i_padded.reshape(n_batches, self.batch_size)
            lon_asc_batched = lon_asc_padded.reshape(n_batches, self.batch_size)

            def body_fn(i, log_liks_batched):
                batch_log_lik = compute_marginal_log_likelihood_astrometry_batch(
                    log_period_batched[i],
                    ecc_batched[i],
                    phase_batched[i],
                    cos_i_batched[i],
                    arg_peri_batched[i],
                    lon_asc_batched[i],
                    data.time,
                    data.scan_angle,
                    data.parallax_factor,
                    data.al_position,
                    data.al_position_err,
                    data.t_ref,
                    linear_prior,
                )
                return log_liks_batched.at[i].set(batch_log_lik)

        # Process batches using fori_loop (memory efficient)
        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )

        # Flatten and trim to original size
        log_likelihoods = log_liks_batched.flatten()[:n_prior_samples]

        return prior_samples, log_likelihoods

    @staticmethod
    @jax.jit
    def _rejection_step(key: jax.Array, log_likelihoods: jax.Array) -> jax.Array:
        """Compute rejection mask.

        Parameters
        ----------
        key : jax.Array
            Random key.
        log_likelihoods : jax.Array
            Log-likelihood values.

        Returns
        -------
        accepted_mask : jax.Array
            Boolean mask of accepted samples.
        """
        weights = jnp.exp(log_likelihoods - jnp.max(log_likelihoods))
        uniform_draws = jr.uniform(key, shape=log_likelihoods.shape)
        return uniform_draws < weights

    def _sample_linear_parameters(
        self,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
        data: "GaiaAstrometryData | RadialVelocityData",
        data_type: DataType,
        n_linear_per_nonlinear: int = 1,
    ) -> jax.Array:
        """Sample linear parameters from conditional posterior.

        For each accepted nonlinear sample, this draws samples from the
        conditional posterior distribution of the linear parameters.

        Parameters
        ----------
        key : jax.Array
            Random key.
        nonlinear_samples : dict[str, jax.Array]
            Accepted nonlinear parameter samples.
        data : GaiaAstrometryData | RadialVelocityData
            Observational data.
        data_type : DataType
            Type of data being processed.
        n_linear_per_nonlinear : int, optional
            Number of linear samples per nonlinear sample. Default: 1.

        Returns
        -------
        linear_samples : jax.Array
            Linear parameter samples, shape depends on data type.
        """
        import numpyro.distributions as dist
        from jaxoplanet.core.kepler import kepler
        from numpyro_ext.distributions import MarginalizedLinear
        from unxt import ustrip

        from epochalypse.likelihood.astrometry import get_astrometry_design_matrix
        from epochalypse.likelihood.rv import get_rv_design_matrix

        n_samples = len(nonlinear_samples["log_period"])

        # Determine number of linear parameters based on data type
        if data_type == "rv":
            n_linear = 2  # K, v0
        elif data_type == "sb2":
            n_linear = 3  # K1, K2, v0
        elif data_type == "astrometry":
            n_linear = (
                6  # alpha_0, delta_0, mu_alpha, mu_delta, parallax, semimajor_axis
            )
        else:  # combined
            n_linear = 8  # astrometry + RV params

        # If no samples accepted, return empty array with correct shape
        if n_samples == 0:
            return jnp.zeros((0, n_linear))

        linear_samples = []
        keys = jr.split(key, n_samples)

        for i in range(n_samples):
            ecc = nonlinear_samples["eccentricity"][i]
            period_day = 10.0 ** nonlinear_samples["log_period"][i]
            phase_peri = nonlinear_samples["phase_peri"][i]
            arg_peri = nonlinear_samples["arg_peri"][i]

            t_peri = phase_peri * period_day
            dt = ustrip("day", data.time) - t_peri

            # Compute true anomaly
            M = 2 * jnp.pi * dt / period_day
            sin_f, cos_f = kepler(M, ecc)

            if data_type == "rv":
                # RV case: 2 linear parameters (K, v0)
                design_matrix = get_rv_design_matrix(sin_f, cos_f, ecc, arg_peri)

                # Sample from conditional posterior
                linear_prior = dist.Normal(0.0, self.prior.linear_prior_scale)
                marg_dist = MarginalizedLinear(
                    design_matrix=design_matrix,
                    prior_distribution=linear_prior,
                    data_distribution=dist.Normal(0.0, ustrip("km/s", data.rv_err)),
                )

                posterior = marg_dist.conditional(ustrip("km/s", data.rv))
            else:
                # Astrometry case: 6 linear parameters
                cos_i = nonlinear_samples["cos_i"][i]
                lon_asc_node = nonlinear_samples["lon_asc_node"][i]

                design_matrix = get_astrometry_design_matrix(
                    data.time,
                    data.scan_angle,
                    data.parallax_factor,
                    sin_f,
                    cos_f,
                    data.t_ref,
                    cos_i,
                    arg_peri,
                    lon_asc_node,
                )

                # Sample from conditional posterior
                linear_prior = dist.Normal(0.0, self.prior.linear_prior_scale)
                marg_dist = MarginalizedLinear(
                    design_matrix=design_matrix,
                    prior_distribution=linear_prior,
                    data_distribution=dist.Normal(
                        0.0, ustrip("mas", data.al_position_err)
                    ),
                )

                posterior = marg_dist.conditional(ustrip("mas", data.al_position))

            sample = posterior.sample(keys[i], sample_shape=(n_linear_per_nonlinear,))
            linear_samples.append(sample)

        # Concatenate linear samples
        # Result shape: (n_samples * n_linear_per_nonlinear, n_linear)
        result = jnp.concatenate(linear_samples, axis=0)
        return result
