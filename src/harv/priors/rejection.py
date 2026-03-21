"""Prior distributions for rejection sampling of Keplerian orbits.

This module implements the RejectionPrior class which manages prior distributions
for both nonlinear and linear parameters in the rejection sampling algorithm.

The prior is agnostic to data type - it simply holds distributions for any/all
parameters. The sampler validates which parameters are needed based on the data.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.random as jr
import numpyro.distributions as dist
import quaxed.numpy as jnp

__all__ = ["RejectionPrior"]


class RejectionPrior(eqx.Module):
    """Prior distribution for rejection sampling of Keplerian orbits.

    This class encapsulates the prior distributions for both nonlinear and linear
    parameters. It is agnostic to data type - the sampler determines which
    parameters are required based on the provided data.

    **Common Parameterizations:**

    **Astrometry:**
        - Nonlinear (6): log(P), e, phase_peri, cos(i), arg_peri, lon_asc_node
        - Linear (6): ra0, dec0, pmra, pmdec, parallax, a

    **Radial Velocity:**
        - Nonlinear (4): log(P), e, arg_peri, phase_peri
        - Linear (2): K, v0

    **Combined (astrometry + RV):**
        - Nonlinear (6): log(P), e, phase_peri, cos(i), arg_peri, lon_asc_node
        - Linear (8): ra0, dec0, pmra, pmdec, parallax, a, K, v0

    Parameters
    ----------
    log_period : dist.Distribution
        Prior on log10(period). Typically Uniform.
    eccentricity : dist.Distribution
        Prior on eccentricity. Typically Beta(0.867, 3.03) from Kipping 2013.
    phase_peri : dist.Distribution
        Prior on phase at pericenter (t_peri / period). Typically Uniform(0, 1).
    linear_prior : dist.MultivariateNormal
        Joint Gaussian prior over all linear parameters for this data type.
        Dimension must match the number of linear parameters for the data type
        (2 for RV, 6 for astrometry, 8 for combined).
    cos_i : dist.Distribution, optional
        Prior on cos(inclination). Typically Uniform(-1, 1).
        Required for astrometry, not used for RV-only.
    arg_peri : dist.Distribution, optional
        Prior on argument of pericenter. Typically Uniform(0, 2pi).
        Required for RV and combined data.
    lon_asc_node : dist.Distribution, optional
        Prior on longitude of ascending node. Typically Uniform(0, 2pi).
        Required for astrometry.
    offsets : dict[str, dist.Distribution | None], optional
        Multi-instrument offset priors. Keys are instrument names, values are
        priors (or None for reference instrument). For RV data only.
    """

    # Nonlinear parameter priors (required first)
    log_period: dist.Distribution
    eccentricity: dist.Distribution
    phase_peri: dist.Distribution

    # Linear parameter prior (required)
    linear_prior: dist.MultivariateNormal

    # Optional nonlinear priors (defaults after required)
    cos_i: dist.Distribution | None = None
    arg_peri: dist.Distribution | None = None
    lon_asc_node: dist.Distribution | None = None

    # Multi-instrument offsets (RV only, optional)
    offsets: dict[str, dist.Distribution | None] | None = None

    @classmethod
    def default_astrometry(
        cls,
        log_period_min: float = -1.0,
        log_period_max: float = 4.0,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale: float = 1000.0,
    ) -> "RejectionPrior":
        """Create default prior for astrometry-only data.

        Parameters
        ----------
        log_period_min : float
            Minimum log10(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log10(period/day). Default: 4.0 (10,000 days ~ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Isotropic 1-sigma scale for the astrometric linear parameters (mas).
            Default: 1000.0.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for astrometry data.
        """
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            cos_i=dist.Uniform(-1.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            lon_asc_node=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior=dist.MultivariateNormal(
                loc=jnp.zeros(6),
                covariance_matrix=linear_prior_scale**2 * jnp.eye(6),
            ),
        )

    @classmethod
    def default_rv(
        cls,
        log_period_min: float = -1.0,
        log_period_max: float = 4.0,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale: float = 100.0,
        offsets: dict[str, dist.Distribution | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for radial velocity data.

        Parameters
        ----------
        log_period_min : float
            Minimum log10(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log10(period/day). Default: 4.0 (10,000 days ~ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Isotropic 1-sigma scale for the RV linear parameters (km/s). Default: 100.0.
        offsets : dict[str, dist.Distribution | None], optional
            Multi-instrument offset priors. Keys are instrument names, values are
            offset priors (or None for reference instrument). Non-reference
            instruments must supply a ``dist.Normal`` prior; its ``loc`` and
            ``scale`` are incorporated into the joint linear prior as additional
            dimensions ``[K, v₀, δ₁, …, δₖ]``.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for RV data.
        """
        # Non-reference instruments are those with a non-None prior.
        non_ref = (
            [(k, v) for k, v in offsets.items() if v is not None] if offsets else []
        )
        if non_ref:
            # Build joint prior: [K, v0, δ_1, ..., δ_k]
            offset_locs = jnp.array([v.loc for _, v in non_ref])
            offset_scales = jnp.array([v.scale for _, v in non_ref])
            loc = jnp.concatenate([jnp.zeros(2), offset_locs])
            scales = jnp.concatenate([jnp.full(2, linear_prior_scale), offset_scales])
            linear_prior: dist.MultivariateNormal = dist.MultivariateNormal(
                loc=loc, covariance_matrix=jnp.diag(scales**2)
            )
        else:
            linear_prior = dist.MultivariateNormal(
                loc=jnp.zeros(2),
                covariance_matrix=linear_prior_scale**2 * jnp.eye(2),
            )
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior=linear_prior,
            offsets=offsets,
        )

    @classmethod
    def default_combined(
        cls,
        log_period_min: float = -1.0,
        log_period_max: float = 4.0,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale_astro: float = 1000.0,
        linear_prior_scale_rv: float = 100.0,
        offsets: dict[str, dist.Distribution | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for combined astrometry + RV data.

        The joint linear prior is block-diagonal: astrometric parameters (6) get
        ``linear_prior_scale_astro``, RV parameters (2) get ``linear_prior_scale_rv``.
        These should be set to physically appropriate scales — astrometry in mas, RV in
        km/s — since the two measurement types live on completely different scales.

        Parameters
        ----------
        log_period_min : float
            Minimum log10(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log10(period/day). Default: 4.0 (10,000 days ~ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale_astro : float
            Isotropic 1-sigma scale for the 6 astrometric linear parameters (mas).
            Default: 1000.0.
        linear_prior_scale_rv : float
            Isotropic 1-sigma scale for the 2 RV linear parameters (km/s).
            Default: 100.0.
        offsets : dict[str, dist.Distribution | None], optional
            Multi-instrument offset priors for RV data.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for combined data.
        """
        # Block-diagonal 8x8 covariance: [astro(6) | rv(2)]
        scales = jnp.concatenate(
            [jnp.full(6, linear_prior_scale_astro), jnp.full(2, linear_prior_scale_rv)]
        )
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            cos_i=dist.Uniform(-1.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            lon_asc_node=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior=dist.MultivariateNormal(
                loc=jnp.zeros(8),
                covariance_matrix=jnp.diag(scales**2),
            ),
            offsets=offsets,
        )

    @classmethod
    def default_sb2(
        cls,
        log_period_min: float = -1.0,
        log_period_max: float = 4.0,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale: float = 100.0,
        offsets: dict[str, dist.Distribution | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for SB2 (double-lined spectroscopic binary) systems.

        SB2 systems have separate RV measurements for both components, requiring
        K1 and K2 as linear parameters.

        Parameters
        ----------
        log_period_min : float
            Minimum log10(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log10(period/day). Default: 4.0 (10,000 days ~ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Isotropic 1-sigma scale for linear parameters (km/s). Default: 100.0.
        offsets : dict[str, dist.Distribution | None], optional
            Multi-instrument offset priors.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for SB2 data.
        """
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior=dist.MultivariateNormal(
                loc=jnp.zeros(3),
                covariance_matrix=linear_prior_scale**2 * jnp.eye(3),
            ),
            offsets=offsets,
        )

    def sample_nonlinear(self, key: jax.Array, n_samples: int) -> dict[str, Any]:
        """Sample nonlinear parameters from priors.

        Parameters
        ----------
        key : jax.Array
            Random key for sampling.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : dict[str, jax.Array]
            Dictionary of parameter samples (dimensionless arrays).
            Keys depend on which priors are defined:
            - "log_period", "eccentricity", "phase_peri" (always)
            - "cos_i" (if cos_i prior is set)
            - "arg_peri" (if arg_peri prior is set)
            - "lon_asc_node" (if lon_asc_node prior is set)
        """
        n_params = 3  # Always have: log_period, eccentricity, phase_peri
        if self.cos_i is not None:
            n_params += 1
        if self.arg_peri is not None:
            n_params += 1
        if self.lon_asc_node is not None:
            n_params += 1

        keys = jr.split(key, n_params)
        key_idx = 0

        samples: dict[str, jax.Array] = {}

        samples["log_period"] = self.log_period.sample(keys[key_idx], (n_samples,))
        key_idx += 1
        samples["eccentricity"] = self.eccentricity.sample(keys[key_idx], (n_samples,))
        key_idx += 1
        samples["phase_peri"] = self.phase_peri.sample(keys[key_idx], (n_samples,))
        key_idx += 1

        if self.cos_i is not None:
            samples["cos_i"] = self.cos_i.sample(keys[key_idx], (n_samples,))
            key_idx += 1

        if self.arg_peri is not None:
            samples["arg_peri"] = self.arg_peri.sample(keys[key_idx], (n_samples,))
            key_idx += 1

        if self.lon_asc_node is not None:
            samples["lon_asc_node"] = self.lon_asc_node.sample(
                keys[key_idx], (n_samples,)
            )
            key_idx += 1

        return samples

    @property
    def n_nonlinear(self) -> int:
        """Number of nonlinear parameters."""
        n = 3  # log_period, eccentricity, phase_peri
        if self.cos_i is not None:
            n += 1
        if self.arg_peri is not None:
            n += 1
        if self.lon_asc_node is not None:
            n += 1
        return n
