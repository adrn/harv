"""Prior distributions for rejection sampling of Keplerian orbits.

This module implements the RejectionPrior class which manages prior distributions
for both nonlinear and linear parameters in the rejection sampling algorithm.

The prior is agnostic to data type - it simply holds distributions for any/all
parameters. The sampler validates which parameters are needed based on the data.
"""

from __future__ import annotations

import equinox as eqx
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
        - Linear (2): K, v₀

    **Combined (astrometry + RV):**
        - Nonlinear (6): log(P), e, phase_peri, cos(i), arg_peri, lon_asc_node
        - Linear (8): ra0, dec0, pmra, pmdec, parallax, a, K, v0

    **SB2 (double-lined spectroscopic binary):**
        - Nonlinear (4): log(P), e, arg_peri, phase_peri
        - Linear (3): K₁, K₂, v₀

    Parameters
    ----------
    log_period : dist.Distribution
        Prior on log₁₀(period). Typically Uniform.
    eccentricity : dist.Distribution
        Prior on eccentricity. Typically Beta(0.867, 3.03) from Kipping 2013.
    phase_peri : dist.Distribution
        Prior on phase at pericenter (t_peri / period). Typically Uniform(0, 1).
    cos_i : dist.Distribution, optional
        Prior on cos(inclination). Typically Uniform(-1, 1).
        Required for astrometry, not used for RV-only.
    arg_peri : dist.Distribution, optional
        Prior on argument of pericenter. Typically Uniform(0, 2π).
        Required for RV and combined data.
    lon_asc_node : dist.Distribution, optional
        Prior on longitude of ascending node. Typically Uniform(0, 2π).
        Required for astrometry.
    linear_prior_scale : float
        Scale parameter for Gaussian prior on linear parameters.
        Default: 1000.0 for astrometry (mas scale), 100.0 for RV (km/s scale).
    offsets : dict[str, dist.Distribution | None], optional
        Multi-instrument offset priors. Keys are instrument names, values are
        priors (or None for reference instrument). For RV data only.
    """

    # Nonlinear parameter priors (required first)
    log_period: dist.Distribution
    eccentricity: dist.Distribution
    phase_peri: dist.Distribution

    # Linear parameter prior (required)
    linear_prior_scale: float

    # Optional nonlinear priors (defaults after required)
    cos_i: dist.Distribution | None = None
    arg_peri: dist.Distribution | None = None
    lon_asc_node: dist.Distribution | None = None

    # Multi-instrument offsets (RV only, optional)
    offsets: dict[str, dist.Distribution | None] | None = None

    def __check_init__(self) -> None:
        """Validate prior configuration."""
        # Validate linear prior scale
        if self.linear_prior_scale <= 0:
            msg = f"linear_prior_scale must be positive, got {self.linear_prior_scale}"
            raise ValueError(msg)

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
            Minimum log₁₀(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log₁₀(period/day). Default: 4.0 (10,000 days ≈ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Scale for Gaussian prior on linear parameters (mas). Default: 1000.0.

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
            linear_prior_scale=linear_prior_scale,
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
            Minimum log₁₀(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log₁₀(period/day). Default: 4.0 (10,000 days ≈ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Scale for Gaussian prior on linear parameters (km/s). Default: 100.0.
        offsets : dict[str, dist.Distribution | None], optional
            Multi-instrument offset priors. Keys are instrument names, values are
            offset priors (or None for reference instrument).

        Returns
        -------
        prior : RejectionPrior
            Prior configured for RV data.
        """
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior_scale=linear_prior_scale,
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
        offsets: dict[str, dist.Distribution | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for combined astrometry + RV data.

        Parameters
        ----------
        log_period_min : float
            Minimum log₁₀(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log₁₀(period/day). Default: 4.0 (10,000 days ≈ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale_astro : float
            Scale for linear parameters. Default: 1000.0.
        offsets : dict[str, dist.Distribution | None], optional
            Multi-instrument offset priors for RV data.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for combined data.
        """
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            cos_i=dist.Uniform(-1.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            lon_asc_node=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior_scale=linear_prior_scale_astro,
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
        K₁ and K₂ as linear parameters.

        Parameters
        ----------
        log_period_min : float
            Minimum log₁₀(period/day). Default: -1.0 (0.1 days).
        log_period_max : float
            Maximum log₁₀(period/day). Default: 4.0 (10,000 days ≈ 27 years).
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Scale for Gaussian prior on linear parameters (km/s). Default: 100.0.
        offsets : dict[str, dist.Distribution | None], optional
            Multi-instrument offset priors.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for SB2 data.
        """
        # SB2 uses same parameterization as RV, but with K1, K2 instead of K
        return cls(
            log_period=dist.Uniform(log_period_min, log_period_max),
            eccentricity=dist.Beta(ecc_alpha, ecc_beta),
            phase_peri=dist.Uniform(0.0, 1.0),
            arg_peri=dist.Uniform(0.0, 2.0 * jnp.pi),
            linear_prior_scale=linear_prior_scale,
            offsets=offsets,
        )

    def sample_nonlinear(
        self, key: jr.PRNGKey, n_samples: int
    ) -> dict[str, jnp.ndarray]:
        """Sample nonlinear parameters from priors.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key for sampling.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : dict[str, jnp.ndarray]
            Dictionary of parameter samples (dimensionless arrays).
            Keys depend on which priors are defined:
            - "log_period", "eccentricity", "phase_peri" (always)
            - "cos_i" (if cos_i prior is set)
            - "arg_peri" (if arg_peri prior is set)
            - "lon_asc_node" (if lon_asc_node prior is set)
        """
        # Split keys for each parameter
        n_params = 3  # Always have: log_period, eccentricity, phase_peri
        if self.cos_i is not None:
            n_params += 1
        if self.arg_peri is not None:
            n_params += 1
        if self.lon_asc_node is not None:
            n_params += 1

        keys = jr.split(key, n_params)
        key_idx = 0

        samples = {}

        # Always present
        samples["log_period"] = self.log_period.sample(keys[key_idx], (n_samples,))
        key_idx += 1
        samples["eccentricity"] = self.eccentricity.sample(keys[key_idx], (n_samples,))
        key_idx += 1
        samples["phase_peri"] = self.phase_peri.sample(keys[key_idx], (n_samples,))
        key_idx += 1

        # Optional parameters
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

    def get_linear_prior_distribution(self) -> dist.Distribution:
        """Get the linear parameter prior distribution.

        Returns
        -------
        prior : dist.Distribution
            Normal distribution with scale = linear_prior_scale.
        """
        return dist.Normal(0.0, self.linear_prior_scale)

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
