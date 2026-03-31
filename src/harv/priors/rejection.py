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


def _or(
    override: dist.Distribution | None, default: dist.Distribution
) -> dist.Distribution:
    """Return *override* if provided, otherwise *default*."""
    return override if override is not None else default


class RejectionPrior(eqx.Module):
    """Prior distribution for rejection sampling of Keplerian orbits.

    This class encapsulates the prior distributions for both nonlinear and linear
    parameters. It is agnostic to data type - the sampler determines which
    parameters are required based on the provided data.

    **Nonlinear parameterization:**

    Parameter names in ``nonlinear_priors`` match the field names of the orbit
    parameter structs (``period``, ``eccentricity``, ``phase_peri``, ``arg_peri``,
    ``cos_i``, ``lon_asc_node``). Distributions are sampled directly and the
    resulting values are used as-is when constructing param structs.

    **Common parameterizations:**

    **Radial Velocity:**
        - Nonlinear keys: ``period``, ``eccentricity``, ``phase_peri``, ``arg_peri``
        - Linear params: K, v0

    **Astrometry:**
        - Nonlinear keys: ``period``, ``eccentricity``, ``phase_peri``, ``cos_i``,
          ``arg_peri``, ``lon_asc_node``
        - Linear params: ra0, dec0, pmra, pmdec, parallax, semi_major_axis

    **Combined (astrometry + RV):**
        - Nonlinear keys: same as astrometry
        - Linear params: ra0, dec0, pmra, pmdec, parallax, semi_major_axis, K, v0

    Parameters
    ----------
    nonlinear_priors : dict[str, dist.Distribution]
        Mapping from parameter name to its prior distribution. The sampler
        checks that this dict contains every field required by the chosen orbit
        param class.
    linear_prior : dist.MultivariateNormal
        Joint Gaussian prior over all linear parameters for this data type.
        Dimension must match the number of linear parameters for the data type
        (2 for RV, 6 for astrometry, 8 for combined).
    offsets : dict[str, dist.Normal | None], optional
        Multi-instrument offset priors. Keys are instrument names, values are
        ``dist.Normal`` priors (or ``None`` for the reference instrument).
        For RV data only.

    Examples
    --------
    >>> from harv.priors.rejection import RejectionPrior
    >>> prior = RejectionPrior.default_rv()
    >>> prior.n_nonlinear
    4
    """

    nonlinear_priors: dict[str, dist.Distribution]
    linear_prior: dist.MultivariateNormal | eqx.Module

    # Multi-instrument offsets (RV only, optional)
    offsets: dict[str, dist.Normal | None] | None = None

    @property
    def n_nonlinear(self) -> int:
        """Number of nonlinear parameters."""
        return len(self.nonlinear_priors)

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
            Dictionary mapping each parameter name to an array of shape
            ``(n_samples,)``.
        """
        keys = jr.split(key, len(self.nonlinear_priors))
        return {
            name: d.sample(k, (n_samples,))
            for (name, d), k in zip(self.nonlinear_priors.items(), keys, strict=True)
        }

    # ------------------------------------------------------------------
    # Default constructors
    # ------------------------------------------------------------------

    @classmethod
    def default_rv(
        cls,
        *,
        # convenience bounds for the default log-uniform period prior
        period_min: float = 0.1,
        period_max: float = 1e4,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale: float = 100.0,
        # per-parameter distribution overrides
        period: dist.Distribution | None = None,
        eccentricity: dist.Distribution | None = None,
        phase_peri: dist.Distribution | None = None,
        arg_peri: dist.Distribution | None = None,
        # linear prior override (skips scale convenience; also accepts callable)
        linear_prior: dist.MultivariateNormal | eqx.Module | None = None,
        offsets: dict[str, dist.Normal | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for radial velocity data.

        Parameters
        ----------
        period_min : float
            Lower bound for the log-uniform period prior (days). Default: 0.1.
        period_max : float
            Upper bound for the log-uniform period prior (days). Default: 1e4.
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Isotropic 1-sigma scale for the RV linear parameters (km/s). Default: 100.0.
        period : dist.Distribution, optional
            Override the period prior entirely.
        eccentricity : dist.Distribution, optional
            Override the eccentricity prior.
        phase_peri : dist.Distribution, optional
            Override the phase-at-periastron prior.
        arg_peri : dist.Distribution, optional
            Override the argument-of-periastron prior.
        linear_prior : dist.MultivariateNormal or eqx.Module, optional
            Override the full linear prior (skips ``linear_prior_scale``).
            May be a fixed ``dist.MultivariateNormal`` **or** a callable
            ``eqx.Module`` whose ``__call__(params) -> dist.MultivariateNormal``
            returns the prior as a function of the nonlinear orbital parameters.
            The callable form enables physically motivated priors such as a
            uniform companion-mass prior, where K depends on period and
            eccentricity::

                import equinox as eqx
                import numpyro.distributions as dist
                import jax.numpy as jnp
                from harv.likelihood._params import RVOrbitParameters
                from harv.priors.rejection import RejectionPrior
                from unxt import ustrip

                class MassBasedKPrior(eqx.Module):
                    m1_solar: float  # primary mass in solar units
                    K_scale: float = 100.0  # km/s

                    def __call__(
                        self, params: RVOrbitParameters
                    ) -> dist.MultivariateNormal:
                        # Rough K upper bound from Kepler's third law:
                        # K ∝ (2π G m₂ / P)^(1/3) (1-e²)^{-1/2}
                        # Use a broad Normal centred on zero with period-scaled width.
                        P_yr = ustrip("yr", params.period)
                        sigma_k = self.K_scale / P_yr ** (1 / 3)
                        loc = jnp.zeros(2)
                        cov = jnp.diag(jnp.array([sigma_k**2, self.K_scale**2]))
                        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

                prior = RejectionPrior.default_rv(
                    linear_prior=MassBasedKPrior(m1_solar=1.0)
                )

        offsets : dict[str, dist.Normal | None], optional
            Multi-instrument offset priors. Keys are instrument names, values are
            ``dist.Normal`` priors (or ``None`` for the reference instrument).
            Non-reference instruments' ``loc`` and ``scale`` are incorporated
            into the joint linear prior as additional dimensions
            ``[K, v₀, δ₁, …, δₖ]``.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for RV data.
        """
        nonlinear = {
            "period": _or(period, dist.LogUniform(period_min, period_max)),
            "eccentricity": _or(eccentricity, dist.Beta(ecc_alpha, ecc_beta)),
            "phase_peri": _or(phase_peri, dist.Uniform(0.0, 1.0)),
            "arg_peri": _or(arg_peri, dist.Uniform(0.0, 2.0 * jnp.pi)),
        }

        if linear_prior is None:
            non_ref = [v for v in (offsets or {}).values() if v is not None]
            if non_ref:
                # Build joint prior: [K, v0, δ_1, ..., δ_k]
                offset_locs = jnp.array([v.loc for v in non_ref])
                offset_scales = jnp.array([v.scale for v in non_ref])
                loc = jnp.concatenate([jnp.zeros(2), offset_locs])
                scales = jnp.concatenate(
                    [jnp.full(2, linear_prior_scale), offset_scales]
                )
                linear_prior = dist.MultivariateNormal(
                    loc=loc, covariance_matrix=jnp.diag(scales**2)
                )
            else:
                linear_prior = dist.MultivariateNormal(
                    loc=jnp.zeros(2),
                    covariance_matrix=linear_prior_scale**2 * jnp.eye(2),
                )

        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            offsets=offsets,
        )

    @classmethod
    def default_astrometry(
        cls,
        *,
        period_min: float = 0.1,
        period_max: float = 1e4,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale: float = 1000.0,
        # per-parameter distribution overrides
        period: dist.Distribution | None = None,
        eccentricity: dist.Distribution | None = None,
        phase_peri: dist.Distribution | None = None,
        cos_i: dist.Distribution | None = None,
        arg_peri: dist.Distribution | None = None,
        lon_asc_node: dist.Distribution | None = None,
        # linear prior override
        linear_prior: dist.MultivariateNormal | None = None,
    ) -> "RejectionPrior":
        """Create default prior for astrometry-only data.

        Parameters
        ----------
        period_min : float
            Lower bound for the log-uniform period prior (days). Default: 0.1.
        period_max : float
            Upper bound for the log-uniform period prior (days). Default: 1e4.
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Isotropic 1-sigma scale for the astrometric linear parameters (mas).
            Default: 1000.0.
        period : dist.Distribution, optional
            Override the period prior.
        eccentricity : dist.Distribution, optional
            Override the eccentricity prior.
        phase_peri : dist.Distribution, optional
            Override the phase-at-periastron prior.
        cos_i : dist.Distribution, optional
            Override the cos(inclination) prior.
        arg_peri : dist.Distribution, optional
            Override the argument-of-periastron prior.
        lon_asc_node : dist.Distribution, optional
            Override the longitude-of-ascending-node prior.
        linear_prior : dist.MultivariateNormal, optional
            Override the full linear prior (skips ``linear_prior_scale``).

        Returns
        -------
        prior : RejectionPrior
            Prior configured for astrometry data.
        """
        nonlinear = {
            "period": _or(period, dist.LogUniform(period_min, period_max)),
            "eccentricity": _or(eccentricity, dist.Beta(ecc_alpha, ecc_beta)),
            "phase_peri": _or(phase_peri, dist.Uniform(0.0, 1.0)),
            "cos_i": _or(cos_i, dist.Uniform(-1.0, 1.0)),
            "arg_peri": _or(arg_peri, dist.Uniform(0.0, 2.0 * jnp.pi)),
            "lon_asc_node": _or(lon_asc_node, dist.Uniform(0.0, 2.0 * jnp.pi)),
        }
        if linear_prior is None:
            linear_prior = dist.MultivariateNormal(
                loc=jnp.zeros(6),
                covariance_matrix=linear_prior_scale**2 * jnp.eye(6),
            )
        return cls(nonlinear_priors=nonlinear, linear_prior=linear_prior)

    @classmethod
    def default_combined(
        cls,
        *,
        period_min: float = 0.1,
        period_max: float = 1e4,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale_astro: float = 1000.0,
        linear_prior_scale_rv: float = 100.0,
        # per-parameter distribution overrides
        period: dist.Distribution | None = None,
        eccentricity: dist.Distribution | None = None,
        phase_peri: dist.Distribution | None = None,
        cos_i: dist.Distribution | None = None,
        arg_peri: dist.Distribution | None = None,
        lon_asc_node: dist.Distribution | None = None,
        # linear prior override
        linear_prior: dist.MultivariateNormal | None = None,
        offsets: dict[str, dist.Normal | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for combined astrometry + RV data.

        The joint linear prior is block-diagonal: astrometric parameters (6) get
        ``linear_prior_scale_astro``, RV parameters (2) get ``linear_prior_scale_rv``.
        These should be set to physically appropriate scales — astrometry in mas, RV in
        km/s — since the two measurement types live on completely different scales.

        Parameters
        ----------
        period_min : float
            Lower bound for the log-uniform period prior (days). Default: 0.1.
        period_max : float
            Upper bound for the log-uniform period prior (days). Default: 1e4.
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
        period : dist.Distribution, optional
            Override the period prior.
        eccentricity : dist.Distribution, optional
            Override the eccentricity prior.
        phase_peri : dist.Distribution, optional
            Override the phase-at-periastron prior.
        cos_i : dist.Distribution, optional
            Override the cos(inclination) prior.
        arg_peri : dist.Distribution, optional
            Override the argument-of-periastron prior.
        lon_asc_node : dist.Distribution, optional
            Override the longitude-of-ascending-node prior.
        linear_prior : dist.MultivariateNormal, optional
            Override the full linear prior (skips scale convenience args).
        offsets : dict[str, dist.Normal | None], optional
            Multi-instrument offset priors for RV data.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for combined data.
        """
        nonlinear = {
            "period": _or(period, dist.LogUniform(period_min, period_max)),
            "eccentricity": _or(eccentricity, dist.Beta(ecc_alpha, ecc_beta)),
            "phase_peri": _or(phase_peri, dist.Uniform(0.0, 1.0)),
            "cos_i": _or(cos_i, dist.Uniform(-1.0, 1.0)),
            "arg_peri": _or(arg_peri, dist.Uniform(0.0, 2.0 * jnp.pi)),
            "lon_asc_node": _or(lon_asc_node, dist.Uniform(0.0, 2.0 * jnp.pi)),
        }
        if offsets is not None:
            msg = (
                "Combined astrometry + multi-survey RV (with per-instrument offsets) "
                "is not yet implemented. Pass offsets=None or use default_rv() for "
                "multi-survey RV without astrometry. "
                "See docs/spec.md §'Combined astrometry + multi-survey RV' for the "
                "planned design."
            )
            raise NotImplementedError(msg)
        if linear_prior is None:
            # Block-diagonal 8x8 covariance: [astro(6) | rv(2)]
            scales = jnp.concatenate(
                [
                    jnp.full(6, linear_prior_scale_astro),
                    jnp.full(2, linear_prior_scale_rv),
                ]
            )
            linear_prior = dist.MultivariateNormal(
                loc=jnp.zeros(8),
                covariance_matrix=jnp.diag(scales**2),
            )
        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
        )

    @classmethod
    def default_sb2(
        cls,
        *,
        period_min: float = 0.1,
        period_max: float = 1e4,
        ecc_alpha: float = 0.867,
        ecc_beta: float = 3.03,
        linear_prior_scale: float = 100.0,
        # per-parameter distribution overrides
        period: dist.Distribution | None = None,
        eccentricity: dist.Distribution | None = None,
        phase_peri: dist.Distribution | None = None,
        arg_peri: dist.Distribution | None = None,
        # linear prior override
        linear_prior: dist.MultivariateNormal | None = None,
        offsets: dict[str, dist.Normal | None] | None = None,
    ) -> "RejectionPrior":
        """Create default prior for SB2 (double-lined spectroscopic binary) systems.

        SB2 systems have separate RV measurements for both components, requiring
        K1 and K2 as linear parameters.

        Parameters
        ----------
        period_min : float
            Lower bound for the log-uniform period prior (days). Default: 0.1.
        period_max : float
            Upper bound for the log-uniform period prior (days). Default: 1e4.
        ecc_alpha : float
            Alpha parameter for Beta eccentricity prior. Default: 0.867 (Kipping 2013).
        ecc_beta : float
            Beta parameter for Beta eccentricity prior. Default: 3.03 (Kipping 2013).
        linear_prior_scale : float
            Isotropic 1-sigma scale for linear parameters (km/s). Default: 100.0.
        period : dist.Distribution, optional
            Override the period prior.
        eccentricity : dist.Distribution, optional
            Override the eccentricity prior.
        phase_peri : dist.Distribution, optional
            Override the phase-at-periastron prior.
        arg_peri : dist.Distribution, optional
            Override the argument-of-periastron prior.
        linear_prior : dist.MultivariateNormal, optional
            Override the full linear prior (skips ``linear_prior_scale``).
        offsets : dict[str, dist.Normal | None], optional
            Multi-instrument offset priors.

        Returns
        -------
        prior : RejectionPrior
            Prior configured for SB2 data.
        """
        nonlinear = {
            "period": _or(period, dist.LogUniform(period_min, period_max)),
            "eccentricity": _or(eccentricity, dist.Beta(ecc_alpha, ecc_beta)),
            "phase_peri": _or(phase_peri, dist.Uniform(0.0, 1.0)),
            "arg_peri": _or(arg_peri, dist.Uniform(0.0, 2.0 * jnp.pi)),
        }
        if linear_prior is None:
            linear_prior = dist.MultivariateNormal(
                loc=jnp.zeros(3),
                covariance_matrix=linear_prior_scale**2 * jnp.eye(3),
            )
        return cls(
            nonlinear_priors=nonlinear,
            linear_prior=linear_prior,
            offsets=offsets,
        )
