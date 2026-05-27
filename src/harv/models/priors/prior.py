"""Prior distributions for rejection sampling of Keplerian orbits.

This module implements the HarvPrior class which manages prior distributions
for both nonlinear and linear parameters in the rejection sampling algorithm.

The prior is agnostic to data type - it simply holds distributions for any/all
parameters. The sampler validates which parameters are needed based on the data.
"""

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.random as jr

from harv.models._helpers import (
    LinearPriorDict,
    LinearPriorDist,
    PriorDist,
    _unwrap_dist,
)

if TYPE_CHECKING:
    from harv.models.parameterizations._base import AbstractParameterization

__all__ = ("HarvPrior",)


class HarvPrior(eqx.Module):
    """Prior distribution for rejection sampling of Keplerian orbits.

    This class encapsulates the prior distributions for both nonlinear and linear
    parameters. It is agnostic to data type - the sampler determines which parameters
    are required based on the provided data.

    We recommend using the "default" factory constructors (e.g. ``default_rv()``,
    ``default_gaia_astrometry()``, etc.), which set up sensible priors for common use
    cases.

    **Nonlinear parameterization:**

    Parameter names in ``nonlinear_priors`` must match the field names of the
    parameterization, for example, ``period``, ``eccentricity``, ``phase_peri``, etc.
    These parameters are sampled explicitly.

    See the options available in `harv.models.parameterizations`.

    **Default parameterizations:**

    Radial Velocity:
        - Nonlinear: ``period``, ``eccentricity``, ``phase_peri``, ``arg_peri``
        - Linear: ``rv_semiamp``, ``v_sys``

    Astrometry:
        - Nonlinear: ``period``, ``eccentricity``, ``phase_peri``, ``cos_i``,
          ``arg_peri``, ``lon_asc_node``
        - Linear params: ``ra0``, ``dec0``, ``pmra``, ``pmdec``, ``parallax``,
          ``semi_major_axis``

    Parameters
    ----------
    nonlinear_priors
        Mapping from parameter name to its prior distribution (a bare
        ``dist.Distribution`` for dimensionless parameters, or a
        :class:`harv.distributions.QuantityDistribution` wrapper for parameters with
        physical units).
    linear_priors
        Per-parameter priors for linear parameters. Each entry is classified:

        - ``dist.Normal`` or ``QD(Normal)`` -- Gaussian, can be analytically
          marginalized.
        - ``LinearPriorCallable`` -- called with nonlinear params to produce a Normal,
          can be marginalized.
        - ``dist.HalfNormal``, ``dist.Delta``, etc. -- non-Gaussian, sampled
          explicitly alongside nonlinear params.

        When using ``default_rv()`` with ``offsets``, the non-reference offset
        priors are automatically included as linear parameters.
    extension_priors
        Priors for extension parameters declared via ``extra_params()``.

    """

    nonlinear_priors: dict[str, PriorDist]
    linear_priors: LinearPriorDict

    # Priors for extension parameters (jitter, offsets, GP hyperparams, etc.).
    # Keys are the parameter names declared by the extension via extra_params().
    # Values are distributions (bare or QuantityDistribution).  These are not
    # validated here -- the sampler checks at run-time that every extension
    # parameter has a matching entry.
    extension_priors: dict[str, PriorDist] = eqx.field(default_factory=dict)

    def sample_nonlinear(self, key: jax.Array, n_samples: int) -> dict[str, Any]:
        """Sample nonlinear parameters from priors.

        Parameters
        ----------
        key
            Random key for sampling.
        n_samples
            Number of samples to draw.

        Returns
        -------
            Dictionary mapping each parameter name to an array of shape
            ``(n_samples,)``.  Values are bare JAX arrays regardless of
            whether the distribution is wrapped in ``QuantityDistribution``.

        Examples
        --------
        >>> import jax
        >>> from unxt import Q
        >>> from harv.samplers import HarvPrior
        >>> sorted(
        ...     HarvPrior.default_rv(
        ...         period_min=Q(2.0, "day"),
        ...         period_max=Q(1000.0, "day"),
        ...         sigma_K0=Q(30.0, "km/s"),
        ...         sigma_v0=Q(50.0, "km/s"),
        ...     ).sample_nonlinear(jax.random.key(0), 10).keys()
        ... )
        ['arg_peri', 'eccentricity', 'period', 'phase_peri']
        >>> HarvPrior.default_rv(
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... ).sample_nonlinear(jax.random.key(0), 10)["period"].shape
        (10,)
        """
        keys = jr.split(key, len(self.nonlinear_priors))
        return {
            name: _unwrap_dist(d).sample(k, (n_samples,))
            for (name, d), k in zip(self.nonlinear_priors.items(), keys, strict=True)
        }

    @classmethod
    def from_parameterization(
        cls,
        parameterization: "AbstractParameterization",
        **kwargs: PriorDist | LinearPriorDist | Any,
    ) -> "HarvPrior":
        """Build a default prior for any supported parameterization.

        Delegates to ``parameterization.default_prior(**kwargs)``.  Each concrete
        parameterization declares its own required scale arguments (e.g.
        ``sigma_K0`` for RV, ``sigma_a0`` for astrometry).

        Parameters
        ----------
        parameterization
            A concrete parameterization (e.g. :class:`StandardRV`,
            :class:`EcoswEsinwRV`, :class:`StandardGaiaAstrometry`,
            :class:`ThieleInnesGaiaAstrometry`).
        **kwargs
            Forwarded to the parameterization's ``default_prior`` method.

        Returns
        -------
        HarvPrior

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.models.parameterizations.rv import EcoswEsinwRV
        >>> from harv.samplers import HarvPrior
        >>> prior = HarvPrior.from_parameterization(
        ...     EcoswEsinwRV(),
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> sorted(prior.nonlinear_priors.keys())
        ['ecosw', 'esinw', 'period', 'phase_peri']
        """
        return parameterization.default_prior(**kwargs)
