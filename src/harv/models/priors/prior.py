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
from unxt import Q

from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    LinearPriorDict,
    LinearPriorDist,
    PriorDist,
    _evaluate_nonlinear_log_prior,
    _needs_explicit_sampling,
    _unwrap_dist,
)

if TYPE_CHECKING:
    from harv.models.component import AbstractComponentModel
    from harv.models.joint import JointModel
    from harv.models.parameterizations._base import AbstractParameterization
    from harv.samplers.samples import Samples

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

    def sample(
        self,
        key: jax.Array,
        n_samples: int,
        *,
        model: "AbstractComponentModel | JointModel",
        return_logprobs: bool = False,
    ) -> "Samples":
        """Draw a complete prior sample for a given model.

        Unlike :meth:`sample_nonlinear`, this draws every parameter the rejection
        sampler would consume for ``model``:

        - base nonlinear params from :attr:`nonlinear_priors`
        - extension nonlinear params (e.g. ``jitter``, GP hypers) discovered by
          walking ``model.extensions``
        - any non-Gaussian linear params from :attr:`linear_priors` that cannot
          be analytically marginalized (e.g. ``HalfNormal``-prior parallax)

        The returned :class:`~harv.samplers.Samples` is the same container the
        rejection sampler produces for posteriors, with units restored from each
        :class:`~harv.distributions.QuantityDistribution`.  Its
        ``linear`` field is empty in the common Gaussian-linear case.
        ``ln_likelihood`` is always ``None`` (no data yet); ``ln_prior`` is
        populated when ``return_logprobs=True``, summing the nonlinear (base +
        extension) prior log-densities — matching the convention used by
        :meth:`RejectionSampler.run`.

        ``model`` is required because the set of extension and explicit-linear
        parameters depends on the extensions attached to the model.

        Parameters
        ----------
        key
            JAX random key.
        n_samples
            Number of prior draws.
        model
            Component or joint model template defining the extensions and
            linear-prior classification.
        return_logprobs
            If ``True``, populate ``Samples.ln_prior`` with the per-sample
            nonlinear-prior log-density.

        Returns
        -------
        Samples
            A container holding the full prior draw, suitable for passing to
            :meth:`RejectionSampler.run_with_samples` or for caching via
            :func:`harv.samplers.make_prior_cache`.

        Examples
        --------
        >>> import jax
        >>> from unxt import Q
        >>> from harv import HarvPrior, RVModel, StandardRV
        >>> prior = StandardRV().default_prior(
        ...     period_min=Q(2.0, "day"),
        ...     period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"),
        ...     sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> samples = prior.sample(jax.random.key(0), 100, model=RVModel())
        >>> samples.n_samples
        100
        >>> samples.data_type
        'RVModel'
        """
        # Local imports break the cycle: ``harv.samplers`` already imports
        # ``HarvPrior`` from this module.
        from harv.samplers._prior_resolution import (  # noqa: PLC0415
            effective_linear_prior_from_prior,
            nonlinear_extension_priors_from_model,
            validate_extension_priors,
        )
        from harv.samplers.samples import Samples  # noqa: PLC0415

        nonlinear_extension_priors, linear_extension_names = (
            nonlinear_extension_priors_from_model(self, model)
        )
        effective_linear_prior = effective_linear_prior_from_prior(self, model) or {}
        validate_extension_priors(self, model, effective_linear_prior)

        # 1. Base nonlinear orbital params (bare arrays).
        key, nl_key = jr.split(key)
        base_nonlinear: dict[str, jax.Array] = self.sample_nonlinear(nl_key, n_samples)

        # 2. Extension nonlinear params (jitter, GP hypers, ...).
        extension_nonlinear: dict[str, jax.Array] = {}
        for name, d in nonlinear_extension_priors.items():
            key, k = jr.split(key)
            extension_nonlinear[name] = _unwrap_dist(d).sample(k, (n_samples,))

        # 3. Explicit (non-Gaussian) linear params -- those that cannot be
        # analytically marginalized.  In the common Gaussian-linear case this
        # is empty.
        explicit_linear: dict[str, jax.Array] = {}
        for name, d in effective_linear_prior.items():
            if _needs_explicit_sampling(d):
                key, k = jr.split(key)
                explicit_linear[name] = _unwrap_dist(d).sample(k, (n_samples,))

        # Restore units onto nonlinear (base + extension) entries.
        all_nonlinear_priors: dict[str, PriorDist] = {
            **self.nonlinear_priors,
            **nonlinear_extension_priors,
        }
        nonlinear_q: dict[str, Q] = {}
        for name, value in {**base_nonlinear, **extension_nonlinear}.items():
            d = all_nonlinear_priors[name]
            unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
            nonlinear_q[name] = Q(value, unit)

        # Restore units onto explicit linear entries.
        linear_q: dict[str, Q] = {}
        for name, value in explicit_linear.items():
            d = effective_linear_prior[name]
            unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
            linear_q[name] = Q(value, unit)

        ln_prior: jax.Array | None = None
        if return_logprobs:
            # Match the rejection sampler's convention: ln_prior sums only the
            # nonlinear (base + extension) contributions; explicit-linear and
            # marginalized-linear priors contribute via ln_likelihood in the
            # posterior path.
            ln_prior = _evaluate_nonlinear_log_prior(
                all_nonlinear_priors,
                {**base_nonlinear, **extension_nonlinear},
            )

        return Samples(
            nonlinear=nonlinear_q,
            linear=linear_q,
            data_type=type(model).__name__,
            linear_extension_names=linear_extension_names,
            ln_prior=ln_prior,
            ln_likelihood=None,
        )

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
