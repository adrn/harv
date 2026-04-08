r"""Likelihood functions for Gaia epoch astrometry data.

This module implements the unified :class:`GaiaAstrometryLikelihood` for Gaia
along-scan astrometry.  The class supports two evaluation modes via the same
``log_prob`` interface:

1. **Marginalized** (``params`` is :class:`MarginalizedParameters`,
   ``linear_prior`` provided): analytically marginalizes over some or all of
   the 6 linear astrometric parameters (ra0, dec0, pmra, pmdec, parallax, a)
   given a Gaussian prior.  Supports partial marginalization via
   ``params.marginalized_names``.

2. **Explicit** (``params`` is :class:`GaiaAstrometryParameters`,
   ``linear_prior`` is ``None``): evaluates the Gaussian data log-likelihood
   directly at the provided linear parameter values.

For the marginalized model, the astrometric model is:

.. math::

    y_\mathrm{AL} &= \alpha_0 \cos\psi + \delta_0 \sin\psi \\
        &+ (\mu_\alpha \cos\psi + \mu_\delta \sin\psi) \, dt \\
        &+ \varpi \, H_\varpi(t) \\
        &+ a \, [(A \sin\psi + B \cos\psi) \cos f
        + (F \sin\psi + G \cos\psi) \sin f]

where :math:`A, B, F, G` are Thiele-Innes constants and :math:`f` is the
true anomaly.
"""

from typing import Any, cast

import jax
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Quantity, ustrip
from unxt.quantity import AllowValue

from harv.data import GaiaAstrometryData
from harv.kepler.orbits import thiele_innes_ABFG
from harv.likelihood.base import AbstractLikelihood
from harv.likelihood.helpers import (
    LinearPriorCallable,
    _resolve_linear_prior,
    _solve_kepler,
)
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    MarginalizedParameters,
)

__all__ = ("GaiaAstrometryLikelihood",)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_design_matrix_gaia_ast(
    data: GaiaAstrometryData,
    params: MarginalizedParameters | GaiaAstrometryParameters,
    sin_f: jax.Array,
    cos_f: jax.Array,
) -> jax.Array:
    """Build the (n_obs, 6) Gaia along-scan design matrix.

    Columns: [ra0, dec0, pmra, pmdec, parallax, a].
    See Appendix A of https://arxiv.org/abs/2206.05726.
    """
    dt_yr = ustrip("yr", data.time - data.t_ref)
    scan_angle_rad = ustrip("rad", data.scan_angle)
    cos_psi = jnp.cos(scan_angle_rad)
    sin_psi = jnp.sin(scan_angle_rad)

    _parallax_factor = ustrip(AllowValue, "", data.parallax_factor)
    _cos_i = ustrip(AllowValue, "", params.cos_i)
    _arg_peri = ustrip(AllowValue, "", params.arg_peri)
    _lon_asc_node = ustrip(AllowValue, "", params.lon_asc_node)

    # Thiele-Innes constants (unit, i.e. semi-major axis = 1)
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(_arg_peri),
        jnp.sin(_arg_peri),
        jnp.cos(_lon_asc_node),
        jnp.sin(_lon_asc_node),
        _cos_i,
    )

    semimaj_term = (A * sin_psi + B * cos_psi) * cos_f + (
        F * sin_psi + G * cos_psi
    ) * sin_f

    return jnp.stack(
        [
            cos_psi,
            sin_psi,
            cos_psi * dt_yr,
            sin_psi * dt_yr,
            _parallax_factor,
            semimaj_term,
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Unified Gaia astrometry likelihood
# ---------------------------------------------------------------------------


class GaiaAstrometryLikelihood(
    AbstractLikelihood[MarginalizedParameters | GaiaAstrometryParameters],
):
    """Unified Gaia astrometry likelihood.

    Supports marginalized and explicit evaluation.

    When ``linear_prior`` is provided and ``params`` is a
    :class:`MarginalizedParameters` instance, the likelihood analytically
    marginalizes over the linear parameters.

    When ``linear_prior`` is ``None``, ``params`` must be a full
    :class:`GaiaAstrometryParameters` and the likelihood is evaluated
    explicitly.

    Parameters
    ----------
    data : GaiaAstrometryData
        Gaia epoch astrometry observations.
    linear_prior : dist.MultivariateNormal or LinearPriorCallable or None
        Gaussian prior over the marginalized linear parameters.  ``None``
        for explicit evaluation.

    Examples
    --------
    Marginalized::

        lik = GaiaAstrometryLikelihood(data=gaia_data, linear_prior=prior)
        log_liks = jax.jit(jax.vmap(lik.log_prob))(params_batch)

    Explicit::

        lik = GaiaAstrometryLikelihood(data=gaia_data)
        log_liks = jax.jit(jax.vmap(lik.log_prob))(full_params_batch)
    """

    data: GaiaAstrometryData
    linear_prior: dist.MultivariateNormal | LinearPriorCallable | None = None

    @property
    def linear_names(self) -> tuple[str, ...]:
        """All linear parameter names in design-matrix column order."""
        return GaiaAstrometryParameters.linear_param_names

    @property
    def linear_units(self) -> tuple[str, ...]:
        """Unit string for each linear parameter.

        Positions and parallax/semi-major-axis are in ``"mas"``; proper
        motions are in ``"mas/yr"`` (matching the design-matrix construction
        where proper-motion columns include ``dt_yr``).
        """
        return ("mas", "mas", "mas/yr", "mas/yr", "mas", "mas")

    def design_matrix(
        self, params: MarginalizedParameters | GaiaAstrometryParameters
    ) -> jax.Array:
        """Build the (n_obs, 6) design matrix for the given parameters."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        return _get_design_matrix_gaia_ast(self.data, params, sin_f, cos_f)

    def log_prob(
        self, params: MarginalizedParameters | GaiaAstrometryParameters
    ) -> jax.Array:
        """Compute the log-likelihood for a single parameter sample.

        Dispatches to marginalized or explicit evaluation based on the
        presence of ``linear_prior``.
        """
        if self.linear_prior is None:
            return self._log_prob_explicit(cast("GaiaAstrometryParameters", params))
        return self._log_prob_marginalized(cast("MarginalizedParameters", params))

    def sample_conditional_linear(
        self, params: MarginalizedParameters, key: jax.Array
    ) -> dict[str, Quantity[Any]]:
        """Sample linear parameters from the conditional posterior.

        Builds a ``MarginalizedLinear`` from the design matrix, the resolved
        linear prior, and the data errors, then draws one sample from the
        posterior conditioned on the observed data.

        Parameters
        ----------
        params : MarginalizedParameters
            Nonlinear orbital parameters (period, eccentricity, etc.).
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        dict[str, Quantity]
            Sampled linear parameters keyed by name following
            ``GaiaAstrometryParameters.linear_param_names``.  Units reflect
            the natural units of each parameter given the design matrix:
            positions/parallax/semi-major-axis in ``"mas"``, proper motions
            in ``"mas/yr"``.
        """
        dm = self.design_matrix(params)
        lp = _resolve_linear_prior(self.linear_prior, params)
        y_obs = ustrip("mas", self.data.al_position)
        y_err = ustrip("mas", self.data.al_position_err)
        marg = MarginalizedLinear(
            design_matrix=dm,
            prior_distribution=lp,
            data_distribution=dist.Normal(0.0, y_err),
        )
        sample = marg.conditional(y_obs).sample(key)

        # Units match how the design matrix is constructed: positional cols are
        # dimensionless so the sample is in mas; proper-motion cols include dt_yr
        # (years) so the sample is in mas/yr.
        return {
            name: Quantity(sample[i], unit)
            for i, (name, unit) in enumerate(
                zip(self.linear_names, self.linear_units, strict=True)
            )
        }

    # -- private helpers ----------------------------------------------------

    def _log_prob_marginalized(self, params: MarginalizedParameters) -> jax.Array:
        """Marginalized log-likelihood."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        X = _get_design_matrix_gaia_ast(self.data, params, sin_f, cos_f)
        y_obs = jnp.asarray(ustrip("mas", self.data.al_position))
        y_err = jnp.asarray(ustrip("mas", self.data.al_position_err))
        return self._marginalize_partial(
            params,
            X,
            y_obs,
            y_err,
            self.linear_names,
            self.linear_units,
            cast("dist.MultivariateNormal", self.linear_prior),
        )

    def _log_prob_explicit(self, params: GaiaAstrometryParameters) -> jax.Array:
        """Explicit log-likelihood with all parameters specified."""
        sin_f, cos_f = _solve_kepler(self.data, params)
        design_matrix = _get_design_matrix_gaia_ast(self.data, params, sin_f, cos_f)

        linear_params = jnp.array(
            [
                params.ra0,
                params.dec0,
                params.pmra,
                params.pmdec,
                params.parallax,
                params.semi_major_axis,
            ]
        )
        y_pred = design_matrix @ linear_params
        y_obs = ustrip("mas", self.data.al_position)
        y_err = ustrip("mas", self.data.al_position_err)

        return dist.Normal(y_pred, y_err).log_prob(y_obs).sum()
