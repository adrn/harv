"""Abstract base for model parameterizations.

A *parameterization* declares the names, units, and roles (linear / nonlinear)
of model parameters and knows how to build the design matrix from data and
nonlinear parameter values.  Concrete subclasses implement :meth:`params` and
:meth:`design_matrix`; the linear/nonlinear split is derived automatically.
"""

__all__ = ("AbstractParameterization",)

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax

from harv.models.extensions.base import ParamInfo

if TYPE_CHECKING:
    from harv.models.priors import HarvPrior


class AbstractParameterization(eqx.Module):
    """Abstract base for model parameterizations.

    Subclasses must implement :meth:`params` (returning all parameter descriptors).  The
    :meth:`nonlinear_params` and :meth:`linear_params` convenience methods are derived
    automatically.

    Concrete parameterizations are also expected to override :meth:`default_prior`
    with a type-narrow signature describing their required scale arguments.
    """

    @abstractmethod
    def params(self) -> tuple[ParamInfo, ...]:
        """All parameters declared by this parameterization (nonlinear first)."""
        ...

    def nonlinear_params(self) -> tuple[ParamInfo, ...]:
        """Return only the nonlinear parameters."""
        return tuple(p for p in self.params() if not p.linear)

    def linear_params(self) -> tuple[ParamInfo, ...]:
        """Return only the linear parameters."""
        return tuple(p for p in self.params() if p.linear)

    def default_prior(self, **kwargs: Any) -> "HarvPrior":
        """Build a :class:`~harv.samplers.HarvPrior` with sensible defaults.

        Each concrete parameterization overrides this with its own type-narrow
        signature for the required scale arguments (e.g. ``sigma_K0`` for RV,
        ``sigma_a0`` for astrometry).  The base implementation raises
        :class:`NotImplementedError`; ``default_prior`` is *not* declared
        ``@abstractmethod`` because subclass signatures legitimately differ.

        Parameters
        ----------
        **kwargs
            Parameterization-specific keyword arguments (scale arguments and
            per-parameter prior overrides).

        Returns
        -------
        HarvPrior
            A prior whose ``nonlinear_priors`` and ``linear_prior`` entries
            match the names declared by ``self.params()``.
        """
        msg = (
            f"{type(self).__name__} does not implement default_prior(...). "
            "Concrete parameterizations should override this method."
        )
        raise NotImplementedError(msg)

    def derived_eccentricity(self, nl_values: dict[str, Any]) -> Any | None:  # noqa: ARG002
        """Eccentricity implied by this parameterization, if it is not a parameter.

        ``LinearPriorCallable`` implementations such as
        :class:`~harv.models.priors.custom_priors.PeriodDependentKPrior` are written
        against the standard parameter names, so a parameterization that encodes
        eccentricity indirectly must say how to recover it or those priors cannot be
        evaluated at all.

        Returns ``None`` by default, which covers both parameterizations that carry
        ``eccentricity`` directly as a nonlinear parameter (nothing to derive) and the
        Kepler-free Fourier bases (no eccentricity exists). Override when the value is
        derivable but absent -- see
        :class:`~harv.models.parameterizations.rv.EcoswEsinwRV`.

        Parameters
        ----------
        nl_values
            Nonlinear parameter values keyed by bare parameter name.

        Returns
        -------
            The derived eccentricity, or ``None`` when this parameterization does not
            imply one.
        """
        return None

    def linear_log_prior_correction(
        self,
        linear_map: dict[str, jax.Array],  # noqa: ARG002
    ) -> jax.Array | None:
        """Optional log-prior correction added to the marginal log-likelihood.

        Called with the conditional-mean linear-parameter values (unit-stripped,
        in the model's observation units) after Gaussian marginalization.  Returns
        ``None`` (no correction) by default.

        Non-trivial only when the parameterization's linear parameters are not the
        natural physical parameters — for example, Thiele-Innes constants
        ``(A, B, F, G)`` instead of the Campbell elements
        ``(a_0, ω, Ω, cos i)``.  In that case the Jacobian of the change of
        variables must be applied to recover the correct posterior when priors
        are specified in the physical (Campbell) space.

        Parameters
        ----------
        linear_map : dict[str, jax.Array]
            Conditional-mean values of the marginalized linear parameters,
            keyed by parameter name.

        Returns
        -------
        jax.Array or None
            Scalar log-correction to add to the marginal log-likelihood, or
            ``None`` if no correction is needed.
        """
        return None
