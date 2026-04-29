"""Abstract base for model parameterizations.

A *parameterization* declares the names, units, and roles (linear / nonlinear)
of model parameters and knows how to build the design matrix from data and
nonlinear parameter values.  Concrete subclasses implement :meth:`params` and
:meth:`design_matrix`; the linear/nonlinear split is derived automatically.
"""

__all__ = ("AbstractParameterization",)

from abc import abstractmethod

import equinox as eqx
import jax

from harv.models.extensions.base import ParamInfo


class AbstractParameterization(eqx.Module):
    """Abstract base for model parameterizations.

    Subclasses must implement :meth:`params` (returning all parameter descriptors).  The
    :meth:`nonlinear_params` and :meth:`linear_params` convenience methods are derived
    automatically.
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

    def linear_log_prior_correction(
        self, linear_map: dict[str, jax.Array]
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
