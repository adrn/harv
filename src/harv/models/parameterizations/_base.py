"""Abstract base for model parameterizations.

A *parameterization* declares the names, units, and roles (linear / nonlinear)
of model parameters and knows how to build the design matrix from data and
nonlinear parameter values.  Concrete subclasses implement :meth:`params` and
:meth:`design_matrix`; the linear/nonlinear split is derived automatically.
"""

__all__ = ("AbstractParameterization",)

from abc import abstractmethod

import equinox as eqx

from harv.models.extensions.base import ParamInfo


class AbstractParameterization(eqx.Module):
    """Abstract base for model parameterizations.

    Subclasses must implement :meth:`params` (returning all parameter
    descriptors) and :meth:`design_matrix`.  The :meth:`nonlinear_params`
    and :meth:`linear_params` convenience methods are derived automatically.
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
