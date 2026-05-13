"""Parameter metadata and the abstract Extension base class.

This module defines:

- :class:`ParamInfo`: a frozen description of a single model parameter (name,
  unit / dimension (e.g., "km/s"), and role in a model (i.e., nonlinear vs linear)).
- :class:`AbstractExtension`: the abstract base that all model extensions must
  subclass.  Extensions can add parameters, modify the design matrix (e.g., polynomial
  trends, or instrumental offsets), and/or modify the data covariance (e.g., to add a
  Gaussian Process noise model).
"""

__all__ = ("AbstractExtension", "ParamInfo")

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax

from harv.data import AbstractData


class ParamInfo(eqx.Module):
    """Immutable descriptor for a single model parameter.

    Parameters
    ----------
    name
        Parameter name.  Must not contain ``"."`` (reserved for
        :class:`~harv.models.joint.JointModel` tied-parameter paths).
    unit
        Unit string (e.g. ``"day"``, ``"km/s"``, ``""`` for dimensionless).
    linear
        Whether the parameter enters the model linearly (default ``False``).

    Examples
    --------
    >>> from harv.models.extensions.base import ParamInfo
    >>> p = ParamInfo("period", "time")
    >>> p.name
    'period'
    >>> p.linear
    False
    """

    name: str
    unit: str
    linear: bool = False

    def __post_init__(self) -> None:
        if "." in self.name:
            msg = (
                f"Parameter name {self.name!r} must not contain '.'; "
                "dots are reserved for JointModel tied-parameter paths."
            )
            raise ValueError(msg)


class AbstractExtension(eqx.Module):
    """Abstract base class for model extensions (jitter, trends, offsets, GP, ...).

    An extension can declare extra parameters (nonlinear and/or linear), modify the
    design matrix (e.g., append trend or offset columns), and/or modify the data
    covariance matrix (e.g., add jitter or a GP kernel).

    Extensions are composed onto a
    :class:`~harv.models.component.AbstractComponentModel` at construction time via the
    ``extensions`` field.  The model calls each hook in the order the extensions are
    listed.

    All extensions receive the raw (unit-stripped) JAX arrays.  Unit handling is the
    model's responsibility.

    Subclasses must implement :meth:`extra_params`. The design-matrix and
    covariance hooks have default no-op implementations that return their input
    unchanged.

    Plot-specific behavior is handled privately by plotting helpers rather than
    through this base extension API.
    """

    @abstractmethod
    def extra_params(self) -> tuple[ParamInfo, ...]:
        """Parameters introduced by this extension.

        Returns
        -------
        tuple of ParamInfo
            May include both nonlinear and linear parameters.
        """
        ...

    def modify_design_matrix(
        self,
        X: jax.Array,
        data: AbstractData | None,  # noqa: ARG002
        nl_values: dict[str, Any],  # noqa: ARG002
    ) -> jax.Array:
        """Optionally append columns to the design matrix.

        Parameters
        ----------
        X
            Current design matrix (base + earlier extensions).
        data
            Observation data (unit-stripped times, etc. accessed via helpers).
        nl_values
            Current nonlinear parameter values (unit-stripped scalars).

        Returns
        -------
        jax.Array, shape (n_obs, n_cols + n_extra)
            Updated design matrix.  Return ``X`` unchanged if not applicable.
        """
        return X

    def modify_covariance(
        self,
        cov: jax.Array,
        data: AbstractData | None,  # noqa: ARG002
        nl_values: dict[str, Any],  # noqa: ARG002
    ) -> jax.Array:
        """Optionally modify the data covariance matrix.

        Parameters
        ----------
        cov
            Current covariance.  Diagonal (1-d) when only measurement errors
            are present; full (2-d) after a GP extension adds off-diagonal
            structure. shape (n_obs,) or (n_obs, n_obs)
        data
            Observation data.
        nl_values
            Current nonlinear parameter values (unit-stripped scalars).

        Returns
        -------
        cov
            Updated covariance (same or promoted shape).  Return ``cov``
            unchanged if not applicable.
        """
        return cov
