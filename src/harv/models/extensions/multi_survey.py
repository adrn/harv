"""Multi-survey offsets."""

__all__ = ("MultiSurveyOffset",)

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp

from harv.models.extensions.base import AbstractExtension, ParamInfo


@final
class MultiSurveyOffset(AbstractExtension):
    """Per-instrument additive offsets to account for instrumental systematics.

    The extension stores a pre-computed indicator matrix and appends it as extra columns
    to the design matrix. Each column corresponds to a non-reference instrument; the
    reference instrument has no explicit offset (absorbed by the systemic velocity or
    baseline parameter).

    Parameters
    ----------
    indicator_matrix : jax.Array, shape (n_obs, n_non_ref)
        Binary matrix: ``indicator_matrix[i, j] == 1`` iff observation *i* belongs to
        the *j*-th non-reference instrument.
    instrument_names : tuple of str
        Ordered names of the non-reference instruments (same column order as
        ``indicator_matrix``). Used as parameter names.
    obs_unit : str
        Physical unit string for the offset parameters (must match the observation unit
        of the model, e.g. ``"km/s"``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from harv.models.extensions import MultiSurveyOffset
    >>> indicator = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> ext = MultiSurveyOffset(indicator, ("espresso", "keck"), "km/s")
    >>> [p.name for p in ext.extra_params()]
    ['espresso', 'keck']
    """

    indicator_matrix: jax.Array
    instrument_names: tuple[str, ...] = eqx.field(static=True)
    obs_unit: str = eqx.field(static=True, default="")

    def __check_init__(self) -> None:
        if self.indicator_matrix.ndim != 2:
            msg = (
                f"indicator_matrix must be 2-d, got shape {self.indicator_matrix.shape}"
            )
            raise ValueError(msg)
        if self.indicator_matrix.shape[1] != len(self.instrument_names):
            msg = (
                f"indicator_matrix has {self.indicator_matrix.shape[1]} columns "
                f"but {len(self.instrument_names)} instrument names given"
            )
            raise ValueError(msg)

    def extra_params(self) -> tuple[ParamInfo, ...]:
        """Parameters introduced by this extension."""
        return tuple(
            ParamInfo(name, self.obs_unit, linear=True)
            for name in self.instrument_names
        )

    def modify_design_matrix(
        self,
        X: jax.Array,
        data: Any,  # noqa: ARG002
        nl_values: dict[str, Any],  # noqa: ARG002
    ) -> jax.Array:
        """Append indicator columns to the design matrix."""
        return jnp.concatenate([X, self.indicator_matrix], axis=-1)
