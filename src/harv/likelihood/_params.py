"""Parameter structs for likelihood functions.

Each likelihood class has a corresponding parameter struct. Structs are
equinox Modules and therefore JAX pytrees, so batching is simply::

    jax.vmap(likelihood.log_prob)(params_batch)

Two levels of parameterization exist for each data type:

- **Full parameters** (nonlinear + linear): the canonical parameter classes.
  Each data type has one ``@final`` class (e.g. ``RVParameters``,
  ``GaiaAstrometryParameters``) that declares all fields and a
  ``linear_param_names`` class variable listing which fields enter the
  forward model linearly.

- **Marginalized parameters**: a ``MarginalizedParameters`` wrapper created
  on-the-fly via ``FullClass.marginalize()`` or ``FullClass.marginalized()``.
  The wrapper drops some or all linear-parameter fields from the pytree,
  telling the likelihood which parameters to analytically integrate out.

Annotations use ``Batchable*`` type aliases (e.g. ``BatchableQTime``,
``BatchableFloat``) which accept both scalar and batched arrays via the
``*batch`` shape wildcard.  The rejection sampler constructs parameter structs
with a leading batch axis; ``jax.vmap`` then slices each leaf to scalar.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, final

import equinox as eqx

from harv.custom_types import (
    BatchableFloat,
    BatchableQAngle,
    BatchableQAngularSpeed,
    BatchableQLength,
    BatchableQSpeed,
    BatchableQTime,
)


# ---------------------------------------------------------------------------
# MarginalizedParameters wrapper
# ---------------------------------------------------------------------------


class MarginalizedParameters(eqx.Module):
    """Wrapper that holds non-marginalized field values as a pytree.

    Created by ``AbstractParameters.marginalize()`` or
    ``AbstractParameters.marginalized()``.  The ``marginalized_names`` tuple
    records which linear parameters have been removed from the pytree and
    should be analytically integrated out by the likelihood.

    Field access is delegated to the internal ``_values`` dict, so
    ``params.period`` works as expected.

    Parameters
    ----------
    _values : dict[str, Any]
        Mapping from field name to value for every *non-marginalized* field.
        These are the pytree leaves that JAX traces through.
    marginalized_names : tuple[str, ...]
        Names of the linear parameters that have been marginalized out.
        Static (not a pytree leaf).
    _source_cls : type or None
        The full parameter class this was derived from, or ``None`` for
        combined (multi-source-class) wrappers.  Static.
    """

    _values: dict[str, Any]
    marginalized_names: tuple[str, ...] = eqx.field(static=True)
    _source_cls: type | None = eqx.field(static=True, default=None)

    @property
    def nonlinear_names(self) -> tuple[str, ...]:
        """Names of the non-marginalized fields present in this wrapper."""
        return tuple(self._values.keys())

    def __getattr__(self, name: str) -> Any:
        # eqx.Module uses __getattr__ only as a fallback, so this won't
        # intercept normal Module attribute access (_values, etc.).
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(  # noqa: B904
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractParameters(eqx.Module):
    """Abstract base for all parameter structs.

    Declares the 4 orbital fields shared by every concrete parameter class,
    and provides ``marginalize()`` / ``marginalized()`` for creating
    ``MarginalizedParameters`` wrappers.
    """

    linear_param_names: ClassVar[tuple[str, ...]] = ()

    period: BatchableQTime
    eccentricity: BatchableFloat
    phase_peri: BatchableFloat
    arg_peri: BatchableFloat

    # -- Marginalization helpers ------------------------------------------

    def marginalize(self, *names: str) -> MarginalizedParameters:
        """Return a ``MarginalizedParameters`` wrapper with *names* removed.

        Parameters
        ----------
        *names : str
            Names of linear parameters to marginalize.  Each must be in
            ``self.linear_param_names``.  If none are given, **all** linear
            parameters are marginalized.

        Returns
        -------
        MarginalizedParameters

        Raises
        ------
        ValueError
            If any name is not a recognised linear parameter.
        """
        cls = type(self)
        if not names:
            names = cls.linear_param_names

        bad = set(names) - set(cls.linear_param_names)
        if bad:
            msg = (
                f"Cannot marginalize {bad}: not in "
                f"{cls.__name__}.linear_param_names = {cls.linear_param_names}"
            )
            raise ValueError(msg)

        keep = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name not in names
        }
        return MarginalizedParameters(
            _values=keep,
            marginalized_names=tuple(names),
            _source_cls=cls,
        )

    @classmethod
    def marginalized(
        cls,
        *names: str,
        **kwargs: Any,
    ) -> MarginalizedParameters:
        """Construct a ``MarginalizedParameters`` directly from keyword args.

        This is the construction path used by the sampler, which does not have
        linear parameter values.

        Parameters
        ----------
        *names : str
            Names of linear parameters to marginalize.  If none are given,
            **all** ``cls.linear_param_names`` are marginalized.
        **kwargs
            Field values for the non-marginalized parameters (typically the
            nonlinear orbital parameters).

        Returns
        -------
        MarginalizedParameters
        """
        if not names:
            names = cls.linear_param_names

        bad = set(names) - set(cls.linear_param_names)
        if bad:
            msg = (
                f"Cannot marginalize {bad}: not in "
                f"{cls.__name__}.linear_param_names = {cls.linear_param_names}"
            )
            raise ValueError(msg)

        return MarginalizedParameters(
            _values=kwargs,
            marginalized_names=tuple(names),
            _source_cls=cls,
        )


# ---------------------------------------------------------------------------
# Concrete parameter classes (one per data type)
# ---------------------------------------------------------------------------


@final
class RVParameters(AbstractParameters):
    """Full parameter set for the RV likelihood.

    Includes both nonlinear orbital parameters and the linear RV parameters
    (semi-amplitude K and systemic velocity v0).
    """

    linear_param_names: ClassVar[tuple[str, ...]] = ("K", "v0")

    K: BatchableQSpeed  # RV semi-amplitude
    v0: BatchableQSpeed  # systemic velocity


@final
class GaiaAstrometryParameters(AbstractParameters):
    """Full parameter set for the Gaia astrometry likelihood.

    Includes both nonlinear orbital parameters and the 6 linear astrometric
    parameters (reference position, proper motion, parallax, semi-major axis).

    ``linear_param_names`` enumerates all linear parameters in this class, in
    the order expected by the design matrix.  Marginalized likelihood classes
    may marginalize over a subset of these; the rejection sampler reads this
    attribute to name the sampled columns consistently.
    """

    linear_param_names: ClassVar[tuple[str, ...]] = (
        "ra0",
        "dec0",
        "pmra",
        "pmdec",
        "parallax",
        "semi_major_axis",
    )

    cos_i: BatchableFloat
    lon_asc_node: BatchableFloat
    ra0: BatchableQAngle  # reference RA offset
    dec0: BatchableQAngle  # reference Dec offset
    pmra: BatchableQAngularSpeed  # proper motion in RA
    pmdec: BatchableQAngularSpeed  # proper motion in Dec
    parallax: BatchableQAngle  # parallax
    semi_major_axis: BatchableQLength  # photocentric semi-major axis
