"""Parameter structs for likelihood functions.

Each likelihood class has a corresponding parameter struct. Structs are
equinox Modules and therefore JAX pytrees, so batching is simply::

    params_batch = RVParameters(period=..., eccentricity=..., ...)
    jax.vmap(likelihood.log_prob)(params_batch)

where the batch axis is automatically sliced by vmap.

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

Annotations use ``Batch*`` type aliases (e.g. ``BatchQTime``,
``BatchFloat``) which accept both scalar and batched arrays via the
``*batch`` shape wildcard.  The rejection sampler constructs parameter structs
with a leading batch axis; ``jax.vmap`` then slices each leaf to scalar.
"""

import dataclasses
from typing import Any, ClassVar, final

import equinox as eqx

from harv.custom_types import (
    DIMENSIONED_BATCH_TYPES,
    BatchFloat,
    BatchQAngle,
    BatchQAngularSpeed,
    BatchQLength,
    BatchQSpeed,
    BatchQTime,
)


class AbstractParameters(eqx.Module):
    """Abstract base for all parameter structs.

    Declares the 4 orbital fields shared by every concrete parameter class,
    and provides ``marginalize()`` / ``marginalized()`` for creating
    ``MarginalizedParameters`` wrappers.
    """

    linear_param_names: ClassVar[tuple[str, ...]] = ()
    nonlinear_param_names: ClassVar[tuple[str, ...]] = ()
    _dimensioned_param_names: ClassVar[tuple[str, ...]] = ()

    period: BatchQTime
    eccentricity: BatchFloat
    phase_peri: BatchFloat  # Fractional orbital phase at pericenter, ∈ [0, 1]
    arg_peri: BatchQAngle

    def __init_subclass__(cls) -> None:
        """Automatically compute nonlinear_param_names for each subclass.

        Note: equinox may not have processed the subclass fields yet when this
        hook fires, so ``dataclasses.fields(cls)`` may only contain parent
        fields.  To be safe, subclasses that add extra nonlinear fields should
        set ``nonlinear_param_names`` explicitly as a ClassVar.
        """
        # NOTE: This is safe to do at the moment because all parameters are defined as
        # plain annotations / dataclass fields. In the future, if we use eqx.field() and
        # any custom arguments, we would have to be more careful here since
        # dataclasses.dataclass() would not know how to handle that.
        dataclasses.dataclass(cls)
        cls.nonlinear_param_names = tuple(
            f.name
            for f in dataclasses.fields(cls)
            if f.name not in cls.linear_param_names and f.init
        )

        # Auto-detect which fields carry physical dimensions by checking
        # their annotation against DIMENSIONED_BATCH_TYPES.
        def _get_annotation(klass: type, name: str) -> Any:
            for base in klass.__mro__:
                if name in getattr(base, "__annotations__", {}):
                    return base.__annotations__[name]
            return None

        cls._dimensioned_param_names = tuple(
            f.name
            for f in dataclasses.fields(cls)
            if f.init and _get_annotation(cls, f.name) in DIMENSIONED_BATCH_TYPES
        )

    @classmethod
    def marginalized(
        cls,
        *names: str,
        **kwargs: Any,
    ) -> "MarginalizedParameters":
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
            values=kwargs,
            marginalized_names=tuple(names),
            source_cls=cls,
        )


# ---------------------------------------------------------------------------
# Concrete parameter classes (one per data type)
# ---------------------------------------------------------------------------


@final
class RVParameters(AbstractParameters):
    """Full parameter set for the RV likelihood.

    Includes both nonlinear orbital parameters, the linear RV parameters
    (semi-amplitude and systemic velocity).

    Parameters
    ----------
    period, eccentricity, phase_peri, arg_peri
        Nonlinear orbital parameters (inherited from ``AbstractParameters``).
    rv_semiamp : BatchQSpeed
        RV semi-amplitude.
    v_sys : BatchQSpeed
        Systemic velocity (for the reference instrument).
    """

    linear_param_names: ClassVar[tuple[str, ...]] = ("rv_semiamp", "v_sys")

    rv_semiamp: BatchQSpeed  # RV semi-amplitude
    v_sys: BatchQSpeed  # systemic velocity (reference instrument)


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

    cos_i: BatchFloat
    lon_asc_node: BatchQAngle
    ra0: BatchQAngle  # reference RA offset
    dec0: BatchQAngle  # reference Dec offset
    pmra: BatchQAngularSpeed  # proper motion in RA
    pmdec: BatchQAngularSpeed  # proper motion in Dec
    parallax: BatchQAngle  # parallax
    semi_major_axis: BatchQLength  # photocentric semi-major axis


@final
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
    values : dict[str, Any]
        Mapping from field name to value for every *non-marginalized* field.
        These are the pytree leaves that JAX traces through.
    marginalized_names : tuple[str, ...]
        Names of the linear parameters that have been marginalized out.
    source_cls : type[AbstractParameters]
        The full parameter class this was derived from.
    """

    values: dict[str, Any]
    marginalized_names: tuple[str, ...] = eqx.field(static=True)
    source_cls: type[AbstractParameters] = eqx.field(static=True)

    def __check_init__(self) -> None:
        for name in self.marginalized_names:
            if name not in self.source_cls.linear_param_names:
                raise ValueError(
                    f"Cannot marginalize {name}: not in "
                    f"{self.source_cls.__name__}.linear_param_names = "
                    f"{self.source_cls.linear_param_names}"
                )

        # TODO: validate that all of the nonlinear parameters are present in values
        for name in self.source_cls.nonlinear_param_names:
            if name not in self.values:
                raise ValueError(
                    f"Missing value for nonlinear parameter {name} "
                    f"in marginalized parameters. Expected keys: "
                    f"{self.source_cls.nonlinear_param_names}"
                )

    @property
    def nonlinear_param_names(self) -> tuple[str, ...]:
        """Names of the non-marginalized fields present in this wrapper."""
        return tuple(self.values.keys())

    def __getattr__(self, name: str) -> Any:
        # eqx.Module uses __getattr__ only as a fallback, so this won't
        # intercept normal Module attribute access (values, etc.).
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(  # noqa: B904
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
