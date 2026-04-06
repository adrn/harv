"""Parameter structs for likelihood functions.

Each likelihood class has a corresponding parameter struct. Structs are
equinox Modules and therefore JAX pytrees, so batching is simply::

    jax.vmap(likelihood.log_prob)(params_batch)

Two levels of parameterization exist for each data type:

- **Orbital parameters** (nonlinear only): used with *marginalized* likelihoods,
  where linear parameters are analytically integrated out given a Gaussian prior.
- **Full parameters** (nonlinear + linear): used with full likelihoods where
  all parameters are specified explicitly.

Annotations use ``Batchable*`` type aliases (e.g. ``BatchableQTime``,
``BatchableFloat``) which accept both scalar and batched arrays via the
``*batch`` shape wildcard.  The rejection sampler constructs parameter structs
with a leading batch axis; ``jax.vmap`` then slices each leaf to scalar.
"""

from typing import ClassVar, final

import equinox as eqx

from harv.custom_types import (
    BatchableFloat,
    BatchableQAngle,
    BatchableQAngularSpeed,
    BatchableQLength,
    BatchableQSpeed,
    BatchableQTime,
)


class AbstractParameters(eqx.Module):
    """Abstract base for all parameter structs.

    Declares the 4 orbital fields shared by every concrete parameter class.
    """

    period: BatchableQTime
    eccentricity: BatchableFloat
    phase_peri: BatchableFloat
    arg_peri: BatchableFloat


@final
class RVMarginalizedParameters(AbstractParameters):
    """Nonlinear orbital parameters for the marginalized RV likelihood.

    The linear parameters (K, v0) are analytically marginalized out.
    """


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
class GaiaAstrometryMarginalizedParameters(AbstractParameters):
    """Nonlinear orbital parameters for the marginalized Gaia astrometry likelihood.

    Also used for combined astrometry + RV runs (the nonlinear parameter set is
    identical; the data type distinction is carried by the sampler/Samples, not here).

    The 6 linear astrometric parameters (ra0, dec0, pmra, pmdec, parallax, a) are
    analytically marginalized out.
    """

    cos_i: BatchableFloat
    lon_asc_node: BatchableFloat


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
