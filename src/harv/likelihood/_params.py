"""Parameter structs for likelihood functions.

Each likelihood class has a corresponding parameter struct. Structs are
equinox Modules and therefore JAX pytrees, so batching is simply::

    jax.vmap(likelihood.log_prob)(params_batch)

Two levels of parameterization exist for each data type:

- **Orbital parameters** (nonlinear only): used with *marginalized* likelihoods,
  where linear parameters are analytically integrated out given a Gaussian prior.
- **Full parameters** (nonlinear + linear): used with full likelihoods where
  all parameters are specified explicitly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import equinox as eqx

if TYPE_CHECKING:
    import jax
    from unxt import Quantity

    from harv.custom_types import Angle, AngularSpeed, Length, Speed, Time


class AbstractBaseKeplerParameters(eqx.Module):
    """Base class for Keplerian orbital parameters shared across data types.

    This includes the 4 nonlinear orbital parameters common to both RV and
    astrometry: period, eccentricity, phase of periastron, and argument of
    periastron.
    """

    period: Quantity[Time]
    eccentricity: float | jax.Array
    phase_peri: float | jax.Array
    arg_peri: float | jax.Array


# ---------------------------------------------------------------------------
# Radial velocity
# ---------------------------------------------------------------------------


class AbstractRVParameters(AbstractBaseKeplerParameters):
    """Abstract base class for RV parameter structs."""

    data_type: ClassVar[str] = "rv"


class RVOrbitParameters(AbstractRVParameters):
    """Nonlinear orbital parameters for the marginalized RV likelihood.

    The linear parameters (K, v0) are analytically marginalized out.
    """


class RVFullParameters(AbstractRVParameters):
    """Full parameter set for the RV likelihood.

    Includes both nonlinear orbital parameters and the linear RV parameters
    (semi-amplitude K and systemic velocity v0).
    """

    linear_param_names: ClassVar[tuple[str, ...]] = ("K", "v0")

    K: Quantity[Speed]  # RV semi-amplitude
    v0: Quantity[Speed]  # systemic velocity


# ---------------------------------------------------------------------------
# Gaia epoch astrometry
# ---------------------------------------------------------------------------


class AbstractGaiaAstrometryParameters(AbstractBaseKeplerParameters):
    """Abstract base class for Gaia astrometry parameter structs."""

    data_type: ClassVar[str] = "astrometry"

    cos_i: float | jax.Array
    lon_asc_node: float | jax.Array


class GaiaAstrometryOrbitParameters(AbstractGaiaAstrometryParameters):
    """Nonlinear orbital parameters for the marginalized Gaia astrometry likelihood.

    The 6 linear astrometric parameters (ra0, dec0, pmra, pmdec, parallax, a) are
    analytically marginalized out.
    """


class CombinedOrbitParameters(AbstractGaiaAstrometryParameters):
    """Nonlinear orbital parameters for combined astrometry + RV data.

    Identical structure to GaiaAstrometryOrbitParameters but tagged with
    data_type = "combined" so that Samples can distinguish combined from
    pure-astrometry posteriors.
    """

    data_type: ClassVar[str] = "combined"


class GaiaAstrometryFullParameters(AbstractGaiaAstrometryParameters):
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

    ra0: Quantity[Angle]  # reference RA offset
    dec0: Quantity[Angle]  # reference Dec offset
    pmra: Quantity[AngularSpeed]  # proper motion in RA
    pmdec: Quantity[AngularSpeed]  # proper motion in Dec
    parallax: Quantity[Angle]  # parallax
    semi_major_axis: Quantity[Length]  # photocentric semi-major axis
