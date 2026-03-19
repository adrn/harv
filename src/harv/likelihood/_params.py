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


class RVOrbitParameters(AbstractRVParameters):
    """Nonlinear orbital parameters for the marginalized RV likelihood.

    The linear parameters (K, v₀) are analytically marginalized out.
    """


class RVFullParameters(AbstractRVParameters):
    """Full parameter set for the RV likelihood.

    Includes both nonlinear orbital parameters and the linear RV parameters
    (semi-amplitude K and systemic velocity v₀).
    """

    linear_param_names: ClassVar[tuple[str, ...]] = ("K", "v0")

    K: Quantity[Speed]  # RV semi-amplitude
    v0: Quantity[Speed]  # systemic velocity


# ---------------------------------------------------------------------------
# Gaia epoch astrometry
# ---------------------------------------------------------------------------


class AbstractGaiaAstrometryParameters(AbstractBaseKeplerParameters):
    """Abstract base class for Gaia astrometry parameter structs."""

    cos_i: float | jax.Array
    lon_asc_node: float | jax.Array


class GaiaAstrometryOrbitParameters(AbstractGaiaAstrometryParameters):
    """Nonlinear orbital parameters for the marginalized Gaia astrometry likelihood.

    The 6 linear astrometric parameters (α₀, δ₀, μ_α, μ_δ, ϖ, a) are
    analytically marginalized out.
    """


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

    ra0: Quantity[Angle]  # α₀: reference RA offset [mas]
    dec0: Quantity[Angle]  # δ₀: reference Dec offset [mas]
    pmra: Quantity[AngularSpeed]  # μ_α: proper motion in RA [mas/yr]
    pmdec: Quantity[AngularSpeed]  # μ_δ: proper motion in Dec [mas/yr]
    parallax: Quantity[Angle]  # ϖ: parallax [mas]
    semi_major_axis: Quantity[Length]  # a: photocentric semi-major axis [mas]
