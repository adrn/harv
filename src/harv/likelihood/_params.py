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

import equinox as eqx


class AbstractBaseKeplerParameters(eqx.Module):
    """Base class for Keplerian orbital parameters shared across data types.

    This includes the 4 nonlinear orbital parameters common to both RV and
    astrometry: period, eccentricity, phase of periastron, and argument of
    periastron.
    """

    period: float
    eccentricity: float
    phase_peri: float
    arg_peri: float


# ---------------------------------------------------------------------------
# Radial velocity
# ---------------------------------------------------------------------------


class AbstractRVParameters(AbstractBaseKeplerParameters):
    """Abstract base class for RV parameter structs."""


class RVOrbitParameters(AbstractRVParameters):
    """Nonlinear orbital parameters for the marginalized RV likelihood.

    The linear parameters (K, v₀) are analytically marginalized out.
    """


class RVParameters(AbstractRVParameters):
    """Full parameter set for the RV likelihood.

    Includes both nonlinear orbital parameters and the linear RV parameters
    (semi-amplitude K and systemic velocity v₀).
    """

    K: float  # RV semi-amplitude
    v0: float  # systemic velocity


# ---------------------------------------------------------------------------
# Gaia epoch astrometry
# ---------------------------------------------------------------------------


class AbstractGaiaAstrometryParameters(AbstractBaseKeplerParameters):
    """Abstract base class for Gaia astrometry parameter structs."""

    cos_i: float
    lon_asc_node: float


class GaiaAstrometryOrbitParameters(AbstractGaiaAstrometryParameters):
    """Nonlinear orbital parameters for the marginalized Gaia astrometry likelihood.

    The 6 linear astrometric parameters (α₀, δ₀, μ_α, μ_δ, ϖ, a) are
    analytically marginalized out.
    """


class GaiaAstrometryParameters(AbstractGaiaAstrometryParameters):
    """Full parameter set for the Gaia astrometry likelihood.

    Includes both nonlinear orbital parameters and the 6 linear astrometric
    parameters (reference position, proper motion, parallax, semi-major axis).
    """

    ra0: float  # α₀: reference RA offset [mas]
    dec0: float  # δ₀: reference Dec offset [mas]
    pmra: float  # μ_α: proper motion in RA [mas/yr]
    pmdec: float  # μ_δ: proper motion in Dec [mas/yr]
    parallax: float  # ϖ: parallax [mas]
    semi_major_axis: float  # a: photocentric semi-major axis [mas]
