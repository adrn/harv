"""harv: Tools for inferring Keplerian orbital parameters.

A JAX-based package for modeling binary-star and star-exoplanet systems with time series
data, such as Gaia DR4 epoch astrometry and radial velocities. The package is units
aware via unxt, supports probabilistic modeling with numpyro, and provides flexible
Keplerian orbit frameworks for single and multi-body systems.
"""

__all__ = (
    # Data containers
    "GaiaAstrometryData",
    "RVData",
    "SourceData",
    # Distributions
    "QD",
    "QuantityDistribution",
    # Models API
    "AbstractComponentModel",
    "GaiaAstrometryModel",
    "JointModel",
    "RVModel",
    # Parameterizations
    "AbstractParameterization",
    "EcoswEsinwRV",
    "StandardGaiaAstrometry",
    "StandardRV",
    # Extensions
    "AbstractExtension",
    "GP",
    "Jitter",
    "MonomialTrend",
    "MultiSurveyOffset",
    "ParamInfo",
    # Samplers
    "AbstractSampler",
    "HarvPrior",
    "NumpyroSampler",
    "RejectionSampler",
    "Samples",
    "make_prior_cache",
    # Modules:
    "data",
    "periodogram",
    "plot",
)

import os
import warnings

import jax

# Double precision is turned on by default. Set JAX_ENABLE_X64 yourself, to anything,
# and harv leaves it alone: "0" opts out of double precision and accepts the
# consequences, and "1" (or an earlier ``jax.config.update``) just means the flip has
# already happened and no warning is needed.
if not jax.config.read("jax_enable_x64") and "JAX_ENABLE_X64" not in os.environ:
    jax.config.update(name="jax_enable_x64", val=True)
    warnings.warn(
        "harv enabled JAX's float64 mode (jax_enable_x64) for this process, "
        "which changes the default dtype of every JAX array created from now "
        "on. harv's likelihoods are not accurate in float32. To silence this, "
        "enable it yourself before importing harv, either with the "
        "JAX_ENABLE_X64=1 environment variable or "
        'jax.config.update("jax_enable_x64", True). Set JAX_ENABLE_X64=0 to '
        "keep float32 and accept degraded results.",
        stacklevel=2,
    )

from harv.data import GaiaAstrometryData, RVData, SourceData
from harv.distributions import QD, QuantityDistribution
from harv.models.extensions import (
    AbstractExtension,
    GP,
    Jitter,
    MultiSurveyOffset,
    ParamInfo,
    MonomialTrend,
)
from harv.models import (
    AbstractComponentModel,
    AbstractParameterization,
    EcoswEsinwRV,
    GaiaAstrometryModel,
    JointModel,
    RVModel,
    StandardGaiaAstrometry,
    StandardRV,
    HarvPrior,
)
from harv.samplers import (
    AbstractSampler,
    make_prior_cache,
    NumpyroSampler,
    RejectionSampler,
    Samples,
)
from harv import data
from harv import periodogram
from harv import plot
from harv._version import __version__
