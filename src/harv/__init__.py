"""harv: Tools for inferring Keplerian orbital parameters.

A JAX-based package for modeling binary-star and star-exoplanet systems with time series
data, such as Gaia DR4 epoch astrometry and radial velocities. The package is units
aware via unxt, supports probabilistic modeling with numpyro, and provides flexible
Keplerian orbit frameworks for single and multi-body systems.
"""

# TODO: need to figure out what is available at top level. Gut feeling is we leave
# harv.io things as harv.io.load_sampler and harv.plot.plot_rv, and Abstract classes
# should be under harv.models, but what should be accessible at top level?
__all__ = (
    # Data containers
    "GaiaAstrometryData",
    "RVData",
    "SourceData",
    # Distributions
    "QD",
    "QuantityDistribution",
    # Models API
    "GaiaAstrometryModel",
    "JointModel",
    "ParamInfo",
    "RVModel",
    # Samplers
    "AbstractSampler",
    "NumpyroSampler",
    "HarvPrior",
    "RejectionSampler",
    "Samples",
    # Modules:
    "data",
    "plot",
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
    NumpyroSampler,
    RejectionSampler,
    Samples,
)
from harv import data
from harv import plot
from harv._version import __version__
