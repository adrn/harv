"""xfail tests for combined astrometry + multi-survey RV (not yet implemented).

Per docs/spec.md §'Combined astrometry + multi-survey RV':

    _CombinedStrategy raises NotImplementedError if SourceData contains both
    GaiaAstrometryData and more than one RadialVelocityData.

These tests are marked xfail(strict=True) so that:
  - they PASS (as expected failures) while the feature is unimplemented, and
  - they FAIL (unexpected pass) the moment the NotImplementedError is removed,
    forcing the developer to update the tests to verify correct behaviour.
"""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from unxt import Quantity

from harv.data import GaiaAstrometryData, RadialVelocityData, SourceData
from harv.priors.rejection import RejectionPrior
from harv.samplers.rejection import RejectionSampler


def _minimal_rv_data(seed: int, n: int = 5) -> RadialVelocityData:
    """Tiny RV dataset for structural tests (not statistically meaningful)."""
    key = jr.PRNGKey(seed)
    times = Quantity(jnp.linspace(0.0, 100.0, n), "day")
    rv = Quantity(jr.normal(key, (n,)) * 2.0, "km/s")
    rv_err = Quantity(jnp.ones(n) * 2.0, "km/s")
    return RadialVelocityData(time=times, rv=rv, rv_err=rv_err)


def _minimal_astro_data(n: int = 10) -> GaiaAstrometryData:
    """Tiny Gaia astrometry dataset for structural tests."""
    times = Quantity(jnp.linspace(0.0, 1000.0, n), "day")
    al_pos = Quantity(jnp.zeros(n), "mas")
    al_pos_err = Quantity(jnp.ones(n) * 0.1, "mas")
    scan_angles = Quantity(jnp.linspace(0.0, 3.14, n), "rad")
    parallax_factors = jnp.zeros(n)
    return GaiaAstrometryData(
        time=times,
        al_position=al_pos,
        al_position_err=al_pos_err,
        scan_angle=scan_angles,
        parallax_factor=parallax_factors,
    )


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "Combined astrometry + multi-survey RV is not yet implemented. "
        "See docs/spec.md §'Combined astrometry + multi-survey RV'."
    ),
)
def test_combined_multisurv_raises_not_implemented():
    """RejectionSampler.run raises NotImplementedError for combined + multi-survey RV.

    This is the guard added in _CombinedStrategy.extract_data. The xfail ensures
    the error stays in place until the feature is fully implemented.
    """
    astro = _minimal_astro_data()
    rv_keck = _minimal_rv_data(seed=0)
    rv_harps = _minimal_rv_data(seed=1)

    source_data = SourceData(
        gaia=astro,
        keck=rv_keck,
        harps=rv_harps,
    )

    prior = RejectionPrior.default_combined(
        period_min=10.0,
        period_max=500.0,
        offsets=None,  # offsets=None also raises because of multi-RV in combined
    )
    sampler = RejectionSampler(prior)
    sampler.run(source_data, n_prior_samples=100, seed=0)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "Combined astrometry + multi-survey RV with offsets is not yet implemented. "
        "See docs/spec.md §'Combined astrometry + multi-survey RV'."
    ),
)
def test_combined_multisurv_with_offsets_raises_not_implemented():
    """default_combined raises NotImplementedError when offsets are passed."""
    RejectionPrior.default_combined(
        period_min=10.0,
        period_max=500.0,
        offsets={"keck": None, "harps": dist.Normal(0.0, 5.0)},
    )
