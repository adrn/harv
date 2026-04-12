"""xfail tests for combined astrometry + multi-survey RV (not yet implemented).

Per docs/spec.md S'Combined astrometry + multi-survey RV':

    CombinedStrategy raises NotImplementedError if SourceData contains both
    GaiaAstrometryData and more than one RVData.

These tests are marked xfail(strict=True) so that:
  - they PASS (as expected failures) while the feature is unimplemented, and
  - they FAIL (unexpected pass) the moment the NotImplementedError is removed,
    forcing the developer to update the tests to verify correct behaviour.

NOTE: default_combined() has been removed from RejectionPrior; the combined
prior must now be constructed manually when implemented.
"""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from unxt import Q

from harv.data import GaiaAstrometryData, RVData, SourceData
from harv.samplers.rejection_prior import RejectionPrior
from harv.distributions import QD
from harv.samplers.rejection import RejectionSampler


def _minimal_rv_data(seed: int, n: int = 5) -> RVData:
    """Tiny RV dataset for structural tests (not statistically meaningful)."""
    key = jr.key(seed)
    times = Q(jnp.linspace(0.0, 100.0, n), "day")
    rv = Q(jr.normal(key, (n,)) * 2.0, "km/s")
    rv_err = Q(jnp.ones(n) * 2.0, "km/s")
    return RVData(time=times, rv=rv, rv_err=rv_err)


def _minimal_astro_data(n: int = 10) -> GaiaAstrometryData:
    """Tiny Gaia astrometry dataset for structural tests."""
    times = Q(jnp.linspace(0.0, 1000.0, n), "day")
    al_pos = Q(jnp.zeros(n), "mas")
    al_pos_err = Q(jnp.ones(n) * 0.1, "mas")
    scan_angles = Q(jnp.linspace(0.0, 3.14, n), "rad")
    parallax_factors = jnp.zeros(n)
    return GaiaAstrometryData(
        time=times,
        al_position=al_pos,
        al_position_err=al_pos_err,
        scan_angle=scan_angles,
        parallax_factor=parallax_factors,
    )


def _make_combined_prior(
    *,
    period_min: float = 10.0,
    period_max: float = 500.0,
    offsets: dict | None = None,
) -> RejectionPrior:
    """Manually construct a combined (astrometry+RV) prior.

    ``default_combined`` was removed from RejectionPrior; this helper builds
    the equivalent prior for testing purposes.
    """
    nonlinear = {
        "period": QD(dist.LogUniform(period_min, period_max), "day"),
        "eccentricity": dist.Beta(0.867, 3.03),
        "phase_peri": dist.Uniform(0.0, 1.0),
        "cos_i": dist.Uniform(-1.0, 1.0),
        "arg_peri": QD(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        "lon_asc_node": QD(dist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
    }
    linear = {
        "ra0": QD(dist.Normal(0.0, 1000.0), "mas"),
        "dec0": QD(dist.Normal(0.0, 1000.0), "mas"),
        "pmra": QD(dist.Normal(0.0, 1000.0), "mas/yr"),
        "pmdec": QD(dist.Normal(0.0, 1000.0), "mas/yr"),
        "parallax": QD(dist.Normal(0.0, 1000.0), "mas"),
        "semi_major_axis": QD(dist.Normal(0.0, 1000.0), "mas"),
        "rv_semiamp": QD(dist.Normal(0.0, 100.0), "km/s"),
        "v_sys": QD(dist.Normal(0.0, 100.0), "km/s"),
    }
    return RejectionPrior(
        nonlinear_priors=nonlinear,
        linear_prior=linear,
        offsets={"rv": offsets} if offsets else None,
    )


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "Combined astrometry + multi-survey RV is not yet implemented. "
        "See docs/spec.md S'Combined astrometry + multi-survey RV'."
    ),
)
def test_combined_multisurv_raises_not_implemented():
    """RejectionSampler.run raises NotImplementedError for combined + multi-survey RV.

    This is the guard added in CombinedStrategy.extract_data. The xfail ensures
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

    prior = _make_combined_prior(period_min=10.0, period_max=500.0)
    sampler = RejectionSampler(prior)
    sampler.run(source_data, n_prior_samples=100, seed=0)


@pytest.mark.xfail(
    strict=True,
    raises=NotImplementedError,
    reason=(
        "Combined astrometry + multi-survey RV with offsets is not yet implemented. "
        "See docs/spec.md S'Combined astrometry + multi-survey RV'."
    ),
)
def test_combined_multisurv_with_offsets_raises_not_implemented():
    """Combined prior with offsets raises NotImplementedError."""
    astro = _minimal_astro_data()
    rv_keck = _minimal_rv_data(seed=0)
    rv_harps = _minimal_rv_data(seed=1)

    source_data = SourceData(
        gaia=astro,
        keck=rv_keck,
        harps=rv_harps,
    )

    prior = _make_combined_prior(
        period_min=10.0,
        period_max=500.0,
        offsets={
            "keck": None,
            "harps": QD(dist.Normal(0.0, 5.0), "km/s"),
        },
    )
    sampler = RejectionSampler(prior)
    sampler.run(source_data, n_prior_samples=100, seed=0)
