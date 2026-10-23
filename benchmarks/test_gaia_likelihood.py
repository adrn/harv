"""Single-device per-sample cost of the Gaia astrometry likelihood.

Unlike ``test_rejection_scaling.py`` (a scaling grid on simulated data), this
times one realistic workload at fixed size: the Gaia-4 DR4 epoch astrometry
(93 epochs, ``docs/tutorials/case-studies/Gaia-4.ipynb``) with the priors from
that case study, for the Standard and Thiele-Innes parameterizations. The
number to track is ``ns_per_sample``, which is what decides how long a 10^7
rejection run takes. A Kepler-solver micro-benchmark sits alongside it because
the solve is about half of that cost.

Results are written outside ``benchmarks/results/*.json`` so ``report.py`` does
not try to merge them into the scaling grid; see ``docs/running-benchmarks.md``.
"""

from pathlib import Path
from typing import Any

import astropy.table as at
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

DATA = (
    Path(__file__).resolve().parent.parent
    / "docs/tutorials/data/gaia-dr4-prerelease/Gaia-4.ecsv"
)
N_PRIOR = 1_000_000
KEPLER_SHAPE = (100_000, 93)


def _gaia4_data() -> Any:
    import harv

    tbl = at.QTable.read(DATA)
    return harv.GaiaAstrometryData(
        time=Q(tbl["relative_time"].to_value("day"), "day"),
        al_position=Q(tbl["pos_al"].to_value("mas"), "mas"),
        al_position_err=Q(tbl["pos_al_err"].to_value("mas"), "mas"),
        scan_angle=Q(tbl["scan_angle"].to_value("deg"), "deg"),
        parallax_factor=jnp.asarray(tbl["parallax_factor"], dtype=np.float64),
    )


def _sampler(parameterization: str, data: Any) -> Any:
    import harv
    import harv.models as hm
    from harv.models.parameterizations import ThieleInnesGaiaAstrometry

    prior_kwargs = {
        "period_min": Q(10.0, "day"),
        "period_max": Q(2000.0, "day"),
        "sigma_a0": Q(1.0, "AU"),
        "sigma_parallax": Q(50.0, "mas"),
        "sigma_pos": Q(100.0, "mas"),
        "sigma_vtan": Q(100.0, "km/s"),
        "jitter": harv.QD(dist.HalfNormal(0.1), "mas"),
    }
    jitter = (harv.Jitter(obs_unit="mas"),)
    if parameterization == "standard":
        param = hm.StandardGaiaAstrometry()
        orbit_names = ("semi_major_axis",)
    else:
        param = ThieleInnesGaiaAstrometry.from_data(data)
        orbit_names = ("ti_A", "ti_B", "ti_F", "ti_G")
    return harv.RejectionSampler(
        param.default_prior(**prior_kwargs),
        harv.GaiaAstrometryModel(parameterization=param, extensions=jitter),
        marginalized_names=("ra0", "dec0", "pmra", "pmdec", *orbit_names),
    )


@pytest.mark.parametrize("parameterization", ["standard", "thiele_innes"])
def test_gaia4_rejection(
    benchmark: Any, parameterization: str, rounds: int, device_info: dict[str, Any]
) -> None:
    """Time ``run_with_samples`` over an in-memory 10^6-draw prior library."""
    data = _gaia4_data()
    sampler = _sampler(parameterization, data)
    library = sampler.prior.sample(N_PRIOR, key=jax.random.key(0), model=sampler.model)

    def call() -> Any:
        return jax.block_until_ready(
            sampler.run_with_samples(
                data, library, top_k=16, key=jax.random.key(1), ignore_non_finite=True
            )
        )

    benchmark.pedantic(call, rounds=rounds, iterations=1, warmup_rounds=1)
    benchmark.extra_info.update(device_info)
    benchmark.extra_info["n_prior_samples"] = N_PRIOR
    benchmark.extra_info["n_obs"] = int(data.n_obs)
    benchmark.extra_info["ns_per_sample"] = benchmark.stats.stats.min / N_PRIOR * 1e9


def _kepler_solvers() -> dict[str, Any]:
    from jaxoplanet.core.kepler import kepler

    from harv.kepler.orbits import true_anomaly_from_mean

    return {
        "jaxoplanet": kepler,
        "harv": lambda M, e: true_anomaly_from_mean(Q(M, "rad"), e),
    }


@pytest.mark.parametrize("solver", ["jaxoplanet", "harv"])
def test_kepler_solver(
    benchmark: Any, solver: str, rounds: int, device_info: dict[str, Any]
) -> None:
    """Time ``(sin f, cos f)`` for 10^5 orbits x 93 epochs, eccentricity up to 0.99."""
    fn = jax.jit(jax.vmap(_kepler_solvers()[solver]))
    M = jax.random.uniform(jax.random.key(0), KEPLER_SHAPE, minval=-800, maxval=300)
    e = jax.random.uniform(jax.random.key(1), KEPLER_SHAPE[:1], maxval=0.99)

    benchmark.pedantic(
        lambda: jax.block_until_ready(fn(M, e)),
        rounds=rounds,
        iterations=1,
        warmup_rounds=1,
    )
    benchmark.extra_info.update(device_info)
    benchmark.extra_info["ns_per_element"] = (
        benchmark.stats.stats.min / np.prod(KEPLER_SHAPE) * 1e9
    )
