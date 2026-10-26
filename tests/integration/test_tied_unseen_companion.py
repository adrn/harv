"""End-to-end: a tied RV + Gaia fit recovers an injected unseen-companion orbit.

Spec §Tied linear parameters. Data are simulated from one orbit with the RV
semi-amplitude set by the classical-convention relation ``K = c * a``.
``JointModel.for_rv_and_gaia`` ties ``K`` to the astrometric orbit by default, so
the fit must recover ``a``, ``K`` and the ascending node (which a free ``K``
leaves ambiguous by pi).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest
from unxt import Q, ustrip

import harv.models as hm
from harv import GaiaAstrometryData, GaiaAstrometryModel, JointModel, RVData, RVModel
from harv.data import SourceData
from harv.distributions import QD
from harv.models import HarvPrior
from harv.samplers import NumpyroSampler, RejectionSampler

TRUTH = {
    "period": Q(400.0, "day"),
    "eccentricity": 0.3,
    "phase_peri": 0.3,
    "arg_peri": Q(1.0, "rad"),
    "lon_asc_node": Q(2.0, "rad"),
    "cos_i": 0.5,
}
PARALLAX = 10.0  # mas
SEMI_MAJOR_AXIS = 1.0  # mas, the star's (= photocenter's) orbit
ASTRO_LINEAR = {
    "ra0": 0.5,
    "dec0": -0.3,
    "pmra": 3.0,
    "pmdec": -2.0,
    "parallax": PARALLAX,
    "semi_major_axis": SEMI_MAJOR_AXIS,
}


def _rv_semiamp() -> float:
    """K in km/s from the spec relation, K = 2 pi a sin(i) / (P sqrt(1 - e^2) plx)."""
    speed = ustrip("km/s", 2 * np.pi * Q(1.0, "AU") / TRUTH["period"])
    sin_i = np.sqrt(1 - TRUTH["cos_i"] ** 2)
    e = TRUTH["eccentricity"]
    return speed * sin_i * SEMI_MAJOR_AXIS / (np.sqrt(1 - e**2) * PARALLAX)


@pytest.fixture(scope="module")
def data() -> SourceData:
    rng = np.random.default_rng(7)
    time_ref = Q(0.0, "day")
    n_ast, n_rv = 60, 25
    astro = GaiaAstrometryData(
        time=Q(jnp.sort(jnp.asarray(rng.uniform(-900, 900, n_ast))), "day"),
        al_position=Q(jnp.zeros(n_ast), "mas"),
        al_position_err=Q(jnp.full(n_ast, 0.05), "mas"),
        scan_angle=Q(jnp.asarray(rng.uniform(0, 2 * np.pi, n_ast)), "rad"),
        parallax_factor=jnp.asarray(rng.uniform(-1, 1, n_ast)),
        time_ref=time_ref,
    )
    rv = RVData(
        time=Q(jnp.sort(jnp.asarray(rng.uniform(-600, 600, n_rv))), "day"),
        rv=Q(jnp.zeros(n_rv), "km/s"),
        rv_err=Q(jnp.full(n_rv, 0.1), "km/s"),
        time_ref=time_ref,
    )
    rv_shared = {
        k: TRUTH[k] for k in ("period", "eccentricity", "phase_peri", "arg_peri")
    }
    astro_linear = {k: jnp.asarray(v) for k, v in ASTRO_LINEAR.items()}
    rv_linear = {"rv_semiamp": jnp.asarray(_rv_semiamp()), "v_sys": jnp.asarray(4.0)}
    al = GaiaAstrometryModel().predict(TRUTH, astro_linear, astro)
    v = RVModel().predict(rv_shared, rv_linear, rv)
    astro = GaiaAstrometryData(
        time=astro.time,
        al_position=Q(al + rng.normal(0.0, 0.05, n_ast), "mas"),
        al_position_err=astro.al_position_err,
        scan_angle=astro.scan_angle,
        parallax_factor=astro.parallax_factor,
        time_ref=time_ref,
    )
    rv = RVData(
        time=rv.time,
        rv=Q(v + rng.normal(0.0, 0.1, n_rv), "km/s"),
        rv_err=rv.rv_err,
        time_ref=time_ref,
    )
    return SourceData(rv=rv, astro=astro)


def test_tied_fit_recovers_injected_orbit(data):
    astro_prior = hm.StandardGaiaAstrometry().default_prior(
        period_min=Q(380.0, "day"),
        period_max=Q(420.0, "day"),
        sigma_a0=Q(1.0, "AU"),
        sigma_pos=Q(10.0, "mas"),
        sigma_vtan=Q(50.0, "km/s"),
        parallax=QD(ndist.Normal(PARALLAX, 0.5), "mas"),
    )
    prior = HarvPrior(
        nonlinear_priors=astro_prior.nonlinear_priors,
        linear_priors={
            **{f"astro.{k}": v for k, v in astro_prior.linear_priors.items()},
            "rv.v_sys": QD(ndist.Normal(0.0, 10.0), "km/s"),
        },
    )
    # No "rv.rv_semiamp" prior: the default unseen-companion tie applies.
    joint = JointModel.for_rv_and_gaia(
        components={"rv": RVModel(), "astro": GaiaAstrometryModel()}
    )
    assert "tied to astro.semi_major_axis" in RejectionSampler(prior, joint).summary()

    # Warm start (spec §Tied linear parameters): fit the astrometry alone, then
    # let the joint model choose between that orbit and its astrometric mirror.
    astro_fit = RejectionSampler(astro_prior, GaiaAstrometryModel()).run(
        data["astro"], key=jax.random.key(0), n_prior_samples=200_000, top_k=32
    )
    joint_sampler = NumpyroSampler(prior, joint)
    candidates = astro_fit.map_sample().with_astrometric_mirror()
    joint_init = joint_sampler.optimize(candidates, data, key=jax.random.key(1))
    posterior = joint_sampler.run(
        data,
        key=jax.random.key(2),
        init_samples=joint_init.map_sample(),
        num_warmup=300,
        num_samples=300,
        num_chains=1,
    ).wrap_angles()

    def median(name: str) -> float:
        return float(np.median(np.asarray(posterior[name].value)))

    assert median("semi_major_axis") == pytest.approx(SEMI_MAJOR_AXIS, rel=0.1)
    assert median("rv_semiamp") == pytest.approx(_rv_semiamp(), rel=0.1)
    assert median("cos_i") == pytest.approx(TRUTH["cos_i"], abs=0.1)
    # The RVs fix the ascending node: no posterior mass at Omega + pi.
    dnode = np.angle(np.exp(1j * (np.asarray(posterior["lon_asc_node"].value) - 2.0)))
    assert np.all(np.abs(dnode) < 0.5)
