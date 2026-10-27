"""Joint priors over dimensioned names: ``joint(names, QD(d, ("day", "")))``."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints
from unxt import Q, ustrip

import harv.models as hm
from harv.distributions import QD, JointPrior, joint
from harv.models.rv import RVModel
from harv.samplers import RejectionSampler, Samples, make_prior_cache
from harv.samplers.numpyro import NumpyroSampler
from harv.simulate.rv import simulate_rv_sb1_data

P_MIN, P_MAX = 2.0, 1000.0


class PeriodEcc(dist.Distribution):
    """log-uniform P, then e | P ~ Uniform(0, 0.2) below 30 d, Uniform(0, 0.9) above."""

    support = constraints.independent(
        constraints.interval(jnp.array([P_MIN, 0.0]), jnp.array([P_MAX, 1.0])), 1
    )

    def __init__(self) -> None:
        super().__init__(batch_shape=(), event_shape=(2,))

    @staticmethod
    def _e_max(period: jax.Array) -> jax.Array:
        return jnp.where(period < 30.0, 0.2, 0.9)

    def sample(self, key, sample_shape=()):
        k1, k2 = jr.split(key)
        period = dist.LogUniform(P_MIN, P_MAX).sample(k1, sample_shape)
        ecc = jr.uniform(k2, sample_shape) * self._e_max(period)
        return jnp.stack([period, ecc], axis=-1)

    def log_prob(self, value):
        period, ecc = value[..., 0], value[..., 1]
        return dist.LogUniform(P_MIN, P_MAX).log_prob(period) + dist.Uniform(
            0.0, self._e_max(period)
        ).log_prob(ecc)


def _prior():
    return hm.StandardRV().default_prior(
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
        **joint(("period", "eccentricity"), QD(PeriodEcc(), ("day", ""))),
    )


def _check_split(period: Q, ecc: Q) -> None:
    p = np.asarray(ustrip("day", period))
    e = np.asarray(ustrip("", ecc))
    assert np.all((p >= P_MIN) & (p <= P_MAX))
    assert np.all(e[p < 30.0] <= 0.2)


@pytest.mark.parametrize(
    "bad", [QD(PeriodEcc(), "day"), QD(PeriodEcc(), ("day", "", "s"))]
)
def test_qd_needs_one_unit_per_name(bad):
    with pytest.raises(TypeError, match="one per name"):
        JointPrior(("period", "eccentricity"), bad)


def test_prior_sample_restores_per_name_units():
    samples = _prior().sample(500, key=jr.key(0), model=RVModel(), return_logprobs=True)
    assert samples["period"].unit == "d"
    _check_split(samples["period"], samples["eccentricity"])
    expected = PeriodEcc().log_prob(
        jnp.stack(
            [ustrip("day", samples["period"]), ustrip("", samples["eccentricity"])], -1
        )
    )
    # ln_prior also has arg_peri + phase_peri; the joint term must be in it.
    assert bool(jnp.all(samples.ln_prior <= expected + 1e-9))


def test_jit_and_vmap():
    prior = _prior()
    draw = eqx.filter_jit(lambda p, k: p.sample_nonlinear(10, key=k))
    out = jax.vmap(lambda k: draw(prior, k))(jr.split(jr.key(3), 4))
    assert out["period"].shape == out["eccentricity"].shape == (4, 10)


class TestSamplers:
    @pytest.fixture(scope="class")
    def setup(self):
        data, _ = simulate_rv_sb1_data(
            seed=3, n_obs=12, period=Q(12.0, "day"), eccentricity=0.1
        )
        prior = _prior()
        model = RVModel()
        init = RejectionSampler(prior, model).run(
            data, key=jr.key(0), n_prior_samples=20_000, top_k=4
        )
        return data, prior, model, init

    def test_rejection(self, setup):
        _, _, _, init = setup
        assert np.isfinite(init.metadata["max_ln_likelihood"])
        assert init["period"].unit == "d"
        _check_split(init["period"], init["eccentricity"])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_nuts_warm_start(self, setup):
        data, prior, model, init = setup
        post = NumpyroSampler(prior, model).run(
            data,
            init_samples=init,
            key=jr.key(1),
            num_chains=1,
            num_warmup=100,
            num_samples=50,
        )
        assert post["period"].unit == "d"
        assert post["period"].shape == (50,)
        _check_split(post["period"], post["eccentricity"])

    def test_prior_cache_round_trip(self, setup, tmp_path: Path):
        data, prior, model, _ = setup
        path = tmp_path / "cache.h5"
        make_prior_cache(prior, model, n_samples=256, filename=path, key=jr.key(0))
        loaded = Samples.from_hdf5(path)
        assert loaded.nonlinear["period"].unit == "d"
        _check_split(loaded.nonlinear["period"], loaded.nonlinear["eccentricity"])
        samples = RejectionSampler(prior, model).run_with_samples(
            data, path, top_k=4, key=jr.key(1)
        )
        assert np.isfinite(samples.metadata["max_ln_likelihood"])
