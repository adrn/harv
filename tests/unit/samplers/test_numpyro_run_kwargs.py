"""NumpyroSampler.run forwards progress_bar, kernel_kwargs and mcmc_kwargs."""

import jax
import jax.numpy as jnp
import numpyro.infer
import pytest
from numpyro.infer import NUTS
from unxt import Q

import harv
import harv.models as hm

TINY = {"num_chains": 1, "num_warmup": 5, "num_samples": 10}


@pytest.fixture(scope="module")
def setup():
    t = jnp.linspace(0.0, 300.0, 12)
    data = harv.RVData(
        time=Q(t, "day"),
        rv=Q(5.0 * jnp.sin(2 * jnp.pi * t / 100.0), "km/s"),
        rv_err=Q(jnp.full(12, 0.5), "km/s"),
    )
    prior = hm.StandardRV().default_prior(
        period_min=Q(50.0, "day"),
        period_max=Q(200.0, "day"),
        sigma_K0=Q(10.0, "km/s"),
        sigma_v0=Q(10.0, "km/s"),
    )
    init = harv.RejectionSampler(prior, harv.RVModel()).run(
        data, key=jax.random.key(0), n_prior_samples=2000, top_k=1
    )
    return harv.NumpyroSampler(prior, harv.RVModel()), data, init


def test_kwargs_reach_kernel_and_mcmc(setup, monkeypatch):
    sampler, data, init = setup
    seen = {}

    def kernel_spy(model, **kw: float):
        seen["kernel"] = kw
        return NUTS(model, **kw)

    real_mcmc = numpyro.infer.MCMC

    def mcmc_spy(*args, **kw):  # noqa: ANN002, ANN003
        seen["mcmc"] = kw
        return real_mcmc(*args, **kw)

    monkeypatch.setattr(numpyro.infer, "MCMC", mcmc_spy)
    post = sampler.run(
        data,
        init_samples=init,
        key=jax.random.key(1),
        kernel=kernel_spy,
        kernel_kwargs={"target_accept_prob": 0.9},
        mcmc_kwargs={"thinning": 2},
        progress_bar=False,
        **TINY,
    )
    assert seen["kernel"] == {"target_accept_prob": 0.9}
    assert seen["mcmc"]["progress_bar"] is False
    assert seen["mcmc"]["thinning"] == 2
    assert post.n_samples == 5  # thinning took effect


def test_duplicate_kwarg_raises(setup):
    sampler, data, init = setup
    with pytest.raises(TypeError, match="progress_bar"):
        sampler.run(
            data,
            init_samples=init,
            key=jax.random.key(1),
            mcmc_kwargs={"progress_bar": False},
            **TINY,
        )
