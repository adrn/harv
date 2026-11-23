"""Rejection -> NUTS from mixed-sign inits warns once, at the caller's line."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from unxt import Q

import harv
import harv.models as hm


def test_run_warns_at_the_callers_line():
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
    rej = harv.RejectionSampler(prior, harv.RVModel()).run(
        data, key=jax.random.key(0), n_prior_samples=2000, top_k=64
    )
    K = np.asarray(rej["rv_semiamp"].value)
    assert K.max() > 0 > K.min()  # the rejection output holds both twins
    init = rej[np.array([K.argmax(), K.argmin()])]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        harv.NumpyroSampler(prior, harv.RVModel()).run(
            data,
            init_samples=init,
            key=jax.random.key(1),
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            progress_bar=False,
        )
    hits = [w for w in caught if "mirror modes" in str(w.message)]
    assert len(hits) == 1
    assert "wrap_angles" in str(hits[0].message)
    assert hits[0].filename == __file__
