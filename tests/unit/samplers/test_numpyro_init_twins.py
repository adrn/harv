"""NumpyroSampler warns when chain inits straddle an amplitude-sign twin."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Q

import harv
import harv.models as hm
from harv.samplers.numpyro import _warn_mixed_sign_inits
from harv.samplers.samples import Samples

MATCH = "mirror modes"


def _caught(fn) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if MATCH in str(w.message)]


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

    hits = _caught(
        lambda: harv.NumpyroSampler(prior, harv.RVModel()).run(
            data,
            init_samples=init,
            key=jax.random.key(1),
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            progress_bar=False,
        )
    )
    assert len(hits) == 1
    assert "wrap_angles" in str(hits[0].message)
    assert hits[0].filename == __file__


def _samples(**linear: list[float]) -> Samples:
    n = len(next(iter(linear.values())))
    return Samples(
        nonlinear={"period": Q(jnp.full(n, 100.0), "day")},
        linear={k: Q(jnp.asarray(v), "km/s") for k, v in linear.items()},
    )


@pytest.mark.parametrize(
    ("samples", "indices", "warns"),
    [
        (_samples(rv_semiamp=[5.0, -5.0]), [0, 1], True),
        (_samples(rv_semiamp=[5.0, -5.0]), [0, 0], False),  # chosen inits agree
        (_samples(rv_semiamp=[5.0, -5.0]), [1], False),  # one chain
        (_samples(semi_major_axis=[2.0, -2.0]), [0, 1], True),  # astrometry only
        # untied joint: K agrees, but a straddles the lon_asc_node twin
        (_samples(rv_semiamp=[5.0, 6.0], semi_major_axis=[2.0, -2.0]), [0, 1], True),
        # signed SB2: K_1 > 0 and K_2 < 0 on every sample
        (
            _samples(
                **{
                    "primary.rv_semiamp": [5.0, 6.0],
                    "secondary.rv_semiamp": [-7.0, -8.0],
                }
            ),
            [0, 1],
            False,
        ),
    ],
)
def test_warns_only_on_mixed_signs(samples, indices, warns):
    hits = _caught(lambda: _warn_mixed_sign_inits(samples, indices))
    assert len(hits) == int(warns)
