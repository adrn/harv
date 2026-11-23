"""_warn_mixed_sign_inits fires only when chain inits straddle a sign twin.

The run-level check (warning location) is in tests/integration.
"""

import warnings

import jax.numpy as jnp
import pytest
from unxt import Q

from harv.samplers.numpyro import _warn_mixed_sign_inits
from harv.samplers.samples import Samples

MATCH = "mirror modes"


def _caught(fn) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if MATCH in str(w.message)]


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
