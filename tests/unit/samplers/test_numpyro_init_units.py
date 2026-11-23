"""NumpyroSampler starts chains at the init samples' values whatever unit they use."""

import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv
import harv.models as hm
from harv import QD
from harv.models.extensions import Jitter
from harv.samplers.samples import Samples

# The prior's units: day, rad, km/s. The samples below use yr, deg and m/s.
PRIOR = hm.StandardRV().default_prior(
    period_min=Q(50.0, "day"),
    period_max=Q(200.0, "day"),
    sigma_K0=Q(10.0, "km/s"),
    sigma_v0=Q(10.0, "km/s"),
    jitter=QD(dist.HalfNormal(1.0), "km/s"),
)
MODEL = harv.RVModel(extensions=(Jitter(obs_unit="km/s"),))


@pytest.mark.parametrize("num_chains", [1, 2])
def test_nonlinear_inits_are_read_in_the_prior_unit(num_chains):
    samples = Samples(
        nonlinear={
            "period": Q([100.0, 150.0], "day").uconvert("yr"),
            "eccentricity": Q([0.1, 0.3], ""),
            "arg_peri": Q([1.0, 4.0], "rad").uconvert("deg"),
            "jitter": Q([0.5, 0.8], "km/s").uconvert("m/s"),
        },
        linear={"rv_semiamp": Q([5.0, 6.0], "km/s"), "v_sys": Q([0.0, 1.0], "km/s")},
        model_type="RVModel",
    )
    init = harv.NumpyroSampler(PRIOR, MODEL)._build_init_params(
        MODEL,
        samples,
        all_priors={**PRIOR.nonlinear_priors, **PRIOR.extension_priors},
        effective_linear_prior=None,
        effective_marginalized_names=None,
        marginalized=True,
        num_chains=num_chains,
    )
    expected = {
        "period": [100.0, 150.0],
        "arg_peri": [1.0, 4.0],
        "jitter": [0.5, 0.8],
        "eccentricity": [0.1, 0.3],
    }
    for name, vals in expected.items():
        np.testing.assert_allclose(init[name], vals[:num_chains], err_msg=name)
