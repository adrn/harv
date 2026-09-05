"""Tests for the ``verbose`` flag that gates harv's advisory warnings.

Advisory warnings report a normal, correctly-handled situation (here: a
non-Gaussian linear prior that cannot be analytically marginalized and so is
sampled explicitly).  Per ``docs/spec.md`` -> "Warnings and verbosity", they are
silent unless the caller opts in with ``verbose=True``.
"""

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from unxt import Q

from harv.data import RVData
from harv.distributions import QD
from harv.models.priors import HarvPrior
from harv.models.rv import RVModel
from harv.samplers.rejection import RejectionSampler

_MATCH = "Non-Gaussian linear prior"


def _halfnormal_prior() -> HarvPrior:
    """An RV prior whose ``rv_semiamp`` prior cannot be marginalized."""
    return HarvPrior(
        nonlinear_priors={
            "period": QD(dist.LogUniform(50.0, 200.0), "day"),
            "eccentricity": dist.Beta(0.867, 3.03),
            "phase_peri": dist.Uniform(0.0, 1.0),
            "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
        },
        linear_priors={
            "rv_semiamp": QD(dist.HalfNormal(30.0), "km/s"),
            "v_sys": QD(dist.Normal(0.0, 50.0), "km/s"),
        },
    )


def _rv_data(n_obs: int = 20, seed: int = 42) -> RVData:
    key = jr.key(seed)
    return RVData(
        time=Q(jnp.linspace(0.0, 200.0, n_obs), "day"),
        rv=Q(jr.normal(key, (n_obs,)) * 5.0, "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 2.0, "km/s"),
    )


class TestHarvPriorSampleVerbose:
    def test_silent_by_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            samples = _halfnormal_prior().sample(jax.random.key(0), 10, model=RVModel())
        assert samples.n_samples == 10

    def test_warns_when_verbose(self):
        with pytest.warns(UserWarning, match=_MATCH):
            samples = _halfnormal_prior().sample(
                jax.random.key(0), 10, model=RVModel(), verbose=True
            )
        # The advisory is informational only: the draw is identical either way.
        assert "rv_semiamp" in samples


class TestRejectionSamplerVerbose:
    def test_silent_by_default(self):
        sampler = RejectionSampler(_halfnormal_prior(), RVModel())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            samples = sampler.run(_rv_data(), n_prior_samples=200, seed=0)
        assert samples.n_samples >= 0

    def test_warns_when_verbose(self):
        sampler = RejectionSampler(_halfnormal_prior(), RVModel(), verbose=True)
        with pytest.warns(UserWarning, match=_MATCH):
            sampler.run(_rv_data(), n_prior_samples=200, seed=0)

    def test_summary_never_warns_even_when_verbose(self):
        """Introspection stays side-effect-free regardless of ``verbose``."""
        sampler = RejectionSampler(_halfnormal_prior(), RVModel(), verbose=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            text = sampler.summary()
        assert "rv_semiamp" in text

    def test_verbose_is_static_and_does_not_change_results(self):
        quiet = RejectionSampler(_halfnormal_prior(), RVModel())
        loud = RejectionSampler(_halfnormal_prior(), RVModel(), verbose=True)
        data = _rv_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = quiet.run(data, n_prior_samples=200, seed=7)
            b = loud.run(data, n_prior_samples=200, seed=7)
        assert a.n_samples == b.n_samples
        # ``verbose`` is a static field, so it lives in the treedef, not the leaves.
        assert jax.tree_util.tree_structure(quiet) != jax.tree_util.tree_structure(loud)
