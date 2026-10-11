"""Tests for ``harv_hq.summary_stats``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Q

from harv.samplers import Samples
from harv_hq.summary_stats import _weighted_percentiles, weighted_resample


def _weighted_samples(ln_likelihood):
    n = len(ln_likelihood)
    return Samples(
        nonlinear={"period": Q(np.arange(1.0, n + 1), "day")},
        linear={},
        model_type="RVModel",
        metadata={"ln_Z_int": 0.0, "n_prior_samples": n},
        ln_likelihood=jnp.asarray(ln_likelihood, dtype=float),
        ln_prior=jnp.zeros(n),
    )


def test_weighted_percentiles_match_unweighted_for_equal_weights():
    values = np.random.default_rng(0).normal(size=10_001)
    weights = np.full(values.size, 1 / values.size)
    np.testing.assert_allclose(
        _weighted_percentiles(values, weights),
        np.percentile(values, [16, 50, 84]),
        atol=1e-3,
    )


def test_weighted_percentiles_follow_the_weights():
    values = np.array([1.0, 2.0, 3.0])
    p16, p50, p84 = _weighted_percentiles(values, np.array([0.0, 0.0, 1.0]))
    assert p16 == p50 == p84 == 3.0


def test_weighted_resample_draws_in_proportion_to_weight():
    # Weights exp(ln L): the third sample carries e^2 / (2 + e^2) ~ 79%.
    samples = _weighted_samples([0.0, 0.0, 2.0])
    resampled = weighted_resample(samples, jax.random.key(0), 20_000)
    periods = np.asarray(resampled["period"].value)
    expected = np.exp(2.0) / (2 + np.exp(2.0))
    assert np.mean(periods == 3.0) == pytest.approx(expected, abs=0.01)


def test_weighted_resample_refuses_zero_weight():
    samples = _weighted_samples([-np.inf, -np.inf])
    with pytest.raises(ValueError, match="no finite posterior weight"):
        weighted_resample(samples, jax.random.key(0), 10)
