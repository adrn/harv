"""Tests for the vendored ``harv.stats.numpyro_ext`` distributions."""

import jax.numpy as jnp
import numpy as np

from harv.stats import UnitDisk
from harv.stats.numpyro_ext import unit_disk


class TestUnitDisk:
    def test_unit_disk_log_prob_is_batched_and_support_aware(self):
        lp = UnitDisk().log_prob(jnp.array([[0.0, 0.0], [0.6, 0.8], [2.0, 2.0]]))
        np.testing.assert_allclose(lp, [-jnp.log(jnp.pi)] * 2 + [-jnp.inf])

    def test_constraint_is_batched(self):
        inside = unit_disk(jnp.array([[0.0, 0.0], [0.6, 0.8], [1.0, 1.0]]))
        np.testing.assert_array_equal(inside, [True, True, False])
