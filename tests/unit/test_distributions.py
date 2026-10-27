"""Tests for ``harv.distributions``: ``JointPrior`` and ``joint``."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

from harv.distributions import QD, JointPrior, joint
from harv.models.priors import HarvPrior
from harv.stats import UnitDisk


class TestJointPrior:
    def test_joint_expands_to_one_entry_per_name(self):
        entries = joint(("ecosw", "esinw"), UnitDisk())
        assert list(entries) == ["ecosw", "esinw"]
        assert entries["ecosw"] is entries["esinw"]

    @pytest.mark.parametrize(
        ("names", "distribution", "match"),
        [
            (("ecosw",), UnitDisk(), "at least two distinct"),
            (("ecosw", "ecosw"), UnitDisk(), "at least two distinct"),
            (("a", "b", "c"), UnitDisk(), "event_shape"),
            (("a", "b"), dist.Normal(0.0, 1.0), "event_shape"),
            (("a", "b"), dist.Normal(jnp.zeros((3, 2)), 1.0).to_event(1), "batch"),
        ],
    )
    def test_joint_prior_validation(self, names, distribution, match):
        with pytest.raises(ValueError, match=match):
            JointPrior(names, distribution)

    @pytest.mark.parametrize("bad", [QD(UnitDisk(), "day"), "not a distribution"])
    def test_joint_prior_rejects_non_distributions(self, bad):
        with pytest.raises(TypeError, match=r"numpyro distribution|one per name"):
            JointPrior(("a", "b"), bad)

    def test_list_names_become_tuple(self):
        prior = HarvPrior(
            nonlinear_priors=joint(["ecosw", "esinw"], UnitDisk()), linear_priors={}
        )
        assert prior.nonlinear_priors["ecosw"].names == ("ecosw", "esinw")
        assert prior.sample_nonlinear(3, key=jr.key(0))["esinw"].shape == (3,)

    def test_bare_string_names_raise(self):
        with pytest.raises(TypeError, match="sequence of names"):
            joint("pe", UnitDisk())

    def test_unit_tuple_must_hold_strings(self):
        with pytest.raises(TypeError, match="unit strings"):
            JointPrior(("a", "b"), QD(UnitDisk(), ("day", None)))
