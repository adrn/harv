"""Tests for joint-model (e.g. SB2) samples: ``Samples.component`` and the
time-based diagnostics, which use every component's observation times."""

import numpy as np
import pytest
from unxt import Q

from harv.data import RVData, SystemData
from harv.samplers import Samples

N = 6


def _joint_samples(period=None):
    rng = np.random.default_rng(0)
    nonlinear = {
        "period": Q(rng.uniform(5.0, 50.0, N) if period is None else period, "day"),
        "eccentricity": Q(rng.uniform(0.0, 0.5, N), ""),
        "phase_peri": Q(rng.uniform(0.0, 1.0, N), ""),
        "arg_peri": Q(rng.uniform(0.0, 6.0, N), "rad"),
    }
    linear = {
        "primary.rv_semiamp": Q(rng.uniform(10.0, 20.0, N), "km/s"),
        "secondary.rv_semiamp": Q(-rng.uniform(15.0, 30.0, N), "km/s"),
        "v_sys": Q(rng.normal(0.0, 5.0, N), "km/s"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        model_type="JointModel",
        metadata={"time_ref": 0.0},
        ln_likelihood=np.linspace(-3.0, 0.0, N),
        ln_prior=np.zeros(N),
    )


def _rv(times):
    times = np.asarray(times, dtype=float)
    return RVData(
        time=Q(times, "day"),
        rv=Q(np.zeros(times.size), "km/s"),
        rv_err=Q(np.ones(times.size), "km/s"),
    )


@pytest.fixture
def sb2_data():
    """An SB2 whose secondary is measured at fewer, different epochs."""
    rng = np.random.default_rng(1)
    primary = np.sort(rng.uniform(0.0, 400.0, 12))
    secondary = np.sort(rng.uniform(100.0, 700.0, 5))
    return SystemData(primary=_rv(primary), secondary=_rv(secondary))


class TestComponent:
    def test_keeps_its_own_and_shared_parameters(self):
        joint = _joint_samples()
        primary = joint.component("primary")
        assert set(primary.nonlinear) == set(joint.nonlinear)  # all shared
        assert set(primary.linear) == {"rv_semiamp", "v_sys"}
        np.testing.assert_array_equal(
            primary["rv_semiamp"].value, joint["primary.rv_semiamp"].value
        )
        secondary = joint.component("secondary")
        assert np.all(secondary["rv_semiamp"].value < 0)
        assert primary.model_type == "JointModel.primary"

    def test_keeps_the_joint_log_probabilities_and_metadata(self):
        joint = _joint_samples()
        primary = joint.component("primary")
        np.testing.assert_array_equal(primary.ln_likelihood, joint.ln_likelihood)
        assert primary.metadata == joint.metadata
        assert primary.map_sample()["period"] == joint.map_sample()["period"]

    def test_an_unknown_component_raises(self):
        with pytest.raises(KeyError, match="tertiary"):
            _joint_samples().component("tertiary")


class TestTimeDiagnostics:
    """On joint samples with a shared period, the time-based diagnostics equal
    those of one component's samples over every component's times."""

    @pytest.fixture
    def pair(self, sb2_data):
        joint = _joint_samples()
        times = np.concatenate([d.time.value for d in sb2_data.values()])
        combined = RVData(
            time=Q(times, "day"),
            rv=Q(np.zeros(times.size), "km/s"),
            rv_err=Q(np.ones(times.size), "km/s"),
            time_ref=sb2_data.time_ref,
        )
        return joint, sb2_data, joint.component("primary"), combined

    @pytest.mark.parametrize(
        "method",
        [
            "max_phase_gap",
            "phase_coverage",
            "periods_spanned",
            "phase_coverage_per_period",
        ],
    )
    def test_per_sample_diagnostics(self, pair, method):
        joint, data, single, combined = pair
        np.testing.assert_allclose(
            getattr(joint, method)(data), getattr(single, method)(combined)
        )

    def test_period_unimodal(self, pair):
        joint, data, single, combined = pair
        assert joint.period_unimodal(data) == single.period_unimodal(combined)
        narrow = _joint_samples(period=np.full(N, 20.0))
        assert narrow.period_unimodal(data)

    def test_the_span_covers_both_components(self, sb2_data):
        """The secondary extends the baseline past the primary's last epoch."""
        joint = _joint_samples(period=np.full(N, 10.0))
        primary_only = joint.periods_spanned(sb2_data["primary"])
        assert np.all(joint.periods_spanned(sb2_data) > primary_only)

    def test_per_component_periods_are_refused(self, sb2_data):
        joint = _joint_samples()
        unshared = Samples(
            nonlinear={
                "primary.period": joint["period"],
                "secondary.period": joint["period"],
            },
            linear=joint.linear,
            model_type="JointModel",
        )
        with pytest.raises(NotImplementedError, match="shared"):
            unshared.max_phase_gap(sb2_data)

    def test_methods_needing_one_model_still_refuse_joint_samples(self, sb2_data):
        with pytest.raises(NotImplementedError, match="single-component"):
            _joint_samples().binary_mass_function()
