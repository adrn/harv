"""Tests for sampler ``data`` validation and AbstractDatasetContainer.t_ref.

Covers the typing-tightening change: ``RejectionSampler.run`` /
``NumpyroSampler.run`` / ``NumpyroSampler.optimize`` reject anything that isn't
an ``AbstractData`` (single-component) or ``AbstractDatasetContainer`` (joint),
and ``t_ref`` is uniformly exposed on every container.
"""

import jax.numpy as jnp
import pytest
from unxt import Q

import harv.models as hm
from harv.data import GaiaAstrometryData, RVData, SourceData, SystemData
from harv.models import JointModel, RVModel
from harv.models.priors import HarvPrior, default_sb2_prior
from harv.samplers import NumpyroSampler, RejectionSampler


def _rv_data(start: float = 0.0, n: int = 6) -> RVData:
    times = jnp.linspace(start, start + 100.0, n)
    return RVData(
        time=Q(times, "day"),
        rv=Q(jnp.zeros(n), "km/s"),
        rv_err=Q(jnp.full(n, 0.5), "km/s"),
    )


def _rv_prior() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


_RUN_KWARGS = {"n_prior_samples": 100, "seed": 0}


class TestSingleComponentValidation:
    """Single-component models must receive an AbstractData."""

    def test_rejection_rejects_bare_dict(self):
        sampler = RejectionSampler(_rv_prior(), RVModel())
        with pytest.raises(TypeError, match="AbstractData"):
            sampler.run({"time": [1, 2, 3]}, **_RUN_KWARGS)

    def test_rejection_rejects_container(self):
        """A multi-component container is wrong for a single-component model."""
        sampler = RejectionSampler(_rv_prior(), RVModel())
        container = SourceData(keck=_rv_data())
        with pytest.raises(TypeError, match="AbstractData"):
            sampler.run(container, **_RUN_KWARGS)

    def test_rejection_accepts_abstractdata(self):
        sampler = RejectionSampler(_rv_prior(), RVModel())
        # Should not raise the validation TypeError (may proceed to sample).
        samples = sampler.run(_rv_data(), **_RUN_KWARGS)
        assert samples is not None

    def test_numpyro_run_rejects_bare_dict(self):
        sampler = NumpyroSampler(_rv_prior(), RVModel())
        with pytest.raises(TypeError, match="AbstractData"):
            sampler.run({"time": [1, 2, 3]}, init_samples=None)


class TestJointValidation:
    """JointModel must receive an AbstractDatasetContainer."""

    def _sb2(self) -> tuple[HarvPrior, JointModel]:
        prior = default_sb2_prior(
            period_min=Q(2.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        joint = JointModel.for_sb2(prior=prior)
        return prior, joint

    def test_rejection_rejects_bare_dict(self):
        prior, joint = self._sb2()
        sampler = RejectionSampler(prior, joint)
        with pytest.raises(TypeError, match="AbstractDatasetContainer"):
            sampler.run(
                {"primary": _rv_data(), "secondary": _rv_data(1.0)}, **_RUN_KWARGS
            )

    def test_rejection_rejects_bare_abstractdata(self):
        prior, joint = self._sb2()
        sampler = RejectionSampler(prior, joint)
        with pytest.raises(TypeError, match="AbstractDatasetContainer"):
            sampler.run(_rv_data(), **_RUN_KWARGS)

    def test_rejection_accepts_container(self):
        prior, joint = self._sb2()
        sampler = RejectionSampler(prior, joint)
        data = SystemData(primary=_rv_data(), secondary=_rv_data(1.0))
        samples = sampler.run(data, **_RUN_KWARGS)
        assert samples is not None


class TestContainerTRef:
    """t_ref is exposed uniformly on AbstractDatasetContainer subclasses."""

    def test_systemdata_t_ref_matches_components(self):
        d1 = RVData(
            time=Q(jnp.array([0.0, 1.0, 2.0]), "day"),
            rv=Q(jnp.zeros(3), "km/s"),
            rv_err=Q(jnp.ones(3), "km/s"),
            t_ref=Q(1.0, "day"),
        )
        d2 = RVData(
            time=Q(jnp.array([0.5, 1.5]), "day"),
            rv=Q(jnp.zeros(2), "km/s"),
            rv_err=Q(jnp.ones(2), "km/s"),
            t_ref=Q(1.0, "day"),
        )
        sys = SystemData(primary=d1, secondary=d2)
        assert sys.t_ref is not None
        # Synchronized to the same value across components.
        assert sys.t_ref == sys["primary"].t_ref
        assert sys.t_ref == sys["secondary"].t_ref

    def test_sourcedata_t_ref_available(self):
        """SourceData previously had no t_ref; the base property now provides it."""
        rv = RVData(
            time=Q(jnp.array([0.0, 1.0]), "day"),
            rv=Q(jnp.zeros(2), "km/s"),
            rv_err=Q(jnp.ones(2), "km/s"),
            t_ref=Q(0.5, "day"),
        )
        gaia = GaiaAstrometryData(
            time=Q(jnp.array([0.0, 1.0, 2.0]), "day"),
            al_position=Q(jnp.array([0.1, -0.2, 0.05]), "mas"),
            al_position_err=Q(jnp.array([0.05, 0.06, 0.04]), "mas"),
            scan_angle=Q(jnp.array([0.5, 1.2, 2.8]), "rad"),
            parallax_factor=jnp.array([0.3, -0.1, 0.4]),
            t_ref=Q(0.5, "day"),
        )
        source = SourceData(rv=rv, gaia=gaia)
        assert source.t_ref is not None
        assert source.t_ref == source["rv"].t_ref
