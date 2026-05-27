"""Tests for ``Sampler.get_extensions()`` on RejectionSampler and NumpyroSampler.

The method walks the attached model:

  - single-component model -> returns ``model.extensions`` (a tuple)
  - JointModel -> returns ``dict[name, tuple[Extension, ...]]``

These tests just construct samplers and assert on the return shape/contents -- no
sampling is performed.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from unxt import Q

import harv.models as hm
from harv.data import RVData
from harv.distributions import QD
from harv.models.extensions import Jitter, MultiSurveyOffset
from harv.models.joint import JointModel
from harv.models.priors import HarvPrior, default_sb2_prior
from harv.models.rv import RVModel
from harv.samplers.base import AbstractSampler
from harv.samplers.numpyro import NumpyroSampler
from harv.samplers.rejection import RejectionSampler


def _rv_data(start: float = 0.0, n: int = 6) -> RVData:
    times = jnp.linspace(start, start + 100.0, n)
    return RVData(
        time=Q(times, "day"),
        rv=Q(jnp.zeros(n), "km/s"),
        rv_err=Q(jnp.full(n, 0.5), "km/s"),
    )


def _basic_prior() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


def _jitter_prior() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
        jitter=QD(dist.HalfNormal(1.0), "km/s"),
    )


# ---------------------------------------------------------------------------
# RejectionSampler
# ---------------------------------------------------------------------------


class TestRejectionSamplerGetExtensions:
    def test_no_extensions(self):
        sampler = RejectionSampler(_basic_prior(), RVModel())
        assert sampler.get_extensions() == ()

    def test_with_jitter(self):
        ext = Jitter(param_unit="km/s")
        sampler = RejectionSampler(_jitter_prior(), RVModel(extensions=(ext,)))
        result = sampler.get_extensions()
        assert isinstance(result, tuple)
        assert result == (ext,)

    def test_constructor_with_component_model(self):
        ext = Jitter(param_unit="km/s")
        model = RVModel(extensions=(ext,))
        sampler = RejectionSampler(_jitter_prior(), model)
        result = sampler.get_extensions()
        assert isinstance(result, tuple)
        assert result == (ext,)

    def test_constructor_with_multisurvey_offset(self):
        """The case that motivated this method: MultiSurveyOffset built from data."""
        from harv.simulate.rv import simulate_rv_multisurv_data  # noqa: PLC0415

        source_data, _ = simulate_rv_multisurv_data(
            instruments={"keck": None, "harps": Q(2.0, "km/s")},
            seed=0,
            n_obs_per_instrument=10,
            period=Q(50.0, "day"),
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(2.0, "km/s"),
        )
        _, indicator, names = source_data.indicator_data_by_type(
            RVData, reference="keck"
        )
        prior = hm.StandardRV().default_prior(
            period_min=Q(40.0, "day"),
            period_max=Q(60.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
            harps=QD(dist.Normal(0.0, 5.0), "km/s"),
        )
        ext = MultiSurveyOffset(indicator, names, "km/s")
        model = RVModel(extensions=(ext,))
        sampler = RejectionSampler(prior, model)
        result = sampler.get_extensions()
        assert isinstance(result, tuple)
        assert result == (ext,)

    def test_constructor_with_joint_model_returns_dict(self):
        """Per-component association is preserved as a dict[name, tuple]."""
        primary_ext = Jitter(param_unit="km/s")
        secondary_ext = Jitter(param_unit="km/s")

        prior = default_sb2_prior(
            period_min=Q(2.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        joint = JointModel.for_sb2(
            prior=prior,
            extensions={"primary": (primary_ext,), "secondary": (secondary_ext,)},
        )
        sampler = RejectionSampler(prior, joint)
        result = sampler.get_extensions()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"primary", "secondary"}
        assert result["primary"] == (primary_ext,)
        assert result["secondary"] == (secondary_ext,)


# ---------------------------------------------------------------------------
# NumpyroSampler
# ---------------------------------------------------------------------------


class TestNumpyroSamplerGetExtensions:
    def test_with_jitter(self):
        ext = Jitter(param_unit="km/s")
        sampler = NumpyroSampler(_jitter_prior(), RVModel(extensions=(ext,)))
        result = sampler.get_extensions()
        assert isinstance(result, tuple)
        assert result == (ext,)

    def test_constructor_with_component_model(self):
        ext = Jitter(param_unit="km/s")
        model = RVModel(extensions=(ext,))
        sampler = NumpyroSampler(_jitter_prior(), model)
        result = sampler.get_extensions()
        assert isinstance(result, tuple)
        assert result == (ext,)


# ---------------------------------------------------------------------------
# Inheritance assertion
# ---------------------------------------------------------------------------


def test_rejection_sampler_inherits_abstract_sampler():
    """Concrete sampler must be a subclass of AbstractSampler."""
    sampler = RejectionSampler(_basic_prior(), RVModel())
    assert isinstance(sampler, AbstractSampler)


def test_numpyro_sampler_inherits_abstract_sampler():
    """Concrete sampler must be a subclass of AbstractSampler."""
    sampler = NumpyroSampler(_basic_prior(), RVModel())
    assert isinstance(sampler, AbstractSampler)
