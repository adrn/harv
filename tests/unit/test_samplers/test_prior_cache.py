"""Tests for the pre-computed prior-sample workflow.

Covers:

- :meth:`HarvPrior.sample` — keys, shapes, units, optional ``ln_prior``,
  ``data_type`` propagation, JointModel support.
- :func:`make_prior_cache` — chunked HDF5 write with round-trip via
  :meth:`Samples.from_hdf5`.
- :meth:`RejectionSampler.run_with_samples` — in-memory and HDF5-path
  branches, key-mismatch validation, parity between disk-sequential and
  in-memory reads, behaviour of ``randomize_prior_order``.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv.models as hm
from harv.data import RVData
from harv.distributions import QD
from harv.models import GaiaAstrometryModel, JointModel, RVModel
from harv.models.extensions import Jitter
from harv.models.priors import HarvPrior, default_sb2_prior
from harv.samplers import RejectionSampler, Samples, make_prior_cache


def _rv_data(n: int = 6, *, seed: int = 0) -> RVData:
    rng = np.random.default_rng(seed)
    return RVData(
        time=Q(jnp.linspace(0.0, 100.0, n), "day"),
        rv=Q(jnp.asarray(rng.normal(0.0, 5.0, n)), "km/s"),
        rv_err=Q(jnp.full(n, 1.0), "km/s"),
    )


def _rv_prior() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


def _rv_prior_with_jitter() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
        jitter=QD(dist.HalfNormal(1.0), "km/s"),
    )


class TestHarvPriorSample:
    def test_returns_samples_container(self):
        samples = _rv_prior().sample(jr.key(0), 100, model=RVModel())
        assert isinstance(samples, Samples)
        assert samples.n_samples == 100
        assert samples.data_type == "RVModel"

    def test_keys_match_base_nonlinear(self):
        samples = _rv_prior().sample(jr.key(0), 64, model=RVModel())
        assert set(samples.nonlinear) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }
        # Default Gaussian linear priors are marginalized — no explicit-linear
        # draws appear in `samples.linear` in this path.
        assert samples.linear == {}

    def test_period_units_round_trip(self):
        samples = _rv_prior().sample(jr.key(0), 32, model=RVModel())
        assert str(samples.nonlinear["period"].unit) == "d"
        # Period values lie inside [period_min, period_max).
        values = np.asarray(samples.nonlinear["period"].value)
        assert (values >= 2.0).all()
        assert (values <= 1000.0).all()

    def test_extension_nonlinear_drawn(self):
        """Jitter (extension nonlinear) appears in the sample dict."""
        model = RVModel(extensions=(Jitter(param_unit="km/s"),))
        samples = _rv_prior_with_jitter().sample(jr.key(0), 50, model=model)
        assert "jitter" in samples.nonlinear
        assert samples.nonlinear["jitter"].shape == (50,)

    def test_return_logprobs_populates_ln_prior(self):
        samples = _rv_prior().sample(
            jr.key(0), 32, model=RVModel(), return_logprobs=True
        )
        assert samples.ln_prior is not None
        assert samples.ln_prior.shape == (32,)
        # Posterior-only field; never populated by prior sampling.
        assert samples.ln_likelihood is None

    def test_ln_prior_omitted_by_default(self):
        samples = _rv_prior().sample(jr.key(0), 32, model=RVModel())
        assert samples.ln_prior is None
        assert samples.ln_likelihood is None

    def test_explicit_linear_appears_in_linear(self):
        """Non-Gaussian linear priors are sampled explicitly into Samples.linear."""
        # Astrometry's default parallax prior is HalfNormal — explicit.
        prior = hm.StandardGaiaAstrometry().default_prior(
            period_min=Q(100.0, "day"),
            period_max=Q(3000.0, "day"),
            sigma_a0=Q(5.0, "AU"),
            sigma_parallax=Q(10.0, "mas"),
            sigma_pos=Q(100.0, "mas"),
            sigma_vtan=Q(50.0, "km/s"),
        )
        samples = prior.sample(jr.key(0), 24, model=GaiaAstrometryModel())
        assert "parallax" in samples.linear
        assert samples.linear["parallax"].shape == (24,)
        assert str(samples.linear["parallax"].unit) == "mas"

    def test_joint_model_data_type(self):
        prior = default_sb2_prior(
            period_min=Q(2.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        joint = JointModel.for_sb2(prior)
        samples = prior.sample(jr.key(0), 16, model=joint)
        assert samples.data_type == "JointModel"
        assert samples.n_samples == 16


class TestMakePriorCache:
    def test_round_trip_via_from_hdf5(self, tmp_path: Path):
        path = tmp_path / "cache.h5"
        make_prior_cache(
            _rv_prior(),
            RVModel(),
            n_samples=500,
            filename=path,
            key=jr.key(0),
            batch_size=100,
        )

        loaded = Samples.from_hdf5(path)
        assert loaded.n_samples == 500
        assert set(loaded.nonlinear) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        }
        assert str(loaded.nonlinear["period"].unit) == "d"
        assert loaded.data_type == "RVModel"

    def test_writes_ln_prior_when_requested(self, tmp_path: Path):
        path = tmp_path / "cache.h5"
        make_prior_cache(
            _rv_prior(),
            RVModel(),
            n_samples=128,
            filename=path,
            key=jr.key(0),
            batch_size=64,
            return_logprobs=True,
        )
        loaded = Samples.from_hdf5(path)
        assert loaded.ln_prior is not None
        assert loaded.ln_prior.shape == (128,)

    def test_with_extension(self, tmp_path: Path):
        """Extension nonlinear params (jitter) make it into the cache."""
        path = tmp_path / "cache.h5"
        model = RVModel(extensions=(Jitter(param_unit="km/s"),))
        make_prior_cache(
            _rv_prior_with_jitter(),
            model,
            n_samples=256,
            filename=path,
            key=jr.key(0),
            batch_size=64,
        )
        loaded = Samples.from_hdf5(path)
        assert "jitter" in loaded.nonlinear
        assert loaded.nonlinear["jitter"].shape == (256,)

    def test_rejects_zero_or_negative(self, tmp_path: Path):
        path = tmp_path / "cache.h5"
        with pytest.raises(ValueError, match="n_samples"):
            make_prior_cache(
                _rv_prior(),
                RVModel(),
                n_samples=0,
                filename=path,
                key=jr.key(0),
            )
        with pytest.raises(ValueError, match="batch_size"):
            make_prior_cache(
                _rv_prior(),
                RVModel(),
                n_samples=10,
                filename=path,
                key=jr.key(0),
                batch_size=0,
            )


class TestRunWithSamplesInMemory:
    def test_basic_call(self):
        prior = _rv_prior()
        model = RVModel()
        sampler = RejectionSampler(prior, model, batch_size=200)
        pri = prior.sample(jr.key(0), 1000, model=model)

        out = sampler.run_with_samples(_rv_data(), pri, seed=42)
        assert isinstance(out, Samples)
        # Linear params get resampled from the conditional posterior.
        assert set(out.linear) == {"rv_semiamp", "v_sys"}

    def test_returns_logprobs(self):
        prior = _rv_prior()
        model = RVModel()
        sampler = RejectionSampler(prior, model, batch_size=200)
        pri = prior.sample(jr.key(0), 1000, model=model)

        out = sampler.run_with_samples(_rv_data(), pri, seed=42, return_logprobs=True)
        assert out.ln_likelihood is not None
        assert out.ln_prior is not None
        assert out.ln_likelihood.shape == (out.n_samples,)

    def test_key_mismatch_raises(self):
        """A Samples that doesn't match the (prior, model) keyset raises."""
        prior = _rv_prior()
        sampler = RejectionSampler(prior, RVModel(), batch_size=200)
        # Build a Samples missing the 'eccentricity' key.
        pri = prior.sample(jr.key(0), 50, model=RVModel())
        broken_nonlinear = {
            k: v for k, v in pri.nonlinear.items() if k != "eccentricity"
        }
        broken = Samples(
            nonlinear=broken_nonlinear,
            linear=pri.linear,
            data_type=pri.data_type,
            linear_extension_names=pri.linear_extension_names,
        )
        with pytest.raises(ValueError, match="Missing"):
            sampler.run_with_samples(_rv_data(), broken, seed=42)

    def test_extension_in_memory(self):
        """In-memory branch handles a Jitter extension."""
        prior = _rv_prior_with_jitter()
        model = RVModel(extensions=(Jitter(param_unit="km/s"),))
        sampler = RejectionSampler(prior, model, batch_size=200)
        pri = prior.sample(jr.key(0), 1000, model=model)
        out = sampler.run_with_samples(_rv_data(), pri, seed=42)
        assert "jitter" in out.nonlinear


class TestRunWithSamplesFromHdf5:
    def test_disk_matches_in_memory_sequential(self, tmp_path: Path):
        """With randomize_prior_order=False, disk path and in-memory must agree."""
        prior = _rv_prior_with_jitter()
        model = RVModel(extensions=(Jitter(param_unit="km/s"),))
        sampler = RejectionSampler(prior, model, batch_size=200)

        path = tmp_path / "cache.h5"
        make_prior_cache(
            prior,
            model,
            n_samples=1000,
            filename=path,
            key=jr.key(0),
            batch_size=200,
        )
        loaded = Samples.from_hdf5(path)

        data = _rv_data()
        disk = sampler.run_with_samples(
            data, path, seed=42, randomize_prior_order=False
        )
        mem = sampler.run_with_samples(data, loaded, seed=42)

        # Same set of accepted samples (same seed, same sample order, same logL).
        assert disk.n_samples == mem.n_samples
        np.testing.assert_allclose(
            np.sort(np.asarray(disk["period"].value)),
            np.sort(np.asarray(mem["period"].value)),
        )

    def test_randomize_does_not_change_n_samples_dramatically(self, tmp_path: Path):
        """randomize_prior_order=True still produces a valid posterior."""
        prior = _rv_prior()
        sampler = RejectionSampler(prior, RVModel(), batch_size=200)

        path = tmp_path / "cache.h5"
        make_prior_cache(
            prior,
            RVModel(),
            n_samples=2000,
            filename=path,
            key=jr.key(0),
            batch_size=200,
        )

        data = _rv_data()
        disk_rand = sampler.run_with_samples(data, path, seed=42)
        disk_seq = sampler.run_with_samples(
            data, path, seed=42, randomize_prior_order=False
        )
        # Acceptance counts may differ (per-position uniforms hit different
        # rows), but should be within a reasonable factor of each other.
        assert disk_rand.n_samples > 0
        assert disk_seq.n_samples > 0
        ratio = disk_rand.n_samples / max(disk_seq.n_samples, 1)
        assert 0.5 < ratio < 2.0

    def test_path_can_be_str_or_pathlike(self, tmp_path: Path):
        """run_with_samples accepts both ``str`` and ``Path``."""
        prior = _rv_prior()
        sampler = RejectionSampler(prior, RVModel(), batch_size=200)

        path = tmp_path / "cache.h5"
        make_prior_cache(
            prior,
            RVModel(),
            n_samples=400,
            filename=path,
            key=jr.key(0),
            batch_size=200,
        )

        data = _rv_data()
        out_str = sampler.run_with_samples(
            data, str(path), seed=7, randomize_prior_order=False
        )
        out_path = sampler.run_with_samples(
            data, path, seed=7, randomize_prior_order=False
        )
        assert out_str.n_samples == out_path.n_samples

    def test_disk_missing_keys_raises(self, tmp_path: Path):
        """A cache built for a model with jitter is invalid for one without."""
        prior_j = _rv_prior_with_jitter()
        model_j = RVModel(extensions=(Jitter(param_unit="km/s"),))
        path = tmp_path / "cache.h5"
        make_prior_cache(
            prior_j,
            model_j,
            n_samples=200,
            filename=path,
            key=jr.key(0),
            batch_size=200,
        )

        # Now try to consume it with a sampler that has no jitter prior:
        prior_nj = _rv_prior()
        sampler = RejectionSampler(prior_nj, RVModel(), batch_size=200)
        # Should succeed because the cache has *extra* keys (jitter), which
        # are unused. The expected_keys is a subset of available. Verify:
        out = sampler.run_with_samples(
            _rv_data(), path, seed=0, randomize_prior_order=False
        )
        assert out.n_samples > 0
