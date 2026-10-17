"""Tests for streaming the prior cache in :meth:`RejectionSampler.run_with_samples`.

Covers the running top-k (identical to selecting over the whole library) and
``stop_at_ess``: where it stops, that it does not stop for constrained data or
before ``top_k`` rows, that the stopped ``ln_Z_int`` agrees with the full one,
and its argument errors.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from unxt import Q

import harv.models as hm
from harv.data import RVData
from harv.models import RVModel
from harv.models.priors import HarvPrior
from harv.samplers import RejectionSampler, Samples, make_prior_cache

N_LIBRARY = 8000
BATCH = 500


def _prior() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


def _sampler() -> RejectionSampler:
    return RejectionSampler(_prior(), RVModel(), batch_size=BATCH)


def _broad_data() -> RVData:
    """Three noisy points: the posterior is close to the prior."""
    return RVData(
        time=Q(jnp.array([0.0, 40.0, 90.0]), "day"),
        rv=Q(jnp.array([1.0, -2.0, 0.5]), "km/s"),
        rv_err=Q(jnp.full(3, 5.0), "km/s"),
    )


def _tight_data() -> RVData:
    """A well-sampled 17-day orbit: a handful of draws carry the evidence."""
    t = jnp.linspace(0.0, 200.0, 25)
    return RVData(
        time=Q(t, "day"),
        rv=Q(20.0 * jnp.sin(2 * jnp.pi * t / 17.0), "km/s"),
        rv_err=Q(jnp.full(25, 0.1), "km/s"),
    )


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cache") / "library.h5"
    make_prior_cache(_prior(), RVModel(), N_LIBRARY, path, key=jr.key(0))
    return path


def _run(path: Path | Samples, data: RVData, **kwargs: object) -> Samples:
    return _sampler().run_with_samples(
        data, path, key=jr.key(1), randomize_prior_order=False, **kwargs
    )


class TestStreamingTopK:
    def test_matches_selection_over_the_whole_library(self, library: Path):
        """Merging per batch picks the same rows, in the same order, as one
        top-k over the whole library (the in-memory path)."""
        streamed = _run(library, _broad_data(), top_k=64)
        whole = _run(Samples.from_hdf5(library), _broad_data(), top_k=64)
        np.testing.assert_array_equal(
            np.asarray(streamed["period"].value), np.asarray(whole["period"].value)
        )
        np.testing.assert_array_equal(
            np.asarray(streamed.ln_likelihood), np.asarray(whole.ln_likelihood)
        )
        for name in ("ln_Z_int", "ln_Z_int_ess", "weight_captured"):
            np.testing.assert_allclose(
                streamed.metadata[name], whole.metadata[name], rtol=1e-10
            )
        assert streamed.metadata["n_prior_samples"] == N_LIBRARY


class TestStopAtESS:
    def test_stops_early_for_a_broad_posterior(self, library: Path):
        samples = _run(library, _broad_data(), top_k=64, stop_at_ess=300)
        n_read = samples.metadata["n_prior_samples"]
        assert n_read < N_LIBRARY
        assert n_read % BATCH == 0
        assert samples.metadata["ln_Z_int_ess"] >= 300
        assert samples.n_samples == 64

    def test_stopped_evidence_agrees_with_the_full_library(self, library: Path):
        stopped = _run(library, _broad_data(), top_k=64, stop_at_ess=300).metadata
        full = _run(library, _broad_data(), top_k=64).metadata
        sigma = np.hypot(stopped["ln_Z_int_mcse"], full["ln_Z_int_mcse"])
        assert abs(stopped["ln_Z_int"] - full["ln_Z_int"]) < 4 * sigma

    def test_constrained_data_read_the_whole_library(self, library: Path):
        samples = _run(library, _tight_data(), top_k=16, stop_at_ess=300)
        assert samples.metadata["n_prior_samples"] == N_LIBRARY

    def test_never_stops_before_top_k_rows(self, library: Path):
        """A threshold below ``top_k`` still fills the fixed-shape output."""
        samples = _run(library, _broad_data(), top_k=1200, stop_at_ess=1)
        assert samples.metadata["n_prior_samples"] == 3 * BATCH
        assert samples.n_samples == 1200

    def test_rejection_path_stops_too(self, library: Path):
        samples = _run(
            library, _broad_data(), stop_at_ess=300, return_evidence_stats=True
        )
        assert samples.metadata["n_prior_samples"] < N_LIBRARY
        assert samples.n_samples > 0

    def test_default_reads_everything(self, library: Path):
        samples = _run(library, _broad_data(), return_evidence_stats=True)
        assert samples.metadata["n_prior_samples"] == N_LIBRARY

    @pytest.mark.parametrize("value", [0, -5.0])
    def test_rejects_non_positive(self, library: Path, value: float):
        with pytest.raises(ValueError, match="must be positive"):
            _run(library, _broad_data(), stop_at_ess=value)

    def test_rejects_an_in_memory_library(self, library: Path):
        with pytest.raises(ValueError, match="in-memory library"):
            _run(Samples.from_hdf5(library), _broad_data(), stop_at_ess=300)
