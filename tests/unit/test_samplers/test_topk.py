"""Tests for top-K-by-weight selection on :class:`RejectionSampler`.

Covers:

- :func:`harv.samplers.rejection._top_k_indices` — descending order, static
  output shape, non-finite handling, behaviour under ``jit`` / ``vmap``.
- :attr:`Samples.weight` and ``samples["weight"]`` — reconstruction from
  ``ln_likelihood`` plus the ``logZ_int`` / ``n_prior_samples`` evidence
  metadata, and the guards for missing metadata and batched ``Samples``.
- :func:`harv.samplers.rejection._prior_monte_carlo_evidence_stats` — the
  ``logZ_int_ess`` Kish effective sample size against a direct computation.
- :meth:`RejectionSampler.run` / :meth:`RejectionSampler.run_with_samples`
  with ``top_k=`` — fixed output length across datasets (vs. rejection's
  data-dependent length), ``weight_captured``, invariance to
  ``randomize_prior_order``, agreement with the rejection path, and the
  mutually-exclusive-argument errors.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from unxt import Q

import harv.models as hm
from harv.data import RVData
from harv.models import RVModel
from harv.models.priors import HarvPrior
from harv.samplers import (
    RejectionSampler,
    Samples,
    make_prior_cache,
    pad_and_stack_samples,
)
from harv.samplers.rejection import _prior_monte_carlo_evidence_stats, _top_k_indices


def _rv_data(n: int = 10, *, seed: int = 0, noise: float = 2.0) -> RVData:
    rng = np.random.default_rng(seed)
    return RVData(
        time=Q(jnp.linspace(0.0, 100.0, n), "day"),
        rv=Q(jnp.asarray(rng.normal(0.0, noise, n)), "km/s"),
        rv_err=Q(jnp.full(n, noise), "km/s"),
    )


def _rv_prior() -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


def _sampler(batch_size: int = 250) -> RejectionSampler:
    return RejectionSampler(_rv_prior(), RVModel(), batch_size=batch_size)


def _weighted_samples(ln_lik: np.ndarray, *, n_prior: int | None = None) -> Samples:
    """A hand-built ``Samples`` carrying the metadata ``weight`` needs.

    ``n_prior`` defaults to ``len(ln_lik)``, i.e. the "nothing was truncated"
    case; pass a larger value to emulate a top-K result drawn from a bigger
    library.
    """
    ln_lik_arr = jnp.asarray(ln_lik)
    stats = _prior_monte_carlo_evidence_stats(ln_lik_arr)
    return Samples(
        nonlinear={"period": Q(jnp.arange(len(ln_lik), dtype=ln_lik_arr.dtype), "day")},
        linear={},
        data_type="RVModel",
        metadata={
            "logZ_int": float(stats["logZ_int"]),
            "n_prior_samples": int(n_prior if n_prior is not None else len(ln_lik)),
        },
        ln_likelihood=ln_lik_arr,
    )


class TestTopKIndices:
    """The selection primitive: static shape, descending weight, -inf last."""

    def test_returns_k_largest_in_descending_order(self):
        """Indices point at the k largest log-likelihoods, largest first."""
        ll = jnp.asarray([1.0, 5.0, 3.0, 4.0, 2.0])
        idx = _top_k_indices(ll, 3)
        assert idx.shape == (3,)
        np.testing.assert_array_equal(np.asarray(idx), [1, 3, 2])

    def test_output_shape_is_static_in_k_not_in_data(self):
        """Output length is exactly k regardless of the input values."""
        for seed in range(3):
            ll = jr.normal(jr.key(seed), (200,))
            assert _top_k_indices(ll, 17).shape == (17,)

    def test_non_finite_sorts_last(self):
        """NaN / +-inf never outrank a finite likelihood."""
        ll = jnp.asarray([jnp.nan, -1.0, jnp.inf, -3.0, -2.0])
        idx = np.asarray(_top_k_indices(ll, 3))
        # The three finite entries, in descending order.
        np.testing.assert_array_equal(idx, [1, 4, 3])

    def test_all_non_finite_does_not_produce_nan(self):
        """An all-non-finite run degrades to the first k indices, not NaN.

        The naive implementation normalizes before selecting, which gives
        ``-inf - -inf = NaN``.  This is the case that occurs for a system whose
        every likelihood underflows, so it must stay well-defined.
        """
        ll = jnp.full((10,), -jnp.inf)
        idx = np.asarray(_top_k_indices(ll, 4))
        assert idx.shape == (4,)
        assert not np.isnan(idx).any()

    def test_k_equal_to_m_is_a_full_descending_sort(self):
        """k == M returns every index, ordered by decreasing likelihood."""
        ll = jr.normal(jr.key(0), (50,))
        idx = np.asarray(_top_k_indices(ll, 50))
        assert sorted(idx.tolist()) == list(range(50))
        selected = np.asarray(ll)[idx]
        assert np.all(np.diff(selected) <= 0.0)

    def test_jit_and_vmap(self):
        """Works under an outer jit and vmapped over a batch of likelihoods."""
        ll = jr.normal(jr.key(0), (4, 30))
        direct = jax.vmap(lambda row: _top_k_indices(row, 5))(ll)
        jitted = jax.jit(jax.vmap(lambda row: _top_k_indices(row, 5)))(ll)
        assert direct.shape == (4, 5)
        np.testing.assert_array_equal(np.asarray(direct), np.asarray(jitted))


class TestEvidenceStats:
    """``logZ_int_ess`` is the Kish ESS of the importance weights."""

    def test_ess_matches_direct_computation(self):
        """``logZ_int_ess`` equals a plain ``(sum w)^2 / sum w^2`` on weights."""
        with jax.enable_x64(new_val=True):
            ll = np.asarray(jr.normal(jr.key(3), (500,), dtype=jnp.float64)) * 3.0
            stats = _prior_monte_carlo_evidence_stats(jnp.asarray(ll))
            w = np.exp(ll - ll.max())
            expected = w.sum() ** 2 / (w**2).sum()
            np.testing.assert_allclose(
                float(stats["logZ_int_ess"]), expected, rtol=1e-10
            )

    def test_flat_likelihood_gives_ess_equal_to_m(self):
        """Equal weights are the maximally efficient case: ESS == M."""
        stats = _prior_monte_carlo_evidence_stats(jnp.zeros(64))
        np.testing.assert_allclose(float(stats["logZ_int_ess"]), 64.0, rtol=1e-5)

    def test_delta_likelihood_gives_ess_of_one(self):
        """One sample dominating everything is the worst case: ESS == 1."""
        ll = jnp.asarray([0.0, *([-200.0] * 99)])
        stats = _prior_monte_carlo_evidence_stats(ll)
        np.testing.assert_allclose(float(stats["logZ_int_ess"]), 1.0, rtol=1e-5)


class TestWeightDerivedKey:
    """``Samples.weight`` reconstructs library-normalized importance weights."""

    def test_flat_likelihood_gives_uniform_weights(self):
        """M equal likelihoods each carry weight 1/M."""
        samples = _weighted_samples(np.zeros(32))
        np.testing.assert_allclose(np.asarray(samples.weight), 1.0 / 32, rtol=1e-5)

    def test_weights_sum_to_one_when_nothing_truncated(self):
        """With ``n_prior_samples == n_samples`` the weights are normalized."""
        samples = _weighted_samples(np.asarray(jr.normal(jr.key(1), (128,))) * 2.0)
        np.testing.assert_allclose(float(samples.weight.sum()), 1.0, rtol=1e-5)

    def test_truncation_shows_up_as_a_deficit(self):
        """Normalizing over the full library means a truncated sum is < 1.

        Half a flat library retains exactly half the posterior mass, which is
        what makes ``weight.sum()`` readable as ``weight_captured``.
        """
        ll = np.zeros(50)
        stats = _prior_monte_carlo_evidence_stats(jnp.asarray(np.zeros(100)))
        samples = Samples(
            nonlinear={"period": Q(jnp.arange(50.0), "day")},
            linear={},
            metadata={
                "logZ_int": float(stats["logZ_int"]),
                "n_prior_samples": 100,
            },
            ln_likelihood=jnp.asarray(ll),
        )
        np.testing.assert_allclose(float(samples.weight.sum()), 0.5, rtol=1e-5)

    def test_getitem_matches_property(self):
        """``samples["weight"]`` is the documented access form."""
        samples = _weighted_samples(np.asarray([0.0, -1.0, -2.0]))
        np.testing.assert_allclose(
            np.asarray(samples["weight"]), np.asarray(samples.weight)
        )

    def test_not_listed_in_keys(self):
        """``weight`` is not a model parameter, so it stays out of ``keys()``.

        ``keys()`` drives the default axes of ``plot_corner`` / ``to_arviz``
        and the all-key form of ``median()``; a weight axis there would be
        wrong.
        """
        samples = _weighted_samples(np.zeros(4))
        parameter_names = samples.keys()
        assert "weight" not in parameter_names
        assert "period" in parameter_names

    def test_all_non_finite_gives_zero_not_nan(self):
        """A run where every likelihood was non-finite has zero weight."""
        samples = _weighted_samples(np.full(8, -np.inf))
        w = np.asarray(samples.weight)
        assert not np.isnan(w).any()
        assert np.all(w == 0.0)

    def test_requires_ln_likelihood(self):
        """Without per-sample log-likelihoods there is nothing to weight."""
        samples = Samples(
            nonlinear={"period": Q(jnp.arange(4.0), "day")},
            linear={},
            metadata={"logZ_int": -1.0, "n_prior_samples": 4},
        )
        with pytest.raises(ValueError, match="requires ln_likelihood"):
            _ = samples.weight

    def test_requires_evidence_metadata(self):
        """Without the library normalization the weights are not defined."""
        samples = Samples(
            nonlinear={"period": Q(jnp.arange(4.0), "day")},
            linear={},
            metadata={},
            ln_likelihood=jnp.zeros(4),
        )
        with pytest.raises(ValueError, match="evidence metadata"):
            _ = samples.weight

    def test_raises_for_batched_samples(self):
        """``pad_and_stack_samples`` keeps only the first entry's metadata.

        Its normalization does not apply to the other entries, so a stacked
        ``weight`` would be silently wrong rather than merely unavailable.
        """
        one = _weighted_samples(np.asarray([0.0, -1.0, -2.0]))
        two = _weighted_samples(np.asarray([0.0, -3.0]))
        stacked, _mask = pad_and_stack_samples([one, two])
        with pytest.raises(ValueError, match="batched Samples"):
            _ = stacked.weight


class TestRunWithTopK:
    """End-to-end ``top_k`` behaviour through the public entry points."""

    def test_returns_exactly_k_samples(self):
        """The output length is ``top_k``, not a function of the data."""
        sampler = _sampler()
        samples = sampler.run(_rv_data(), n_prior_samples=1000, top_k=16, seed=0)
        assert samples.n_samples == 16
        assert samples["period"].shape == (16,)

    def test_output_length_is_stable_across_datasets(self):
        """Every dataset yields identically shaped arrays -- the whole point.

        This is the observable form of the performance claim.  The leading axis
        of the returned arrays *is* the ``jax.vmap`` shape that
        ``_sample_linear_parameters`` compiles against, so an invariant shape
        across datasets is exactly the condition for reusing the compiled
        conditional Gaussian solve.  Rejection cannot promise it: its output
        length is ``accepted_mask.sum()``.

        Asserted on shapes rather than on a jit cache counter deliberately --
        ``jax.jit``'s ``_cache_size()`` reports process-global compilation
        state, so it is not a per-function count and drifts (even negative)
        once other tests in the session compile anything.
        """
        sampler = _sampler()
        library = _rv_prior().sample(jr.key(1), 1000, model=RVModel())

        shapes = []
        for seed in range(3):
            samples = sampler.run_with_samples(
                _rv_data(seed=seed), library, top_k=16, seed=0
            )
            assert samples.n_samples == 16
            shapes.append(
                {k: v.shape for k, v in (samples.nonlinear | samples.linear).items()}
            )
        assert shapes[0] == shapes[1] == shapes[2]
        assert set(shapes[0].values()) == {(16,)}

    def test_rejection_output_length_varies_across_datasets(self):
        """The baseline ``top_k`` exists to fix: rejection's length is data-dependent.

        Guards the premise of the feature, so the contrast is asserted rather
        than only asserted about in prose.
        """
        sampler = _sampler()
        library = _rv_prior().sample(jr.key(1), 1000, model=RVModel())
        lengths = {
            sampler.run_with_samples(
                _rv_data(seed=seed, noise=noise), library, seed=0
            ).n_samples
            for seed, noise in ((0, 2.0), (1, 10.0), (2, 30.0))
        }
        assert len(lengths) > 1

    def test_weights_are_non_increasing(self):
        """Rows come back ordered by decreasing weight."""
        sampler = _sampler()
        samples = sampler.run(_rv_data(), n_prior_samples=1000, top_k=32, seed=0)
        w = np.asarray(samples["weight"])
        assert np.all(np.diff(w) <= 1e-12)

    def test_logprobs_and_evidence_stats_are_forced_on(self):
        """``top_k`` populates what ``weight`` needs without being asked.

        Requiring the caller to also pass ``return_logprobs=True`` would be
        pure boilerplate, so the flag is overridden rather than validated.
        """
        sampler = _sampler()
        samples = sampler.run(
            _rv_data(),
            n_prior_samples=1000,
            top_k=8,
            seed=0,
            return_logprobs=False,
            return_evidence_stats=False,
        )
        assert samples.ln_likelihood is not None
        assert samples.ln_prior is not None
        for key in ("logZ_int", "logZ_int_ess", "n_prior_samples", "weight_captured"):
            assert key in samples.metadata

    def test_weight_captured_equals_weight_sum(self):
        """The metadata scalar is exactly the sum of the returned weights."""
        sampler = _sampler()
        samples = sampler.run(_rv_data(), n_prior_samples=1000, top_k=32, seed=0)
        np.testing.assert_allclose(
            samples.metadata["weight_captured"],
            float(samples["weight"].sum()),
            rtol=1e-4,
        )

    def test_weight_captured_is_one_when_k_equals_library_size(self):
        """Nothing truncated means all of the posterior mass survives."""
        sampler = _sampler()
        samples = sampler.run(_rv_data(), n_prior_samples=1000, top_k=1000, seed=0)
        np.testing.assert_allclose(samples.metadata["weight_captured"], 1.0, rtol=1e-4)

    def test_low_capture_is_reported_not_hidden(self):
        """A too-small ``top_k`` shows up as a small ``weight_captured``.

        This is the diagnostic that says the returned weighted samples are a
        biased view of the posterior: with a broad likelihood and ``k << M``,
        ``weight_captured`` collapses towards ``k / M`` and estimates built from
        these samples are *not* posterior estimates.  See ``docs/sharp-bits.md``.
        """
        sampler = _sampler(batch_size=1000)
        library = _rv_prior().sample(jr.key(1), 4000, model=RVModel())
        # Few, noisy observations -> broad likelihood -> large ESS.
        data = _rv_data(4, seed=7, noise=10.0)
        samples = sampler.run_with_samples(data, library, top_k=16, seed=0)
        assert samples.metadata["logZ_int_ess"] > 100.0
        assert samples.metadata["weight_captured"] < 0.5

    def test_selection_is_invariant_to_prior_order(self, tmp_path: Path):
        """Selection depends only on the likelihoods, not on read order.

        ``randomize_prior_order`` permutes which contiguous HDF5 slice is read
        when, so it changes row positions but not the set of library draws with
        the largest weights.  Rejection cannot make this promise.
        """
        sampler = _sampler()
        path = tmp_path / "library.h5"
        make_prior_cache(
            _rv_prior(), RVModel(), 1000, path, key=jr.key(1), batch_size=250
        )
        data = _rv_data(seed=7)
        shuffled = sampler.run_with_samples(
            data, path, top_k=16, seed=3, randomize_prior_order=True
        )
        sequential = sampler.run_with_samples(
            data, path, top_k=16, seed=3, randomize_prior_order=False
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(shuffled["period"].value)),
            np.sort(np.asarray(sequential["period"].value)),
            rtol=1e-6,
        )

    def test_agrees_with_rejection_when_nothing_is_truncated(self):
        """At ``k == M`` the weighted mean matches the rejection mean.

        Both paths consume the same library and the same importance weights --
        rejection by accept/reject, this path by carrying the weights forward --
        so their posterior means must agree up to the rejection path's Monte
        Carlo error.  This is what ties the new path to the already-validated
        one.  It is asserted at ``k == M`` on purpose: with ``k < M`` the
        truncation bias is real and this comparison would (correctly) fail,
        which is what ``weight_captured`` exists to report.
        """
        n_library = 5000
        sampler = _sampler(batch_size=1000)
        library = _rv_prior().sample(jr.key(1), n_library, model=RVModel())
        data = _rv_data(10, seed=7, noise=2.0)

        topk = sampler.run_with_samples(data, library, top_k=n_library, seed=0)
        rejected = sampler.run_with_samples(data, library, seed=0)

        np.testing.assert_allclose(topk.metadata["weight_captured"], 1.0, rtol=1e-4)
        w = np.asarray(topk["weight"])
        w = w / w.sum()
        weighted_mean = float((w * np.asarray(topk["period"].value)).sum())

        periods = np.asarray(rejected["period"].value)
        rejection_mean = float(periods.mean())
        mc_error = float(periods.std() / np.sqrt(periods.size))
        assert abs(weighted_mean - rejection_mean) < 4.0 * mc_error

    def test_preserves_x64_dtype(self):
        """Weights and parameters keep float64 under x64, not silent float32."""
        with jax.enable_x64(new_val=True):
            sampler = _sampler()
            library = _rv_prior().sample(jr.key(1), 500, model=RVModel())
            samples = sampler.run_with_samples(_rv_data(), library, top_k=8, seed=0)
            assert samples["period"].value.dtype == np.float64
            assert samples["weight"].dtype == np.float64

    def test_rejection_path_is_unchanged(self):
        """Omitting ``top_k`` still gives a data-dependent rejection result."""
        sampler = _sampler()
        samples = sampler.run(_rv_data(), n_prior_samples=1000, top_k=None, seed=0)
        assert samples.n_samples > 0
        assert "weight_captured" not in samples.metadata


class TestTopKErrors:
    """Argument validation, at the entry point rather than after evaluation."""

    def test_rejects_max_posterior_samples_and_top_k_together(self):
        """Two conflicting output-shape policies is always a mistake."""
        sampler = _sampler()
        with pytest.raises(ValueError, match="mutually exclusive"):
            sampler.run(
                _rv_data(),
                n_prior_samples=500,
                top_k=8,
                max_posterior_samples=8,
                seed=0,
            )

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_rejects_non_positive_top_k(self, top_k: int):
        """``top_k`` must ask for at least one sample."""
        sampler = _sampler()
        with pytest.raises(ValueError, match="positive integer"):
            sampler.run(_rv_data(), n_prior_samples=500, top_k=top_k, seed=0)

    def test_rejects_top_k_larger_than_library(self):
        """A short return would defeat the fixed-shape contract."""
        sampler = _sampler()
        with pytest.raises(ValueError, match="exceeds the number of prior samples"):
            sampler.run(_rv_data(), n_prior_samples=500, top_k=501, seed=0)

    def test_run_with_samples_rejects_conflicting_arguments(self):
        """The same validation applies on the pre-computed-library path."""
        sampler = _sampler()
        library = _rv_prior().sample(jr.key(1), 500, model=RVModel())
        with pytest.raises(ValueError, match="mutually exclusive"):
            sampler.run_with_samples(
                _rv_data(), library, top_k=8, max_posterior_samples=8, seed=0
            )
