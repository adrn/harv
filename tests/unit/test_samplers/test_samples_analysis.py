"""Tests for Samples analysis methods and derived physical quantities.

Covers the functionality ported from ``thejoker.samples_analysis`` plus the
mass-function / physical-orbit-size helpers and the optional per-sample
log-probability storage.
"""

import shutil
import uuid
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Q, ustrip

from harv import RVData, RejectionPrior, RejectionSampler
from harv.kepler.orbits import rv_at_times
from harv.models.extensions import Jitter
from harv.models.rv import RVModel
from harv.samplers.samples import Samples


def _rv_samples(n=120, *, period=None, with_logprobs=False, seed=0):
    """Build a minimal RV Samples object with ``n`` draws."""
    rng = np.random.default_rng(seed)
    if period is None:
        period = rng.uniform(40.0, 60.0, n)
    nonlinear = {
        "period": Q(np.asarray(period, dtype=float), "day"),
        "eccentricity": Q(rng.uniform(0.0, 0.3, n), ""),
        "phase_peri": Q(rng.uniform(0.0, 1.0, n), ""),
        "arg_peri": Q(rng.uniform(0.0, 2 * np.pi, n), "rad"),
    }
    linear = {
        "rv_semiamp": Q(rng.uniform(5.0, 15.0, n), "km/s"),
        "v_sys": Q(rng.uniform(-1.0, 1.0, n), "km/s"),
    }
    kwargs = {}
    if with_logprobs:
        kwargs["ln_likelihood"] = jnp.asarray(rng.normal(size=n))
        kwargs["ln_prior"] = jnp.asarray(rng.normal(size=n))
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="RVModel",
        metadata={"t_ref": 0.0, "t_ref_unit": "day"},
        **kwargs,
    )


def _astro_samples(n=60, seed=1):
    """Build a minimal Gaia-astrometry Samples object."""
    rng = np.random.default_rng(seed)
    nonlinear = {
        "period": Q(rng.uniform(300.0, 500.0, n), "day"),
        "eccentricity": Q(rng.uniform(0.0, 0.3, n), ""),
        "phase_peri": Q(rng.uniform(0.0, 1.0, n), ""),
        "arg_peri": Q(rng.uniform(0.0, 2 * np.pi, n), "rad"),
        "cos_i": Q(rng.uniform(0.1, 0.9, n), ""),
        "lon_asc_node": Q(rng.uniform(0.0, 2 * np.pi, n), "rad"),
    }
    linear = {
        "ra0": Q(np.zeros(n), "mas"),
        "dec0": Q(np.zeros(n), "mas"),
        "pmra": Q(np.full(n, 10.0), "mas/yr"),
        "pmdec": Q(np.full(n, -5.0), "mas/yr"),
        "parallax": Q(np.full(n, 20.0), "mas"),
        "semi_major_axis": Q(rng.uniform(1.0, 4.0, n), "mas"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="GaiaAstrometryModel",
        metadata={"t_ref": 0.0, "t_ref_unit": "day"},
    )


@pytest.fixture
def rv_data():
    rng = np.random.default_rng(42)
    t = np.sort(rng.uniform(0.0, 200.0, 16))
    return RVData(
        time=Q(t, "day"),
        rv=Q(rng.normal(0.0, 10.0, 16), "km/s"),
        rv_err=Q(np.full(16, 1.0), "km/s"),
    )


# ---------------------------------------------------------------------------
# Log-probability storage + MAP sample
# ---------------------------------------------------------------------------


class TestLogProbStorage:
    def test_fields_default_none(self):
        s = _rv_samples()
        assert s.ln_likelihood is None
        assert s.ln_prior is None

    def test_ln_posterior_requires_both(self):
        s = _rv_samples()
        with pytest.raises(ValueError, match="return_logprobs=True"):
            _ = s.ln_posterior

    def test_ln_posterior_sum(self):
        s = _rv_samples(with_logprobs=True)
        assert jnp.allclose(s.ln_posterior, s.ln_prior + s.ln_likelihood)

    def test_map_sample(self):
        s = _rv_samples(n=50, with_logprobs=True)
        expected = int(jnp.argmax(s.ln_prior + s.ln_likelihood))
        map_s, idx = s.map_sample(return_index=True)
        assert idx == expected
        assert map_s.n_samples == 1
        assert jnp.allclose(map_s["period"].value, s["period"].value[expected])

    def test_map_sample_without_logprobs_raises(self):
        with pytest.raises(ValueError, match="return_logprobs=True"):
            _rv_samples().map_sample()

    def test_slicing_carries_logprobs(self):
        s = _rv_samples(n=40, with_logprobs=True)
        sub = s[:10]
        assert sub.ln_likelihood.shape == (10,)
        assert sub.ln_prior.shape == (10,)
        assert jnp.allclose(sub.ln_likelihood, s.ln_likelihood[:10])

    def test_hdf5_roundtrip(self, tmp_path_in_repo):
        s = _rv_samples(n=30, with_logprobs=True)
        path = tmp_path_in_repo / "samples_logprob.h5"
        s.to_hdf5(path)
        loaded = Samples.from_hdf5(path)
        assert jnp.allclose(loaded.ln_likelihood, s.ln_likelihood)
        assert jnp.allclose(loaded.ln_prior, s.ln_prior)

    def test_hdf5_roundtrip_without_logprobs(self, tmp_path_in_repo):
        s = _rv_samples(n=30)
        path = tmp_path_in_repo / "samples_plain.h5"
        s.to_hdf5(path)
        loaded = Samples.from_hdf5(path)
        assert loaded.ln_likelihood is None
        assert loaded.ln_prior is None


@pytest.fixture
def tmp_path_in_repo():
    """A temporary directory inside the repository (CLAUDE.md file rules)."""
    d = Path(__file__).parent / f"_tmp_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sample analysis (ported from thejoker.samples_analysis)
# ---------------------------------------------------------------------------


class TestPeriodModality:
    def test_unimodal_true_for_tight_periods(self, rv_data):
        jitter = np.random.default_rng(0).normal(0.0, 1e-4, 50)
        s = _rv_samples(n=50, period=np.full(50, 50.0) + jitter)
        assert s.period_unimodal(rv_data) is True

    def test_unimodal_false_for_spread_periods(self, rv_data):
        s = _rv_samples(n=50, period=np.linspace(40.0, 60.0, 50))
        assert s.period_unimodal(rv_data) is False

    def test_period_modes_finds_two_clusters(self, rv_data):
        periods = np.concatenate([np.full(40, 50.0), np.full(40, 200.0)])
        s = _rv_samples(n=80, period=periods)
        all_unimodal, mode_periods, n_per_mode = s.period_modes(rv_data, n_clusters=2)
        assert mode_periods.shape == (2,)
        assert int(n_per_mode.sum()) == 80
        assert sorted(int(c) for c in n_per_mode) == [40, 40]
        # Each cluster has identical periods, so every mode is unimodal.
        assert all_unimodal is True


class TestPhaseStatistics:
    def test_max_phase_gap_shape_and_range(self, rv_data):
        gaps = _rv_samples(n=30).max_phase_gap(rv_data)
        assert gaps.shape == (30,)
        assert np.all((gaps >= 0.0) & (gaps <= 1.0))

    def test_phase_coverage_shape_and_range(self, rv_data):
        cov = _rv_samples(n=30).phase_coverage(rv_data, n_bins=10)
        assert cov.shape == (30,)
        assert np.all((cov >= 0.0) & (cov <= 1.0))

    def test_phase_coverage_known_value(self):
        # period = 10 d, t_ref = 0: phase = (time / 10) mod 1. Times 0.5, 1.5,
        # 2.5 d give phases 0.05, 0.15, 0.25 -> bins 0, 1, 2 of 10 occupied.
        data = RVData(
            time=Q(np.array([0.5, 1.5, 2.5]), "day"),
            rv=Q(np.zeros(3), "km/s"),
            rv_err=Q(np.ones(3), "km/s"),
            t_ref=Q(0.0, "day"),
        )
        s = _rv_samples(n=5, period=np.full(5, 10.0))
        cov = s.phase_coverage(data, n_bins=10)
        assert cov.shape == (5,)
        # 3 of 10 bins occupied, identically for every sample.
        assert np.allclose(cov, 0.3)

    def test_periods_spanned(self, rv_data):
        s = _rv_samples(n=20, period=np.full(20, 50.0))
        spanned = s.periods_spanned(rv_data)
        span = float(np.ptp(ustrip("day", rv_data.time)))
        assert np.allclose(spanned, span / 50.0)

    def test_phase_coverage_per_period_counts_all_obs(self, rv_data):
        # A period longer than the data span puts every obs in one window.
        s = _rv_samples(n=10, period=np.full(10, 1e5))
        counts = s.phase_coverage_per_period(rv_data)
        assert counts.shape == (10,)
        assert np.all(counts == rv_data.n_times)


class TestSingleComponentGuard:
    def test_namespaced_samples_raise(self, rv_data):
        s = Samples(
            nonlinear={"primary.period": Q(jnp.full(5, 50.0), "day")},
            linear={"primary.rv_semiamp": Q(jnp.full(5, 5.0), "km/s")},
            data_type="JointModel",
            metadata={"t_ref": 0.0},
        )
        with pytest.raises(NotImplementedError, match="single-component"):
            s.periods_spanned(rv_data)


# ---------------------------------------------------------------------------
# Derived physical quantities
# ---------------------------------------------------------------------------


class TestDerivedQuantities:
    def test_binary_mass_function_rv(self):
        bmf = _rv_samples(n=20).binary_mass_function()
        assert bmf.unit.is_equivalent(Q(1.0, "Msun").unit)
        assert bmf.shape == (20,)
        assert jnp.all(ustrip("Msun", bmf) > 0)

    def test_binary_mass_function_requires_rv(self):
        with pytest.raises(KeyError, match="rv_semiamp"):
            _astro_samples().binary_mass_function()

    def test_semi_major_axis_au(self):
        a = _astro_samples(n=15).semi_major_axis_AU()
        assert a.unit.is_equivalent(Q(1.0, "AU").unit)
        assert a.shape == (15,)
        assert jnp.all(ustrip("AU", a) > 0)

    def test_semi_major_axis_au_requires_astrometry(self):
        with pytest.raises(KeyError, match="semi_major_axis"):
            _rv_samples().semi_major_axis_AU()

    def test_companion_mass_rv(self):
        s = _rv_samples(n=20)
        m2 = s.companion_mass(Q(1.0, "Msun"))
        assert m2.unit.is_equivalent(Q(1.0, "Msun").unit)
        assert m2.shape == (20,)
        assert jnp.all(ustrip("Msun", m2) > 0)

    def test_minimum_companion_mass_is_smallest(self):
        s = _rv_samples(n=20)
        m1 = Q(1.0, "Msun")
        m2_min = s.minimum_companion_mass(m1)
        m2_incl = s.companion_mass(m1, sini=0.5)
        assert jnp.all(ustrip("Msun", m2_incl) >= ustrip("Msun", m2_min))

    def test_companion_mass_astrometry(self):
        m2 = _astro_samples(n=15).companion_mass(Q(1.2, "Msun"))
        assert m2.shape == (15,)
        assert jnp.all(ustrip("Msun", m2) > 0)

    def test_derived_keys_present(self):
        rv_keys = _rv_samples().keys()
        astro_keys = _astro_samples().keys()
        assert "binary_mass_function" in rv_keys
        assert "semi_major_axis_AU" in astro_keys
        assert "binary_mass_function" not in astro_keys

    def test_derived_key_getitem(self):
        s = _rv_samples(n=12)
        assert jnp.allclose(
            ustrip("Msun", s["binary_mass_function"]),
            ustrip("Msun", s.binary_mass_function()),
        )


# ---------------------------------------------------------------------------
# Integration: RejectionSampler return_logprobs
# ---------------------------------------------------------------------------


class TestRejectionSamplerLogProbs:
    def test_return_logprobs_populates_fields(self, rv_data):
        prior = RejectionPrior.default_rv(
            period_min=Q(10.0, "day"),
            period_max=Q(400.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        sampler = RejectionSampler(prior, RVModel())
        s = sampler.run(rv_data, n_prior_samples=20_000, seed=7, return_logprobs=True)
        assert s.ln_likelihood is not None
        assert s.ln_prior is not None
        assert s.ln_likelihood.shape == (s.n_samples,)
        assert s.ln_prior.shape == (s.n_samples,)
        assert jnp.all(jnp.isfinite(s.ln_posterior))

    def test_default_run_has_no_logprobs(self, rv_data):
        prior = RejectionPrior.default_rv(
            period_min=Q(10.0, "day"),
            period_max=Q(400.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        sampler = RejectionSampler(prior, RVModel())
        s = sampler.run(rv_data, n_prior_samples=5_000, seed=7)
        assert s.ln_likelihood is None
        assert s.ln_prior is None


# ---------------------------------------------------------------------------
# Goodness of fit: chi2 / reduced_chi2
# ---------------------------------------------------------------------------

# Known RV orbit used to build self-consistent data + samples.
_ORBIT = {"P": 60.0, "ecc": 0.15, "tp": 8.0, "w": 1.1, "K": 9.0, "v0": 2.0}


def _orbit_data_and_samples(n_obs=20, *, rv_offset=0.0, jitter=None):
    """RV data built from ``_ORBIT`` plus a Samples object matching it exactly.

    ``t_ref`` is fixed to 0 so that the model's ``phase_peri`` (relative to
    ``t_ref``) equals ``t_peri / period``.
    """
    o = _ORBIT
    t = np.sort(np.random.default_rng(0).uniform(0.0, 240.0, n_obs))
    rv = rv_at_times(
        Q(t, "day"),
        period=Q(o["P"], "day"),
        eccentricity=Q(o["ecc"], ""),
        t_peri=Q(o["tp"], "day"),
        arg_peri=Q(o["w"], "rad"),
        rv_semiamp=Q(o["K"], "km/s"),
        v_sys=Q(o["v0"], "km/s"),
    )
    data = RVData(
        time=Q(t, "day"),
        rv=rv + Q(rv_offset, "km/s"),
        rv_err=Q(np.full(n_obs, 1.5), "km/s"),
        t_ref=Q(0.0, "day"),
    )
    n = 4

    def col(value: float, unit: str) -> Q:
        return Q(np.full(n, value, dtype=float), unit)

    nonlinear = {
        "period": col(o["P"], "day"),
        "eccentricity": col(o["ecc"], ""),
        "phase_peri": col(o["tp"] / o["P"], ""),
        "arg_peri": col(o["w"], "rad"),
    }
    if jitter is not None:
        nonlinear["jitter"] = col(jitter, "km/s")
    samples = Samples(
        nonlinear=nonlinear,
        linear={"rv_semiamp": col(o["K"], "km/s"), "v_sys": col(o["v0"], "km/s")},
        data_type="RVModel",
        metadata={"t_ref": 0.0},
    )
    return data, samples


class TestChiSquared:
    def test_chi2_zero_for_exact_fit(self):
        data, samples = _orbit_data_and_samples()
        chi2 = samples.chi2(data, RVModel())
        assert chi2.shape == (samples.n_samples,)
        assert np.allclose(np.asarray(chi2), 0.0, atol=1e-3)

    def test_chi2_known_offset(self):
        # Every point offset by 2*sigma -> chi2 = n_obs * 2**2.
        data, samples = _orbit_data_and_samples(n_obs=20, rv_offset=2.0 * 1.5)
        chi2 = samples.chi2(data, RVModel())
        assert np.allclose(np.asarray(chi2), 20 * 4.0, rtol=1e-4)

    def test_reduced_chi2_default_dof(self):
        data, samples = _orbit_data_and_samples(n_obs=20, rv_offset=2.0 * 1.5)
        model = RVModel()
        reduced = samples.reduced_chi2(data, model)
        n_params = len(model._all_nonlinear_names()) + len(model._all_linear_names())
        assert np.allclose(np.asarray(reduced), (20 * 4.0) / (20 - n_params))

    def test_reduced_chi2_custom_dof(self):
        data, samples = _orbit_data_and_samples(n_obs=20, rv_offset=2.0 * 1.5)
        reduced = samples.reduced_chi2(data, RVModel(), dof=10)
        assert np.allclose(np.asarray(reduced), (20 * 4.0) / 10)

    def test_reduced_chi2_rejects_nonpositive_dof(self):
        data, samples = _orbit_data_and_samples()
        with pytest.raises(ValueError, match="Degrees of freedom must be positive"):
            samples.reduced_chi2(data, RVModel(), dof=0)

    def test_chi2_with_jitter_uses_inflated_errors(self):
        # Same residuals, but jitter inflates the error budget -> smaller chi2.
        data, samples_plain = _orbit_data_and_samples(n_obs=20, rv_offset=3.0)
        _, samples_jit = _orbit_data_and_samples(n_obs=20, rv_offset=3.0, jitter=2.0)
        chi2_plain = samples_plain.chi2(data, RVModel())
        chi2_jit = samples_jit.chi2(
            data, RVModel(extensions=(Jitter(param_unit="km/s"),))
        )
        assert np.all(np.asarray(chi2_jit) < np.asarray(chi2_plain))
        # chi2 = sum r^2 / (sigma^2 + jitter^2) = 20 * 3^2 / (1.5^2 + 2^2).
        assert np.allclose(np.asarray(chi2_jit), 20 * 9.0 / (1.5**2 + 2.0**2))

    def test_chi2_namespaced_samples_raise(self):
        data, _ = _orbit_data_and_samples()
        joint = Samples(
            nonlinear={"primary.period": Q(jnp.full(3, 60.0), "day")},
            linear={"primary.rv_semiamp": Q(jnp.full(3, 9.0), "km/s")},
            data_type="JointModel",
            metadata={"t_ref": 0.0},
        )
        with pytest.raises(NotImplementedError, match="single-component"):
            joint.chi2(data, RVModel())
