"""Tests for the rejection acceptance-resolution diagnostic and warning."""

import warnings

import pytest
from unxt import Q

import harv.models as hm
from harv.samplers import RejectionSampler
from harv.samplers.samples import MIN_EVIDENCE_ESS, Samples, _assess_resolution
from harv.simulate import simulate_rv_sb1_data

RV_SCALES = {"sigma_K0": Q(30.0, "km/s"), "sigma_v0": Q(10.0, "km/s")}


def _prior():
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"), period_max=Q(2000.0, "day"), **RV_SCALES
    )


def _peaked_data():
    # High SNR, densely sampled -> sharply peaked likelihood -> under-resolved
    # rejection with a broad log-uniform prior.
    data, _ = simulate_rv_sb1_data(
        seed=42,
        n_obs=16,
        baseline=Q(100.0, "day"),
        period=Q(35.0, "day"),
        eccentricity=0.3,
        rv_semiamp=Q(10.0, "km/s"),
    )
    return data


def _broad_data():
    # Low SNR -> broad likelihood -> many accepted samples, high evidence ESS.
    data, _ = simulate_rv_sb1_data(
        seed=1,
        n_obs=40,
        baseline=Q(400.0, "day"),
        period=Q(120.0, "day"),
        eccentricity=0.1,
        rv_semiamp=Q(2.0, "km/s"),
        rv_err=Q(1.5, "km/s"),
    )
    return data


class TestAssessResolution:
    def test_low_ess_is_under_resolved(self):
        resolved, msg = _assess_resolution(
            n_prior=1_000_000, n_accepted=1, evidence_ess=1.0, max_log_likelihood=-50.0
        )
        assert resolved is False
        assert "Under-resolved" in msg

    def test_high_ess_is_resolved(self):
        resolved, msg = _assess_resolution(
            n_prior=1_000_000,
            n_accepted=5000,
            evidence_ess=3000.0,
            max_log_likelihood=-8.0,
        )
        assert resolved is True
        assert "Resolved" in msg

    def test_threshold_boundary(self):
        assert _assess_resolution(
            n_prior=10,
            n_accepted=1,
            evidence_ess=MIN_EVIDENCE_ESS,
            max_log_likelihood=0.0,
        )[0]
        assert not _assess_resolution(
            n_prior=10,
            n_accepted=1,
            evidence_ess=MIN_EVIDENCE_ESS - 0.1,
            max_log_likelihood=0.0,
        )[0]


class TestAcceptanceDiagnostics:
    def test_reports_under_resolved(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = RejectionSampler(_prior(), hm.RVModel()).run(
                _peaked_data(),
                n_prior_samples=1_000_000,
                seed=0,
                return_evidence_stats=True,
            )
        diag = s.acceptance_diagnostics()
        assert diag["well_resolved"] is False
        assert diag["evidence_ess"] < MIN_EVIDENCE_ESS
        assert diag["n_prior_samples"] == 1_000_000
        assert diag["n_accepted"] == s.n_samples
        assert "Under-resolved" in diag["message"]

    def test_reports_resolved(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = RejectionSampler(_prior(), hm.RVModel()).run(
                _broad_data(),
                n_prior_samples=2_000_000,
                seed=0,
                return_evidence_stats=True,
            )
        diag = s.acceptance_diagnostics()
        assert diag["well_resolved"] is True
        assert diag["evidence_ess"] >= MIN_EVIDENCE_ESS

    def test_requires_evidence_stats(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = RejectionSampler(_prior(), hm.RVModel()).run(
                _peaked_data(), n_prior_samples=200_000, seed=0
            )
        with pytest.raises(ValueError, match="return_evidence_stats=True"):
            s.acceptance_diagnostics()

    def test_missing_key_lists_missing(self):
        # A Samples built by hand (e.g. loaded from an older file) without stats.
        s = Samples(
            nonlinear={"period": Q([100.0, 101.0], "day")},
            linear={},
            data_type="RVModel",
            metadata={"t_ref": 0.0, "t_ref_unit": "day"},
        )
        with pytest.raises(ValueError, match="return_evidence_stats"):
            s.acceptance_diagnostics()


class TestSamplerWarning:
    def test_warns_when_under_resolved(self):
        with pytest.warns(UserWarning, match="Under-resolved rejection run"):
            RejectionSampler(_prior(), hm.RVModel()).run(
                _peaked_data(), n_prior_samples=1_000_000, seed=0
            )

    def test_warning_fires_without_evidence_stats(self):
        # The warning must not depend on return_evidence_stats.
        with pytest.warns(UserWarning, match="Under-resolved"):
            RejectionSampler(_prior(), hm.RVModel()).run(
                _peaked_data(), n_prior_samples=500_000, seed=0
            )

    def test_no_warning_when_resolved(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RejectionSampler(_prior(), hm.RVModel()).run(
                _broad_data(), n_prior_samples=2_000_000, seed=0
            )
        assert not any("Under-resolved" in str(w.message) for w in caught)
