"""Tests for ``RejectionSampler.summary()``.

``summary()`` returns a plain-ASCII string describing how the configured sampler will
treat each parameter (marginalized / explicitly sampled / Gaussian-but-sampled). These
tests construct samplers and assert on the returned string -- no sampling is performed.
The method must also be side-effect-free: it runs no sampling and emits no warnings.
"""

import warnings
from typing import Any

import numpyro.distributions as dist
from unxt import Q

import harv.models as hm
from harv.distributions import QD
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.extensions import Jitter
from harv.models.joint import JointModel
from harv.models.priors import HarvPrior, default_sb2_prior
from harv.models.rv import RVModel
from harv.samplers.rejection import RejectionSampler


def _rv_prior(**kwargs: Any) -> HarvPrior:
    return hm.StandardRV().default_prior(
        period_min=Q(2.0, "day"),
        period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
        **kwargs,
    )


def _row(text: str, name: str) -> str:
    """Return the (stripped-prefix) table row whose name column is ``name``."""
    return next(ln for ln in text.splitlines() if ln.strip().startswith(name))


def _gaia_prior() -> HarvPrior:
    return hm.StandardGaiaAstrometry().default_prior(
        period_min=Q(100.0, "day"),
        period_max=Q(3000.0, "day"),
        sigma_a0=Q(5.0, "AU"),
        sigma_parallax=Q(10.0, "mas"),
        sigma_pos=Q(100.0, "mas"),
        sigma_vtan=Q(50.0, "km/s"),
    )


class TestRVSummary:
    def test_returns_string_with_core_content(self):
        sampler = RejectionSampler(_rv_prior(), RVModel())
        text = sampler.summary()

        assert isinstance(text, str)
        # Header metadata.
        assert "RejectionSampler" in text
        assert "RVModel" in text
        assert "StandardRV" in text
        # Nonlinear params (always sampled) appear.
        for name in ("period", "eccentricity", "phase_peri", "arg_peri"):
            assert name in text
        # Linear params with default Gaussian priors are marginalized.
        assert "rv_semiamp" in text
        assert "v_sys" in text
        assert "marginalized" in text

    def test_has_section_headings(self):
        text = RejectionSampler(_rv_prior(), RVModel()).summary()
        assert "Nonlinear parameters (sampled)" in text
        assert "Linear parameters" in text

    def test_header_parameter_counts(self):
        """Header reports sampled vs. marginalized counts. Default RV: 4 NL sampled,
        2 linear marginalized (rv_semiamp, v_sys)."""
        text = RejectionSampler(_rv_prior(), RVModel()).summary()
        assert "4 sampled, 2 marginalized" in _row(text, "parameters")

    def test_counts_reflect_override(self):
        """Excluding v_sys leaves rv_semiamp sampled: 5 sampled, 1 marginalized."""
        text = RejectionSampler(
            _rv_prior(), RVModel(), marginalized_names=("v_sys",)
        ).summary()
        assert "5 sampled, 1 marginalized" in _row(text, "parameters")

    def test_jitter_extension_marked(self):
        ext = Jitter(param_unit="km/s")
        prior = _rv_prior(jitter=QD(dist.HalfNormal(1.0), "km/s"))
        sampler = RejectionSampler(prior, RVModel(extensions=(ext,)))
        text = sampler.summary()

        # Extension appears in header and the jitter param is a marked nonlinear row.
        assert "Jitter(km/s)" in text
        assert "jitter (ext)" in text

    def test_marginalized_names_override_flips_status(self):
        """A Gaussian linear param excluded from marginalization is 'could marg.'."""
        sampler = RejectionSampler(
            _rv_prior(), RVModel(), marginalized_names=("v_sys",)
        )
        text = sampler.summary()

        assert "sampled (could marg.)" in text
        # The excluded rv_semiamp row carries the "could marg." status; v_sys is
        # still marginalized.
        assert "sampled (could marg.)" in _row(text, "rv_semiamp")
        assert "marginalized" in _row(text, "v_sys")


class TestGaiaSummary:
    def test_nongaussian_parallax_is_sampled(self):
        text = RejectionSampler(_gaia_prior(), GaiaAstrometryModel()).summary()

        parallax_line = _row(text, "parallax")
        # HalfNormal parallax cannot be marginalized -> non-Gaussian "sampled".
        assert "sampled" in parallax_line
        assert "could marg." not in parallax_line
        # Other astrometric linear params remain marginalized.
        assert "ra0" in text
        assert "semi_major_axis" in text

    def test_no_warning_emitted(self):
        """Introspection must be side-effect-free even with non-Gaussian priors."""
        sampler = RejectionSampler(_gaia_prior(), GaiaAstrometryModel())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            text = sampler.summary()
        assert isinstance(text, str)


class TestJointSummary:
    def test_sb2_components_and_namespaced_params(self):
        prior = default_sb2_prior(
            period_min=Q(2.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        joint = JointModel.for_sb2(prior=prior)
        text = RejectionSampler(prior, joint).summary()

        assert "JointModel" in text
        assert "primary" in text
        assert "secondary" in text
        # Namespaced linear param appears for the per-component semi-amplitudes.
        assert "primary.rv_semiamp" in text
        assert "secondary.rv_semiamp" in text
        # Shared systemic velocity is bare (not namespaced).
        assert "v_sys" in text
