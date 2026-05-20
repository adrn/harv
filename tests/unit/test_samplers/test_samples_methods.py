"""Tests for NumpyroSampler.run() and Samples.plot.

Both methods operate on an existing Samples object. Rather than running the
full rejection sampler (slow), tests build a minimal Samples instance directly
using the constructor. NumpyroSampler.run() returns a Samples object.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as ndist
import pytest
from unxt import Q, ustrip

from harv.data import GaiaAstrometryData, RVData, SystemData
from harv.distributions import QD
from harv.kepler.orbits import (
    astrometric_orbit_at_times,
    rv_at_times,
    thiele_innes_ABFG,
)
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.extensions import Jitter, MonomialTrend
from harv.models.extensions.base import ParamInfo
from harv.models.extensions.gp import GP
from harv.models.joint import JointModel
from harv.models.rv import RVModel
from harv.plot import get_alpha, plot_gaia_astrometry, plot_gaia_sky_orbit, plot_rv
from harv.samplers.numpyro import NumpyroSampler
from harv.samplers.rejection_prior import RejectionPrior
from harv.samplers.samples import Samples

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Fixtures: minimal Samples objects for each data type
# ---------------------------------------------------------------------------

N = 10  # number of mock posterior samples


@pytest.fixture
def rv_samples() -> Samples:
    """Minimal RV Samples with N draws."""
    nonlinear = {
        "period": Q(jnp.linspace(90.0, 110.0, N), "day"),
        "eccentricity": Q(jnp.linspace(0.0, 0.3, N), ""),
        "phase_peri": Q(jnp.linspace(0.0, 1.0, N), ""),
        "arg_peri": Q(jnp.linspace(0.0, 3.14, N), "rad"),
    }
    linear = {
        "rv_semiamp": Q(jnp.linspace(3.0, 7.0, N), "km/s"),
        "v_sys": Q(jnp.linspace(-1.0, 1.0, N), "km/s"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="RVModel",
        metadata={"t_ref": 0.0},
    )


@pytest.fixture
def astro_samples() -> Samples:
    """Minimal astrometry Samples with N draws."""
    nonlinear = {
        "period": Q(jnp.linspace(250.0, 350.0, N), "day"),
        "eccentricity": Q(jnp.linspace(0.0, 0.3, N), ""),
        "phase_peri": Q(jnp.linspace(0.0, 1.0, N), ""),
        "arg_peri": Q(jnp.linspace(0.0, 3.14, N), "rad"),
        "cos_i": Q(jnp.linspace(0.2, 0.8, N), ""),
        "lon_asc_node": Q(jnp.linspace(0.0, 6.28, N), "rad"),
    }
    linear = {
        "ra0": Q(jnp.zeros(N), "mas"),
        "dec0": Q(jnp.zeros(N), "mas"),
        "pmra": Q(jnp.ones(N) * 10.0, "mas/yr"),
        "pmdec": Q(jnp.ones(N) * -5.0, "mas/yr"),
        "parallax": Q(jnp.ones(N) * 5.0, "mas"),
        "semi_major_axis": Q(jnp.linspace(1.0, 3.0, N), "mas"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="GaiaAstrometryModel",
        metadata={"t_ref": 0.0},
    )


@pytest.fixture
def combined_samples() -> Samples:
    """Minimal combined (astrometry + RV) Samples with N draws."""
    nonlinear = {
        "period": Q(jnp.linspace(250.0, 350.0, N), "day"),
        "eccentricity": Q(jnp.linspace(0.0, 0.2, N), ""),
        "phase_peri": Q(jnp.linspace(0.0, 1.0, N), ""),
        "arg_peri": Q(jnp.linspace(0.0, 3.14, N), "rad"),
        "cos_i": Q(jnp.linspace(0.2, 0.8, N), ""),
        "lon_asc_node": Q(jnp.linspace(0.0, 6.28, N), "rad"),
    }
    linear = {
        "ra0": Q(jnp.zeros(N), "mas"),
        "dec0": Q(jnp.zeros(N), "mas"),
        "pmra": Q(jnp.ones(N) * 10.0, "mas/yr"),
        "pmdec": Q(jnp.ones(N) * -5.0, "mas/yr"),
        "parallax": Q(jnp.ones(N) * 5.0, "mas"),
        "semi_major_axis": Q(jnp.linspace(1.0, 3.0, N), "mas"),
        "rv_semiamp": Q(jnp.linspace(3.0, 7.0, N), "km/s"),
        "v_sys": Q(jnp.zeros(N), "km/s"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="JointModel",
        metadata={"t_ref": 0.0},
    )


@pytest.fixture
def empty_rv_samples() -> Samples:
    """RV Samples with zero accepted draws."""
    return Samples(
        nonlinear={
            "period": Q(jnp.array([]), "day"),
            "eccentricity": Q(jnp.array([]), ""),
            "phase_peri": Q(jnp.array([]), ""),
            "arg_peri": Q(jnp.array([]), "rad"),
        },
        linear={
            "rv_semiamp": Q(jnp.array([]), "km/s"),
            "v_sys": Q(jnp.array([]), "km/s"),
        },
        data_type="RVModel",
        metadata={},
    )


@pytest.fixture
def rv_sampler_and_data():
    """NumpyroSampler + RVData used for NumpyroSampler.run() tests."""
    times = Q(jnp.linspace(0.0, 100.0, 20), "day")
    rv = Q(jnp.zeros(20), "km/s")
    rv_err = Q(jnp.ones(20) * 2.0, "km/s")
    data = RVData(time=times, rv=rv, rv_err=rv_err)
    prior = RejectionPrior.default_rv(
        period_min=Q(50.0, "day"),
        period_max=Q(200.0, "day"),
        sigma_K0=Q(30.0, "km/s"),
        sigma_v0=Q(30.0, "km/s"),
    )
    model = RVModel()
    return NumpyroSampler(prior, model), data


# ---------------------------------------------------------------------------
# Tests: rv_at_times and astrometric_orbit_at_times
# ---------------------------------------------------------------------------


class TestRvAtTimes:
    """Unit tests for rv_at_times."""

    def test_circular_orbit_shape(self):
        """Output has the same shape as the input times."""
        times = Q(np.linspace(0.0, 200.0, 50), "day")
        rv = rv_at_times(
            times,
            period=Q(200.0, "day"),
            eccentricity=0.0,
            t_peri=Q(0.0, "day"),
            arg_peri=Q(0.0, "rad"),
            rv_semiamp=Q(10.0, "km/s"),
            v_sys=Q(0.0, "km/s"),
        )
        assert rv.shape == (50,)

    def test_unit_preserved(self):
        """Output unit matches rv_semiamp and v_sys unit."""
        times = Q(np.array([0.0, 50.0, 100.0]), "day")
        rv = rv_at_times(
            times,
            period=Q(200.0, "day"),
            eccentricity=0.3,
            t_peri=Q(50.0, "day"),
            arg_peri=Q(1.2, "rad"),
            rv_semiamp=Q(8.0, "km/s"),
            v_sys=Q(-5.0, "km/s"),
        )
        assert rv.unit.physical_type == "speed"

    def test_v_sys_offset(self):
        """Systemic velocity shifts every sample by v_sys."""
        times = Q(np.array([0.0, 50.0, 100.0]), "day")
        kwargs = {
            "period": Q(200.0, "day"),
            "eccentricity": 0.0,
            "t_peri": Q(0.0, "day"),
            "arg_peri": Q(0.0, "rad"),
            "rv_semiamp": Q(10.0, "km/s"),
        }
        rv0 = rv_at_times(times, v_sys=Q(0.0, "km/s"), **kwargs)
        rv5 = rv_at_times(times, v_sys=Q(5.0, "km/s"), **kwargs)
        np.testing.assert_allclose(np.asarray(rv5.value - rv0.value), 5.0, atol=1e-6)


class TestAstrometricOrbitAtTimes:
    """Unit tests for astrometric_orbit_at_times."""

    def test_output_shape_and_unit(self):
        """Both outputs have the input shape and semi_major_axis unit."""
        times = Q(np.linspace(0.0, 300.0, 40), "day")
        dra, ddec = astrometric_orbit_at_times(
            times,
            period=Q(300.0, "day"),
            eccentricity=0.3,
            t_peri=Q(0.0, "day"),
            arg_peri=Q(1.2, "rad"),
            cos_i=0.5,
            lon_asc_node=Q(0.8, "rad"),
            semi_major_axis=Q(3.0, "mas"),
        )
        assert dra.shape == (40,)
        assert ddec.shape == (40,)
        assert dra.unit.physical_type == "angle"

    def test_circular_face_on_orbit_is_circle(self):
        """Face-on circular orbit traces a circle: Deltara^2+Deltadec^2 = const."""
        times = Q(np.linspace(0.0, 1.0, 500), "day")
        dra, ddec = astrometric_orbit_at_times(
            times,
            period=Q(1.0, "day"),
            eccentricity=0.0,
            t_peri=Q(0.0, "day"),
            arg_peri=Q(0.0, "rad"),
            cos_i=1.0,  # face-on
            lon_asc_node=Q(0.0, "rad"),
            semi_major_axis=Q(1.0, "mas"),
        )
        r2 = np.asarray(dra.value) ** 2 + np.asarray(ddec.value) ** 2
        np.testing.assert_allclose(r2, 1.0, atol=1e-5)

    def test_eccentric_face_on_orbit_pericenter_distance(self):
        """Face-on eccentric orbit: r(pericenter) = a(1-e), r(apocenter) = a(1+e).

        At pericenter (true anomaly f=0, i.e. t=t_peri), the distance from
        the focus should be a*(1-e). At apocenter (f=pi), a*(1+e). This
        verifies the r/a = (1-e^2)/(1+e*cos f) factor is applied.
        """
        e = 0.6
        a = 5.0  # mas
        period = Q(100.0, "day")
        # At t_peri, f=0, so r = a(1-e)
        t_peri_val = Q(0.0, "day")
        dra_peri, ddec_peri = astrometric_orbit_at_times(
            Q(np.array([0.0]), "day"),
            period=period,
            eccentricity=e,
            t_peri=t_peri_val,
            arg_peri=Q(0.0, "rad"),
            cos_i=1.0,  # face-on
            lon_asc_node=Q(0.0, "rad"),
            semi_major_axis=Q(a, "mas"),
        )
        r_peri = np.sqrt(
            np.asarray(dra_peri.value) ** 2 + np.asarray(ddec_peri.value) ** 2
        )
        np.testing.assert_allclose(r_peri, a * (1 - e), atol=1e-8)

        # At apocenter (half period later), f=pi, so r = a(1+e)
        dra_apo, ddec_apo = astrometric_orbit_at_times(
            Q(np.array([50.0]), "day"),
            period=period,
            eccentricity=e,
            t_peri=t_peri_val,
            arg_peri=Q(0.0, "rad"),
            cos_i=1.0,  # face-on
            lon_asc_node=Q(0.0, "rad"),
            semi_major_axis=Q(a, "mas"),
        )
        r_apo = np.sqrt(
            np.asarray(dra_apo.value) ** 2 + np.asarray(ddec_apo.value) ** 2
        )
        np.testing.assert_allclose(r_apo, a * (1 + e), atol=1e-8)


# ---------------------------------------------------------------------------
# Tests: NumpyroSampler.run() -- marginalized (default)
# ---------------------------------------------------------------------------


class TestNumpyroSamplerRun:
    """Tests for NumpyroSampler.run() with marginalized=True (default)."""

    def test_returns_samples(self, rv_samples, rv_sampler_and_data):
        """run() returns a Samples container."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=0,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)

    def test_nonlinear_keys_match_prior(self, rv_samples, rv_sampler_and_data):
        """Returned Samples has nonlinear keys matching the prior."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=1,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        for key in sampler.prior.nonlinear_priors:
            assert key in result.nonlinear, f"Missing nonlinear key '{key}'"

    def test_linear_keys_present(self, rv_samples, rv_sampler_and_data):
        """Marginalized run() conditionally samples linear params in output."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=2,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert "rv_semiamp" in result.linear
        assert "v_sys" in result.linear

    def test_output_shape(self, rv_samples, rv_sampler_and_data):
        """Output has num_chains * num_samples total samples."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=3,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert result.n_samples == 10  # 2 chains x 5 samples

    def test_data_type_preserved(self, rv_samples, rv_sampler_and_data):
        """Output data_type matches the model's data_type."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=4,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert result.data_type == "RVModel"

    def test_return_logprobs(self, rv_samples, rv_sampler_and_data):
        """return_logprobs populates ln_likelihood / ln_prior on the output."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=5,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
            return_logprobs=True,
        )
        assert result.ln_likelihood is not None
        assert result.ln_prior is not None
        assert result.ln_likelihood.shape == (result.n_samples,)
        assert result.ln_prior.shape == (result.n_samples,)
        assert jnp.all(jnp.isfinite(result.ln_posterior))

    def test_default_run_has_no_logprobs(self, rv_samples, rv_sampler_and_data):
        """Without return_logprobs the log-prob fields stay None."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=6,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert result.ln_likelihood is None
        assert result.ln_prior is None

    def test_raises_for_empty_samples(self, empty_rv_samples, rv_sampler_and_data):
        """run() raises ValueError when there are no posterior samples."""
        sampler, data = rv_sampler_and_data
        with pytest.raises(ValueError, match="no posterior samples"):
            sampler.run(
                data,
                init_samples=empty_rv_samples,
                seed=5,
                num_chains=2,
                num_warmup=5,
                num_samples=5,
            )

    def test_single_chain(self, rv_samples, rv_sampler_and_data):
        """run() with num_chains=1 produces expected output."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=6,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        assert result.n_samples == 3

    def test_sampler_owned_marginalized_subset(self, rv_samples):
        """Sampler-owned marginalized_names is honored in marginalized mode."""
        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        data = RVData(time=times, rv=rv, rv_err=rv_err)
        prior = RejectionPrior.default_rv(
            period_min=Q(50.0, "day"),
            period_max=Q(200.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
        )
        sampler = NumpyroSampler(prior, RVModel(), marginalized_names=("v_sys",))

        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=7,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )

        assert isinstance(result, Samples)
        assert "rv_semiamp" in result.linear
        assert "v_sys" in result.linear


class TestNumpyroSamplerRunFull:
    """Tests for NumpyroSampler.run() with marginalized=False."""

    def test_returns_samples(self, rv_samples, rv_sampler_and_data):
        """run(marginalized=False) returns a Samples container."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=10,
            marginalized=False,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)

    def test_has_linear_params(self, rv_samples, rv_sampler_and_data):
        """Full model output includes named linear params."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=11,
            marginalized=False,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        for key in sampler.prior.nonlinear_priors:
            assert key in result.nonlinear, f"Missing nonlinear key '{key}'"
        for name in rv_samples.linear:
            assert name in result.linear, f"Missing linear key '{name}'"


# ---------------------------------------------------------------------------
# Tests: NumpyroSampler.run() with extra_model
# ---------------------------------------------------------------------------


class TestNumpyroSamplerRunExtraModel:
    """Tests for NumpyroSampler.run() with extra_model reparameterization."""

    def _make_extra_model(self):
        """Return a minimal extra_model that fixes K to a constant."""

        def extra_model(pars):
            K_scale = numpyro.sample("K_scale", ndist.HalfNormal(10.0))
            return {"rv_semiamp": K_scale}

        return extra_model

    def test_raises_without_extra_init_params(self, rv_samples, rv_sampler_and_data):
        """extra_model without extra_init_params raises ValueError."""
        sampler, data = rv_sampler_and_data
        with pytest.raises(ValueError, match="extra_init_params is required"):
            sampler.run(
                data,
                init_samples=rv_samples,
                seed=20,
                extra_model=self._make_extra_model(),
                num_chains=2,
                num_warmup=5,
                num_samples=5,
            )

    def test_run_with_extra_model(self, rv_samples, rv_sampler_and_data):
        """run() with extra_model completes and returns Samples."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=21,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)

    def test_run_extra_model_marginalized(self, rv_samples, rv_sampler_and_data):
        """extra_model + marginalized=True runs and returns Samples."""
        sampler, data = rv_sampler_and_data
        result = sampler.run(
            data,
            init_samples=rv_samples,
            seed=22,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            marginalized=True,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        # Nonlinear sites must be present.
        for key in sampler.prior.nonlinear_priors:
            assert key in result.nonlinear

    def test_extra_model_raises_for_unknown_param(
        self, rv_samples, rv_sampler_and_data
    ):
        """extra_model returning an unknown linear param name raises ValueError."""

        def bad_extra_model(pars):
            x = numpyro.sample("x", ndist.Normal(0.0, 1.0))
            return {"not_a_real_param": x}

        sampler, data = rv_sampler_and_data
        with pytest.raises(ValueError, match="unknown linear parameter name"):
            sampler.run(
                data,
                init_samples=rv_samples,
                seed=23,
                extra_model=bad_extra_model,
                extra_init_params={"x": jnp.zeros(2)},
                num_chains=2,
                num_warmup=1,
                num_samples=1,
                chain_method="sequential",
            )


# ---------------------------------------------------------------------------
# Tests: NumpyroSampler.run() with non-Gaussian linear priors (regression)
# ---------------------------------------------------------------------------


class TestNumpyroSamplerNonGaussianLinear:
    """Regression tests for NumpyroSampler.run().

    When linear priors include non-Gaussian distributions (e.g. HalfNormal for parallax)
    that must be explicitly sampled rather than analytically marginalized.
    """

    @pytest.fixture
    def astro_sampler_and_data(self):
        """NumpyroSampler + GaiaAstrometryData with HalfNormal parallax."""
        n_obs = 15
        times = Q(jnp.linspace(0.0, 600.0, n_obs), "day")
        data = GaiaAstrometryData(
            time=times,
            al_position=Q(jnp.zeros(n_obs), "mas"),
            al_position_err=Q(jnp.ones(n_obs) * 0.1, "mas"),
            scan_angle=Q(jnp.linspace(0.0, 3.14, n_obs), "rad"),
            parallax_factor=jnp.full(n_obs, 0.5),
        )
        prior = RejectionPrior.default_gaia_astrometry(
            period_min=Q(100.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_a0=Q(5.0, "AU"),
            sigma_parallax=Q(10.0, "mas"),
            sigma_pos=Q(100.0, "mas"),
            sigma_vtan=Q(50.0, "km/s"),
        )
        return NumpyroSampler(prior, GaiaAstrometryModel()), data

    def test_run_marginalized_with_halfnormal_parallax(
        self, astro_samples, astro_sampler_and_data
    ):
        """MCMC runs and output includes parallax when it has a HalfNormal prior."""
        sampler, data = astro_sampler_and_data
        result = sampler.run(
            data,
            init_samples=astro_samples,
            seed=30,
            num_chains=2,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        assert "parallax" in result.linear
        assert result.n_samples == 6  # 2 chains x 3 samples

    def test_run_single_chain_with_halfnormal_parallax(
        self, astro_samples, astro_sampler_and_data
    ):
        """MCMC with num_chains=1 and HalfNormal parallax produces valid output."""
        sampler, data = astro_sampler_and_data
        result = sampler.run(
            data,
            init_samples=astro_samples,
            seed=31,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        assert "parallax" in result.linear
        assert result.n_samples == 3


# ---------------------------------------------------------------------------
# Tests: NumpyroSampler.run() with combined (astro + RV) model + jitter
# ---------------------------------------------------------------------------


class TestNumpyroSamplerCombinedWithJitter:
    """Regression tests for NumpyroSampler.run().

    On a combined astrometry+RV model with jitter priors.

    These catch issues that previously only surfaced when running the full
    getting-started notebook, such as init_params in the wrong (constrained) space
    producing garbage MCMC output.
    """

    @pytest.fixture
    def combined_sampler_and_data(self):
        """Combined NumpyroSampler + JointModel with jitter priors."""
        # Minimal astrometry data
        n_ast = 15
        times_ast = Q(jnp.linspace(0.0, 1000.0, n_ast), "day")
        astro_data = GaiaAstrometryData(
            time=times_ast,
            al_position=Q(jnp.zeros(n_ast), "mas"),
            al_position_err=Q(jnp.ones(n_ast) * 0.1, "mas"),
            scan_angle=Q(jnp.linspace(0.0, 3.14, n_ast), "rad"),
            parallax_factor=jnp.full(n_ast, 0.5),
        )

        # Minimal RV data
        n_rv = 8
        times_rv = Q(jnp.linspace(0.0, 800.0, n_rv), "day")
        rv_data = RVData(
            time=times_rv,
            rv=Q(jnp.zeros(n_rv), "km/s"),
            rv_err=Q(jnp.ones(n_rv) * 1.0, "km/s"),
        )

        # Split linear priors per component (qualified with component name)
        astro_linear = {
            "astro.ra0": QD(ndist.Normal(0.0, 100.0), "mas"),
            "astro.dec0": QD(ndist.Normal(0.0, 100.0), "mas"),
            "astro.pmra": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
            "astro.pmdec": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
            "astro.parallax": QD(ndist.HalfNormal(10.0), "mas"),
            "astro.semi_major_axis": QD(ndist.Normal(0.0, 50.0), "mas"),
        }
        rv_linear = {
            "rv.rv_semiamp": QD(ndist.Normal(0.0, 30.0), "km/s"),
            "rv.v_sys": QD(ndist.Normal(0.0, 30.0), "km/s"),
        }

        astro_model = GaiaAstrometryModel()
        rv_model_inst = RVModel(extensions=(Jitter(param_unit="km/s"),))

        joint = JointModel.for_rv_and_gaia(
            components={"astro": astro_model, "rv": rv_model_inst}
        )

        # Combined prior (nonlinear priors only, linear handled by components)
        nonlinear = {
            "period": QD(ndist.Normal(300.0, 50.0), "day"),
            "eccentricity": ndist.TruncatedNormal(0.3, 0.2, low=0.0, high=1.0),
            "phase_peri": ndist.Uniform(0.0, 1.0),
            "cos_i": ndist.Uniform(-1.0, 1.0),
            "arg_peri": QD(ndist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
            "lon_asc_node": QD(ndist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        }
        # Merged linear prior dict for the RejectionPrior
        linear = {**astro_linear, **rv_linear}
        prior = RejectionPrior(
            nonlinear_priors=nonlinear,
            linear_prior=linear,
            extension_priors={"jitter": QD(ndist.HalfNormal(4.0), "km/s")},
        )
        joint_data = {"astro": astro_data, "rv": rv_data}
        return NumpyroSampler(prior, joint), joint_data

    @pytest.fixture
    def combined_samples_with_jitter(self) -> Samples:
        """Minimal combined Samples that include rv.jitter in nonlinear.

        ``RejectionSampler.run`` flattens unique joint linear names to bare keys
        (for example ``"parallax"`` instead of ``"astro.parallax"``) while
        keeping colliding names qualified. Warm-started MCMC must accept that
        output shape when building init values for joint numpyro sites.
        """
        nonlinear = {
            "period": Q(jnp.linspace(280.0, 320.0, N), "day"),
            "eccentricity": Q(jnp.linspace(0.1, 0.5, N), ""),
            "phase_peri": Q(jnp.linspace(0.1, 0.9, N), ""),
            "arg_peri": Q(jnp.linspace(0.5, 5.5, N), "rad"),
            "cos_i": Q(jnp.linspace(-0.5, 0.5, N), ""),
            "lon_asc_node": Q(jnp.linspace(0.5, 5.5, N), "rad"),
            "rv.jitter": Q(jnp.linspace(1.0, 5.0, N), "km/s"),
        }
        linear = {
            "ra0": Q(jnp.zeros(N), "mas"),
            "dec0": Q(jnp.zeros(N), "mas"),
            "pmra": Q(jnp.ones(N) * 10.0, "mas/yr"),
            "pmdec": Q(jnp.ones(N) * -5.0, "mas/yr"),
            "parallax": Q(jnp.ones(N) * 3.0, "mas"),
            "semi_major_axis": Q(jnp.linspace(1.0, 3.0, N), "mas"),
            "rv_semiamp": Q(jnp.linspace(3.0, 7.0, N), "km/s"),
            "v_sys": Q(jnp.zeros(N), "km/s"),
        }
        return Samples(
            nonlinear=nonlinear,
            linear=linear,
            data_type="combined",
            metadata={"t_ref": 0.0},
        )

    def test_run_marginalized_completes(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Combined marginalized MCMC accepts bare unique joint warm-start keys."""
        sampler, data = combined_sampler_and_data
        result = sampler.run(
            data,
            init_samples=combined_samples_with_jitter,
            seed=40,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        # All nonlinear params present
        for key in sampler.prior.nonlinear_priors:
            assert key in result.nonlinear, f"Missing nonlinear key: {key}"
        # Jitter site present (model-key convention for JointModel: "rv.jitter")
        assert "rv.jitter" in result.nonlinear
        # Explicit linear (HalfNormal parallax) present
        assert "parallax" in result.linear

    def test_run_marginalized_accepts_bare_joint_linear_warm_start_keys(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Regression: joint warm starts may flatten unique explicit linear names."""
        sampler, data = combined_sampler_and_data
        result = sampler.run(
            data,
            init_samples=combined_samples_with_jitter,
            seed=43,
            num_chains=2,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        assert "parallax" in result.linear

    def test_run_marginalized_sample_count(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Combined marginalized MCMC produces expected number of samples."""
        sampler, data = combined_sampler_and_data
        result = sampler.run(
            data,
            init_samples=combined_samples_with_jitter,
            seed=41,
            num_chains=2,
            num_warmup=3,
            num_samples=4,
            chain_method="sequential",
        )
        assert result.n_samples == 8  # 2 chains x 4 samples

    def test_return_logprobs_joint_model(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """return_logprobs=True works for JointModel (marginalized path)."""
        sampler, data = combined_sampler_and_data
        num_chains = 2
        num_samples = 3
        result = sampler.run(
            data,
            init_samples=combined_samples_with_jitter,
            seed=44,
            num_chains=num_chains,
            num_warmup=3,
            num_samples=num_samples,
            chain_method="sequential",
            return_logprobs=True,
        )
        assert isinstance(result, Samples)
        assert result.ln_likelihood is not None
        assert result.ln_prior is not None
        assert result.ln_likelihood.shape == (num_chains * num_samples,)
        assert result.ln_prior.shape == (num_chains * num_samples,)

    def test_run_full_completes(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Combined full (non-marginalized) MCMC runs and returns Samples.

        Regression: the full model previously failed on combined data because
        (a) callable linear priors received a composite dict instead of a
        single params object, and (b) joint warm-start samples may flatten
        unique linear keys (for example ``parallax`` rather than
        ``astro.parallax``), so `_build_init_params()` must normalize them
        before packing the ``_linear`` site.
        """
        sampler, data = combined_sampler_and_data
        result = sampler.run(
            data,
            init_samples=combined_samples_with_jitter,
            seed=42,
            marginalized=False,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        assert isinstance(result, Samples)
        # All nonlinear params present
        for key in sampler.prior.nonlinear_priors:
            assert key in result.nonlinear, f"Missing nonlinear key: {key}"
        # Linear parameters present
        assert len(result.linear) > 0


# ---------------------------------------------------------------------------
# Tests: plot_rv and plot_astrometry (D4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotRV:
    """Tests for harv.plot.plot_rv."""

    def test_returns_axes(self, rv_samples):
        """plot_rv() returns a matplotlib Axes when ax=None."""
        ax = plot_rv(rv_samples)
        assert hasattr(ax, "get_xlabel")
        plt.close("all")

    def test_figure_has_one_axes(self, rv_samples):
        """RV plot produces exactly one Axes."""
        ax = plot_rv(rv_samples)
        assert len(ax.figure.axes) == 1
        plt.close("all")

    def test_plot_with_rv_data(self, rv_samples):
        """plot_rv(samples, data) plots data points without error."""
        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        fig = plot_rv(rv_samples, rv_data)
        assert fig is not None
        plt.close("all")

    def test_n_samples_limits_model_curves(self, rv_samples):
        """n_samples controls how many posterior curves are drawn."""
        ax = plot_rv(rv_samples, n_samples=3)
        expected_alpha = get_alpha(3)
        model_lines = [
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        ]
        assert len(model_lines) == 3
        plt.close("all")

    def test_raises_for_bad_data_type(self, rv_samples):
        """plot_rv() raises ValueError for unrecognised data argument."""
        with pytest.raises(ValueError, match="must be"):
            plot_rv(rv_samples, "not_a_data_object")

    def test_xlabel_is_phase(self, rv_samples):
        """X-axis label mentions orbital phase when phase_fold_median=True."""
        ax = plot_rv(rv_samples, phase_fold_median=True)
        assert "phase" in ax.get_xlabel().lower()
        plt.close("all")

    def test_xlabel_is_time_by_default(self, rv_samples):
        """Default (phase_fold_median=False) x-axis label mentions time."""
        ax = plot_rv(rv_samples)
        assert "time" in ax.get_xlabel().lower()
        plt.close("all")

    def test_plot_with_jitter_extension_widens_error_bars(self, rv_samples):
        """Jitter extension widens plotted RV error bars."""
        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        jitter_samples = Samples(
            nonlinear={**rv_samples.nonlinear, "jitter": Q(jnp.ones(N), "km/s")},
            linear=rv_samples.linear,
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )

        ax_plain = plot_rv(rv_samples, rv_data, n_samples=1)
        ax_jitter = plot_rv(
            jitter_samples,
            rv_data,
            extensions=(Jitter(param_unit="km/s"),),
            n_samples=1,
        )

        plain_segments = ax_plain.collections[0].get_segments()
        # The widened error bars are rendered as a second collection on top of the base.
        jitter_segments = ax_jitter.collections[-1].get_segments()
        plain_height = plain_segments[0][1, 1] - plain_segments[0][0, 1]
        jitter_height = jitter_segments[0][1, 1] - jitter_segments[0][0, 1]

        assert jitter_height > plain_height
        plt.close("all")

    def test_plot_handles_more_instruments_than_color_cycle(self, rv_samples):
        """Many instruments must not crash plot_rv on the orbit-color zip.

        Regression: ``zip(items, colors[1:N+1], strict=True)`` raised a
        ValueError once N exceeded ``len(colors) - 1`` (i.e. 9 for the default
        matplotlib palette).  Orbit colors must cycle modulo the palette.
        """
        n_instr = 12  # > default 10-color matplotlib palette
        rv_datasets: dict[str, RVData] = {}
        linear: dict[str, Q] = {}
        for i in range(n_instr):
            name = f"inst{i}"
            rv_datasets[name] = RVData(
                time=Q(jnp.linspace(0.0, 100.0, 6) + i, "day"),
                rv=Q(jnp.zeros(6) + i * 0.1, "km/s"),
                rv_err=Q(jnp.ones(6) * 0.5, "km/s"),
            )
            linear[f"{name}.rv_semiamp"] = Q(jnp.linspace(3.0, 7.0, N), "km/s")
            linear[f"{name}.v_sys"] = Q(jnp.linspace(-1.0, 1.0, N) + i * 0.05, "km/s")

        samples = Samples(
            nonlinear=rv_samples.nonlinear,
            linear=linear,
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        ax = plot_rv(samples, SystemData(**rv_datasets), n_samples=1)
        assert ax is not None
        plt.close("all")

    def test_plot_with_gp_extension_changes_time_domain_curve(self, rv_samples):
        """GP plotting support modifies the time-domain RV overlay."""
        tinygp = pytest.importorskip("tinygp")

        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        gp_samples = Samples(
            nonlinear={
                **rv_samples.nonlinear,
                "gp_amp": Q(jnp.ones(N) * 2.0, "km/s"),
                "gp_scale": Q(jnp.ones(N) * 10.0, "day"),
            },
            linear=rv_samples.linear,
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        gp = GP(
            kernel_builder=lambda hp: (
                tinygp.kernels.ExpSquared(hp["gp_scale"]) * hp["gp_amp"] ** 2
            ),
            hyperparams=(
                ParamInfo("gp_amp", "km/s"),
                ParamInfo("gp_scale", "day"),
            ),
            time_unit="day",
        )

        ax_plain = plot_rv(rv_samples, rv_data, n_samples=1)
        ax_gp = plot_rv(gp_samples, rv_data, extensions=(gp,), n_samples=1)

        expected_alpha = get_alpha(1)
        plain_line = next(
            line
            for line in ax_plain.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )
        gp_line = next(
            line
            for line in ax_gp.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )

        assert not np.allclose(plain_line.get_ydata(), gp_line.get_ydata())
        plt.close("all")

    def test_plot_with_trend_extension_changes_time_domain_curve(self, rv_samples):
        """Trend plotting support modifies the time-domain RV overlay."""
        times = Q(jnp.linspace(0.0, 4.0, 5), "day")
        rv = Q(jnp.zeros(5), "km/s")
        rv_err = Q(jnp.ones(5) * 0.5, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        time_grid = Q(jnp.linspace(0.0, 4.0, 9), "day")
        trend_samples = Samples(
            nonlinear=rv_samples.nonlinear,
            linear={
                **rv_samples.linear,
                "trend_1": Q(jnp.ones(N) * 0.25, "km/s"),
            },
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        trend = MonomialTrend(order=1, time_unit="day", obs_unit="km/s")

        ax_plain = plot_rv(rv_samples, rv_data, n_samples=1, time_grid=time_grid)
        ax_trend = plot_rv(
            trend_samples,
            rv_data,
            extensions=(trend,),
            n_samples=1,
            time_grid=time_grid,
        )

        expected_alpha = get_alpha(1)
        plain_line = next(
            line
            for line in ax_plain.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )
        trend_line = next(
            line
            for line in ax_trend.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )

        assert not np.allclose(plain_line.get_ydata(), trend_line.get_ydata())
        plt.close("all")

    def test_plot_with_trend_extension_can_show_signal_components(self, rv_samples):
        """Trend plotting can decompose Keplerian and extension contributions."""
        times = Q(jnp.linspace(0.0, 4.0, 5), "day")
        rv = Q(jnp.zeros(5), "km/s")
        rv_err = Q(jnp.ones(5) * 0.5, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        time_grid = Q(jnp.linspace(0.0, 4.0, 9), "day")
        trend_samples = Samples(
            nonlinear=rv_samples.nonlinear,
            linear={
                **rv_samples.linear,
                "trend_1": Q(jnp.ones(N) * 0.25, "km/s"),
            },
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        trend = MonomialTrend(order=1, time_unit="day", obs_unit="km/s")

        ax = plot_rv(
            trend_samples,
            rv_data,
            extensions=(trend,),
            n_samples=1,
            time_grid=time_grid,
            show_signal_components=True,
        )

        expected_alpha = get_alpha(1)
        component_lines = [
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        ]

        assert len(component_lines) == 2
        assert {line.get_label() for line in component_lines} == {
            "Extensions",
            "Keplerian",
        }
        assert not np.allclose(
            component_lines[0].get_ydata(),
            component_lines[1].get_ydata(),
        )
        plt.close("all")

    def test_phase_fold_subtracts_trend_signal_from_data(self, rv_samples):
        """Phase-folded plots subtract the reference trend signal from data points."""
        times = Q(jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), "day")
        rv = Q(jnp.zeros(5), "km/s")
        rv_err = Q(jnp.ones(5) * 0.5, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        trend_samples = Samples(
            nonlinear=rv_samples.nonlinear,
            linear={
                **rv_samples.linear,
                "trend_1": Q(jnp.ones(N) * 0.25, "km/s"),
            },
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        trend = MonomialTrend(order=1, time_unit="day", obs_unit="km/s")

        ax_plain = plot_rv(rv_samples, rv_data, n_samples=1, phase_fold_median=True)
        ax_trend = plot_rv(
            trend_samples,
            rv_data,
            extensions=(trend,),
            n_samples=1,
            phase_fold_median=True,
        )

        plain_data = next(line for line in ax_plain.lines if line.get_marker() == "o")
        trend_data = next(line for line in ax_trend.lines if line.get_marker() == "o")

        assert not np.allclose(plain_data.get_ydata(), trend_data.get_ydata())
        plt.close("all")

    def test_plot_with_gp_extension_can_show_signal_components(self, rv_samples):
        """GP plotting can decompose Keplerian and extension contributions."""
        tinygp = pytest.importorskip("tinygp")

        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        gp_samples = Samples(
            nonlinear={
                **rv_samples.nonlinear,
                "gp_amp": Q(jnp.ones(N) * 2.0, "km/s"),
                "gp_scale": Q(jnp.ones(N) * 10.0, "day"),
            },
            linear=rv_samples.linear,
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        gp = GP(
            kernel_builder=lambda hp: (
                tinygp.kernels.ExpSquared(hp["gp_scale"]) * hp["gp_amp"] ** 2
            ),
            hyperparams=(
                ParamInfo("gp_amp", "km/s"),
                ParamInfo("gp_scale", "day"),
            ),
            time_unit="day",
        )

        ax = plot_rv(
            gp_samples,
            rv_data,
            extensions=(gp,),
            n_samples=1,
            show_signal_components=True,
        )

        expected_alpha = get_alpha(1)
        component_lines = [
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        ]

        assert len(component_lines) == 2
        assert {line.get_label() for line in component_lines} == {
            "Extensions",
            "Keplerian",
        }
        assert not np.allclose(
            component_lines[0].get_ydata(),
            component_lines[1].get_ydata(),
        )
        plt.close("all")

    def test_plot_with_quasisep_gp_unsorted_times(self, rv_samples):
        """Quasisep GP plotting sorts unsorted RV times before conditioning."""
        tinygp = pytest.importorskip("tinygp")

        times = Q(jnp.array([20.0, 5.0, 35.0, 10.0, 25.0]), "day")
        rv = Q(jnp.zeros(5), "km/s")
        rv_err = Q(jnp.ones(5) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        gp_samples = Samples(
            nonlinear={
                **rv_samples.nonlinear,
                "gp_omega": Q(jnp.ones(N) * 0.2, "1/day"),
                "gp_Q": Q(jnp.ones(N) * 2.0, ""),
                "gp_sigma": Q(jnp.ones(N) * 1.0, "km/s"),
            },
            linear=rv_samples.linear,
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        gp = GP(
            kernel_builder=lambda hp: tinygp.kernels.quasisep.SHO(
                omega=hp["gp_omega"],
                quality=hp["gp_Q"],
                sigma=hp["gp_sigma"],
            ),
            hyperparams=(
                ParamInfo("gp_omega", "1/day"),
                ParamInfo("gp_Q", ""),
                ParamInfo("gp_sigma", "km/s"),
            ),
            time_unit="day",
        )

        ax = plot_rv(gp_samples, rv_data, extensions=(gp,), n_samples=1)

        expected_alpha = get_alpha(1)
        gp_line = next(
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )

        assert np.all(np.isfinite(gp_line.get_ydata()))
        plt.close("all")

    def test_plot_uses_explicit_time_grid(self, rv_samples):
        """Supplying time_grid bypasses the default get_t_grid path."""
        times = Q(jnp.array([0.0, 50.0, 100.0]), "day")
        rv = Q(jnp.zeros(3), "km/s")
        rv_err = Q(jnp.ones(3), "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        time_grid = Q(jnp.array([2.0, 4.0, 8.0, 16.0]), "day")

        with patch(
            "harv.plot.get_t_grid", side_effect=AssertionError("should not be called")
        ):
            ax = plot_rv(rv_samples, rv_data, n_samples=1, time_grid=time_grid)

        expected_alpha = get_alpha(1)
        orbit_line = next(
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )

        np.testing.assert_allclose(orbit_line.get_xdata(), np.asarray(time_grid.value))
        plt.close("all")

    def test_plot_with_gp_large_grid_chunks_conditioning(self, rv_samples):
        """GP plotting chunks large prediction grids to avoid dense blowups."""
        pytest.importorskip("tinygp")

        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        gp_samples = Samples(
            nonlinear={
                **rv_samples.nonlinear,
                "gp_amp": Q(jnp.ones(N) * 2.0, "km/s"),
                "gp_scale": Q(jnp.ones(N) * 10.0, "day"),
            },
            linear=rv_samples.linear,
            data_type=rv_samples.data_type,
            metadata=rv_samples.metadata,
        )
        gp = GP(
            kernel_builder=lambda hp: (
                hp["gp_amp"] ** 2 * jnp.exp(-0.5 * hp["gp_scale"])
            ),
            hyperparams=(
                ParamInfo("gp_amp", "km/s"),
                ParamInfo("gp_scale", "day"),
            ),
            time_unit="day",
        )

        chunk_lengths: list[int] = []

        class FakeGaussianProcess:
            def __init__(self, kernel, data_times, diag):
                del kernel, data_times, diag

            def condition(self, residuals, X_test):
                del residuals
                chunk_lengths.append(len(X_test))
                return None, SimpleNamespace(loc=jnp.zeros(len(X_test)))

        with (
            patch("harv.plot.tinygp.GaussianProcess", FakeGaussianProcess),
            patch(
                "harv.plot.get_t_grid",
                return_value=Q(jnp.linspace(-10.0, 110.0, 5000), "day"),
            ),
        ):
            ax = plot_rv(gp_samples, rv_data, extensions=(gp,), n_samples=1)

        expected_alpha = get_alpha(1)
        gp_line = next(
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        )

        assert np.all(np.isfinite(gp_line.get_ydata()))
        assert len(chunk_lengths) > 1
        assert max(chunk_lengths) <= 2048
        assert sum(chunk_lengths) == 5000
        plt.close("all")


@pytest.fixture
def gaia_data() -> GaiaAstrometryData:
    """Minimal GaiaAstrometryData spanning a few orbital periods."""
    n_obs = 24
    return GaiaAstrometryData(
        time=Q(jnp.linspace(0.0, 800.0, n_obs), "day"),
        al_position=Q(jnp.zeros(n_obs), "mas"),
        al_position_err=Q(jnp.ones(n_obs) * 0.1, "mas"),
        scan_angle=Q(jnp.linspace(0.0, 6.0, n_obs), "rad"),
        parallax_factor=jnp.linspace(-1.0, 1.0, n_obs),
    )


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotCornerTruths:
    """Truth-value handling in Samples.plot_corner.

    Regression: ``plot_corner`` unwrapped each Q-valued sample column to a
    raw array before checking the truth, so a stale ``isinstance(values, Q)``
    branch never fired and the fallback ``float(truth_val)`` raised a
    UnitConversionError on any Q truth with non-dimensionless units.
    """

    def _capture_truth_overlay(self, samples, truths, params):
        fake_axes = np.empty((len(params), len(params)), dtype=object)
        for idx in np.ndindex(fake_axes.shape):
            fake_axes[idx] = MagicMock()
        fake_plot_matrix = SimpleNamespace(
            backend="matplotlib",
            viz={"plot": SimpleNamespace(values=fake_axes)},
        )
        with patch("harv.samplers.samples.az.plot_pair", return_value=fake_plot_matrix):
            result = samples.plot_corner(params=params, truths=truths)
        return result, fake_axes

    def test_q_truth_converted_to_sample_unit(self, rv_samples):
        """Q truth in a different but compatible unit converts correctly."""
        truths = {"period": Q(0.27, "yr")}  # ~98.6 day
        _, axes = self._capture_truth_overlay(rv_samples, truths, ["period"])
        assert axes[0, 0].axvline.call_args.args[0] == pytest.approx(
            0.27 * 365.25, abs=0.5
        )

    def test_q_truth_same_unit_as_sample(self, rv_samples):
        """Q truth already in the sample's unit is returned unchanged."""
        truths = {"rv_semiamp": Q(5.0, "km/s")}
        _, axes = self._capture_truth_overlay(rv_samples, truths, ["rv_semiamp"])
        assert axes[0, 0].axvline.call_args.args[0] == pytest.approx(5.0)

    def test_plain_float_truth_with_dimensionless_sample(self, rv_samples):
        """Plain-float truth still works for a dimensionless sample column."""
        truths = {"eccentricity": 0.15}
        _, axes = self._capture_truth_overlay(rv_samples, truths, ["eccentricity"])
        assert axes[0, 0].axvline.call_args.args[0] == pytest.approx(0.15)

    def test_truths_overlay_on_lower_triangle_pair_plot(self, rv_samples):
        """Two-parameter corner plots draw lower-triangle truth crosshairs."""
        truths = {"period": Q(100.0, "day"), "eccentricity": 0.2}
        _, axes = self._capture_truth_overlay(
            rv_samples, truths, ["period", "eccentricity"]
        )

        axes[0, 0].axvline.assert_called_once()
        axes[1, 1].axvline.assert_called_once()
        axes[1, 0].axvline.assert_called_once()
        axes[1, 0].axhline.assert_called_once()
        axes[1, 0].scatter.assert_called_once()
        axes[0, 1].scatter.assert_not_called()


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotGaiaAstrometry:
    """Tests for harv.plot.plot_gaia_astrometry."""

    def test_returns_figure(self, astro_samples, gaia_data):
        """plot_gaia_astrometry() returns a matplotlib Figure when axes=None."""
        fig = plot_gaia_astrometry(astro_samples[0], data=gaia_data)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_figure_has_two_axes(self, astro_samples, gaia_data):
        """The two-panel astrometry plot produces exactly two Axes."""
        fig = plot_gaia_astrometry(astro_samples[0], data=gaia_data)
        assert len(fig.axes) == 2
        plt.close("all")

    def test_sky_panel_equal_aspect(self, astro_samples, gaia_data):
        """The sky-projection panel (axes[0]) uses equal aspect ratio."""
        fig = plot_gaia_astrometry(astro_samples[0], data=gaia_data)
        assert fig.axes[0].get_aspect() != "auto"
        plt.close("all")

    def test_multiple_samples_raises(self, astro_samples, gaia_data):
        """Passing more than one sample raises a friendly ValueError."""
        with pytest.raises(ValueError, match="exactly one posterior sample"):
            plot_gaia_astrometry(astro_samples, data=gaia_data)
        plt.close("all")

    def test_residual_panel_labelled_with_zero_line(self, astro_samples, gaia_data):
        """The residual panel (axes[1]) is labelled and has a y=0 reference line."""
        fig = plot_gaia_astrometry(astro_samples[0], data=gaia_data)
        ax_resid = fig.axes[1]
        assert ax_resid.get_xlabel().startswith("time")
        assert "residual" in ax_resid.get_ylabel()
        has_zero_line = any(
            len(line.get_ydata()) > 0 and np.allclose(line.get_ydata(), 0.0)
            for line in ax_resid.get_lines()
        )
        assert has_zero_line
        plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotGaiaSkyOrbit:
    """Tests for harv.plot.plot_gaia_sky_orbit."""

    def _orbit_params(self, astro_samples) -> dict:
        return {
            "period": astro_samples["period"][0],
            "eccentricity": astro_samples["eccentricity"][0],
            "t_peri": astro_samples["t_peri"][0],
            "arg_peri": astro_samples["arg_peri"][0],
            "cos_i": astro_samples["cos_i"][0],
            "lon_asc_node": astro_samples["lon_asc_node"][0],
            "semi_major_axis": astro_samples["semi_major_axis"][0],
        }

    def test_returns_figure_no_data(self, astro_samples):
        """Without data, only the orbit ellipse is drawn."""
        fig = plot_gaia_sky_orbit(self._orbit_params(astro_samples), data=None)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_returns_figure_with_data(self, astro_samples, gaia_data):
        """With data, scan-direction segments are drawn at each epoch."""
        fig = plot_gaia_sky_orbit(self._orbit_params(astro_samples), data=gaia_data)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_equal_aspect(self, astro_samples):
        """The sky-orbit axes use equal aspect ratio."""
        fig = plot_gaia_sky_orbit(self._orbit_params(astro_samples))
        assert fig.axes[0].get_aspect() != "auto"
        plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotCombined:
    """Tests for plot_rv and plot_gaia_astrometry on combined (joint) samples."""

    def test_plot_rv_on_combined_samples(self, combined_samples):
        """plot_rv works on combined samples that include RV parameters."""
        ax = plot_rv(combined_samples)
        assert ax is not None
        plt.close("all")

    def test_plot_gaia_astrometry_on_combined_samples(
        self, combined_samples, gaia_data
    ):
        """plot_gaia_astrometry works on combined samples with astrometry params."""
        fig = plot_gaia_astrometry(combined_samples[0], data=gaia_data)
        assert hasattr(fig, "savefig")
        plt.close("all")


# ---------------------------------------------------------------------------
# Thiele-Innes to Campbell conversion
# ---------------------------------------------------------------------------

_ARG_PERI = 0.7
_LON_ASC_NODE = 1.3
_COS_I = 0.5
_A = 2.5  # mas


@pytest.fixture
def ti_samples() -> Samples:
    """Samples with Thiele-Innes linear params built from known Campbell elements."""
    A, B, F, G = thiele_innes_ABFG(
        jnp.cos(_ARG_PERI),
        jnp.sin(_ARG_PERI),
        jnp.cos(_LON_ASC_NODE),
        jnp.sin(_LON_ASC_NODE),
        _COS_I,
    )
    nonlinear = {
        "period": Q(jnp.ones(N) * 90.0, "day"),
        "eccentricity": Q(jnp.ones(N) * 0.1, ""),
        "phase_peri": Q(jnp.ones(N) * 0.3, ""),
    }
    linear = {
        "ra0": Q(jnp.zeros(N), "mas"),
        "dec0": Q(jnp.zeros(N), "mas"),
        "pmra": Q(jnp.ones(N) * 10.0, "mas/yr"),
        "pmdec": Q(jnp.ones(N) * -5.0, "mas/yr"),
        "parallax": Q(jnp.ones(N) * 5.0, "mas"),
        "ti_A": Q(jnp.ones(N) * float(_A * A), "mas"),
        "ti_B": Q(jnp.ones(N) * float(_A * B), "mas"),
        "ti_F": Q(jnp.ones(N) * float(_A * F), "mas"),
        "ti_G": Q(jnp.ones(N) * float(_A * G), "mas"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="GaiaAstrometryModel",
        metadata={"t_ref": 0.0},
    )


class TestThieleInnesToCampbell:
    def test_output_keys(self, ti_samples: Samples) -> None:
        result = ti_samples.thiele_innes_to_campbell()
        assert set(result.nonlinear.keys()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
            "lon_asc_node",
            "cos_i",
        }
        assert "semi_major_axis" in result.linear
        for name in ("ti_A", "ti_B", "ti_F", "ti_G"):
            assert name not in result.linear

    def test_preserved_params(self, ti_samples: Samples) -> None:
        result = ti_samples.thiele_innes_to_campbell()
        assert "ra0" in result.linear
        assert "parallax" in result.linear

    def test_recovered_semi_major_axis(self, ti_samples: Samples) -> None:
        result = ti_samples.thiele_innes_to_campbell()

        a_out = ustrip("mas", result.linear["semi_major_axis"])
        assert jnp.allclose(a_out, _A, atol=1e-5)

    def test_recovered_cos_i(self, ti_samples: Samples) -> None:
        result = ti_samples.thiele_innes_to_campbell()

        cos_i_out = ustrip("", result.nonlinear["cos_i"])
        assert jnp.allclose(cos_i_out, _COS_I, atol=1e-5)

    def test_recovered_arg_peri(self, ti_samples: Samples) -> None:
        # (ω, Ω) and (ω+π, Ω+π) are degenerate; verify via TI round-trip
        result = ti_samples.thiele_innes_to_campbell()

        arg_peri_out = ustrip("rad", result.nonlinear["arg_peri"])[0]
        lon_asc_node_out = ustrip("rad", result.nonlinear["lon_asc_node"])[0]
        cos_i_out = ustrip("", result.nonlinear["cos_i"])[0]
        A_rt, B_rt, *_ = thiele_innes_ABFG(
            jnp.cos(arg_peri_out),
            jnp.sin(arg_peri_out),
            jnp.cos(lon_asc_node_out),
            jnp.sin(lon_asc_node_out),
            cos_i_out,
        )
        # Must recover original unit TI constants (a0 cancels)
        A, B, *_ = thiele_innes_ABFG(
            jnp.cos(_ARG_PERI),
            jnp.sin(_ARG_PERI),
            jnp.cos(_LON_ASC_NODE),
            jnp.sin(_LON_ASC_NODE),
            _COS_I,
        )
        assert jnp.allclose(A_rt, A, atol=1e-5)
        assert jnp.allclose(B_rt, B, atol=1e-5)

    def test_missing_ti_raises(self) -> None:
        # Partial TI constants (present but incomplete) raise; a complete
        # absence of TI constants is a no-op (see test_samples.py).
        samples = Samples(
            nonlinear={
                "period": Q(jnp.ones(3), "day"),
                "eccentricity": Q(jnp.zeros(3), ""),
                "phase_peri": Q(jnp.zeros(3), ""),
            },
            linear={
                "ra0": Q(jnp.zeros(3), "mas"),
                "ti_A": Q(jnp.zeros(3), "mas"),
                "ti_B": Q(jnp.zeros(3), "mas"),
            },
            data_type="GaiaAstrometryModel",
        )
        with pytest.raises(RuntimeError, match="ti_"):
            samples.thiele_innes_to_campbell()
