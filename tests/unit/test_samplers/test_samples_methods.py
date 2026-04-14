"""Tests for Samples.init_mcmc (D3) and Samples.plot (D4).

Both methods operate on an existing Samples object.  Rather than running the
full rejection sampler (slow), tests build a minimal Samples instance directly
using the constructor.  init_mcmc lives on NumpyroSampler.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as ndist
import pytest
from numpyro import infer
from unxt import Q

from harv.data import RVData
from harv.kepler.orbits import astrometric_orbit_at_times, rv_at_times
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    RVParameters,
)
from harv.model import Model
from harv.samplers.numpyro import NumpyroSampler
from harv.samplers.rejection_prior import RejectionPrior
from harv.samplers.samples import Samples, WarmStartMCMC

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
        orbit_cls=RVParameters,
        full_cls=(RVParameters,),
        data_type="rv",
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
        orbit_cls=GaiaAstrometryParameters,
        full_cls=(GaiaAstrometryParameters,),
        data_type="astrometry",
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
        orbit_cls=GaiaAstrometryParameters,
        full_cls=(GaiaAstrometryParameters, RVParameters),
        data_type="combined",
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
        orbit_cls=RVParameters,
        full_cls=(RVParameters,),
        data_type="rv",
        metadata={},
    )


@pytest.fixture
def rv_sampler_and_data():
    """NumpyroSampler + RVData used for init_mcmc tests."""
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
    sampler = NumpyroSampler(Model(prior, data))
    return sampler


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
        kwargs = dict(
            period=Q(200.0, "day"),
            eccentricity=0.0,
            t_peri=Q(0.0, "day"),
            arg_peri=Q(0.0, "rad"),
            rv_semiamp=Q(10.0, "km/s"),
        )
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
# Tests: Samples.init_mcmc (D3)
# ---------------------------------------------------------------------------


class TestInitMcmc:
    """Tests for Samples.init_mcmc."""

    def test_returns_warm_start_mcmc(self, rv_samples, rv_sampler_and_data):
        """init_mcmc returns a WarmStartMCMC wrapping a numpyro MCMC."""
        sampler = rv_sampler_and_data
        result = sampler.init_mcmc(
            rv_samples, num_chains=2, num_warmup=10, num_samples=10
        )
        assert isinstance(result, WarmStartMCMC)

    def test_init_params_keys_match_nonlinear(self, rv_samples, rv_sampler_and_data):
        """init_params dict contains all nonlinear parameter keys."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, num_chains=2, num_warmup=10, num_samples=10
        )
        params = mcmc._init_params
        for key in rv_samples.nonlinear:
            assert key in params, f"Missing key '{key}' in init_params"

    def test_init_params_shape_equals_num_chains(self, rv_samples, rv_sampler_and_data):
        """Each init_params array has shape (num_chains,)."""
        sampler = rv_sampler_and_data
        num_chains = 3
        mcmc = sampler.init_mcmc(
            rv_samples, num_chains=num_chains, num_warmup=10, num_samples=10
        )
        for key, arr in mcmc._init_params.items():
            assert arr.shape == (num_chains,), (
                f"Expected shape ({num_chains},) for '{key}', got {arr.shape}"
            )

    def test_init_params_values_from_posterior(self, rv_samples, rv_sampler_and_data):
        """Starting positions, when transformed back, match the posterior."""
        from numpyro.distributions import biject_to

        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, num_chains=3, num_warmup=10, num_samples=10
        )
        # init_params are in unconstrained space.  Round-trip through the
        # forward transform to recover the original constrained values.
        from harv.likelihood.helpers import _unwrap_dist

        d = _unwrap_dist(sampler.model.prior.nonlinear_priors["period"])
        transform = biject_to(d.support)
        recovered = np.asarray(transform(mcmc._init_params["period"]))
        expected = np.asarray(rv_samples.nonlinear["period"].value)[:3]
        np.testing.assert_allclose(recovered, expected, rtol=1e-5)

    def test_raises_for_empty_samples(self, empty_rv_samples, rv_sampler_and_data):
        """init_mcmc raises ValueError when there are no posterior samples."""
        sampler = rv_sampler_and_data
        with pytest.raises(ValueError, match="no posterior samples"):
            sampler.init_mcmc(
                empty_rv_samples, num_chains=2, num_warmup=10, num_samples=10
            )

    def test_broadcast_init_when_fewer_samples_than_chains(
        self, rv_samples, rv_sampler_and_data
    ):
        """When n_samples < num_chains, init_params are scalar (broadcast)."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            num_chains=N + 5,  # more chains than samples
            num_warmup=10,
            num_samples=10,
        )
        # Scalar init values are broadcast by numpyro to all chains.
        for key, val in mcmc._init_params.items():
            assert val.ndim == 0, (
                f"init_params['{key}'] should be scalar when broadcasting"
            )

    def test_warm_start_mcmc_delegates_attributes(
        self, rv_samples, rv_sampler_and_data
    ):
        """WarmStartMCMC delegates unknown attributes to the underlying MCMC."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, num_chains=2, num_warmup=10, num_samples=10
        )
        # num_chains is an attribute of the underlying numpyro MCMC object.
        assert mcmc.num_chains == 2

    def test_default_kernel_is_nuts(self, rv_samples, rv_sampler_and_data):
        """When kernel is omitted the default is NUTS."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, num_warmup=10, num_samples=10, num_chains=2
        )
        assert isinstance(mcmc._mcmc.sampler, infer.NUTS)

    def test_run_produces_posterior_samples(self, rv_samples, rv_sampler_and_data):
        """mcmc.run() completes and get_samples() returns the expected keys."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        mcmc.run(jr.key(0))
        posterior = mcmc.get_samples()
        # The auto-generated model samples all keys from prior.nonlinear_priors.
        for key in sampler.model.prior.nonlinear_priors:
            assert key in posterior, f"Missing site '{key}' in posterior"
        assert posterior["period"].shape == (10,)  # 2 chains x 5 samples

    def test_single_chain_produces_scalar_init_params(
        self, rv_samples, rv_sampler_and_data
    ):
        """For num_chains=1, all init_params values are 0-d (scalar)."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(rv_samples, num_chains=1, num_warmup=5, num_samples=5)
        for key, val in mcmc._init_params.items():
            assert val.ndim == 0, (
                f"init_params['{key}'] should be scalar for single chain"
            )

    def test_run_single_chain(self, rv_samples, rv_sampler_and_data):
        """MCMC with num_chains=1 completes and produces expected shapes."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(rv_samples, num_chains=1, num_warmup=3, num_samples=3)
        mcmc.run(jr.key(0))
        posterior = mcmc.get_samples()
        for key in sampler.model.prior.nonlinear_priors:
            assert key in posterior
        assert posterior["period"].shape == (3,)  # 1 chain x 3 samples


class TestInitMcmcFull:
    """Tests for NumpyroSampler.init_mcmc with marginalized=False."""

    def test_returns_warm_start_mcmc(self, rv_samples, rv_sampler_and_data):
        """init_mcmc(marginalized=False) returns a WarmStartMCMC."""
        sampler = rv_sampler_and_data
        result = sampler.init_mcmc(
            rv_samples,
            marginalized=False,
            num_chains=2,
            num_warmup=10,
            num_samples=10,
        )
        assert isinstance(result, WarmStartMCMC)

    def test_init_params_has_linear_site(self, rv_samples, rv_sampler_and_data):
        """Full model init_params includes '_linear', not individual named sites."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            marginalized=False,
            num_chains=2,
            num_warmup=10,
            num_samples=10,
        )
        assert "_linear" in mcmc._init_params
        # Named linear params are deterministic sites, not init_params entries.
        assert "rv_semiamp" not in mcmc._init_params
        assert "v_sys" not in mcmc._init_params

    def test_linear_init_shape(self, rv_samples, rv_sampler_and_data):
        """'_linear' init has shape (num_chains, n_linear)."""
        sampler = rv_sampler_and_data
        num_chains = 2
        mcmc = sampler.init_mcmc(
            rv_samples,
            marginalized=False,
            num_chains=num_chains,
            num_warmup=10,
            num_samples=10,
        )
        n_linear = len(rv_samples.linear)
        assert mcmc._init_params["_linear"].shape == (num_chains, n_linear)

    def test_run_produces_named_linear_params(self, rv_samples, rv_sampler_and_data):
        """mcmc.run() produces named deterministic sites K, v0, etc."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            marginalized=False,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        mcmc.run(jr.key(0))
        posterior = mcmc.get_samples()
        # Nonlinear sites must be present.
        for key in sampler.model.prior.nonlinear_priors:
            assert key in posterior, f"Missing nonlinear site '{key}'"
        # Named linear deterministic sites must be present.
        for name in rv_samples.linear:
            assert name in posterior, f"Missing linear site '{name}'"

    def test_marginalized_true_has_no_linear_site(
        self, rv_samples, rv_sampler_and_data
    ):
        """Marginalized model (default) does not put '_linear' in init_params."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            num_chains=2,
            num_warmup=10,
            num_samples=10,
        )
        assert "_linear" not in mcmc._init_params


# ---------------------------------------------------------------------------
# Tests: NumpyroSampler.init_mcmc with extra_model
# ---------------------------------------------------------------------------


class TestInitMcmcExtraModel:
    """Tests for NumpyroSampler.init_mcmc with a physical reparameterization via extra_model."""

    def _make_extra_model(self):
        """Return a minimal extra_model that fixes K to a constant."""

        def extra_model(pars):
            # Sample a dummy physical parameter, compute K from it.
            K_scale = numpyro.sample("K_scale", ndist.HalfNormal(10.0))
            return {"rv_semiamp": K_scale}

        return extra_model

    def test_raises_without_extra_init_params(self, rv_samples, rv_sampler_and_data):
        """extra_model without extra_init_params raises ValueError."""
        sampler = rv_sampler_and_data
        with pytest.raises(ValueError, match="extra_init_params is required"):
            sampler.init_mcmc(
                rv_samples,
                extra_model=self._make_extra_model(),
                num_chains=2,
                num_warmup=5,
                num_samples=5,
            )

    def test_returns_warm_start_mcmc(self, rv_samples, rv_sampler_and_data):
        """init_mcmc with extra_model returns a WarmStartMCMC."""
        sampler = rv_sampler_and_data
        result = sampler.init_mcmc(
            rv_samples,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            num_chains=2,
            num_warmup=5,
            num_samples=5,
        )
        assert isinstance(result, WarmStartMCMC)

    def test_init_params_includes_extra(self, rv_samples, rv_sampler_and_data):
        """init_params contains both nonlinear and extra_init_params entries."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            num_chains=2,
            num_warmup=5,
            num_samples=5,
        )
        assert "period" in mcmc._init_params
        assert "K_scale" in mcmc._init_params
        # Linear params are not explicitly sampled in the marginalized case.
        assert "_linear" not in mcmc._init_params

    def test_run_extra_model_marginalized(self, rv_samples, rv_sampler_and_data):
        """extra_model + marginalized=True runs and returns expected sites."""
        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            marginalized=True,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        mcmc.run(jr.key(42))
        posterior = mcmc.get_samples()

        # Nonlinear sites must be present.
        for key in sampler.model.prior.nonlinear_priors:
            assert key in posterior
        # The extra-model site is present.
        assert "K_scale" in posterior
        # K is exposed as a deterministic site.
        assert "rv_semiamp" in posterior
        # v_sys is analytically marginalized -- not a sample site.
        assert "v_sys" not in posterior

    def test_extra_model_raises_for_unknown_param(
        self, rv_samples, rv_sampler_and_data
    ):
        """extra_model returning an unknown linear param name raises ValueError."""

        def bad_extra_model(pars):
            x = numpyro.sample("x", ndist.Normal(0.0, 1.0))
            return {"not_a_real_param": x}

        sampler = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            extra_model=bad_extra_model,
            extra_init_params={"x": jnp.zeros(2)},
            num_chains=2,
            num_warmup=1,
            num_samples=1,
            chain_method="sequential",
        )
        with pytest.raises(ValueError, match="unknown linear parameter name"):
            mcmc.run(jr.key(0))


# ---------------------------------------------------------------------------
# Tests: init_mcmc with non-Gaussian linear priors (regression)
# ---------------------------------------------------------------------------


class TestInitMcmcNonGaussianLinear:
    """Regression tests for init_mcmc when linear priors include non-Gaussian
    distributions (e.g. HalfNormal for parallax) that must be explicitly
    sampled rather than analytically marginalized.
    """

    @pytest.fixture
    def astro_sampler_and_data(self):
        """NumpyroSampler + GaiaAstrometryData with HalfNormal parallax."""
        from harv.data import GaiaAstrometryData

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
        sampler = NumpyroSampler(Model(prior, data))
        return sampler

    def test_init_params_includes_explicit_linear(
        self, astro_samples, astro_sampler_and_data
    ):
        """Non-Gaussian linear params appear in init_params when marginalized=True."""
        sampler = astro_sampler_and_data
        mcmc = sampler.init_mcmc(
            astro_samples, num_chains=2, num_warmup=5, num_samples=5
        )
        # HalfNormal parallax is not analytically marginalized, so it must
        # appear as an explicit init_params entry.
        assert "parallax" in mcmc._init_params

    def test_init_params_excludes_marginalized_linear(
        self, astro_samples, astro_sampler_and_data
    ):
        """Analytically marginalized linear params are NOT in init_params."""
        sampler = astro_sampler_and_data
        mcmc = sampler.init_mcmc(
            astro_samples, num_chains=2, num_warmup=5, num_samples=5
        )
        # ra0, dec0 have Normal priors and should be marginalized out.
        assert "ra0" not in mcmc._init_params
        assert "dec0" not in mcmc._init_params

    def test_explicit_linear_shape(self, astro_samples, astro_sampler_and_data):
        """Explicit linear init values have shape (num_chains,)."""
        sampler = astro_sampler_and_data
        num_chains = 3
        mcmc = sampler.init_mcmc(
            astro_samples, num_chains=num_chains, num_warmup=5, num_samples=5
        )
        assert mcmc._init_params["parallax"].shape == (num_chains,)

    def test_run_marginalized_with_halfnormal_parallax(
        self, astro_samples, astro_sampler_and_data
    ):
        """MCMC runs without error when parallax has a HalfNormal prior."""
        sampler = astro_sampler_and_data
        mcmc = sampler.init_mcmc(
            astro_samples,
            num_chains=2,
            num_warmup=3,
            num_samples=3,
            chain_method="sequential",
        )
        mcmc.run(jr.key(99))
        posterior = mcmc.get_samples()
        # parallax must appear in posterior as an explicit site.
        assert "parallax" in posterior
        assert posterior["parallax"].shape == (6,)  # 2 chains x 3 samples

    def test_single_chain_init_params_are_scalar(
        self, astro_samples, astro_sampler_and_data
    ):
        """For num_chains=1, init_params values must be 0-d (scalar) arrays."""
        sampler = astro_sampler_and_data
        mcmc = sampler.init_mcmc(
            astro_samples, num_chains=1, num_warmup=3, num_samples=3
        )
        for key, val in mcmc._init_params.items():
            assert val.ndim == 0, (
                f"init_params['{key}'] has ndim={val.ndim} (shape {val.shape}), "
                "expected scalar (0-d) for num_chains=1"
            )

    def test_run_single_chain_with_halfnormal_parallax(
        self, astro_samples, astro_sampler_and_data
    ):
        """MCMC with num_chains=1 and HalfNormal parallax runs without error."""
        sampler = astro_sampler_and_data
        mcmc = sampler.init_mcmc(
            astro_samples,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
        )
        mcmc.run(jr.key(42))
        posterior = mcmc.get_samples()
        assert "parallax" in posterior
        assert posterior["parallax"].shape == (3,)  # 1 chain x 3 samples


# ---------------------------------------------------------------------------
# Tests: init_mcmc with combined (astrometry + RV) model + jitter (regression)
# ---------------------------------------------------------------------------


class TestInitMcmcCombinedWithJitter:
    """Regression tests for init_mcmc on a combined astrometry+RV model.

    These catch issues that previously only surfaced when running the full
    getting-started notebook, such as:
    - Constrained init_params causing pathologically tiny step sizes
    - Shape mismatches for single-chain init with HalfNormal jitter priors
    """

    @pytest.fixture
    def combined_sampler_and_data(self):
        """Combined NumpyroSampler + SourceData with jitter priors."""
        import numpyro.distributions as ndist

        from harv.data import GaiaAstrometryData, SourceData
        from harv.distributions import QD

        # Minimal astrometry data
        n_ast = 15
        times_ast = Q(jnp.linspace(0.0, 1000.0, n_ast), "day")
        astro = GaiaAstrometryData(
            time=times_ast,
            al_position=Q(jnp.zeros(n_ast), "mas"),
            al_position_err=Q(jnp.ones(n_ast) * 0.1, "mas"),
            scan_angle=Q(jnp.linspace(0.0, 3.14, n_ast), "rad"),
            parallax_factor=jnp.full(n_ast, 0.5),
        )

        # Minimal RV data
        n_rv = 8
        times_rv = Q(jnp.linspace(0.0, 800.0, n_rv), "day")
        rv = RVData(
            time=times_rv,
            rv=Q(jnp.zeros(n_rv), "km/s"),
            rv_err=Q(jnp.ones(n_rv) * 1.0, "km/s"),
        )

        source_data = SourceData(gaia=astro, rv=rv)

        # Combined prior with mixed transform types:
        # - Normal (unconstrained), Uniform (bounded), HalfNormal (positive)
        nonlinear = {
            "period": QD(ndist.Normal(300.0, 50.0), "day"),
            "eccentricity": ndist.TruncatedNormal(0.3, 0.2, low=0.0, high=1.0),
            "phase_peri": ndist.Uniform(0.0, 1.0),
            "cos_i": ndist.Uniform(-1.0, 1.0),
            "arg_peri": QD(ndist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
            "lon_asc_node": QD(ndist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        }
        linear = {
            "ra0": QD(ndist.Normal(0.0, 100.0), "mas"),
            "dec0": QD(ndist.Normal(0.0, 100.0), "mas"),
            "pmra": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
            "pmdec": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
            "parallax": QD(ndist.HalfNormal(10.0), "mas"),
            "semi_major_axis": QD(ndist.Normal(0.0, 50.0), "mas"),
            "rv_semiamp": QD(ndist.Normal(0.0, 30.0), "km/s"),
            "v_sys": QD(ndist.Normal(0.0, 30.0), "km/s"),
        }
        jitter_priors = {
            "rv": QD(ndist.HalfNormal(4.0), "km/s"),
        }
        prior = RejectionPrior(
            nonlinear_priors=nonlinear,
            linear_prior=linear,
            jitter_priors=jitter_priors,
        )
        sampler = NumpyroSampler(Model(prior, source_data))
        return sampler

    @pytest.fixture
    def combined_samples_with_jitter(self) -> Samples:
        """Minimal combined Samples that include jitter_rv in nonlinear."""
        nonlinear = {
            "period": Q(jnp.linspace(280.0, 320.0, N), "day"),
            "eccentricity": Q(jnp.linspace(0.1, 0.5, N), ""),
            "phase_peri": Q(jnp.linspace(0.1, 0.9, N), ""),
            "arg_peri": Q(jnp.linspace(0.5, 5.5, N), "rad"),
            "cos_i": Q(jnp.linspace(-0.5, 0.5, N), ""),
            "lon_asc_node": Q(jnp.linspace(0.5, 5.5, N), "rad"),
            "jitter_rv": Q(jnp.linspace(1.0, 5.0, N), "km/s"),
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
            orbit_cls=GaiaAstrometryParameters,
            full_cls=(GaiaAstrometryParameters, RVParameters),
            data_type="combined",
            metadata={"t_ref": 0.0},
        )

    def test_init_params_include_jitter(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Jitter site appears in init_params for combined model."""
        sampler = combined_sampler_and_data
        mcmc = sampler.init_mcmc(
            combined_samples_with_jitter,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
        )
        assert "jitter_rv" in mcmc._init_params

    def test_init_params_are_unconstrained(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Init values for bounded params differ from raw constrained values.

        This catches the bug where constrained values were passed directly,
        causing numpyro to misinterpret them and produce tiny step sizes.
        """
        sampler = combined_sampler_and_data
        mcmc = sampler.init_mcmc(
            combined_samples_with_jitter,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
        )
        # For Uniform(0, 2*pi) arg_peri, unconstrained != constrained.
        unconstrained = np.asarray(mcmc._init_params["arg_peri"])
        constrained = np.asarray(
            combined_samples_with_jitter.nonlinear["arg_peri"].value[:2]
        )
        # Unconstrained values should be logit-transformed, not raw angles.
        assert not np.allclose(unconstrained, constrained, atol=0.01)

    def test_run_marginalized_completes(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Combined marginalized MCMC with jitter runs without error."""
        sampler = combined_sampler_and_data
        mcmc = sampler.init_mcmc(
            combined_samples_with_jitter,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
        )
        mcmc.run(jr.key(0))
        posterior = mcmc.get_samples()
        # All nonlinear sites present
        for key in sampler.model.prior.nonlinear_priors:
            assert key in posterior
        # Jitter site present
        assert "jitter_rv" in posterior
        # Explicit linear (HalfNormal parallax) present
        assert "parallax" in posterior

    def test_step_size_is_reasonable(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """MCMC step size should not be pathologically tiny.

        Regression: when init_params were in constrained (instead of
        unconstrained) space, numpyro applied the forward transform to
        already-constrained values, placing the chain at a terrible point
        with step_size ~ 1e-14.
        """
        sampler = combined_sampler_and_data
        mcmc = sampler.init_mcmc(
            combined_samples_with_jitter,
            num_chains=1,
            num_warmup=5,
            num_samples=3,
        )
        mcmc.run(jr.key(42))
        last_state = mcmc._mcmc.last_state
        step_size = float(last_state.adapt_state.step_size)
        # A healthy NUTS step size is typically 0.001–1.0.
        # The old bug produced ~5e-14. Anything > 1e-6 is fine.
        assert step_size > 1e-6, (
            f"Step size {step_size:.2e} is pathologically small — "
            "init_params may be in the wrong (constrained) space"
        )

    def test_run_full_completes(
        self, combined_samples_with_jitter, combined_sampler_and_data
    ):
        """Combined full (non-marginalized) MCMC runs without error.

        Regression: the full model previously failed on combined data because
        (a) callable linear priors received a composite dict instead of a
        single params object, and (b) init_params included explicit linear
        keys (e.g. ``parallax``) that belong in ``_linear`` for the full model.
        """
        sampler = combined_sampler_and_data
        mcmc = sampler.init_mcmc(
            combined_samples_with_jitter,
            num_chains=1,
            num_warmup=3,
            num_samples=3,
            marginalized=False,
        )
        # ``parallax`` is an explicit (HalfNormal) linear param sampled as a
        # separate numpyro site, not part of ``_linear``.
        assert "parallax" in mcmc._init_params
        assert "_linear" in mcmc._init_params
        mcmc.run(jr.key(0))
        posterior = mcmc.get_samples()
        # All nonlinear sites present
        for key in sampler.model.prior.nonlinear_priors:
            assert key in posterior
        # Linear parameters exposed as deterministic sites
        assert "_linear" in posterior


# ---------------------------------------------------------------------------
# Tests: Samples.plot (D4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotRV:
    """Tests for Samples.plot with data_type='rv'."""

    def test_returns_figure(self, rv_samples):
        """plot() returns a matplotlib Figure."""
        fig = rv_samples.plot()
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_figure_has_one_axes(self, rv_samples):
        """RV plot produces exactly one Axes."""
        fig = rv_samples.plot()
        assert len(fig.axes) == 1
        plt.close("all")

    def test_plot_with_rv_data(self, rv_samples):
        """plot(data=rv_data) plots data points without error."""
        times = Q(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Q(jnp.zeros(20), "km/s")
        rv_err = Q(jnp.ones(20) * 2.0, "km/s")
        rv_data = RVData(time=times, rv=rv, rv_err=rv_err)
        fig = rv_samples.plot(data=rv_data)
        assert fig is not None
        plt.close("all")

    def test_n_samples_limits_model_curves(self, rv_samples):
        """n_samples controls how many posterior curves are drawn."""
        fig = rv_samples.plot(n_samples=3)
        ax = fig.axes[0]
        # Adaptive alpha = max(0.08, min(0.6, 8/n)); for n=3 this is 0.6.
        expected_alpha = max(0.08, min(0.6, 8.0 / 3))
        model_lines = [
            line
            for line in ax.lines
            if line.get_alpha() is not None
            and abs(line.get_alpha() - expected_alpha) < 0.01
        ]
        assert len(model_lines) == 3
        plt.close("all")

    def test_raises_for_bad_data_type(self, rv_samples):
        """plot() raises ValueError for unrecognised data argument."""
        with pytest.raises(ValueError, match="must be"):
            rv_samples.plot(data="not_a_data_object")

    def test_xlabel_is_phase(self, rv_samples):
        """X-axis label mentions orbital phase when phase_fold=True."""
        fig = rv_samples.plot(phase_fold=True)
        assert "phase" in fig.axes[0].get_xlabel().lower()
        plt.close("all")

    def test_xlabel_is_time_by_default(self, rv_samples):
        """Default (phase_fold=False) x-axis label mentions time."""
        fig = rv_samples.plot()
        assert "time" in fig.axes[0].get_xlabel().lower()
        plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotAstrometry:
    """Tests for Samples.plot with data_type='astrometry'."""

    def test_returns_figure(self, astro_samples):
        """plot() returns a matplotlib Figure."""
        fig = astro_samples.plot()
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_figure_has_one_axes(self, astro_samples):
        """Astrometry plot produces exactly one Axes."""
        fig = astro_samples.plot()
        assert len(fig.axes) == 1
        plt.close("all")

    def test_axes_equal_aspect(self, astro_samples):
        """On-sky plot uses equal aspect ratio (not 'auto')."""
        fig = astro_samples.plot()
        assert fig.axes[0].get_aspect() != "auto"
        plt.close("all")

    def test_data_argument_ignored(self, astro_samples):
        """Astrometry plot accepts data=None without error."""
        fig = astro_samples.plot(data=None)
        assert fig is not None
        plt.close("all")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
class TestPlotCombined:
    """Tests for Samples.plot with data_type='combined'."""

    def test_returns_figure(self, combined_samples):
        """plot() returns a matplotlib Figure."""
        fig = combined_samples.plot()
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_figure_has_two_axes(self, combined_samples):
        """Combined plot produces exactly two Axes (RV + sky orbit)."""
        fig = combined_samples.plot()
        assert len(fig.axes) == 2
        plt.close("all")


class TestPlotUnknownDataType:
    """Tests for Samples.plot error handling."""

    def test_raises_for_unknown_data_type(self):
        """plot() raises ValueError for an unknown data_type."""
        bad_samples = Samples(
            nonlinear={
                "period": Q(jnp.ones(5) * 100.0, "day"),
                "eccentricity": Q(jnp.zeros(5), ""),
                "phase_peri": Q(jnp.zeros(5), ""),
                "arg_peri": Q(jnp.zeros(5), "rad"),
            },
            linear={
                "rv_semiamp": Q(jnp.zeros(5), "km/s"),
                "v_sys": Q(jnp.zeros(5), "km/s"),
            },
            orbit_cls=RVParameters,
            full_cls=(RVParameters,),
            data_type="unknown",
            metadata={},
        )
        with pytest.raises(ValueError, match="Unknown data_type"):
            bad_samples.plot()
