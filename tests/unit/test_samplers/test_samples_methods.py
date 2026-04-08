"""Tests for Samples.init_mcmc (D3) and Samples.plot (D4).

Both methods operate on an existing Samples object.  Rather than running the
full rejection sampler (slow), tests build a minimal Samples instance directly
using the constructor.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as ndist
import pytest
from numpyro import infer
from unxt import Quantity

from harv.data import RadialVelocityData
from harv.kepler.orbits import astrometric_orbit_at_times, rv_at_times
from harv.likelihood.params import (
    GaiaAstrometryParameters,
    RVParameters,
)
from harv.priors.rejection import RejectionPrior
from harv.samplers.rejection import RejectionSampler
from harv.samplers.samples import Samples, _WarmStartMCMC

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
        "period": Quantity(jnp.linspace(90.0, 110.0, N), "day"),
        "eccentricity": Quantity(jnp.linspace(0.0, 0.3, N), ""),
        "phase_peri": Quantity(jnp.linspace(0.0, 1.0, N), ""),
        "arg_peri": Quantity(jnp.linspace(0.0, 3.14, N), "rad"),
    }
    linear = {
        "K": Quantity(jnp.linspace(3.0, 7.0, N), "km/s"),
        "v0": Quantity(jnp.linspace(-1.0, 1.0, N), "km/s"),
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
        "period": Quantity(jnp.linspace(250.0, 350.0, N), "day"),
        "eccentricity": Quantity(jnp.linspace(0.0, 0.3, N), ""),
        "phase_peri": Quantity(jnp.linspace(0.0, 1.0, N), ""),
        "arg_peri": Quantity(jnp.linspace(0.0, 3.14, N), "rad"),
        "cos_i": Quantity(jnp.linspace(0.2, 0.8, N), ""),
        "lon_asc_node": Quantity(jnp.linspace(0.0, 6.28, N), "rad"),
    }
    linear = {
        "ra0": Quantity(jnp.zeros(N), "mas"),
        "dec0": Quantity(jnp.zeros(N), "mas"),
        "pmra": Quantity(jnp.ones(N) * 10.0, "mas/yr"),
        "pmdec": Quantity(jnp.ones(N) * -5.0, "mas/yr"),
        "parallax": Quantity(jnp.ones(N) * 5.0, "mas"),
        "semi_major_axis": Quantity(jnp.linspace(1.0, 3.0, N), "mas"),
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
        "period": Quantity(jnp.linspace(250.0, 350.0, N), "day"),
        "eccentricity": Quantity(jnp.linspace(0.0, 0.2, N), ""),
        "phase_peri": Quantity(jnp.linspace(0.0, 1.0, N), ""),
        "arg_peri": Quantity(jnp.linspace(0.0, 3.14, N), "rad"),
        "cos_i": Quantity(jnp.linspace(0.2, 0.8, N), ""),
        "lon_asc_node": Quantity(jnp.linspace(0.0, 6.28, N), "rad"),
    }
    linear = {
        "ra0": Quantity(jnp.zeros(N), "mas"),
        "dec0": Quantity(jnp.zeros(N), "mas"),
        "pmra": Quantity(jnp.ones(N) * 10.0, "mas/yr"),
        "pmdec": Quantity(jnp.ones(N) * -5.0, "mas/yr"),
        "parallax": Quantity(jnp.ones(N) * 5.0, "mas"),
        "semi_major_axis": Quantity(jnp.linspace(1.0, 3.0, N), "mas"),
        "K": Quantity(jnp.linspace(3.0, 7.0, N), "km/s"),
        "v0": Quantity(jnp.zeros(N), "km/s"),
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
            "period": Quantity(jnp.array([]), "day"),
            "eccentricity": Quantity(jnp.array([]), ""),
            "phase_peri": Quantity(jnp.array([]), ""),
            "arg_peri": Quantity(jnp.array([]), "rad"),
        },
        linear={
            "K": Quantity(jnp.array([]), "km/s"),
            "v0": Quantity(jnp.array([]), "km/s"),
        },
        orbit_cls=RVParameters,
        full_cls=(RVParameters,),
        data_type="rv",
        metadata={},
    )


@pytest.fixture
def rv_sampler_and_data():
    """RejectionSampler + RadialVelocityData used for init_mcmc tests."""
    times = Quantity(jnp.linspace(0.0, 100.0, 20), "day")
    rv = Quantity(jnp.zeros(20), "km/s")
    rv_err = Quantity(jnp.ones(20) * 2.0, "km/s")
    data = RadialVelocityData(time=times, rv=rv, rv_err=rv_err)
    prior = RejectionPrior.default_rv(period_min=50.0, period_max=200.0)
    sampler = RejectionSampler(prior)
    return sampler, data


# ---------------------------------------------------------------------------
# Tests: rv_at_times and astrometric_orbit_at_times
# ---------------------------------------------------------------------------


class TestRvAtTimes:
    """Unit tests for rv_at_times."""

    def test_circular_orbit_shape(self):
        """Output has the same shape as the input times."""
        times = Quantity(np.linspace(0.0, 200.0, 50), "day")
        rv = rv_at_times(
            times,
            period=Quantity(200.0, "day"),
            eccentricity=0.0,
            t_peri=Quantity(0.0, "day"),
            arg_peri=Quantity(0.0, "rad"),
            K=Quantity(10.0, "km/s"),
            v0=Quantity(0.0, "km/s"),
        )
        assert rv.shape == (50,)

    def test_unit_preserved(self):
        """Output unit matches K and v0 unit."""
        times = Quantity(np.array([0.0, 50.0, 100.0]), "day")
        rv = rv_at_times(
            times,
            period=Quantity(200.0, "day"),
            eccentricity=0.3,
            t_peri=Quantity(50.0, "day"),
            arg_peri=Quantity(1.2, "rad"),
            K=Quantity(8.0, "km/s"),
            v0=Quantity(-5.0, "km/s"),
        )
        assert rv.unit.physical_type == "speed"

    def test_v0_offset(self):
        """Systemic velocity shifts every sample by v0."""
        times = Quantity(np.array([0.0, 50.0, 100.0]), "day")
        kwargs = dict(
            period=Quantity(200.0, "day"),
            eccentricity=0.0,
            t_peri=Quantity(0.0, "day"),
            arg_peri=Quantity(0.0, "rad"),
            K=Quantity(10.0, "km/s"),
        )
        rv0 = rv_at_times(times, v0=Quantity(0.0, "km/s"), **kwargs)
        rv5 = rv_at_times(times, v0=Quantity(5.0, "km/s"), **kwargs)
        np.testing.assert_allclose(np.asarray(rv5.value - rv0.value), 5.0, atol=1e-6)


class TestAstrometricOrbitAtTimes:
    """Unit tests for astrometric_orbit_at_times."""

    def test_output_shape_and_unit(self):
        """Both outputs have the input shape and semi_major_axis unit."""
        times = Quantity(np.linspace(0.0, 300.0, 40), "day")
        dra, ddec = astrometric_orbit_at_times(
            times,
            period=Quantity(300.0, "day"),
            eccentricity=0.3,
            t_peri=Quantity(0.0, "day"),
            arg_peri=Quantity(1.2, "rad"),
            cos_i=0.5,
            lon_asc_node=Quantity(0.8, "rad"),
            semi_major_axis=Quantity(3.0, "mas"),
        )
        assert dra.shape == (40,)
        assert ddec.shape == (40,)
        assert dra.unit.physical_type == "angle"

    def test_circular_face_on_orbit_is_circle(self):
        """Face-on circular orbit traces a circle: Δra²+Δdec² = const."""
        times = Quantity(np.linspace(0.0, 1.0, 500), "day")
        dra, ddec = astrometric_orbit_at_times(
            times,
            period=Quantity(1.0, "day"),
            eccentricity=0.0,
            t_peri=Quantity(0.0, "day"),
            arg_peri=Quantity(0.0, "rad"),
            cos_i=1.0,  # face-on
            lon_asc_node=Quantity(0.0, "rad"),
            semi_major_axis=Quantity(1.0, "mas"),
        )
        r2 = np.asarray(dra.value) ** 2 + np.asarray(ddec.value) ** 2
        np.testing.assert_allclose(r2, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Samples.init_mcmc (D3)
# ---------------------------------------------------------------------------


class TestInitMcmc:
    """Tests for Samples.init_mcmc."""

    def test_returns_warm_start_mcmc(self, rv_samples, rv_sampler_and_data):
        """init_mcmc returns a _WarmStartMCMC wrapping a numpyro MCMC."""
        sampler, data = rv_sampler_and_data
        result = sampler.init_mcmc(
            rv_samples, data, num_chains=2, num_warmup=10, num_samples=10
        )
        assert isinstance(result, _WarmStartMCMC)

    def test_init_params_keys_match_nonlinear(self, rv_samples, rv_sampler_and_data):
        """init_params dict contains all nonlinear parameter keys."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, data, num_chains=2, num_warmup=10, num_samples=10
        )
        params = mcmc._init_params
        for key in rv_samples.nonlinear:
            assert key in params, f"Missing key '{key}' in init_params"

    def test_init_params_shape_equals_num_chains(self, rv_samples, rv_sampler_and_data):
        """Each init_params array has shape (num_chains,)."""
        sampler, data = rv_sampler_and_data
        num_chains = 3
        mcmc = sampler.init_mcmc(
            rv_samples, data, num_chains=num_chains, num_warmup=10, num_samples=10
        )
        for key, arr in mcmc._init_params.items():
            assert arr.shape == (
                num_chains,
            ), f"Expected shape ({num_chains},) for '{key}', got {arr.shape}"

    def test_init_params_values_from_posterior(self, rv_samples, rv_sampler_and_data):
        """Starting positions are the first num_chains posterior samples."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, data, num_chains=3, num_warmup=10, num_samples=10
        )
        expected = np.asarray(rv_samples.nonlinear["period"].value)[:3]
        np.testing.assert_array_equal(np.asarray(mcmc._init_params["period"]), expected)

    def test_raises_for_empty_samples(self, empty_rv_samples, rv_sampler_and_data):
        """init_mcmc raises ValueError when there are no posterior samples."""
        sampler, data = rv_sampler_and_data
        with pytest.raises(ValueError, match="no posterior samples"):
            sampler.init_mcmc(
                empty_rv_samples, data, num_chains=2, num_warmup=10, num_samples=10
            )

    def test_raises_when_fewer_samples_than_chains(
        self, rv_samples, rv_sampler_and_data
    ):
        """init_mcmc raises ValueError when n_samples < num_chains."""
        sampler, data = rv_sampler_and_data
        with pytest.raises(ValueError, match="Fewer posterior samples"):
            sampler.init_mcmc(
                rv_samples,
                data,
                num_chains=N + 1,  # more chains than samples
                num_warmup=10,
                num_samples=10,
            )

    def test_warm_start_mcmc_delegates_attributes(
        self, rv_samples, rv_sampler_and_data
    ):
        """_WarmStartMCMC delegates unknown attributes to the underlying MCMC."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, data, num_chains=2, num_warmup=10, num_samples=10
        )
        # num_chains is an attribute of the underlying numpyro MCMC object.
        assert mcmc.num_chains == 2

    def test_default_kernel_is_nuts(self, rv_samples, rv_sampler_and_data):
        """When kernel is omitted the default is NUTS."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples, data, num_warmup=10, num_samples=10, num_chains=2
        )
        assert isinstance(mcmc._mcmc.sampler, infer.NUTS)

    def test_run_produces_posterior_samples(self, rv_samples, rv_sampler_and_data):
        """mcmc.run() completes and get_samples() returns the expected keys."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        mcmc.run(jr.PRNGKey(0))
        posterior = mcmc.get_samples()
        # The auto-generated model samples all keys from prior.nonlinear_priors.
        for key in sampler.prior.nonlinear_priors:
            assert key in posterior, f"Missing site '{key}' in posterior"
        assert posterior["period"].shape == (10,)  # 2 chains x 5 samples


class TestInitMcmcFull:
    """Tests for RejectionSampler.init_mcmc with marginalized=False."""

    def test_returns_warm_start_mcmc(self, rv_samples, rv_sampler_and_data):
        """init_mcmc(marginalized=False) returns a _WarmStartMCMC."""
        sampler, data = rv_sampler_and_data
        result = sampler.init_mcmc(
            rv_samples,
            data,
            marginalized=False,
            num_chains=2,
            num_warmup=10,
            num_samples=10,
        )
        assert isinstance(result, _WarmStartMCMC)

    def test_init_params_has_linear_site(self, rv_samples, rv_sampler_and_data):
        """Full model init_params includes '_linear', not individual named sites."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            marginalized=False,
            num_chains=2,
            num_warmup=10,
            num_samples=10,
        )
        assert "_linear" in mcmc._init_params
        # Named linear params are deterministic sites, not init_params entries.
        assert "K" not in mcmc._init_params
        assert "v0" not in mcmc._init_params

    def test_linear_init_shape(self, rv_samples, rv_sampler_and_data):
        """'_linear' init has shape (num_chains, n_linear)."""
        sampler, data = rv_sampler_and_data
        num_chains = 2
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            marginalized=False,
            num_chains=num_chains,
            num_warmup=10,
            num_samples=10,
        )
        n_linear = len(rv_samples.linear)
        assert mcmc._init_params["_linear"].shape == (num_chains, n_linear)

    def test_run_produces_named_linear_params(self, rv_samples, rv_sampler_and_data):
        """mcmc.run() produces named deterministic sites K, v0, etc."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            marginalized=False,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        mcmc.run(jr.PRNGKey(0))
        posterior = mcmc.get_samples()
        # Nonlinear sites must be present.
        for key in sampler.prior.nonlinear_priors:
            assert key in posterior, f"Missing nonlinear site '{key}'"
        # Named linear deterministic sites must be present.
        for name in rv_samples.linear:
            assert name in posterior, f"Missing linear site '{name}'"

    def test_marginalized_true_has_no_linear_site(
        self, rv_samples, rv_sampler_and_data
    ):
        """Marginalized model (default) does not put '_linear' in init_params."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            num_chains=2,
            num_warmup=10,
            num_samples=10,
        )
        assert "_linear" not in mcmc._init_params


# ---------------------------------------------------------------------------
# Tests: RejectionSampler.init_mcmc with extra_model
# ---------------------------------------------------------------------------


class TestInitMcmcExtraModel:
    """Tests for init_mcmc with a physical reparameterization via extra_model."""

    def _make_extra_model(self):
        """Return a minimal extra_model that fixes K to a constant."""

        def extra_model(pars):
            # Sample a dummy physical parameter, compute K from it.
            K_scale = numpyro.sample("K_scale", ndist.HalfNormal(10.0))
            return {"K": K_scale}

        return extra_model

    def test_raises_without_extra_init_params(self, rv_samples, rv_sampler_and_data):
        """extra_model without extra_init_params raises ValueError."""
        sampler, data = rv_sampler_and_data
        with pytest.raises(ValueError, match="extra_init_params is required"):
            sampler.init_mcmc(
                rv_samples,
                data,
                extra_model=self._make_extra_model(),
                num_chains=2,
                num_warmup=5,
                num_samples=5,
            )

    def test_returns_warm_start_mcmc(self, rv_samples, rv_sampler_and_data):
        """init_mcmc with extra_model returns a _WarmStartMCMC."""
        sampler, data = rv_sampler_and_data
        result = sampler.init_mcmc(
            rv_samples,
            data,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            num_chains=2,
            num_warmup=5,
            num_samples=5,
        )
        assert isinstance(result, _WarmStartMCMC)

    def test_init_params_includes_extra(self, rv_samples, rv_sampler_and_data):
        """init_params contains both nonlinear and extra_init_params entries."""
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
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
        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            extra_model=self._make_extra_model(),
            extra_init_params={"K_scale": jnp.full(2, 5.0)},
            marginalized=True,
            num_chains=2,
            num_warmup=5,
            num_samples=5,
            chain_method="sequential",
        )
        mcmc.run(jr.PRNGKey(42))
        posterior = mcmc.get_samples()

        # Nonlinear sites must be present.
        for key in sampler.prior.nonlinear_priors:
            assert key in posterior
        # The extra-model site is present.
        assert "K_scale" in posterior
        # K is exposed as a deterministic site.
        assert "K" in posterior
        # v0 is analytically marginalized — not a sample site.
        assert "v0" not in posterior

    def test_extra_model_raises_for_unknown_param(
        self, rv_samples, rv_sampler_and_data
    ):
        """extra_model returning an unknown linear param name raises ValueError."""

        def bad_extra_model(pars):
            x = numpyro.sample("x", ndist.Normal(0.0, 1.0))
            return {"not_a_real_param": x}

        sampler, data = rv_sampler_and_data
        mcmc = sampler.init_mcmc(
            rv_samples,
            data,
            extra_model=bad_extra_model,
            extra_init_params={"x": jnp.zeros(2)},
            num_chains=2,
            num_warmup=1,
            num_samples=1,
            chain_method="sequential",
        )
        with pytest.raises(ValueError, match="unknown linear parameter name"):
            mcmc.run(jr.PRNGKey(0))


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
        times = Quantity(jnp.linspace(0.0, 100.0, 20), "day")
        rv = Quantity(jnp.zeros(20), "km/s")
        rv_err = Quantity(jnp.ones(20) * 2.0, "km/s")
        rv_data = RadialVelocityData(time=times, rv=rv, rv_err=rv_err)
        fig = rv_samples.plot(data=rv_data)
        assert fig is not None
        plt.close("all")

    def test_n_samples_limits_model_curves(self, rv_samples):
        """n_samples controls how many posterior curves are drawn."""
        fig = rv_samples.plot(n_samples=3)
        ax = fig.axes[0]
        model_lines = [
            line
            for line in ax.lines
            if line.get_alpha() is not None and line.get_alpha() < 0.5  # type: ignore[operator]
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
                "period": Quantity(jnp.ones(5) * 100.0, "day"),
                "eccentricity": Quantity(jnp.zeros(5), ""),
                "phase_peri": Quantity(jnp.zeros(5), ""),
                "arg_peri": Quantity(jnp.zeros(5), "rad"),
            },
            linear={
                "K": Quantity(jnp.zeros(5), "km/s"),
                "v0": Quantity(jnp.zeros(5), "km/s"),
            },
            orbit_cls=RVParameters,
            full_cls=(RVParameters,),
            data_type="unknown",
            metadata={},
        )
        with pytest.raises(ValueError, match="Unknown data_type"):
            bad_samples.plot()
