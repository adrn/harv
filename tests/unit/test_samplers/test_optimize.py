"""Tests for NumpyroSampler.optimize().

Refines posterior samples to the local MAP using ``numpyro.optim.Minimize``
(BFGS) with an :class:`AutoDelta` guide. Tests verify that refined samples
have higher posterior density than the warm-start input and that the joint
model path is supported.
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest
from unxt import Q

from harv.data import GaiaAstrometryData, RVData
from harv.distributions import QD
from harv.kepler.orbits import (
    astrometric_orbit_at_times,
    campbell_from_thiele_innes,
    mean_anomaly,
    rv_at_times,
    thiele_innes_from_campbell,
    true_anomaly_from_mean,
)
from harv.kepler.orbits import (
    astrometric_orbit_at_times as _astrom,
)
from harv.models._helpers import _evaluate_nonlinear_log_prior
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.joint import JointModel
from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry
from harv.models.rv import RVModel
from harv.samplers.numpyro import NumpyroSampler
from harv.samplers.rejection_prior import RejectionPrior
from harv.samplers.samples import Samples

# ---------------------------------------------------------------------------
# Synthetic RV fixture: a single off-mode "warm-start" sample
# ---------------------------------------------------------------------------


TRUE_PERIOD = Q(73.2, "day")
TRUE_ECC = 0.18
TRUE_PHASE_PERI = 0.42
TRUE_ARG_PERI = Q(1.1, "rad")
TRUE_K = Q(8.0, "km/s")
TRUE_V_SYS = Q(-0.3, "km/s")


@pytest.fixture
def rv_data_and_truth():
    """Synthetic RV data generated from a known orbit."""
    times = Q(jnp.linspace(0.0, 200.0, 40), "day")
    t_peri = TRUE_PHASE_PERI * TRUE_PERIOD
    rv_true = rv_at_times(
        times, TRUE_PERIOD, TRUE_ECC, t_peri, TRUE_ARG_PERI, TRUE_K, TRUE_V_SYS
    )
    rng = np.random.default_rng(0)
    noise = Q(jnp.asarray(rng.normal(0.0, 0.5, size=40)), "km/s")
    rv = rv_true + noise
    rv_err = Q(jnp.ones(40) * 0.5, "km/s")
    return RVData(time=times, rv=rv, rv_err=rv_err)


@pytest.fixture
def rv_sampler(rv_data_and_truth):
    """NumpyroSampler configured for the synthetic RV data."""
    prior = RejectionPrior.default_rv(
        period_min=Q(30.0, "day"),
        period_max=Q(150.0, "day"),
        sigma_K0=Q(20.0, "km/s"),
        sigma_v0=Q(20.0, "km/s"),
    )
    return NumpyroSampler(prior, RVModel())


@pytest.fixture
def off_mode_samples() -> Samples:
    """Two warm-start samples near (but not at) the true MAP."""
    nonlinear = {
        "period": Q(jnp.array([74.5, 72.0]), "day"),
        "eccentricity": Q(jnp.array([0.25, 0.10]), ""),
        "phase_peri": Q(jnp.array([0.50, 0.35]), ""),
        "arg_peri": Q(jnp.array([1.4, 0.8]), "rad"),
    }
    linear = {
        "rv_semiamp": Q(jnp.array([7.0, 9.0]), "km/s"),
        "v_sys": Q(jnp.array([0.5, -0.8]), "km/s"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="RVModel",
        metadata={"t_ref": 0.0},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNumpyroSamplerOptimize:
    """Tests for NumpyroSampler.optimize()."""

    def test_refines_to_higher_logposterior(
        self, rv_sampler, rv_data_and_truth, off_mode_samples
    ):
        """Refined samples have higher (or equal) ln_posterior than warm starts."""

        refined = rv_sampler.optimize(off_mode_samples, rv_data_and_truth, seed=0)

        model = rv_sampler.model
        prior_nl = rv_sampler.prior.nonlinear_priors
        eff_lp = rv_sampler.prior.linear_prior

        for i in range(off_mode_samples.n_samples):
            warm_nl = {
                "period": off_mode_samples.nonlinear["period"][i],
                "eccentricity": off_mode_samples.nonlinear["eccentricity"][i].value,
                "phase_peri": off_mode_samples.nonlinear["phase_peri"][i].value,
                "arg_peri": off_mode_samples.nonlinear["arg_peri"][i],
            }
            warm_lik = float(
                model.log_prob(warm_nl, rv_data_and_truth, linear_prior=eff_lp)
            )
            warm_prior = float(
                _evaluate_nonlinear_log_prior(
                    prior_nl,
                    {
                        "period": warm_nl["period"].value,
                        "eccentricity": warm_nl["eccentricity"],
                        "phase_peri": warm_nl["phase_peri"],
                        "arg_peri": warm_nl["arg_peri"].value,
                    },
                )
            )
            warm_lpost = warm_lik + warm_prior
            refined_lpost = float(refined.ln_posterior[i])
            assert refined_lpost >= warm_lpost - 1e-3, (
                f"sample {i}: refined ln_post={refined_lpost} did not improve "
                f"over warm-start ln_post={warm_lpost}"
            )

    def test_refines_toward_truth(
        self, rv_sampler, rv_data_and_truth, off_mode_samples
    ):
        """Period of refined samples is closer to the true period."""
        refined = rv_sampler.optimize(off_mode_samples, rv_data_and_truth, seed=0)
        true_p = float(TRUE_PERIOD.value)
        warm_dp = np.abs(
            np.asarray(off_mode_samples.nonlinear["period"].value) - true_p
        )
        refined_dp = np.abs(np.asarray(refined.nonlinear["period"].value) - true_p)
        # At least one of the warm samples should refine closer to truth.
        assert (refined_dp <= warm_dp + 1e-3).any()

    def test_preserves_sample_count_and_keys(
        self, rv_sampler, rv_data_and_truth, off_mode_samples
    ):
        """Refined Samples has same shape and key set as input."""
        refined = rv_sampler.optimize(off_mode_samples, rv_data_and_truth, seed=0)
        assert refined.n_samples == off_mode_samples.n_samples
        assert set(refined.nonlinear) == set(off_mode_samples.nonlinear)
        assert set(refined.linear) == set(off_mode_samples.linear)
        assert refined.ln_likelihood is not None
        assert refined.ln_prior is not None
        assert refined.ln_likelihood.shape == (off_mode_samples.n_samples,)

    def test_linear_params_are_deterministic(
        self, rv_sampler, rv_data_and_truth, off_mode_samples
    ):
        """Linear params at the MAP are the conditional mean (no RNG wobble).

        Different ``seed`` arguments should produce identical linear values
        because ``optimize`` returns the conditional posterior mean for the
        marginalized linear parameters.
        """
        refined_a = rv_sampler.optimize(off_mode_samples, rv_data_and_truth, seed=1)
        refined_b = rv_sampler.optimize(off_mode_samples, rv_data_and_truth, seed=999)
        for name in refined_a.linear:
            assert jnp.allclose(
                refined_a.linear[name].value,
                refined_b.linear[name].value,
                rtol=1e-5,
                atol=1e-6,
            ), f"linear param {name} differs between seeds"

    def test_empty_samples_raises(self, rv_sampler, rv_data_and_truth):
        """Calling optimize() on an empty Samples raises ValueError."""
        empty = Samples(
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
        with pytest.raises(ValueError, match="no samples"):
            rv_sampler.optimize(empty, rv_data_and_truth)


# ---------------------------------------------------------------------------
# JointModel path
# ---------------------------------------------------------------------------


@pytest.fixture
def joint_sampler_and_data():
    """Combined RV + astrometry JointModel sampler + data."""
    n_ast = 15
    times_ast = Q(jnp.linspace(0.0, 1000.0, n_ast), "day")
    astro_data = GaiaAstrometryData(
        time=times_ast,
        al_position=Q(jnp.zeros(n_ast), "mas"),
        al_position_err=Q(jnp.ones(n_ast) * 0.1, "mas"),
        scan_angle=Q(jnp.linspace(0.0, 3.14, n_ast), "rad"),
        parallax_factor=jnp.full(n_ast, 0.5),
    )
    n_rv = 8
    times_rv = Q(jnp.linspace(0.0, 800.0, n_rv), "day")
    rv_data = RVData(
        time=times_rv,
        rv=Q(jnp.zeros(n_rv), "km/s"),
        rv_err=Q(jnp.ones(n_rv) * 1.0, "km/s"),
    )

    linear = {
        "astro.ra0": QD(ndist.Normal(0.0, 100.0), "mas"),
        "astro.dec0": QD(ndist.Normal(0.0, 100.0), "mas"),
        "astro.pmra": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
        "astro.pmdec": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
        "astro.parallax": QD(ndist.Normal(3.0, 1.0), "mas"),
        "astro.semi_major_axis": QD(ndist.Normal(0.0, 50.0), "mas"),
        "rv.rv_semiamp": QD(ndist.Normal(0.0, 30.0), "km/s"),
        "rv.v_sys": QD(ndist.Normal(0.0, 30.0), "km/s"),
    }
    nonlinear = {
        "period": QD(ndist.Normal(300.0, 50.0), "day"),
        "eccentricity": ndist.TruncatedNormal(0.3, 0.2, low=0.0, high=1.0),
        "phase_peri": ndist.Uniform(0.0, 1.0),
        "cos_i": ndist.Uniform(-1.0, 1.0),
        "arg_peri": QD(ndist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
        "lon_asc_node": QD(ndist.Uniform(0.0, 2.0 * jnp.pi), "rad"),
    }
    prior = RejectionPrior(nonlinear_priors=nonlinear, linear_prior=linear)
    joint = JointModel.for_rv_and_gaia(
        components={"astro": GaiaAstrometryModel(), "rv": RVModel()}
    )
    return NumpyroSampler(prior, joint), {"astro": astro_data, "rv": rv_data}


@pytest.fixture
def joint_off_mode_samples() -> Samples:
    """One warm-start sample for the JointModel."""
    nonlinear = {
        "period": Q(jnp.array([310.0]), "day"),
        "eccentricity": Q(jnp.array([0.20]), ""),
        "phase_peri": Q(jnp.array([0.50]), ""),
        "arg_peri": Q(jnp.array([1.4]), "rad"),
        "cos_i": Q(jnp.array([0.30]), ""),
        "lon_asc_node": Q(jnp.array([2.0]), "rad"),
    }
    linear = {
        "ra0": Q(jnp.array([0.5]), "mas"),
        "dec0": Q(jnp.array([0.5]), "mas"),
        "pmra": Q(jnp.array([1.0]), "mas/yr"),
        "pmdec": Q(jnp.array([1.0]), "mas/yr"),
        "parallax": Q(jnp.array([3.5]), "mas"),
        "semi_major_axis": Q(jnp.array([2.0]), "mas"),
        "rv_semiamp": Q(jnp.array([4.0]), "km/s"),
        "v_sys": Q(jnp.array([0.0]), "km/s"),
    }
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        data_type="JointModel",
        metadata={"t_ref": 0.0},
    )


class TestNumpyroSamplerOptimizeJoint:
    """Tests for NumpyroSampler.optimize() with a JointModel."""

    def test_optimize_jointmodel(self, joint_sampler_and_data, joint_off_mode_samples):
        """optimize() works for JointModel and populates logprobs."""
        sampler, data = joint_sampler_and_data
        refined = sampler.optimize(joint_off_mode_samples, data, seed=0)
        assert refined.n_samples == 1
        assert refined.ln_likelihood is not None
        assert refined.ln_prior is not None
        assert jnp.isfinite(refined.ln_posterior).all()


# ---------------------------------------------------------------------------
# Thiele-Innes parameterization path -- regression for Gaia BH3-style bug
# ---------------------------------------------------------------------------


class TestNumpyroSamplerOptimizeThieleInnes:
    """Regression tests for the TI parameterization + sub-orbit coverage case.

    The Gaia BH3 case study uses ThieleInnesGaiaAstrometry with sub-orbit data
    coverage. The conditional posterior over (ti_A, ti_B, ti_F, ti_G) is highly
    correlated; a random draw can be far from the conditional mean along the
    degenerate direction, and the nonlinear TI -> Campbell conversion amplifies
    that noise. `optimize` must return the conditional posterior mean (= MAP)
    for linear parameters, not a random draw.
    """

    @pytest.fixture
    def ti_data_and_truth(self):
        """Synthetic Gaia astrometry data from a known orbit, TI design."""
        n = 30
        # Baseline ~600 days, period ~400 days -> ~1.5 orbits (well-conditioned
        # enough that BFGS converges reliably; the test still exercises the
        # conditional-mean code path).
        times = Q(jnp.linspace(0.0, 600.0, n), "day")
        scan_angle = Q(jnp.linspace(0.0, 6.0 * jnp.pi, n) % (2 * jnp.pi), "rad")
        parallax_factor = jnp.sin(jnp.linspace(0.0, 4.0 * jnp.pi, n))

        true_period = Q(400.0, "day")
        true_ecc = 0.3
        true_phase = 0.4
        true_arg_peri = Q(0.8, "rad")
        true_cos_i = Q(0.6, "")
        true_Omega = Q(1.5, "rad")
        true_a = Q(2.0, "mas")
        true_pmra = Q(8.0, "mas/yr")
        true_pmdec = Q(-4.0, "mas/yr")
        true_parallax = Q(3.0, "mas")

        t_peri = true_phase * true_period
        dra, ddec = astrometric_orbit_at_times(
            times,
            true_period,
            true_ecc,
            t_peri,
            true_arg_peri,
            true_cos_i,
            true_Omega,
            true_a,
        )
        sin_psi = jnp.sin(scan_angle.value)
        cos_psi = jnp.cos(scan_angle.value)
        dt_yr = (times.value - 0.0) / 365.25
        al_truth = (
            dra.value * sin_psi
            + ddec.value * cos_psi
            + true_pmra.value * dt_yr * sin_psi
            + true_pmdec.value * dt_yr * cos_psi
            + true_parallax.value * parallax_factor
        )
        rng = np.random.default_rng(123)
        noise = rng.normal(0.0, 0.05, size=n)
        al = Q(jnp.asarray(al_truth + noise), "mas")
        al_err = Q(jnp.ones(n) * 0.05, "mas")
        data = GaiaAstrometryData(
            time=times,
            al_position=al,
            al_position_err=al_err,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
        )
        # Pre-compute true TI constants for the warm-start fixture.
        ti_A, ti_B, ti_F, ti_G = thiele_innes_from_campbell(
            true_a, true_arg_peri, true_Omega, true_cos_i
        )
        return data, {
            "period": true_period,
            "eccentricity": true_ecc,
            "phase_peri": true_phase,
            "arg_peri": true_arg_peri,
            "cos_i": true_cos_i,
            "lon_asc_node": true_Omega,
            "semi_major_axis": true_a,
            "pmra": true_pmra,
            "pmdec": true_pmdec,
            "parallax": true_parallax,
            "ti_A": ti_A,
            "ti_B": ti_B,
            "ti_F": ti_F,
            "ti_G": ti_G,
        }

    @pytest.fixture
    def ti_sampler(self, ti_data_and_truth):
        """NumpyroSampler with TI parameterization."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(ndist.Normal(400.0, 50.0), "day"),
                "eccentricity": ndist.TruncatedNormal(0.3, 0.2, low=0.0, high=1.0),
                "phase_peri": ndist.Uniform(0.0, 1.0),
            },
            linear_prior={
                "ra0": QD(ndist.Normal(0.0, 100.0), "mas"),
                "dec0": QD(ndist.Normal(0.0, 100.0), "mas"),
                "pmra": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
                "pmdec": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
                "parallax": QD(ndist.Normal(3.0, 1.0), "mas"),
                "ti_A": QD(ndist.Normal(0.0, 10.0), "mas"),
                "ti_B": QD(ndist.Normal(0.0, 10.0), "mas"),
                "ti_F": QD(ndist.Normal(0.0, 10.0), "mas"),
                "ti_G": QD(ndist.Normal(0.0, 10.0), "mas"),
            },
        )
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(apply_jacobian_correction=False)
        )
        return NumpyroSampler(prior, model)

    @pytest.fixture
    def ti_off_mode_samples(self, ti_data_and_truth) -> Samples:
        """Warm-start sample near truth in TI coordinates."""
        _, truth = ti_data_and_truth
        nonlinear = {
            "period": Q(jnp.array([float(truth["period"].value) * 1.02]), "day"),
            "eccentricity": Q(jnp.array([float(truth["eccentricity"]) + 0.05]), ""),
            "phase_peri": Q(jnp.array([float(truth["phase_peri"]) + 0.03]), ""),
        }
        linear = {
            "ra0": Q(jnp.array([0.0]), "mas"),
            "dec0": Q(jnp.array([0.0]), "mas"),
            "pmra": Q(jnp.array([float(truth["pmra"].value) + 0.5]), "mas/yr"),
            "pmdec": Q(jnp.array([float(truth["pmdec"].value) - 0.5]), "mas/yr"),
            "parallax": Q(jnp.array([float(truth["parallax"].value) + 0.2]), "mas"),
            "ti_A": Q(jnp.array([float(truth["ti_A"].value) * 1.1]), "mas"),
            "ti_B": Q(jnp.array([float(truth["ti_B"].value) * 1.1]), "mas"),
            "ti_F": Q(jnp.array([float(truth["ti_F"].value) * 1.1]), "mas"),
            "ti_G": Q(jnp.array([float(truth["ti_G"].value) * 1.1]), "mas"),
        }
        return Samples(
            nonlinear=nonlinear,
            linear=linear,
            data_type="GaiaAstrometryModel",
            metadata={"t_ref": 0.0},
        )

    def test_optimize_ti_returns_valid_campbell(
        self, ti_sampler, ti_data_and_truth, ti_off_mode_samples
    ):
        """After optimize, TI -> Campbell conversion gives valid elements.

        With the ``use_mean=True`` policy the converted Campbell elements are
        deterministic and physically valid: ``semi_major_axis > 0``, ``cos_i``
        in ``[-1, 1]`` (signed so prograde/retrograde orbits round-trip), and
        angles in ``[0, 2pi]``.
        """
        data, _ = ti_data_and_truth
        refined = ti_sampler.optimize(ti_off_mode_samples, data, seed=0)
        campbell = refined.thiele_innes_to_campbell()
        a = campbell.linear["semi_major_axis"][0].value
        cos_i = campbell.nonlinear["cos_i"][0].value
        arg_peri = campbell.nonlinear["arg_peri"][0].value
        omega = campbell.nonlinear["lon_asc_node"][0].value
        assert float(a) > 0.0
        assert -1.0 <= float(cos_i) <= 1.0
        assert 0.0 <= float(arg_peri) < 2 * float(jnp.pi) + 1e-6
        assert 0.0 <= float(omega) < 2 * float(jnp.pi) + 1e-6

    def test_optimize_ti_linear_params_deterministic(
        self, ti_sampler, ti_data_and_truth, ti_off_mode_samples
    ):
        """TI constants returned by optimize are RNG-free (conditional mean)."""
        data, _ = ti_data_and_truth
        refined_a = ti_sampler.optimize(ti_off_mode_samples, data, seed=1)
        refined_b = ti_sampler.optimize(ti_off_mode_samples, data, seed=999)
        for name in ("ti_A", "ti_B", "ti_F", "ti_G", "ra0", "dec0", "parallax"):
            assert jnp.allclose(
                refined_a.linear[name].value,
                refined_b.linear[name].value,
                rtol=1e-5,
                atol=1e-6,
            ), f"{name} differs across seeds"


class TestNumpyroSamplerOptimizeThieleInnesSubOrbit:
    """Regression for the Gaia BH3 case: TI fit with sub-orbit baseline.

    Generates synthetic Gaia AL data with a known orbit, observation baseline
    much shorter than the period (~30% coverage), and verifies that after
    `optimize`:

    1. The design-matrix prediction at the refined TI values matches the data.
    2. The Campbell-path prediction (`thiele_innes_to_campbell` ->
       `astrometric_orbit_at_times`) ALSO matches the data.

    If (1) passes but (2) fails, the bug is in the TI -> Campbell conversion
    or `astrometric_orbit_at_times`. If neither passes, the bug is in the
    marginalization / prior alignment.
    """

    @pytest.fixture
    def bh3_like_data_and_truth(self):
        """Synthetic Gaia AL data with sub-orbit baseline (~30% of one period)."""

        n = 60
        true_period = Q(4000.0, "day")
        baseline = Q(1200.0, "day")  # ~30% of one orbit
        times = Q(jnp.linspace(0.0, float(baseline.value), n), "day")
        # Use explicit t_ref=0 so phase_peri matches the data-generation convention.
        # (Default t_ref is mean(times), which would shift the orbital phase.)
        t_ref = Q(0.0, "day")
        rng = np.random.default_rng(42)
        scan_angle = Q(jnp.asarray(rng.uniform(0.0, 2 * jnp.pi, n)), "rad")
        parallax_factor = jnp.asarray(rng.uniform(-1.0, 1.0, n))

        true_ecc = 0.7
        true_phase = 0.1
        true_arg_peri = Q(1.5, "rad")
        true_cos_i = Q(0.93, "")  # ~22 deg, similar to BH3
        true_Omega = Q(2.2, "rad")
        true_a = Q(3.0, "mas")
        true_pmra = Q(-5.0, "mas/yr")
        true_pmdec = Q(3.0, "mas/yr")
        true_parallax = Q(1.67, "mas")
        true_ra0 = Q(0.0, "mas")
        true_dec0 = Q(0.0, "mas")

        # Note: the model internally interprets phase_peri as relative to
        # data.t_ref, so the absolute t_peri used here must be t_ref + phase*period.
        t_peri = t_ref + true_phase * true_period
        dra, ddec = _astrom(
            times,
            true_period,
            true_ecc,
            t_peri,
            true_arg_peri,
            true_cos_i,
            true_Omega,
            true_a,
        )
        sin_psi = jnp.sin(scan_angle.value)
        cos_psi = jnp.cos(scan_angle.value)
        dt_yr = (times.value - float(t_ref.value)) / 365.25
        al_truth = (
            true_ra0.value * sin_psi
            + true_dec0.value * cos_psi
            + dra.value * sin_psi
            + ddec.value * cos_psi
            + true_pmra.value * dt_yr * sin_psi
            + true_pmdec.value * dt_yr * cos_psi
            + true_parallax.value * parallax_factor
        )
        sigma_al = 0.1  # mas
        noise = rng.normal(0.0, sigma_al, size=n)
        al = Q(jnp.asarray(al_truth + noise), "mas")
        al_err = Q(jnp.ones(n) * sigma_al, "mas")
        data = GaiaAstrometryData(
            time=times,
            al_position=al,
            al_position_err=al_err,
            scan_angle=scan_angle,
            parallax_factor=parallax_factor,
            t_ref=t_ref,
        )

        ti_A, ti_B, ti_F, ti_G = thiele_innes_from_campbell(
            true_a, true_arg_peri, true_Omega, true_cos_i
        )
        return data, {
            "period": true_period,
            "eccentricity": true_ecc,
            "phase_peri": true_phase,
            "arg_peri": true_arg_peri,
            "cos_i": true_cos_i,
            "lon_asc_node": true_Omega,
            "semi_major_axis": true_a,
            "ra0": true_ra0,
            "dec0": true_dec0,
            "pmra": true_pmra,
            "pmdec": true_pmdec,
            "parallax": true_parallax,
            "ti_A": ti_A,
            "ti_B": ti_B,
            "ti_F": ti_F,
            "ti_G": ti_G,
            "sigma_al": sigma_al,
        }

    @pytest.fixture
    def bh3_ti_sampler(self, bh3_like_data_and_truth):
        """TI sampler with BH3-like setup."""
        prior = RejectionPrior(
            nonlinear_priors={
                "period": QD(ndist.LogUniform(1000.0, 10_000.0), "day"),
                "eccentricity": ndist.TruncatedNormal(0.5, 0.4, low=0.0, high=1.0),
                "phase_peri": ndist.Uniform(0.0, 1.0),
            },
            linear_prior={
                "ra0": QD(ndist.Normal(0.0, 100.0), "mas"),
                "dec0": QD(ndist.Normal(0.0, 100.0), "mas"),
                "pmra": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
                "pmdec": QD(ndist.Normal(0.0, 50.0), "mas/yr"),
                "parallax": QD(ndist.Normal(1.67, 0.07), "mas"),
                "ti_A": QD(ndist.Normal(0.0, 100.0), "mas"),
                "ti_B": QD(ndist.Normal(0.0, 100.0), "mas"),
                "ti_F": QD(ndist.Normal(0.0, 100.0), "mas"),
                "ti_G": QD(ndist.Normal(0.0, 100.0), "mas"),
            },
        )
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(apply_jacobian_correction=False)
        )
        return NumpyroSampler(prior, model)

    @pytest.fixture
    def bh3_warm_start(self, bh3_like_data_and_truth) -> Samples:
        """Warm-start near truth in TI coordinates (off by a few %)."""
        _, t = bh3_like_data_and_truth
        return Samples(
            nonlinear={
                "period": Q(jnp.array([float(t["period"].value) * 1.02]), "day"),
                "eccentricity": Q(jnp.array([float(t["eccentricity"]) - 0.05]), ""),
                "phase_peri": Q(jnp.array([float(t["phase_peri"]) + 0.02]), ""),
            },
            linear={
                "ra0": Q(jnp.array([0.0]), "mas"),
                "dec0": Q(jnp.array([0.0]), "mas"),
                "pmra": Q(jnp.array([float(t["pmra"].value) + 0.3]), "mas/yr"),
                "pmdec": Q(jnp.array([float(t["pmdec"].value) - 0.3]), "mas/yr"),
                "parallax": Q(jnp.array([float(t["parallax"].value)]), "mas"),
                "ti_A": Q(jnp.array([float(t["ti_A"].value) * 1.05]), "mas"),
                "ti_B": Q(jnp.array([float(t["ti_B"].value) * 1.05]), "mas"),
                "ti_F": Q(jnp.array([float(t["ti_F"].value) * 1.05]), "mas"),
                "ti_G": Q(jnp.array([float(t["ti_G"].value) * 1.05]), "mas"),
            },
            data_type="GaiaAstrometryModel",
            metadata={"t_ref": 0.0},
        )

    def test_design_matrix_prediction_matches_data(
        self, bh3_ti_sampler, bh3_like_data_and_truth, bh3_warm_start
    ):
        """The orbit predicted by design-matrix @ refined TI values fits data."""

        data, truth = bh3_like_data_and_truth
        refined = bh3_ti_sampler.optimize(bh3_warm_start, data, seed=0)

        # Build the same TI design matrix the likelihood uses
        period = float(refined.nonlinear["period"][0].value)
        ecc = float(refined.nonlinear["eccentricity"][0].value)
        phase = float(refined.nonlinear["phase_peri"][0].value)
        t_peri = phase * period
        dt = data.time.value - 0.0 - t_peri  # t_ref=0 by default
        M = mean_anomaly(Q(dt, "day"), Q(period, "day"))
        sin_f, cos_f = true_anomaly_from_mean(M, ecc)
        sin_f = jnp.asarray(
            jnp.asarray(sin_f) if not hasattr(sin_f, "value") else sin_f.value
        )
        cos_f = jnp.asarray(
            jnp.asarray(cos_f) if not hasattr(cos_f, "value") else cos_f.value
        )

        psi = data.scan_angle.value
        sin_psi = jnp.sin(psi)
        cos_psi = jnp.cos(psi)
        dt_yr = data.time.value / 365.25
        pf = data.parallax_factor

        param = ThieleInnesGaiaAstrometry(apply_jacobian_correction=False)
        X = param.design_matrix(
            sin_f, cos_f, dt_yr, sin_psi, cos_psi, pf, {"eccentricity": ecc}
        )
        lin = jnp.array(
            [
                float(refined.linear["ra0"][0].value),
                float(refined.linear["dec0"][0].value),
                float(refined.linear["pmra"][0].value),
                float(refined.linear["pmdec"][0].value),
                float(refined.linear["parallax"][0].value),
                float(refined.linear["ti_A"][0].value),
                float(refined.linear["ti_B"][0].value),
                float(refined.linear["ti_F"][0].value),
                float(refined.linear["ti_G"][0].value),
            ]
        )
        al_pred_dm = X @ lin
        resid_dm = jnp.asarray(data.al_position.value) - al_pred_dm
        max_resid_dm = float(jnp.max(jnp.abs(resid_dm)))
        sigma = truth["sigma_al"]
        assert max_resid_dm < 10.0 * sigma, (
            f"Design-matrix path: max |residual| = {max_resid_dm:.3f} mas "
            f">> {10 * sigma:.3f} mas (10*sigma). The TI fit itself is wrong."
        )

    def test_retrograde_orbit_roundtrip(self, bh3_like_data_and_truth):
        """Regression: TI -> Campbell -> TI must be exact for cos_i < 0.

        The Gaia BH3 case showed a 'diagonal flip' in the plotted sky orbit
        when the fitted TI corresponded to a retrograde orbit (cos_i < 0).
        Root cause was ``cos_i = |v/a_0^2|`` in ``campbell_from_thiele_innes``
        silently flipping the sign. Verify the round-trip is exact for an
        explicitly negative-cos_i orbit.
        """

        _, truth = bh3_like_data_and_truth
        a = truth["semi_major_axis"]
        arg_peri = truth["arg_peri"]
        omega = truth["lon_asc_node"]
        # Flip cos_i to the retrograde branch (mirror across i=90 deg).
        cos_i_retro = Q(-float(truth["cos_i"].value), "")

        ti_A, ti_B, ti_F, ti_G = thiele_innes_from_campbell(
            a, arg_peri, omega, cos_i_retro
        )
        camp = campbell_from_thiele_innes(ti_A, ti_B, ti_F, ti_G)
        assert float(camp["cos_i"].value) < 0.0, (
            "campbell_from_thiele_innes must preserve cos_i sign for retrograde "
            "orbits; got "
            f"cos_i = {float(camp['cos_i'].value)}"
        )
        A_rt, B_rt, F_rt, G_rt = thiele_innes_from_campbell(
            camp["semi_major_axis"],
            camp["arg_peri"],
            camp["lon_asc_node"],
            camp["cos_i"],
        )
        for name, orig, rt in (
            ("A", ti_A, A_rt),
            ("B", ti_B, B_rt),
            ("F", ti_F, F_rt),
            ("G", ti_G, G_rt),
        ):
            assert jnp.allclose(orig.value, rt.value, atol=1e-5), (
                f"TI {name} did not round-trip: orig={orig.value}, rt={rt.value}"
            )

    def test_campbell_path_prediction_matches_data(
        self, bh3_ti_sampler, bh3_like_data_and_truth, bh3_warm_start
    ):
        """After TI -> Campbell, astrometric_orbit_at_times fits data.

        If `test_design_matrix_prediction_matches_data` passes but this one
        fails, the bug is in `thiele_innes_to_campbell` /
        `campbell_from_thiele_innes` / `astrometric_orbit_at_times` consistency.
        """

        data, truth = bh3_like_data_and_truth
        refined = bh3_ti_sampler.optimize(bh3_warm_start, data, seed=0)
        campbell = refined.thiele_innes_to_campbell()

        dra, ddec = astrometric_orbit_at_times(
            data.time,
            campbell.nonlinear["period"][0],
            campbell.nonlinear["eccentricity"][0],
            campbell["t_peri"][0],
            campbell.nonlinear["arg_peri"][0],
            campbell.nonlinear["cos_i"][0],
            campbell.nonlinear["lon_asc_node"][0],
            campbell.linear["semi_major_axis"][0],
        )
        sin_psi = jnp.sin(data.scan_angle.value)
        cos_psi = jnp.cos(data.scan_angle.value)
        dt_yr = data.time.value / 365.25

        ra0 = float(campbell.linear["ra0"][0].value)
        dec0 = float(campbell.linear["dec0"][0].value)
        pmra = float(campbell.linear["pmra"][0].value)
        pmdec = float(campbell.linear["pmdec"][0].value)
        parallax = float(campbell.linear["parallax"][0].value)

        al_pred = (
            ra0 * sin_psi
            + dec0 * cos_psi
            + (pmra * dt_yr) * sin_psi
            + (pmdec * dt_yr) * cos_psi
            + parallax * data.parallax_factor
            + dra.value * sin_psi
            + ddec.value * cos_psi
        )
        resid = jnp.asarray(data.al_position.value) - jnp.asarray(al_pred)
        max_resid = float(jnp.max(jnp.abs(resid)))
        sigma = truth["sigma_al"]
        assert max_resid < 10.0 * sigma, (
            f"Campbell path: max |residual| = {max_resid:.3f} mas "
            f">> {10 * sigma:.3f} mas (10*sigma)."
        )
