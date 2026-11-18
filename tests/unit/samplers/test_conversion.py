"""Tests for parameterization conversion of sampled parameter values.

Covers the standalone :func:`harv.samplers.convert_parameterization` and the
:meth:`harv.samplers.Samples.convert_parameterization` wrapper.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Q

from harv.kepler.orbits import (
    campbell_from_thiele_innes,
    ecc_omega_from_sqrte_cosw_sinw,
    sqrte_cosw_sinw_from_ecc_omega,
    thiele_innes_from_campbell,
    thiele_innes_unit,
)
from harv.models.parameterizations.gaia import (
    StandardGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.models.parameterizations.rv import SqrtEccRV, StandardRV
from harv.samplers import convert_parameterization
from harv.samplers.samples import Samples


def _rv_nonlinear() -> dict[str, Q]:
    return {
        "period": Q(jnp.array([100.0, 120.0]), "day"),
        "eccentricity": Q(jnp.array([0.3, 0.1]), ""),
        "phase_peri": Q(jnp.array([0.0, 0.25]), ""),
        "arg_peri": Q(jnp.array([0.5, 1.2]), "rad"),
    }


def _rv_linear() -> dict[str, Q]:
    return {
        "rv_semiamp": Q(jnp.array([5.0, 6.0]), "km/s"),
        "v_sys": Q(jnp.array([0.0, 1.0]), "km/s"),
    }


def _gaia_nonlinear() -> dict[str, Q]:
    return {
        "period": Q(jnp.array([300.0, 280.0]), "day"),
        "eccentricity": Q(jnp.array([0.3, 0.2]), ""),
        "phase_peri": Q(jnp.array([0.2, 0.6]), ""),
        "arg_peri": Q(jnp.array([0.8, 2.1]), "rad"),
        "lon_asc_node": Q(jnp.array([1.1, 0.4]), "rad"),
        "cos_i": Q(jnp.array([0.6, 0.3]), ""),
    }


def _gaia_linear() -> dict[str, Q]:
    return {
        "ra0": Q(jnp.array([0.0, 0.1]), "mas"),
        "dec0": Q(jnp.array([0.0, -0.1]), "mas"),
        "pmra": Q(jnp.array([1.0, 2.0]), "mas/yr"),
        "pmdec": Q(jnp.array([-1.0, 0.5]), "mas/yr"),
        "parallax": Q(jnp.array([5.0, 8.0]), "mas"),
        "semi_major_axis": Q(jnp.array([2.0, 1.5]), "mas"),
    }


class TestConvertRV:
    def test_standard_to_sqrt_ecc(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        out_nl, out_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=SqrtEccRV()
        )

        assert list(out_nl) == ["period", "sqrte_cosw", "sqrte_sinw", "mean_longitude"]
        assert list(out_lin) == ["rv_semiamp", "v_sys"]

        exp_x, exp_y = sqrte_cosw_sinw_from_ecc_omega(
            nl["eccentricity"], nl["arg_peri"]
        )
        assert jnp.allclose(out_nl["sqrte_cosw"].value, exp_x.value)
        assert jnp.allclose(out_nl["sqrte_sinw"].value, exp_y.value)
        # mean longitude at time_ref: omega - 2*pi*phase_peri, wrapped to [0, 2pi)
        exp_lon = jnp.mod(
            nl["arg_peri"].value - 2 * jnp.pi * nl["phase_peri"].value, 2 * jnp.pi
        )
        assert jnp.allclose(out_nl["mean_longitude"].value, exp_lon)
        assert out_nl["mean_longitude"].unit == "rad"
        # shared params and units pass through unchanged
        assert jnp.allclose(out_nl["period"].value, nl["period"].value)
        assert out_lin["rv_semiamp"].unit == lin["rv_semiamp"].unit

    def test_sqrt_ecc_to_standard(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        sqe_nl, _ = convert_parameterization(
            nl, lin, source=StandardRV(), target=SqrtEccRV()
        )
        out_nl, _ = convert_parameterization(
            sqe_nl, lin, source=SqrtEccRV(), target=StandardRV()
        )

        assert list(out_nl) == ["period", "eccentricity", "phase_peri", "arg_peri"]
        exp_ecc, exp_omega = ecc_omega_from_sqrte_cosw_sinw(
            sqe_nl["sqrte_cosw"], sqe_nl["sqrte_sinw"]
        )
        assert jnp.allclose(out_nl["eccentricity"].value, exp_ecc.value)
        assert jnp.allclose(out_nl["arg_peri"].value, exp_omega.value)

    def test_arg_peri_wrapped_into_standard_prior_range(self):
        # atan2 gives (-pi, pi]; StandardRV's default arg_peri prior is [0, 2pi).
        nl = {
            "period": Q(jnp.array([100.0]), "day"),
            "sqrte_cosw": Q(jnp.array([0.3]), ""),
            "sqrte_sinw": Q(jnp.array([-0.4]), ""),
            "mean_longitude": Q(jnp.array([1.0]), "rad"),
        }
        out_nl, _ = convert_parameterization(
            nl, _rv_linear(), source=SqrtEccRV(), target=StandardRV()
        )
        arg_peri = out_nl["arg_peri"].value
        assert jnp.all((arg_peri >= 0) & (arg_peri < 2 * jnp.pi))
        assert jnp.allclose(arg_peri, 2 * jnp.pi + jnp.arctan2(-0.4, 0.3))

    def test_round_trip_recovers_inputs(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        sqe_nl, sqe_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=SqrtEccRV()
        )
        back_nl, back_lin = convert_parameterization(
            sqe_nl, sqe_lin, source=SqrtEccRV(), target=StandardRV()
        )
        assert jnp.allclose(
            back_nl["eccentricity"].value, nl["eccentricity"].value, atol=1e-6
        )
        assert jnp.allclose(back_nl["arg_peri"].value, nl["arg_peri"].value, atol=1e-6)
        # Phases are circular: 1 - eps and 0 are the same point.
        dphase = back_nl["phase_peri"].value - nl["phase_peri"].value
        assert jnp.allclose((dphase + 0.5) % 1.0 - 0.5, 0.0, atol=1e-6)
        assert jnp.allclose(back_lin["v_sys"].value, lin["v_sys"].value)

    def test_extra_params_preserved(self):
        nl = _rv_nonlinear()
        lin = {**_rv_linear(), "jitter": Q(jnp.array([0.2, 0.3]), "km/s")}
        _out_nl, out_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=SqrtEccRV()
        )
        assert "jitter" in out_lin
        assert jnp.allclose(out_lin["jitter"].value, lin["jitter"].value)

    def test_same_type_is_identity(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        out_nl, out_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=StandardRV()
        )
        for key, value in nl.items():
            assert jnp.array_equal(out_nl[key].value, value.value)
        for key, value in lin.items():
            assert jnp.array_equal(out_lin[key].value, value.value)

    def test_missing_key_raises_value_error(self):
        nl = _rv_nonlinear()
        lin = {"rv_semiamp": Q(jnp.array([5.0, 6.0]), "km/s")}  # missing v_sys
        with pytest.raises(ValueError, match="Missing required source parameters"):
            convert_parameterization(nl, lin, source=StandardRV(), target=SqrtEccRV())

    def test_under_jit(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        fn = jax.jit(
            lambda n, ln: convert_parameterization(
                n, ln, source=StandardRV(), target=SqrtEccRV()
            )
        )
        out_nl, _out_lin = fn(nl, lin)
        exp_x, _ = sqrte_cosw_sinw_from_ecc_omega(nl["eccentricity"], nl["arg_peri"])
        assert jnp.allclose(out_nl["sqrte_cosw"].value, exp_x.value)

    def test_under_vmap(self):
        nl, lin = _rv_nonlinear(), _rv_linear()

        def fn(n, ln):
            return convert_parameterization(
                n, ln, source=StandardRV(), target=SqrtEccRV()
            )

        out_nl, out_lin = jax.vmap(fn)(nl, lin)
        assert out_nl["sqrte_cosw"].shape == (2,)
        assert out_lin["v_sys"].shape == (2,)


class TestConvertGaia:
    def test_standard_to_thiele_innes(self):
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        out_nl, out_lin = convert_parameterization(
            nl,
            lin,
            source=StandardGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(),
        )
        assert list(out_nl) == ["period", "eccentricity", "phase_peri"]
        assert list(out_lin) == [
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "ti_A",
            "ti_B",
            "ti_F",
            "ti_G",
        ]

        exp_A, exp_B, exp_F, exp_G = thiele_innes_from_campbell(
            lin["semi_major_axis"],
            nl["arg_peri"],
            nl["lon_asc_node"],
            nl["cos_i"],
        )
        assert jnp.allclose(out_lin["ti_A"].value, exp_A.value, atol=1e-6)
        assert jnp.allclose(out_lin["ti_B"].value, exp_B.value, atol=1e-6)
        assert jnp.allclose(out_lin["ti_F"].value, exp_F.value, atol=1e-6)
        assert jnp.allclose(out_lin["ti_G"].value, exp_G.value, atol=1e-6)

    def test_thiele_innes_to_standard(self):
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        ti_nl, ti_lin = convert_parameterization(
            nl,
            lin,
            source=StandardGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(),
        )
        out_nl, out_lin = convert_parameterization(
            ti_nl,
            ti_lin,
            source=ThieleInnesGaiaAstrometry(),
            target=StandardGaiaAstrometry(),
        )
        assert "semi_major_axis" in out_lin
        for key in ("arg_peri", "lon_asc_node", "cos_i"):
            assert key in out_nl

        campbell = campbell_from_thiele_innes(
            A=ti_lin["ti_A"], B=ti_lin["ti_B"], F=ti_lin["ti_F"], G=ti_lin["ti_G"]
        )
        assert jnp.allclose(
            out_lin["semi_major_axis"].value, campbell["semi_major_axis"].value
        )
        assert jnp.allclose(out_nl["cos_i"].value, campbell["cos_i"].value)

    def test_round_trip_recovers_inputs(self):
        # prograde orbit with in-range angles -> exact round trip
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        ti_nl, ti_lin = convert_parameterization(
            nl,
            lin,
            source=StandardGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(),
        )
        back_nl, back_lin = convert_parameterization(
            ti_nl,
            ti_lin,
            source=ThieleInnesGaiaAstrometry(),
            target=StandardGaiaAstrometry(),
        )
        assert jnp.allclose(
            back_lin["semi_major_axis"].value, lin["semi_major_axis"].value, atol=1e-5
        )
        assert jnp.allclose(back_nl["arg_peri"].value, nl["arg_peri"].value, atol=1e-5)
        assert jnp.allclose(
            back_nl["lon_asc_node"].value, nl["lon_asc_node"].value, atol=1e-5
        )
        assert jnp.allclose(back_nl["cos_i"].value, nl["cos_i"].value, atol=1e-5)

    def test_thiele_innes_same_type_is_lossless_identity(self):
        # TI -> TI must be an exact no-op: routing through Campbell would force
        # cos_i >= 0 and wrap angles, mangling retrograde orbits.
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        ti_nl, ti_lin = convert_parameterization(
            nl,
            lin,
            source=StandardGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(),
        )
        _out_nl, out_lin = convert_parameterization(
            ti_nl,
            ti_lin,
            source=ThieleInnesGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(),
        )
        for key in ("ti_A", "ti_B", "ti_F", "ti_G"):
            assert jnp.array_equal(out_lin[key].value, ti_lin[key].value)

    def test_extra_params_preserved(self):
        nl = _gaia_nonlinear()
        lin = {**_gaia_linear(), "excess_noise": Q(jnp.array([0.1, 0.2]), "mas")}
        _out_nl, out_lin = convert_parameterization(
            nl,
            lin,
            source=StandardGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(),
        )
        assert jnp.allclose(out_lin["excess_noise"].value, lin["excess_noise"].value)

    def test_under_jit(self):
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        fn = jax.jit(
            lambda n, ln: convert_parameterization(
                n,
                ln,
                source=StandardGaiaAstrometry(),
                target=ThieleInnesGaiaAstrometry(),
            )
        )
        _, out_lin = fn(nl, lin)
        assert "ti_A" in out_lin
        assert out_lin["ti_A"].shape == (2,)

    def test_ti_vs_standard_design_matrix_agree(self):
        """Standard and TI design matrices must produce the same AL signal.

        Smoking-gun test for a TI structural bug. For a fixed orbit (Campbell
        elements and equivalent TI constants), the predicted along-scan
        position must be **identical** between the two parameterizations,
        regardless of marginalization, priors, or MCMC. If this fails, the TI
        design matrix has a column/sign bug.
        """
        # Pick a fixed orbit
        a = jnp.array(2.5)  # mas
        arg_peri = jnp.array(0.8)  # rad
        lon_asc_node = jnp.array(1.7)  # rad
        cos_i = jnp.array(0.55)
        ecc = jnp.array(0.42)
        ra0 = jnp.array(0.3)  # mas
        dec0 = jnp.array(-0.4)  # mas
        pmra = jnp.array(2.0)  # mas/yr
        pmdec = jnp.array(-1.5)  # mas/yr
        parallax = jnp.array(1.7)  # mas

        # Arbitrary observation geometry
        n = 25
        rng = np.random.default_rng(7)
        sin_psi = jnp.asarray(jnp.sin(jnp.linspace(0.0, 5.0 * jnp.pi, n)))
        cos_psi = jnp.asarray(jnp.cos(jnp.linspace(0.0, 5.0 * jnp.pi, n)))
        dt = jnp.linspace(-0.5, 0.5, n)  # years
        parallax_factor = jnp.asarray(rng.uniform(-1.0, 1.0, n))

        # True anomaly geometry (independent of parameterization)
        sin_f = jnp.asarray(rng.uniform(-1.0, 1.0, n))
        cos_f = jnp.sqrt(jnp.maximum(1.0 - sin_f**2, 0.0)) * jnp.where(
            jnp.asarray(rng.integers(0, 2, n)) == 0, 1.0, -1.0
        )

        # Standard design matrix: (n, 6); rightmost column is semi_major_axis
        std_X = StandardGaiaAstrometry().design_matrix(
            sin_f,
            cos_f,
            dt,
            sin_psi,
            cos_psi,
            parallax_factor,
            {
                "eccentricity": ecc,
                "arg_peri": arg_peri,
                "lon_asc_node": lon_asc_node,
                "cos_i": cos_i,
            },
        )
        std_lin = jnp.array([ra0, dec0, pmra, pmdec, parallax, a])
        al_std = std_X @ std_lin

        # TI design matrix: (n, 9); TI columns at the end
        # Compute equivalent ti_A..G = a * unit_TI_constants
        A_u, B_u, F_u, G_u = thiele_innes_unit(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )
        ti_A = a * A_u
        ti_B = a * B_u
        ti_F = a * F_u
        ti_G = a * G_u
        ti_X = ThieleInnesGaiaAstrometry(apply_jacobian_correction=False).design_matrix(
            sin_f,
            cos_f,
            dt,
            sin_psi,
            cos_psi,
            parallax_factor,
            {"eccentricity": ecc},
        )
        ti_lin = jnp.array([ra0, dec0, pmra, pmdec, parallax, ti_A, ti_B, ti_F, ti_G])
        al_ti = ti_X @ ti_lin

        assert jnp.allclose(al_std, al_ti, atol=1e-8), (
            f"Standard AL prediction does not match TI prediction. "
            f"max abs diff = {float(jnp.max(jnp.abs(al_std - al_ti)))}"
        )


class TestConvertErrors:
    def test_namespaced_samples_raise(self):
        nl = _rv_nonlinear()
        lin = {
            "primary.rv_semiamp": Q(jnp.array([5.0, 6.0]), "km/s"),
            "v_sys": Q(jnp.array([0.0, 1.0]), "km/s"),
        }
        with pytest.raises(NotImplementedError, match="single-component"):
            convert_parameterization(nl, lin, source=StandardRV(), target=SqrtEccRV())

    def test_mixed_family_raises(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        with pytest.raises(NotImplementedError, match="Unsupported"):
            convert_parameterization(
                nl, lin, source=StandardRV(), target=StandardGaiaAstrometry()
            )


class TestSamplesWrapper:
    @pytest.mark.parametrize(
        "nonlinear",
        [
            # mean_longitude alone (e.g. a user subset) cannot give phase_peri.
            {
                "period": Q(jnp.array([100.0]), "day"),
                "mean_longitude": Q(jnp.array([1.0]), "rad"),
            },
            # A full periastron phase but no period.
            {
                "sqrte_cosw": jnp.array([0.3]),
                "sqrte_sinw": jnp.array([0.4]),
                "mean_longitude": Q(jnp.array([1.0]), "rad"),
            },
        ],
    )
    def test_time_peri_needs_all_inputs(self, nonlinear):
        samples = Samples(nonlinear=nonlinear, linear={}, model_type="RVModel")
        derivable = samples.keys()
        assert "time_peri" not in derivable
        with pytest.raises(KeyError, match="time_peri needs"):
            samples["time_peri"]

    def test_rv_wrapper_preserves_metadata_and_extensions(self):
        samples = Samples(
            nonlinear=_rv_nonlinear(),
            linear={**_rv_linear(), "jitter": Q(jnp.array([0.2, 0.3]), "km/s")},
            model_type="RVModel",
            metadata={"time_ref": 0.0, "time_ref_unit": "day", "num_chains": 2},
            linear_extension_names=("jitter",),
        )
        converted = samples.convert_parameterization(
            source=StandardRV(), target=SqrtEccRV()
        )
        assert isinstance(converted, Samples)
        assert set(converted.nonlinear) == {
            "period",
            "sqrte_cosw",
            "sqrte_sinw",
            "mean_longitude",
        }
        # time_peri is derivable from the mean longitude and matches the source,
        # modulo one period.
        period = samples["period"].value
        dt = converted["time_peri"].value - samples["time_peri"].value
        assert jnp.allclose((dt / period + 0.5) % 1.0 - 0.5, 0.0, atol=1e-9)
        assert "jitter" in converted.linear
        assert converted.model_type == "RVModel"
        assert converted.metadata == samples.metadata
        assert converted.linear_extension_names == ("jitter",)

        round_tripped = converted.convert_parameterization(
            source=SqrtEccRV(), target=StandardRV()
        )
        assert jnp.allclose(
            round_tripped["eccentricity"].value,
            samples["eccentricity"].value,
            atol=1e-6,
        )
        assert jnp.allclose(round_tripped["jitter"].value, samples["jitter"].value)

    def test_gaia_wrapper_matches_thiele_innes_to_campbell(self):
        n = 4
        a0 = jnp.ones(n) * 2.0
        arg_peri = jnp.linspace(0.2, 1.5, n)
        lon_asc_node = jnp.linspace(0.5, 2.0, n)
        cos_i = jnp.linspace(0.3, 0.8, n)
        ti_A, ti_B, ti_F, ti_G = thiele_innes_from_campbell(
            Q(a0, "mas"),
            Q(arg_peri, "rad"),
            Q(lon_asc_node, "rad"),
            Q(cos_i, ""),
        )
        samples = Samples(
            nonlinear={
                "period": Q(jnp.ones(n) * 365.0, "day"),
                "eccentricity": Q(jnp.zeros(n), ""),
                "phase_peri": Q(jnp.zeros(n), ""),
            },
            linear={
                "ra0": Q(jnp.zeros(n), "mas"),
                "dec0": Q(jnp.zeros(n), "mas"),
                "pmra": Q(jnp.zeros(n), "mas/yr"),
                "pmdec": Q(jnp.zeros(n), "mas/yr"),
                "parallax": Q(jnp.ones(n), "mas"),
                "ti_A": ti_A,
                "ti_B": ti_B,
                "ti_F": ti_F,
                "ti_G": ti_G,
            },
            model_type="gaia_astro",
            metadata={},
        )
        converted = samples.convert_parameterization(
            source=ThieleInnesGaiaAstrometry(),
            target=StandardGaiaAstrometry(),
        )
        convenience = samples.thiele_innes_to_campbell()
        for key in ("semi_major_axis", "arg_peri", "lon_asc_node", "cos_i"):
            assert jnp.allclose(converted[key].value, convenience[key].value)

    def test_wrapper_rejects_namespaced_samples(self):
        samples = Samples(
            nonlinear=_rv_nonlinear(),
            linear={
                "primary.rv_semiamp": Q(jnp.array([5.0, 6.0]), "km/s"),
                "v_sys": Q(jnp.array([0.0, 1.0]), "km/s"),
            },
            model_type="JointModel",
            metadata={},
        )
        with pytest.raises(NotImplementedError, match="single-component"):
            samples.convert_parameterization(source=StandardRV(), target=SqrtEccRV())
