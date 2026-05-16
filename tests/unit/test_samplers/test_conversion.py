"""Tests for parameterization conversion of sampled parameter values.

Covers the standalone :func:`harv.samplers.convert_parameterization` and the
:meth:`harv.samplers.Samples.convert_parameterization` wrapper.
"""

import jax
import jax.numpy as jnp
import pytest
from unxt import Q

from harv.kepler.orbits import (
    campbell_from_thiele_innes,
    ecc_omega_from_ecosw_esinw,
    ecosw_esinw_from_ecc_omega,
    thiele_innes_from_campbell,
)
from harv.models.parameterizations.gaia import (
    StandardGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV
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
    def test_standard_to_ecosw_esinw(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        out_nl, out_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=EcoswEsinwRV()
        )

        assert list(out_nl) == ["period", "ecosw", "esinw", "phase_peri"]
        assert list(out_lin) == ["rv_semiamp", "v_sys"]

        exp_ecosw, exp_esinw = ecosw_esinw_from_ecc_omega(
            nl["eccentricity"], nl["arg_peri"]
        )
        assert jnp.allclose(out_nl["ecosw"].value, exp_ecosw.value)
        assert jnp.allclose(out_nl["esinw"].value, exp_esinw.value)
        # shared params and units pass through unchanged
        assert jnp.allclose(out_nl["period"].value, nl["period"].value)
        assert out_lin["rv_semiamp"].unit == lin["rv_semiamp"].unit

    def test_ecosw_esinw_to_standard(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        ecosw_nl, _ = convert_parameterization(
            nl, lin, source=StandardRV(), target=EcoswEsinwRV()
        )
        out_nl, _ = convert_parameterization(
            ecosw_nl, lin, source=EcoswEsinwRV(), target=StandardRV()
        )

        assert list(out_nl) == ["period", "eccentricity", "phase_peri", "arg_peri"]
        exp_ecc, exp_omega = ecc_omega_from_ecosw_esinw(
            ecosw_nl["ecosw"], ecosw_nl["esinw"]
        )
        assert jnp.allclose(out_nl["eccentricity"].value, exp_ecc.value)
        assert jnp.allclose(out_nl["arg_peri"].value, exp_omega.value)

    def test_round_trip_recovers_inputs(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        ecosw_nl, ecosw_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=EcoswEsinwRV()
        )
        back_nl, back_lin = convert_parameterization(
            ecosw_nl, ecosw_lin, source=EcoswEsinwRV(), target=StandardRV()
        )
        assert jnp.allclose(
            back_nl["eccentricity"].value, nl["eccentricity"].value, atol=1e-6
        )
        assert jnp.allclose(back_nl["arg_peri"].value, nl["arg_peri"].value, atol=1e-6)
        assert jnp.allclose(back_lin["v_sys"].value, lin["v_sys"].value)

    def test_extra_params_preserved(self):
        nl = _rv_nonlinear()
        lin = {**_rv_linear(), "jitter": Q(jnp.array([0.2, 0.3]), "km/s")}
        _out_nl, out_lin = convert_parameterization(
            nl, lin, source=StandardRV(), target=EcoswEsinwRV()
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
            convert_parameterization(
                nl, lin, source=StandardRV(), target=EcoswEsinwRV()
            )

    def test_under_jit(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        fn = jax.jit(
            lambda n, ln: convert_parameterization(
                n, ln, source=StandardRV(), target=EcoswEsinwRV()
            )
        )
        out_nl, _out_lin = fn(nl, lin)
        exp_ecosw, _ = ecosw_esinw_from_ecc_omega(nl["eccentricity"], nl["arg_peri"])
        assert jnp.allclose(out_nl["ecosw"].value, exp_ecosw.value)

    def test_under_vmap(self):
        nl, lin = _rv_nonlinear(), _rv_linear()

        def fn(n, ln):
            return convert_parameterization(
                n, ln, source=StandardRV(), target=EcoswEsinwRV()
            )

        out_nl, out_lin = jax.vmap(fn)(nl, lin)
        assert out_nl["ecosw"].shape == (2,)
        assert out_lin["v_sys"].shape == (2,)


class TestConvertGaia:
    def test_standard_to_thiele_innes(self):
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        out_nl, out_lin = convert_parameterization(
            nl,
            lin,
            source=StandardGaiaAstrometry(),
            target=ThieleInnesGaiaAstrometry(a_floor=0.01),
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
            target=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )
        out_nl, out_lin = convert_parameterization(
            ti_nl,
            ti_lin,
            source=ThieleInnesGaiaAstrometry(a_floor=0.01),
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
            target=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )
        back_nl, back_lin = convert_parameterization(
            ti_nl,
            ti_lin,
            source=ThieleInnesGaiaAstrometry(a_floor=0.01),
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
            target=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )
        _out_nl, out_lin = convert_parameterization(
            ti_nl,
            ti_lin,
            source=ThieleInnesGaiaAstrometry(a_floor=0.01),
            target=ThieleInnesGaiaAstrometry(a_floor=0.05),
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
            target=ThieleInnesGaiaAstrometry(a_floor=0.01),
        )
        assert jnp.allclose(out_lin["excess_noise"].value, lin["excess_noise"].value)

    def test_under_jit(self):
        nl, lin = _gaia_nonlinear(), _gaia_linear()
        fn = jax.jit(
            lambda n, ln: convert_parameterization(
                n,
                ln,
                source=StandardGaiaAstrometry(),
                target=ThieleInnesGaiaAstrometry(a_floor=0.01),
            )
        )
        _, out_lin = fn(nl, lin)
        assert "ti_A" in out_lin
        assert out_lin["ti_A"].shape == (2,)


class TestConvertErrors:
    def test_namespaced_samples_raise(self):
        nl = _rv_nonlinear()
        lin = {
            "primary.rv_semiamp": Q(jnp.array([5.0, 6.0]), "km/s"),
            "v_sys": Q(jnp.array([0.0, 1.0]), "km/s"),
        }
        with pytest.raises(NotImplementedError, match="single-component"):
            convert_parameterization(
                nl, lin, source=StandardRV(), target=EcoswEsinwRV()
            )

    def test_mixed_family_raises(self):
        nl, lin = _rv_nonlinear(), _rv_linear()
        with pytest.raises(NotImplementedError, match="Unsupported"):
            convert_parameterization(
                nl, lin, source=StandardRV(), target=StandardGaiaAstrometry()
            )


class TestSamplesWrapper:
    def test_rv_wrapper_preserves_metadata_and_extensions(self):
        samples = Samples(
            nonlinear=_rv_nonlinear(),
            linear={**_rv_linear(), "jitter": Q(jnp.array([0.2, 0.3]), "km/s")},
            data_type="RVModel",
            metadata={"t_ref": Q(0.0, "day"), "num_chains": 2},
            linear_extension_names=("jitter",),
        )
        converted = samples.convert_parameterization(
            source=StandardRV(), target=EcoswEsinwRV()
        )
        assert isinstance(converted, Samples)
        assert set(converted.nonlinear) == {"period", "ecosw", "esinw", "phase_peri"}
        assert "jitter" in converted.linear
        assert converted.data_type == "RVModel"
        assert converted.metadata == samples.metadata
        assert converted.linear_extension_names == ("jitter",)

        round_tripped = converted.convert_parameterization(
            source=EcoswEsinwRV(), target=StandardRV()
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
            data_type="gaia_astro",
            metadata={},
        )
        converted = samples.convert_parameterization(
            source=ThieleInnesGaiaAstrometry(a_floor=0.0),
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
            data_type="JointModel",
            metadata={},
        )
        with pytest.raises(NotImplementedError, match="single-component"):
            samples.convert_parameterization(source=StandardRV(), target=EcoswEsinwRV())
