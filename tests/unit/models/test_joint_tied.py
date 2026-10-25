"""Unit tests for tied linear parameters in :class:`harv.models.JointModel`.

Spec §Tied linear parameters. ``JointModel.for_rv_and_gaia`` ties
``rv.rv_semiamp`` to ``astro.semi_major_axis`` by default (unseen companion):
``K = c * a`` with ``c = 2 pi sin(i) / (P sqrt(1 - e^2) parallax)``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest
from jax.scipy.stats import multivariate_normal
from unxt import Q, ustrip

from harv import GaiaAstrometryData, GaiaAstrometryModel, JointModel, RVData, RVModel
from harv.data import SourceData
from harv.distributions import QD
from harv.models import (
    EcoswEsinwRV,
    HarvPrior,
    Jitter,
    rv_semiamp_per_semi_major_axis,
    tied,
)
from harv.models._helpers import LinearRole, classify_linear_prior, pinned_linear_names
from harv.models.parameterizations import ThieleInnesGaiaAstrometry
from harv.models.ties import TiedPrior
from harv.samplers import NumpyroSampler, RejectionSampler, Samples
from harv.samplers._prior_resolution import (
    effective_linear_prior_from_prior,
    explicit_linear_names,
    resolve_effective_marginalized_names,
)

PARALLAX = 5.0  # mas
MARGINALIZED = (
    "astro.ra0",
    "astro.dec0",
    "astro.pmra",
    "astro.pmdec",
    "astro.semi_major_axis",
    "rv.v_sys",
    # Free K must be marginalized (it is not in _values()); a tied K folds into
    # its source's column either way.
    "rv.rv_semiamp",
)
FREE_K = QD(ndist.Normal(0.0, 30.0), "km/s")


def _data() -> SourceData:
    rng = np.random.default_rng(42)
    n_ast, n_rv = 30, 12
    astro = GaiaAstrometryData(
        time=Q(jnp.linspace(0.0, 1500.0, n_ast), "day"),
        al_position=Q(jnp.asarray(rng.normal(0.0, 0.3, n_ast)), "mas"),
        al_position_err=Q(jnp.full(n_ast, 0.1), "mas"),
        scan_angle=Q(jnp.asarray(rng.uniform(0, 2 * np.pi, n_ast)), "rad"),
        parallax_factor=jnp.asarray(rng.uniform(-1, 1, n_ast)),
    )
    rv = RVData(
        time=Q(jnp.linspace(200.0, 1700.0, n_rv), "day"),
        rv=Q(jnp.asarray(rng.normal(0.0, 3.0, n_rv)), "km/s"),
        rv_err=Q(jnp.full(n_rv, 0.5), "km/s"),
    )
    return SourceData(rv=rv, astro=astro)


def _linear_priors() -> dict:
    """No ``rv.rv_semiamp`` entry: the model's default tie applies."""
    return {
        "astro.ra0": QD(ndist.Normal(0.0, 10.0), "mas"),
        "astro.dec0": QD(ndist.Normal(0.0, 10.0), "mas"),
        "astro.pmra": QD(ndist.Normal(0.0, 20.0), "mas/yr"),
        "astro.pmdec": QD(ndist.Normal(0.0, 20.0), "mas/yr"),
        "astro.parallax": QD(ndist.Normal(PARALLAX, 1.0), "mas"),
        "astro.semi_major_axis": QD(ndist.Normal(0.0, 2.0), "mas"),
        "rv.v_sys": QD(ndist.Normal(0.0, 10.0), "km/s"),
    }


def _values(cos_i: float = 0.4) -> dict:
    return {
        "period": Q(300.0, "day"),
        "eccentricity": 0.3,
        "phase_peri": 0.2,
        "arg_peri": Q(1.0, "rad"),
        "lon_asc_node": Q(2.0, "rad"),
        "cos_i": cos_i,
        "astro.parallax": jnp.asarray(PARALLAX),
    }


def _model() -> JointModel:
    return JointModel.for_rv_and_gaia(
        components={"rv": RVModel(), "astro": GaiaAstrometryModel()}
    )


def _log_prob(model, priors, values=None, marginalized_names=MARGINALIZED):
    return model.log_prob(
        _values() if values is None else values,
        _data(),
        linear_priors=priors,
        marginalized_names=marginalized_names,
    )


def _factor(values: dict) -> float:
    """The factor c, in km/s per mas, straight from the spec formula."""
    e, cos_i = values["eccentricity"], values["cos_i"]
    speed = ustrip("km/s", 2 * np.pi * Q(1.0, "AU") / values["period"])
    return speed * np.sqrt(1 - cos_i**2) / (np.sqrt(1 - e**2) * PARALLAX)


def test_for_rv_and_gaia_ties_rv_semiamp_by_default():
    default = _model().default_linear_priors["rv.rv_semiamp"]
    assert default.source == "astro.semi_major_axis"
    assert default.scale is rv_semiamp_per_semi_major_axis


@pytest.mark.parametrize("marginalized_names", [MARGINALIZED, None])
def test_log_prob_matches_closed_form_marginal(marginalized_names):
    data = _data()
    model = _model()
    values = _values()
    priors = _linear_priors()
    if marginalized_names is None:
        # Auto mode: parallax is still explicit, here by its Delta-free Normal
        # being read by the tie's scale (pinned via `requires`).
        assert "astro.parallax" in pinned_linear_names(
            model._merged_linear_priors(priors)
        )
    got = model.log_prob(
        values, data, linear_priors=priors, marginalized_names=marginalized_names
    )

    # y = X beta + noise, beta = (ra0, dec0, pmra, pmdec, a, v_sys), with the RV
    # rows of the `a` column equal to c times the RV-shape column.
    rv_model, astro_model = model.components["rv"], model.components["astro"]
    rv_names = ("period", "eccentricity", "phase_peri", "arg_peri")
    x_rv = rv_model._full_design_matrix({k: values[k] for k in rv_names}, data["rv"])
    x_astro = astro_model._full_design_matrix(values, data["astro"])
    # astro columns: (ra0, dec0, pmra, pmdec, parallax, semi_major_axis)
    n_ast, n_rv = x_astro.shape[0], x_rv.shape[0]
    X = jnp.zeros((n_ast + n_rv, 6))
    X = X.at[:n_ast, :4].set(x_astro[:, :4]).at[:n_ast, 4].set(x_astro[:, 5])
    X = X.at[n_ast:, 4].set(_factor(values) * x_rv[:, 0])
    X = X.at[n_ast:, 5].set(x_rv[:, 1])
    y = jnp.concatenate(
        [
            ustrip("mas", data["astro"].al_position) - PARALLAX * x_astro[:, 4],
            ustrip("km/s", data["rv"].rv),
        ]
    )
    noise = jnp.concatenate(
        [
            ustrip("mas", data["astro"].al_position_err),
            ustrip("km/s", data["rv"].rv_err),
        ]
    )
    prior_scale = jnp.array([10.0, 10.0, 20.0, 20.0, 2.0, 10.0])
    cov = jnp.diag(noise**2) + X @ jnp.diag(prior_scale**2) @ X.T
    expected = multivariate_normal.logpdf(y, jnp.zeros_like(y), cov)

    assert jnp.allclose(got, expected, rtol=1e-10)


def test_explicit_tie_matches_default():
    explicit = {
        **_linear_priors(),
        "rv.rv_semiamp": tied(
            "astro.semi_major_axis", scale=rv_semiamp_per_semi_major_axis
        ),
    }
    assert jnp.allclose(
        _log_prob(_model(), explicit), _log_prob(_model(), _linear_priors())
    )


def test_user_prior_frees_rv_semiamp():
    """A prior on rv.rv_semiamp replaces the default tie (the untied model)."""
    untied = JointModel(
        components={"rv": RVModel(), "astro": GaiaAstrometryModel()},
        shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
    )
    priors = {**_linear_priors(), "rv.rv_semiamp": FREE_K}
    assert jnp.allclose(_log_prob(_model(), priors), _log_prob(untied, priors))
    assert not jnp.allclose(
        _log_prob(_model(), priors), _log_prob(_model(), _linear_priors())
    )


@pytest.mark.parametrize("use_mean", [False, True])
def test_conditional_tied_value_is_derived(use_mean):
    values = _values()
    out = _model().sample_conditional_linear(
        values,
        _data(),
        key=jax.random.key(0),
        linear_priors=_linear_priors(),
        marginalized_names=MARGINALIZED,
        use_mean=use_mean,
    )
    expected = _factor(values) * out["astro"]["semi_major_axis"]
    assert jnp.allclose(out["rv"]["rv_semiamp"], expected, rtol=1e-12)


def test_breaks_astrometric_mirror_degeneracy():
    """(omega + pi, Omega + pi) leaves astrometry unchanged but flips K's sign."""
    values = _values()
    mirror = {
        **values,
        "arg_peri": values["arg_peri"] + Q(np.pi, "rad"),
        "lon_asc_node": values["lon_asc_node"] + Q(np.pi, "rad"),
    }
    free = {**_linear_priors(), "rv.rv_semiamp": FREE_K}
    tied_lp = [_log_prob(_model(), _linear_priors(), v) for v in (values, mirror)]
    free_lp = [_log_prob(_model(), free, v) for v in (values, mirror)]
    # Free K: the mirror is the same orbit with K -> -K, so equally likely.
    assert jnp.allclose(free_lp[0], free_lp[1], rtol=1e-10)
    # Tied: K's sign is fixed by a, so the mirror is a different model.
    assert not jnp.allclose(tied_lp[0], tied_lp[1], rtol=1e-3)


def test_tied_prior_role_and_requires():
    prior = tied("astro.semi_major_axis", scale=rv_semiamp_per_semi_major_axis)
    assert classify_linear_prior(prior) is LinearRole.TIED
    assert prior.requires == ("period", "eccentricity", "cos_i", "parallax")


def test_tie_pins_the_parallax_it_reads_across_components():
    """The RV tie reads astro.parallax, so it is explicit, not marginalized."""
    model, priors = _model(), _linear_priors()
    assert "parallax" in model.explicit_params(priors)
    assert "parallax" not in model.marginalized_params(priors)
    assert "rv_semiamp" not in model.explicit_params(priors)


def test_log_prob_under_jit_and_vmap():
    data = _data()
    model = _model()
    priors = _linear_priors()

    def ln_l(cos_i):
        return model.log_prob(
            _values(cos_i), data, linear_priors=priors, marginalized_names=MARGINALIZED
        )

    cos_i = jnp.array([-0.5, 0.1, 0.8])
    batched = jax.jit(jax.vmap(ln_l))(cos_i)
    looped = jnp.stack([ln_l(c) for c in cos_i])
    assert jnp.allclose(batched, looped, rtol=1e-10)


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("semi_major_axis", "component-qualified"),
        ("astro.nope", "component-qualified"),
        ("other.semi_major_axis", "component-qualified"),
    ],
)
def test_rejects_bad_tie_source(source, match):
    priors = {
        **_linear_priors(),
        "rv.rv_semiamp": tied(source, scale=rv_semiamp_per_semi_major_axis),
    }
    with pytest.raises(ValueError, match=match):
        _log_prob(_model(), priors)


def test_rejects_explicit_tie_source():
    with pytest.raises(ValueError, match="must be analytically marginalized"):
        _log_prob(
            _model(),
            _linear_priors(),
            marginalized_names=tuple(
                n for n in MARGINALIZED if n != "astro.semi_major_axis"
            ),
        )


def test_rejects_marginalized_value_the_scale_reads():
    values = {k: v for k, v in _values().items() if k != "astro.parallax"}
    with pytest.raises(ValueError, match="sampled explicitly"):
        _log_prob(
            _model(),
            _linear_priors(),
            values,
            marginalized_names=(*MARGINALIZED, "astro.parallax"),
        )


def test_rejects_chained_tie():
    scale = rv_semiamp_per_semi_major_axis
    priors = {
        **_linear_priors(),
        "astro.semi_major_axis": tied("rv.v_sys", scale=scale),
        "rv.rv_semiamp": tied("astro.semi_major_axis", scale=scale),
    }
    with pytest.raises(ValueError, match="itself tied"):
        _log_prob(_model(), priors)


def test_rejects_tie_in_single_component_model():
    priors = {
        **{k.removeprefix("rv."): v for k, v in _linear_priors().items()},
        "rv_semiamp": tied("rv.v_sys", scale=rv_semiamp_per_semi_major_axis),
    }
    with pytest.raises(NotImplementedError, match="need a JointModel"):
        RVModel().log_prob(
            {
                k: _values()[k]
                for k in ("period", "eccentricity", "phase_peri", "arg_peri")
            },
            _data()["rv"],
            linear_priors=priors,
            marginalized_names=("rv_semiamp", "v_sys"),
        )


@pytest.mark.parametrize(
    "components",
    [
        {
            "rv": RVModel(),
            "astro": GaiaAstrometryModel(
                parameterization=ThieleInnesGaiaAstrometry(
                    apply_jacobian_correction=False
                )
            ),
        },
        {
            "rv": RVModel(parameterization=EcoswEsinwRV()),
            "astro": GaiaAstrometryModel(),
        },
    ],
)
def test_for_rv_and_gaia_requires_standard_parameterizations(components):
    with pytest.raises(TypeError, match="thiele_innes_to_campbell"):
        JointModel.for_rv_and_gaia(components=components)


def test_sampler_resolution_keeps_tied_target_marginalized():
    """Samplers see the model's default tie and never sample the tied target."""
    prior = HarvPrior(
        nonlinear_priors={"period": QD(ndist.Normal(300.0, 5.0), "day")},
        linear_priors=_linear_priors(),
    )
    effective = effective_linear_prior_from_prior(prior, _model())
    assert classify_linear_prior(effective["rv.rv_semiamp"]) is LinearRole.TIED

    without_target = tuple(n for n in MARGINALIZED if n != "rv.rv_semiamp")
    resolved = resolve_effective_marginalized_names(effective, without_target)
    assert "rv.rv_semiamp" in resolved
    assert "rv.rv_semiamp" not in explicit_linear_names(effective, resolved)
    # Auto mode: the tie's scale reads parallax, so it stays explicit.
    assert explicit_linear_names(effective, None) == ("astro.parallax",)


def test_rejects_fixed_tie_source():
    priors = {
        **_linear_priors(),
        "astro.semi_major_axis": QD(ndist.Delta(1.0), "mas"),
    }
    with pytest.raises(ValueError, match="must be analytically marginalized"):
        _log_prob(_model(), priors)


def test_rejects_tie_in_single_component_numpyro_model():
    priors = {
        "rv_semiamp": tied("rv.v_sys", scale=rv_semiamp_per_semi_major_axis),
        "v_sys": QD(ndist.Normal(0.0, 10.0), "km/s"),
    }
    with pytest.raises(NotImplementedError, match="need a JointModel"):
        RVModel().numpyro_model(
            {"period": QD(ndist.Normal(300.0, 5.0), "day")},
            _data()["rv"],
            priors,
            marginalized=False,
        )


def test_summary_counts_tied_parameters_separately():
    prior = HarvPrior(
        nonlinear_priors={"period": QD(ndist.Normal(300.0, 5.0), "day")},
        linear_priors=_linear_priors(),
    )
    summary = RejectionSampler(prior, _model()).summary()
    # Sampled: period + parallax; tied: rv_semiamp; the rest are marginalized.
    assert "2 sampled, 6 marginalized, 1 tied" in summary
    assert "tied to astro.semi_major_axis" in summary


def test_joint_path_applies_parameterization_prior_correction():
    """A tie forces the joint path, which must keep e.g. the Thiele-Innes Jacobian."""
    ti = ThieleInnesGaiaAstrometry(a_floor=0.01, apply_jacobian_correction=True)
    model = JointModel(
        components={"rv": RVModel(), "astro": GaiaAstrometryModel(parameterization=ti)},
        shared_params=("period", "eccentricity", "phase_peri"),
    )
    unit_ratio = Q(1.0, "km/(s mas)")
    priors = {
        **{k: v for k, v in _linear_priors().items() if "semi_major_axis" not in k},
        **{f"astro.ti_{c}": QD(ndist.Normal(0.0, 2.0), "mas") for c in "ABFG"},
        "rv.rv_semiamp": tied("astro.ti_A", scale=lambda _: unit_ratio),
    }
    values = {k: v for k, v in _values().items() if k not in ("lon_asc_node", "cos_i")}
    data = _data()
    marg = (*MARGINALIZED[:4], "astro.ti_A", "astro.ti_B", "astro.ti_F", "astro.ti_G")
    marg = (*marg, "rv.v_sys")
    got = model.log_prob(values, data, linear_priors=priors, marginalized_names=marg)

    comp_nl = {
        "rv": {
            k: values[k] for k in ("period", "eccentricity", "phase_peri", "arg_peri")
        },
        "astro": {
            **{k: values[k] for k in ("period", "eccentricity", "phase_peri")},
            "parallax": values["astro.parallax"],
        },
    }
    per_comp_marg, _ = model._resolve_marginalization(marg, priors)
    marg_dist, y, cols, _ = model._build_joint_marginalized_linear(
        comp_nl, per_comp_marg, data, priors
    )
    mean = marg_dist.conditional(y).mean
    astro_map = {nm: mean[j] for j, (nm, owner) in enumerate(cols) if owner == "astro"}
    expected = marg_dist.log_prob(y) + ti.linear_log_prior_correction(astro_map)
    assert jnp.allclose(got, expected, rtol=1e-12)
    assert abs(float(got - marg_dist.log_prob(y))) > 1e-6  # the correction is there


def test_rejects_tie_into_a_prior_corrected_component():
    """The Thiele-Innes Jacobian assumes four independent constants."""
    ti = ThieleInnesGaiaAstrometry(a_floor=0.01, apply_jacobian_correction=True)
    model = JointModel(
        components={"rv": RVModel(), "astro": GaiaAstrometryModel(parameterization=ti)},
        shared_params=("period", "eccentricity", "phase_peri"),
    )
    priors = {
        **{k: v for k, v in _linear_priors().items() if "semi_major_axis" not in k},
        **{f"astro.ti_{c}": QD(ndist.Normal(0.0, 2.0), "mas") for c in "BFG"},
        "astro.ti_A": tied("astro.ti_B", scale=lambda _: Q(0.5, "")),
        "rv.rv_semiamp": FREE_K,
    }
    values = {k: v for k, v in _values().items() if k not in ("lon_asc_node", "cos_i")}
    marg = (*MARGINALIZED, "astro.ti_A", "astro.ti_B", "astro.ti_F", "astro.ti_G")
    with pytest.raises(NotImplementedError, match="prior correction"):
        model.log_prob(values, _data(), linear_priors=priors, marginalized_names=marg)


def test_tie_scale_reads_a_shared_linear_parameter():
    """A tie whose scale reads a shared (bare-keyed) linear parameter pins it."""

    def scale(params):
        return Q(-2.0 * (1.0 + 0.1 * ustrip("km/s", params["v_sys"])), "")

    scale.requires = ("v_sys",)
    model = JointModel(
        components={"primary": RVModel(), "secondary": RVModel()},
        shared_params=("period", "eccentricity", "phase_peri", "arg_peri"),
        shared_linear_params=("v_sys",),
    )
    priors = {
        "primary.rv_semiamp": FREE_K,
        "secondary.rv_semiamp": tied("primary.rv_semiamp", scale=scale),
        "v_sys": QD(ndist.Normal(0.0, 10.0), "km/s"),
    }
    assert pinned_linear_names(priors) == {"v_sys"}
    assert model.marginalized_params(priors) == ("rv_semiamp",)

    rv = _data()["rv"]
    data = SourceData(primary=rv, secondary=rv)
    orbit = ("period", "eccentricity", "phase_peri", "arg_peri")
    values = {**{k: _values()[k] for k in orbit}, "v_sys": jnp.asarray(1.0)}
    got = jax.jit(lambda v: model.log_prob(v, data, linear_priors=priors))(values)
    # The scale at v_sys = 1 km/s is -2.2.
    constant = {
        **priors,
        "secondary.rv_semiamp": tied("primary.rv_semiamp", scale=lambda _: Q(-2.2, "")),
    }
    expected = model.log_prob(
        values,
        data,
        linear_priors=constant,
        marginalized_names=("primary.rv_semiamp", "secondary.rv_semiamp"),
    )
    assert jnp.allclose(got, expected, rtol=1e-12)

    out = model.sample_conditional_linear(
        values, data, key=jax.random.key(0), linear_priors=priors, use_mean=True
    )
    assert out["v_sys"] == 1.0  # shared explicit values stay at the top level
    assert jnp.allclose(
        out["secondary"]["rv_semiamp"], -2.2 * out["primary"]["rv_semiamp"]
    )


def test_component_fit_warm_starts_joint_extension_sites():
    """An astrometry-only fit's ``jitter`` initializes ``astro.jitter``."""
    joint = JointModel.for_rv_and_gaia(
        components={
            "rv": RVModel(extensions=(Jitter(obs_unit="km/s"),)),
            "astro": GaiaAstrometryModel(extensions=(Jitter(obs_unit="mas"),)),
        }
    )
    prior = HarvPrior(
        nonlinear_priors={"period": QD(ndist.Normal(300.0, 5.0), "day")},
        linear_priors=_linear_priors(),
        extension_priors={
            "rv.jitter": QD(ndist.HalfNormal(1.0), "km/s"),
            "astro.jitter": QD(ndist.HalfNormal(0.1), "mas"),
        },
    )
    astro_fit = Samples(
        nonlinear={"period": Q([300.0], "day"), "jitter": Q([0.02], "mas")},
        linear={"parallax": Q([PARALLAX], "mas")},
        model_type="GaiaAstrometryModel",
    )
    init = NumpyroSampler(prior, joint)._build_init_params(
        joint,
        astro_fit,
        effective_linear_prior=None,
        effective_marginalized_names=None,
        marginalized=True,
        num_chains=1,
    )
    assert float(init["astro.jitter"]) == pytest.approx(0.02)
    assert "jitter" not in init
    assert "rv.jitter" not in init


def test_no_default_tie_without_a_shared_orbit():
    model = JointModel.for_rv_and_gaia(
        components={"rv": RVModel(), "astro": GaiaAstrometryModel()},
        shared_params=("period",),
    )
    assert model.default_linear_priors == {}


def test_tie_pins_only_in_the_source_namespace():
    """A per-component dict must not pin the tied component's same-named param."""
    tie = tied("a.y", scale=rv_semiamp_per_semi_major_axis)  # reads "parallax"
    flat = {"b.x": tie, "a.parallax": FREE_K, "b.parallax": FREE_K}
    assert pinned_linear_names(flat) == {"a.parallax"}
    assert pinned_linear_names({"x": tie, "parallax": FREE_K}) == frozenset()


def test_scale_may_be_a_module_holding_arrays():
    class Constant(eqx.Module):
        value: Q

        def __call__(self, params):
            return self.value

    prior = tied("astro.ti_A", scale=Constant(Q(2.0, "km/(s mas)")))
    assert isinstance(prior, TiedPrior)
    assert float(prior.scale({}).value) == 2.0


def test_scale_accepts_integer_cos_i():
    params = {
        "period": Q(365.25, "day"),
        "eccentricity": 0,
        "cos_i": 0,
        "parallax": Q(1.0, "mas"),
    }
    assert float(rv_semiamp_per_semi_major_axis(params).to("km/(s mas)").value) > 29


def test_component_fit_warm_starts_qualified_base_sites():
    """A joint prior keyed "astro.cos_i" receives the astrometry fit's cos_i."""
    joint = _model()
    prior = HarvPrior(
        nonlinear_priors={
            "period": QD(ndist.Normal(300.0, 5.0), "day"),
            "astro.cos_i": ndist.Uniform(-1.0, 1.0),
        },
        linear_priors=_linear_priors(),
    )
    astro_fit = Samples(
        nonlinear={"period": Q([300.0], "day"), "cos_i": Q([0.4], "")},
        linear={},
        model_type="GaiaAstrometryModel",
    )
    init = NumpyroSampler(prior, joint)._build_init_params(
        joint,
        astro_fit,
        effective_linear_prior=None,
        effective_marginalized_names=None,
        marginalized=True,
        num_chains=1,
    )
    assert float(init["astro.cos_i"]) == pytest.approx(0.4)
    assert "cos_i" not in init
