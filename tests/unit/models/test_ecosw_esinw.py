"""Tests for the EcoswEsinwRV parameterization and its use in RVModel."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

# quaxed for anything touching Quantity: `Samples.__getitem__` returns Q, and
# jax.numpy rejects it.
import quaxed.numpy as qnp
from unxt import Q

import harv.models as hm
from harv.data import RVData
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV
from harv.models.rv import RVModel
from harv.samplers.numpyro import NumpyroSampler, _unconstrain_init_params
from harv.samplers.rejection import RejectionSampler
from harv.simulate.rv import simulate_rv_sb1_data


def _make_rv_data(n_obs=20):
    return RVData(
        time=Q(jnp.linspace(0, 100, n_obs), "day"),
        rv=Q(jnp.zeros(n_obs), "km/s"),
        rv_err=Q(jnp.ones(n_obs) * 0.1, "km/s"),
    )


def _rv_prior():
    return {
        "rv_semiamp": dist.Normal(0.0, 100.0),
        "v_sys": dist.Normal(0.0, 100.0),
    }


# Reference orbital parameters (standard parameterization)
_ECC = 0.3
_ARG_PERI = 1.0  # rad
_ECOSW = _ECC * float(jnp.cos(_ARG_PERI))
_ESINW = _ECC * float(jnp.sin(_ARG_PERI))
_PHASE_PERI = 0.3
_MEAN_LON = Q(_ARG_PERI - 2 * float(jnp.pi) * _PHASE_PERI, "rad")


class TestEcoswEsinwRVParameterization:
    """Unit tests for the EcoswEsinwRV parameterization object."""

    def test_params_names(self):
        p = EcoswEsinwRV()
        names = [pi.name for pi in p.params()]
        assert names == [
            "period",
            "ecosw",
            "esinw",
            "mean_longitude",
            "rv_semiamp",
            "v_sys",
        ]

    def test_nonlinear_linear_split(self):
        p = EcoswEsinwRV()
        nl = p.nonlinear_params()
        lin = p.linear_params()
        assert len(nl) == 4
        assert len(lin) == 2
        assert all(not pi.linear for pi in nl)
        assert all(pi.linear for pi in lin)

    def test_eccentricity(self):
        p = EcoswEsinwRV()
        nl = {"ecosw": _ECOSW, "esinw": _ESINW}
        ecc = p.eccentricity(nl)
        assert jnp.allclose(ecc, _ECC, atol=1e-6)

    def test_eccentricity_zero(self):
        p = EcoswEsinwRV()
        nl = {"ecosw": 0.0, "esinw": 0.0}
        assert float(p.eccentricity(nl)) == 0.0

    def test_strip_nl_for_design(self):
        p = EcoswEsinwRV()
        nl = {"ecosw": 0.3, "esinw": 0.1, "other": "kept"}
        stripped = p.strip_nonlinear_for_design(nl)
        assert "ecosw" in stripped
        assert "esinw" in stripped
        assert "other" in stripped

    def test_design_matrix_shape(self):
        p = EcoswEsinwRV()
        n_obs = 10
        key = jax.random.key(42)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))
        nonlinear_values = {"ecosw": _ECOSW, "esinw": _ESINW}
        X = p.design_matrix(sin_f, cos_f, nonlinear_values)
        assert X.shape == (n_obs, 2)

    def test_design_matrix_second_col_ones(self):
        p = EcoswEsinwRV()
        n_obs = 5
        sin_f = jnp.zeros(n_obs)
        cos_f = jnp.ones(n_obs)
        nonlinear_values = {"ecosw": 0.0, "esinw": 0.0}
        X = p.design_matrix(sin_f, cos_f, nonlinear_values)
        assert jnp.allclose(X[:, 1], 1.0)

    def test_design_matrix_matches_standard(self):
        """EcoswEsinwRV must produce the same design matrix as StandardRV."""
        std = StandardRV()
        eco = EcoswEsinwRV()
        n_obs = 20
        key = jax.random.key(7)
        sin_f = jax.random.normal(key, (n_obs,))
        cos_f = jax.random.normal(key, (n_obs,))

        dm_std = std.design_matrix(
            sin_f, cos_f, {"eccentricity": _ECC, "arg_peri": _ARG_PERI}
        )
        dm_eco = eco.design_matrix(sin_f, cos_f, {"ecosw": _ECOSW, "esinw": _ESINW})
        assert jnp.allclose(dm_std, dm_eco, atol=1e-6)

    def test_static_registered(self):
        p = EcoswEsinwRV()
        leaves, _ = jax.tree.flatten(p)
        assert leaves == []

    def test_design_matrix_jit(self):
        p = EcoswEsinwRV()
        sin_f = jnp.array([0.1, 0.2, 0.3])
        cos_f = jnp.array([0.9, 0.8, 0.7])
        nonlinear_values = {"ecosw": 0.2, "esinw": 0.1}

        @jax.jit
        def fn(sf, cf):
            return p.design_matrix(sf, cf, nonlinear_values)

        X = fn(sin_f, cos_f)
        assert X.shape == (3, 2)


class TestStandardRVHelpers:
    """Verify the new helper methods on StandardRV too."""

    def test_eccentricity(self):
        p = StandardRV()
        nl = {"eccentricity": 0.4}
        assert float(p.eccentricity(nl)) == 0.4

    def test_strip_nl_for_design(self):
        p = StandardRV()
        nl = {
            "eccentricity": Q(0.3, ""),
            "arg_peri": Q(1.0, "rad"),
            "period": Q(100.0, "day"),
        }
        stripped = p.strip_nonlinear_for_design(nl)
        # Should be plain floats after stripping
        assert not hasattr(stripped["eccentricity"], "unit")
        assert not hasattr(stripped["arg_peri"], "unit")


class TestRVModelWithEcoswEsinw:
    """RVModel constructed with EcoswEsinwRV parameterization."""

    def test_construction(self):
        model = RVModel(parameterization=EcoswEsinwRV())
        assert isinstance(model.parameterization, EcoswEsinwRV)

    def test_param_names(self):
        model = RVModel(parameterization=EcoswEsinwRV())
        assert set(model._all_nonlinear_names()) == {
            "period",
            "ecosw",
            "esinw",
            "mean_longitude",
        }
        assert set(model._all_linear_names()) == {"rv_semiamp", "v_sys"}

    def test_base_design_matrix_shape(self):
        data = _make_rv_data(n_obs=10)
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "mean_longitude": _MEAN_LON,
        }
        X = model._base_design_matrix(nl, data)
        assert X.shape == (10, 2)

    def test_explicit_is_finite(self):
        data = _make_rv_data()
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "mean_longitude": _MEAN_LON,
        }
        linear = {
            "rv_semiamp": jnp.float32(5.0),
            "v_sys": jnp.float32(0.0),
        }
        ll = model.log_prob(nl, data, linear_values=linear)
        assert jnp.isfinite(ll)

    def test_marginalized_is_finite(self):
        data = _make_rv_data()
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "mean_longitude": _MEAN_LON,
        }
        ll = model.log_prob(nl, data, linear_priors=_rv_prior())
        assert jnp.isfinite(ll)

    def test_jit(self):
        data = _make_rv_data()
        model = RVModel(parameterization=EcoswEsinwRV())
        nl = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "mean_longitude": _MEAN_LON,
        }

        @jax.jit
        def fn():
            return model.log_prob(nl, data, linear_priors=_rv_prior())

        ll = fn()
        assert jnp.isfinite(ll)


class TestNumericalEquivalence:
    """EcoswEsinw model must match StandardRV for equivalent parameters."""

    def test_explicit_equivalence(self):
        data = _make_rv_data(n_obs=30)

        std_model = RVModel()
        eco_model = RVModel(parameterization=EcoswEsinwRV())

        nl_std = {
            "period": Q(100.0, "day"),
            "eccentricity": _ECC,
            "phase_peri": _PHASE_PERI,
            "arg_peri": _ARG_PERI,
        }
        nl_eco = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "mean_longitude": _MEAN_LON,
        }
        linear = {
            "rv_semiamp": jnp.float32(5.0),
            "v_sys": jnp.float32(0.0),
        }

        ll_std = std_model.log_prob(nl_std, data, linear_values=linear)
        ll_eco = eco_model.log_prob(nl_eco, data, linear_values=linear)

        assert jnp.allclose(ll_std, ll_eco, atol=1e-6), (
            f"std={float(ll_std)}, eco={float(ll_eco)}"
        )

    def test_marginalized_equivalence(self):
        data = _make_rv_data(n_obs=30)
        prior = _rv_prior()

        std_model = RVModel()
        eco_model = RVModel(parameterization=EcoswEsinwRV())

        nl_std = {
            "period": Q(100.0, "day"),
            "eccentricity": _ECC,
            "phase_peri": _PHASE_PERI,
            "arg_peri": _ARG_PERI,
        }
        nl_eco = {
            "period": Q(100.0, "day"),
            "ecosw": _ECOSW,
            "esinw": _ESINW,
            "mean_longitude": _MEAN_LON,
        }

        ll_std = std_model.log_prob(nl_std, data, linear_priors=prior)
        ll_eco = eco_model.log_prob(nl_eco, data, linear_priors=prior)

        assert jnp.allclose(ll_std, ll_eco, atol=1e-6), (
            f"std={float(ll_std)}, eco={float(ll_eco)}"
        )


class TestEcoswEsinwDefaultPriorIsEvaluable:
    """The default prior must survive being *evaluated*, not just constructed.

    ``EcoswEsinwRV.default_prior`` gives ``rv_semiamp`` a
    :class:`~harv.models.priors.callables.PeriodDependentKPrior`, which reads
    ``params["eccentricity"]`` -- a key this parameterization does not have, since it
    carries ``(ecosw, esinw)``. The existing prior tests only inspect keys, so the
    whole combination raised ``KeyError: 'eccentricity'`` on every sampler run.
    """

    @staticmethod
    def _prior_and_model(**overrides: Any) -> tuple[Any, RVModel]:
        param = hm.EcoswEsinwRV()
        prior = param.default_prior(
            period_min=Q(10.0, "day"),
            period_max=Q(3000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
            **overrides,
        )
        return prior, RVModel(parameterization=param)

    def test_derived_eccentricity_matches_ecosw_esinw(self):
        """The parameterization can recover eccentricity from its own params."""
        param = hm.EcoswEsinwRV()
        nl = {"ecosw": jnp.asarray(0.3), "esinw": jnp.asarray(0.4)}
        assert param.derived_eccentricity(nl) == pytest.approx(0.5)

    def test_base_parameterizations_derive_nothing(self):
        """Parameterizations that carry eccentricity (or have none) return None."""
        assert hm.StandardRV().derived_eccentricity({}) is None
        assert hm.FourierRV(n_terms=2).derived_eccentricity({}) is None

    @pytest.mark.filterwarnings("ignore:Under-resolved rejection run:UserWarning")
    def test_rejection_sampler_runs(self):
        """The regression: this raised KeyError('eccentricity') before the fix."""
        data, _ = simulate_rv_sb1_data(seed=42, n_obs=16)
        prior, model = self._prior_and_model()
        sampler = RejectionSampler(prior, model, batch_size=2000)
        samples = sampler.run_with_samples(
            data,
            prior.sample(2000, key=jr.key(0), model=model),
            top_k=8,
            key=jax.random.key(0),
        )
        assert samples.n_samples == 8
        assert jnp.isfinite(samples.metadata["max_ln_likelihood"])

    def test_default_prior_stays_in_unit_disk(self):
        """The joint UnitDisk default never draws an unbound orbit."""
        prior, model = self._prior_and_model()
        cache = prior.sample(4000, key=jr.key(0), model=model)
        ecc = qnp.sqrt(cache["ecosw"] ** 2 + cache["esinw"] ** 2)
        assert qnp.all(ecc < 1.0)

    @pytest.mark.filterwarnings("ignore:Under-resolved rejection run:UserWarning")
    def test_square_override_propagates_nan(self):
        """Independent Uniform(-1, 1) overrides escape the disk and poison the max.

        Pins the sharp edge documented in docs/sharp-bits.md so a change in either
        direction is visible.
        """
        data, _ = simulate_rv_sb1_data(seed=42, n_obs=16)
        prior, model = self._prior_and_model(
            ecosw=dist.Uniform(-1.0, 1.0), esinw=dist.Uniform(-1.0, 1.0)
        )
        cache = prior.sample(2000, key=jr.key(0), model=model)
        ecc = qnp.sqrt(cache["ecosw"] ** 2 + cache["esinw"] ** 2)
        assert qnp.any(ecc >= 1.0), "expected the square prior to escape the unit disk"

        sampler = RejectionSampler(prior, model, batch_size=2000)
        without = sampler.run_with_samples(data, cache, top_k=8, key=jax.random.key(0))
        assert not jnp.isfinite(without.metadata["max_ln_likelihood"])


@pytest.mark.filterwarnings("ignore:Under-resolved rejection run:UserWarning")
class TestEcoswEsinwNumpyro:
    """The joint (ecosw, esinw) prior on the numpyro path: one site, named values."""

    @pytest.fixture(scope="class")
    def setup(self):
        data, _ = simulate_rv_sb1_data(seed=42, n_obs=16)
        param = hm.EcoswEsinwRV()
        prior = param.default_prior(
            period_min=Q(10.0, "day"),
            period_max=Q(3000.0, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(50.0, "km/s"),
        )
        model = RVModel(parameterization=param)
        init = RejectionSampler(prior, model).run(
            data, key=jr.key(0), n_prior_samples=4000, top_k=2
        )
        return data, prior, model, init

    @staticmethod
    def _check(samples: Any) -> None:
        # Closed disk: a constrained optimizer can saturate the bijector onto
        # the boundary, but never past it.
        ecosw, esinw = samples["ecosw"], samples["esinw"]
        assert "ecosw,esinw" not in samples.nonlinear
        assert bool(qnp.all(ecosw**2 + esinw**2 <= 1))
        assert bool(jnp.all(jnp.isfinite(samples.ln_prior)))

    def test_partial_warm_start_raises(self, setup):
        data, prior, model, init = setup
        nonlinear = {k: v for k, v in init.nonlinear.items() if k != "esinw"}
        partial = eqx.tree_at(lambda s: s.nonlinear, init, nonlinear)
        with pytest.raises(ValueError, match=r"missing \['esinw'\]"):
            NumpyroSampler(prior, model).run(
                data, init_samples=partial, key=jr.key(1), num_chains=1
            )

    def test_nuts_warm_start(self, setup):
        data, prior, model, init = setup
        post = NumpyroSampler(prior, model).run(
            data,
            init_samples=init,
            key=jr.key(1),
            num_chains=1,
            num_warmup=200,
            num_samples=100,
            return_logprobs=True,
        )
        assert post["ecosw"].shape == (100,)
        self._check(post)
        # Warm-started at the right mode: recovers the injected e ~ 0.05.
        ecc = qnp.sqrt(post["ecosw"] ** 2 + post["esinw"] ** 2)
        assert float(qnp.median(ecc)) < 0.2

    def test_optimize(self, setup):
        data, prior, model, init = setup
        refined = NumpyroSampler(prior, model).optimize(init, data, key=jr.key(2))
        assert refined["ecosw"].shape == (init.n_samples,)
        self._check(refined)


def test_warm_start_on_disk_edge_is_finite():
    # A MAP on the rim (e = 1) unconstrains to inf; it must be pulled finite.
    prior, _ = TestEcoswEsinwDefaultPriorIsEvaluable._prior_and_model()
    init = {"ecosw,esinw": jnp.array([0.6, 0.8])}
    out = _unconstrain_init_params(init, prior)
    assert bool(jnp.all(jnp.isfinite(out["ecosw,esinw"])))


def test_rim_point_unconstraining_to_nan_is_clamped():
    # (1, 0) unconstrains to [inf, nan] (0 / sqrt(0)); both must come back finite.
    prior, _ = TestEcoswEsinwDefaultPriorIsEvaluable._prior_and_model()
    out = _unconstrain_init_params({"ecosw,esinw": jnp.array([1.0, 0.0])}, prior)
    assert bool(jnp.all(jnp.isfinite(out["ecosw,esinw"])))


def test_non_finite_init_is_not_hidden():
    prior, _ = TestEcoswEsinwDefaultPriorIsEvaluable._prior_and_model()
    out = _unconstrain_init_params({"ecosw,esinw": jnp.array([jnp.nan, 0.0])}, prior)
    assert not bool(jnp.all(jnp.isfinite(out["ecosw,esinw"])))


def test_phase_peri_override_raises():
    # EcoswEsinwRV samples mean_longitude; a stale phase_peri= must not be dropped.
    with pytest.raises(TypeError, match="mean_longitude"):
        TestEcoswEsinwDefaultPriorIsEvaluable._prior_and_model(
            phase_peri=dist.Uniform(0.0, 0.5)
        )
