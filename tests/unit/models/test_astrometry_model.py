"""Unit tests for GaiaAstrometryModel."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from unxt import Q, ustrip

import harv.models as hm
from harv.data import GaiaAstrometryData
from harv.kepler.orbits import thiele_innes_unit
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry


def _make_astro_data(n_obs=50):
    return GaiaAstrometryData(
        time=Q(jnp.linspace(0, 1000, n_obs), "day"),
        al_position=Q(jnp.zeros(n_obs), "mas"),
        al_position_err=Q(jnp.ones(n_obs) * 0.1, "mas"),
        scan_angle=Q(jnp.linspace(0, 2 * jnp.pi, n_obs), "rad"),
        parallax_factor=jnp.ones(n_obs) * 0.5,
        time_ref=Q(0.0, "day"),
    )


def _astro_prior():
    return {
        "ra0": dist.Normal(0.0, 1000.0),
        "dec0": dist.Normal(0.0, 1000.0),
        "pmra": dist.Normal(0.0, 1000.0),
        "pmdec": dist.Normal(0.0, 1000.0),
        "parallax": dist.Normal(0.0, 1000.0),
        "semi_major_axis": dist.Normal(0.0, 1000.0),
    }


def _nl_values():
    return {
        "period": Q(100.0, "day"),
        "eccentricity": 0.3,
        "phase_peri": 0.0,
        "arg_peri": Q(1.0, "rad"),
        "lon_asc_node": Q(2.0, "rad"),
        "cos_i": 0.5,
    }


class TestGaiaAstrometryModelBasic:
    def test_construction(self):
        model = GaiaAstrometryModel()
        assert model.parameterization is not None

    def test_param_names(self):
        model = GaiaAstrometryModel()
        assert set(model._all_nonlinear_names()) == {
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
            "lon_asc_node",
            "cos_i",
        }
        assert set(model._all_linear_names()) == {
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "semi_major_axis",
        }

    def test_obs_unit(self):
        model = GaiaAstrometryModel()
        assert model._obs_unit(_make_astro_data()) == "mas"

    def test_design_matrix_shape(self):
        data = _make_astro_data(n_obs=15)
        model = GaiaAstrometryModel()
        X = model._base_design_matrix(_nl_values(), data)
        assert X.shape == (15, 6)


class TestGaiaAstrometryModelMarginalized:
    def test_marginalized_is_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        ll = model.log_prob(_nl_values(), data, linear_priors=_astro_prior())
        assert jnp.isfinite(ll)

    def test_marginalized_jit(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        nl = _nl_values()

        @jax.jit
        def fn():
            return model.log_prob(nl, data, linear_priors=_astro_prior())

        ll = fn()
        assert jnp.isfinite(ll)


class TestGaiaAstrometryModelSampleConditional:
    def test_sample_returns_all_linear(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        key = jax.random.key(42)
        samples = model.sample_conditional_linear(
            _nl_values(), data, key=key, linear_priors=_astro_prior()
        )
        expected = {"ra0", "dec0", "pmra", "pmdec", "parallax", "semi_major_axis"}
        assert set(samples.keys()) == expected

    def test_sample_values_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel()
        key = jax.random.key(0)
        samples = model.sample_conditional_linear(
            _nl_values(), data, key=key, linear_priors=_astro_prior()
        )
        for name, val in samples.items():
            assert jnp.isfinite(val), f"{name} is not finite"


class TestGaiaAstrometryModelThieleInnes:
    """Tests for GaiaAstrometryModel with ThieleInnesGaiaAstrometry."""

    def _ti_prior(self):
        return {
            "ra0": dist.Normal(0.0, 1000.0),
            "dec0": dist.Normal(0.0, 1000.0),
            "pmra": dist.Normal(0.0, 1000.0),
            "pmdec": dist.Normal(0.0, 1000.0),
            "parallax": dist.Normal(0.0, 1000.0),
            "ti_A": dist.Normal(0.0, 10.0),
            "ti_B": dist.Normal(0.0, 10.0),
            "ti_F": dist.Normal(0.0, 10.0),
            "ti_G": dist.Normal(0.0, 10.0),
        }

    def _ti_nl_values(self):
        return {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
        }

    def test_construction(self):
        p = ThieleInnesGaiaAstrometry()
        model = GaiaAstrometryModel(parameterization=p)
        assert isinstance(model.parameterization, ThieleInnesGaiaAstrometry)

    def test_param_names(self):
        model = GaiaAstrometryModel(parameterization=ThieleInnesGaiaAstrometry())
        assert set(model._all_nonlinear_names()) == {
            "period",
            "eccentricity",
            "phase_peri",
        }
        assert set(model._all_linear_names()) == {
            "ra0",
            "dec0",
            "pmra",
            "pmdec",
            "parallax",
            "ti_A",
            "ti_B",
            "ti_F",
            "ti_G",
        }

    def test_design_matrix_shape(self):
        data = _make_astro_data(n_obs=20)
        model = GaiaAstrometryModel(parameterization=ThieleInnesGaiaAstrometry())
        X = model._base_design_matrix(self._ti_nl_values(), data=data)
        assert X.shape == (20, 9)

    def test_linear_param_units(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(parameterization=ThieleInnesGaiaAstrometry())
        units = model._linear_param_units(data)
        assert units["pmra"] == "mas/yr"
        assert units["ti_A"] == "mas"
        assert units["ti_B"] == "mas"
        assert units["ti_F"] == "mas"
        assert units["ti_G"] == "mas"

    def test_log_prob_finite(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(),
        )
        ll = model.log_prob(
            self._ti_nl_values(), data=data, linear_priors=self._ti_prior()
        )
        assert jnp.isfinite(ll)

    def test_log_prob_jit(self):
        data = _make_astro_data()
        model = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(),
        )
        nl = self._ti_nl_values()

        @jax.jit
        def fn():
            return model.log_prob(nl, data=data, linear_priors=self._ti_prior())

        ll = fn()
        assert jnp.isfinite(ll)

    def test_orbit_contribution_matches_standard(self):
        """TI and Standard models produce identical orbit contributions."""
        a0 = 1.5
        arg_peri, lon_asc_node, cos_i = 0.8, 1.1, 0.6
        A, B, F, G = thiele_innes_unit(
            jnp.cos(arg_peri),
            jnp.sin(arg_peri),
            jnp.cos(lon_asc_node),
            jnp.sin(lon_asc_node),
            cos_i,
        )

        data = _make_astro_data(n_obs=20)

        nl_std = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
            "arg_peri": Q(arg_peri, "rad"),
            "lon_asc_node": Q(lon_asc_node, "rad"),
            "cos_i": cos_i,
        }
        nl_ti = {
            "period": Q(100.0, "day"),
            "eccentricity": 0.3,
            "phase_peri": 0.0,
        }

        model_std = GaiaAstrometryModel()
        model_ti = GaiaAstrometryModel(
            parameterization=ThieleInnesGaiaAstrometry(),
        )

        X_std = model_std._base_design_matrix(nl_std, data=data)
        X_ti = model_ti._base_design_matrix(nl_ti, data=data)

        # Standard orbit column (col 5): scaled by a0 externally
        orbit_std = X_std[:, 5] * a0

        # TI orbit: X_ti cols 5-8 @ [A, B, F, G] * a0
        orbit_ti = (
            X_ti[:, 5] * A * a0
            + X_ti[:, 6] * B * a0
            + X_ti[:, 7] * F * a0
            + X_ti[:, 8] * G * a0
        )

        assert jnp.allclose(orbit_std, orbit_ti, atol=1e-5)


class TestFullNumpyroModelWithTheGaiaDefaults:
    """The `marginalized=False` path under the stock Gaia defaults.

    Three of the six default linear priors are callables, and two of those read
    ``parallax``, whose own prior is a ``HalfNormal``. That combination used to
    raise ``KeyError``: ``parallax`` was drawn into the explicit group while the
    callables were resolved against the *nonlinear* values only, so the value
    they needed was never in the dict they were handed. Now every linear
    parameter is one site, plain priors are drawn before callables, and the
    drawn values are passed along unit-aware.
    """

    @staticmethod
    def _default_prior() -> hm.HarvPrior:
        return hm.StandardGaiaAstrometry().default_prior(
            period_min=Q(100.0, "day"),
            period_max=Q(1000.0, "day"),
            sigma_a0=Q(1.0, "AU"),
            sigma_parallax=Q(10.0, "mas"),
            sigma_pos=Q(100.0, "mas"),
            sigma_vtan=Q(50.0, "km/s"),
        )

    def test_every_linear_param_is_its_own_sample_site(self):
        data = _make_astro_data()
        prior = self._default_prior()
        model_fn = GaiaAstrometryModel().numpyro_model(
            prior.nonlinear_priors, data, prior.linear_priors, marginalized=False
        )
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model_fn).get_trace()

        for name in GaiaAstrometryModel()._all_linear_names():
            assert trace[name]["type"] == "sample", name
        assert "_linear" not in trace
        assert jnp.isfinite(trace["ln_lik"]["fn"].log_prob(trace["ln_lik"]["value"]))

    def test_parallax_is_positive_and_read_by_the_callables(self):
        """`parallax` must be drawn before the priors that scale with it."""
        data = _make_astro_data()
        prior = self._default_prior()
        model_fn = GaiaAstrometryModel().numpyro_model(
            prior.nonlinear_priors, data, prior.linear_priors, marginalized=False
        )
        with handlers.seed(rng_seed=3):
            trace = handlers.trace(model_fn).get_trace()

        parallax = float(trace["parallax"]["value"])
        assert parallax > 0.0
        # sigma_mu = sigma_v0[AU/yr] * parallax, so the proper-motion site's own
        # scale pins that it saw the drawn parallax rather than a default.
        sigma_pm = float(trace["pmra"]["fn"].scale)
        expected = float(ustrip("AU/yr", Q(50.0, "km/s")) * parallax)
        assert sigma_pm == pytest.approx(expected, rel=1e-10)


class TestMarginalizedLikelihoodMatchesDenseOracle:
    """Gaia + jitter marginal likelihood against a dense n x n Gaussian.

    With an explicit parallax and Normal priors on the marginalized columns,
    ``y - X_plx * plx ~ N(X_m mu, diag(err^2 + jitter^2) + X_m S X_m^T)`` and the
    conditional mean of the marginalized params is the usual ridge solution.
    Both are computed here directly, so any change to how harv assembles or
    solves the marginalization must still reproduce them.
    """

    JITTER = 0.07
    PARALLAX = 3.0

    @staticmethod
    def _data() -> GaiaAstrometryData:
        n = 40
        k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
        return GaiaAstrometryData(
            time=Q(jnp.sort(jax.random.uniform(k1, (n,), maxval=1500.0)), "day"),
            al_position=Q(2.0 * jax.random.normal(k2, (n,)), "mas"),
            al_position_err=Q(jnp.linspace(0.05, 0.2, n), "mas"),
            scan_angle=Q(jax.random.uniform(k3, (n,), maxval=2 * jnp.pi), "rad"),
            parallax_factor=jnp.linspace(-0.8, 0.9, n),
            time_ref=Q(700.0, "day"),
        )

    @staticmethod
    def _prior(marg: tuple[str, ...]) -> dict[str, dist.Normal]:
        # Nonzero means, scales within ~10x of each other and of the data: a
        # 1000-mas prior next to 0.1-mas errors makes the dense n x n covariance
        # ill-conditioned enough (~1e8) that the oracle, not harv, loses digits.
        return {n: dist.Normal(0.3 * i - 1.0, 2.0 + i) for i, n in enumerate(marg)}

    def _dense(self, model, nl, data, prior, marg):
        names = model._all_linear_names()
        X = model._full_design_matrix(nl, data)
        y = ustrip("mas", data.al_position) - X[:, names.index("parallax")] * (
            self.PARALLAX
        )
        Xm = X[:, [names.index(n) for n in marg]]
        mu = jnp.array([prior[n].loc for n in marg])
        s2 = jnp.array([prior[n].scale for n in marg]) ** 2
        var = ustrip("mas", data.al_position_err) ** 2 + self.JITTER**2
        cov = jnp.diag(var) + (Xm * s2) @ Xm.T
        lp = dist.MultivariateNormal(Xm @ mu, cov).log_prob(y)
        precision = jnp.diag(1 / s2) + Xm.T @ (Xm / var[:, None])
        mean = jnp.linalg.solve(precision, mu / s2 + Xm.T @ (y / var))
        return lp, mean

    def _check(self, param, nl, prior, marg):
        data = self._data()
        model = GaiaAstrometryModel(
            parameterization=param, extensions=(hm.Jitter(obs_unit="mas"),)
        )
        nl = {**nl, "jitter": self.JITTER}
        explicit = {"parallax": jnp.asarray(self.PARALLAX)}
        lp = model.log_prob(
            nl,
            data,
            linear_priors=prior,
            marginalized_names=marg,
            linear_values=explicit,
        )
        c = model._build_marginalized_linear(nl, marg, explicit, data, prior)
        mean = c.dist.conditional(c.obs).mean

        lp_dense, mean_dense = self._dense(model, nl, data, prior, marg)
        correction = param.linear_log_prior_correction(
            dict(zip(marg, mean_dense, strict=True))
        )
        expected = lp_dense if correction is None else lp_dense + correction
        assert jnp.allclose(mean, mean_dense, rtol=1e-9, atol=1e-9)
        assert lp == pytest.approx(float(expected), rel=1e-10, abs=1e-8)

    def test_standard(self):
        marg = ("ra0", "dec0", "pmra", "pmdec", "semi_major_axis")
        self._check(hm.StandardGaiaAstrometry(), _nl_values(), self._prior(marg), marg)

    @pytest.mark.parametrize("correct", [False, True])
    def test_thiele_innes(self, correct):
        param = (
            ThieleInnesGaiaAstrometry(a_floor=0.01, apply_jacobian_correction=True)
            if correct
            else ThieleInnesGaiaAstrometry()
        )
        marg = ("ra0", "dec0", "pmra", "pmdec", "ti_A", "ti_B", "ti_F", "ti_G")
        nl = {"period": Q(300.0, "day"), "eccentricity": 0.6, "phase_peri": 0.2}
        self._check(param, nl, self._prior(marg), marg)
