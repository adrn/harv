"""Unit tests for harv.stats.grid_density.LogGridDensity."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from unxt import Q, ustrip

import harv.models as hm
from harv.distributions import QD
from harv.samplers import RejectionSampler
from harv.simulate import simulate_rv_sb1_data
from harv.stats import LogGridDensity


def _random_grid_density(seed: int, n: int) -> LogGridDensity:
    rng = np.random.default_rng(seed)
    ln_grid = jnp.asarray(np.sort(rng.uniform(-1.0, 6.0, size=n)))
    log_density = jnp.asarray(rng.normal(0.0, 2.0, size=n))
    return LogGridDensity(ln_grid, log_density)


class TestConstruction:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal shape"):
            LogGridDensity(jnp.zeros(3), jnp.zeros(4))

    def test_too_few_knots_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            LogGridDensity(jnp.zeros(1), jnp.zeros(1))

    def test_non_increasing_grid_raises(self):
        # eqx.error_if raises eagerly for concrete inputs:
        with pytest.raises(Exception, match="strictly increasing"):
            LogGridDensity(jnp.array([0.0, 2.0, 1.0]), jnp.zeros(3))


class TestDensity:
    @pytest.mark.parametrize(
        ("seed", "n"), [(0, 2), (1, 3), (2, 5), (3, 17), (4, 64), (5, 128)]
    )
    def test_normalization(self, seed: int, n: int):
        """int p(x) dx = 1 for random knot configurations."""
        d = _random_grid_density(seed, n)
        u = jnp.linspace(d.ln_grid[0], d.ln_grid[-1], 20_001)
        x = jnp.exp(u)
        # Integrate in ln-space: int p(x) dx = int p(x) x d(ln x)
        integrand = jnp.exp(d.log_prob(x)) * x
        total = jnp.trapezoid(integrand, u)
        assert jnp.isclose(total, 1.0, atol=1e-4)

    def test_flat_density_matches_loguniform(self):
        low, high = 2.0, 500.0
        d = LogGridDensity(jnp.log(jnp.array([low, 10.0, high])), jnp.full(3, -3.2))
        lu = dist.LogUniform(low, high)
        x = jnp.geomspace(low * 1.001, high * 0.999, 101)
        assert jnp.allclose(d.log_prob(x), lu.log_prob(x), atol=1e-6)
        assert jnp.allclose(d.cdf(x), lu.cdf(x), atol=1e-6)

    def test_log_prob_outside_support(self):
        d = _random_grid_density(0, 8)
        lo = float(d.low)
        hi = float(d.high)
        vals = jnp.array([-1.0, 0.0, lo * 0.5, hi * 2.0])
        assert bool(jnp.all(jnp.isneginf(d.log_prob(vals))))
        assert bool(jnp.all(jnp.isneginf(d.log_prob_ln(vals))))
        assert jnp.allclose(d.cdf(jnp.array([0.0, lo * 0.5])), 0.0)
        assert jnp.allclose(d.cdf(jnp.array([hi * 2.0])), 1.0)

    def test_log_prob_ln_identity(self):
        d = _random_grid_density(3, 16)
        x = jnp.exp(jnp.linspace(d.ln_grid[0] + 1e-3, d.ln_grid[-1] - 1e-3, 57))
        assert jnp.allclose(d.log_prob_ln(x), d.log_prob(x) + jnp.log(x), atol=1e-6)

    def test_knot_values(self):
        """log_prob at the knots equals the normalized knot density."""
        d = _random_grid_density(7, 12)
        expected = jnp.log(d._rho) - d.ln_grid
        assert jnp.allclose(d.log_prob(jnp.exp(d.ln_grid)), expected, atol=1e-5)

    def test_mean_matches_numerical(self):
        d = _random_grid_density(11, 10)
        u = jnp.linspace(d.ln_grid[0], d.ln_grid[-1], 40_001)
        x = jnp.exp(u)
        numerical = jnp.trapezoid(x * jnp.exp(d.log_prob(x)) * x, u)
        assert jnp.isclose(d.mean, numerical, rtol=1e-4)


class TestSampling:
    @pytest.mark.parametrize(
        ("seed", "n"), [(0, 2), (1, 3), (2, 5), (3, 17), (4, 64), (5, 128)]
    )
    def test_cdf_icdf_roundtrip(self, seed: int, n: int):
        d = _random_grid_density(seed, n)
        q = jnp.linspace(0.0, 1.0, 101)
        assert jnp.allclose(d.cdf(d.icdf(q)), q, atol=1e-5)

    def test_icdf_endpoints(self):
        d = _random_grid_density(5, 20)
        assert jnp.isclose(d.icdf(0.0), d.low, rtol=1e-6)
        assert jnp.isclose(d.icdf(1.0), d.high, rtol=1e-6)

    def test_samples_in_support(self):
        d = _random_grid_density(1, 30)
        x = d.sample(jr.key(0), (10_000,))
        assert x.shape == (10_000,)
        assert bool(jnp.all((x >= d.low) & (x <= d.high)))
        assert bool(jnp.all(jnp.isfinite(d.log_prob(x))))

    def test_sample_histogram_matches_cdf(self):
        """Empirical CDF at the knots matches the analytic CDF."""
        d = _random_grid_density(2, 9)
        x = d.sample(jr.key(1), (200_000,))
        knot_x = jnp.exp(d.ln_grid[1:-1])
        empirical = jnp.mean(x[None, :] <= knot_x[:, None], axis=1)
        assert jnp.allclose(empirical, d.cdf(knot_x), atol=5e-3)


class TestJax:
    def test_log_prob_jit_and_vmap(self):
        d = _random_grid_density(4, 15)
        x = d.sample(jr.key(2), (64,))
        eager = d.log_prob(x)
        jitted = jax.jit(d.log_prob)(x)
        vmapped = jax.vmap(d.log_prob)(x)
        assert jnp.allclose(eager, jitted)
        assert jnp.allclose(eager, vmapped)

    def test_sample_under_jit(self):
        d = _random_grid_density(4, 15)

        @jax.jit
        def draw(key):
            return d.sample(key, (100,))

        x = draw(jr.key(3))
        assert bool(jnp.all((x >= d.low) & (x <= d.high)))

    def test_pytree_roundtrip(self):
        d = _random_grid_density(6, 11)
        d2 = jax.tree.map(lambda a: a, d)
        x = jnp.exp(jnp.linspace(d.ln_grid[0], d.ln_grid[-1], 33))
        assert jnp.allclose(d.log_prob(x), d2.log_prob(x))

    def test_equal_knot_count_same_tree_structure(self):
        """Two instances with equal knot counts share a pytree structure."""
        d1 = _random_grid_density(0, 25)
        d2 = _random_grid_density(99, 25)
        s1 = jax.tree_util.tree_structure(d1)
        s2 = jax.tree_util.tree_structure(d2)
        assert s1 == s2


class TestPriorIntegration:
    def test_qd_wrapped_sampling_units(self):
        d = _random_grid_density(8, 10)
        qd = QD(d, "day")
        x = qd.sample(jr.key(4), (16,))
        assert str(x.unit) == "d"
        lp = qd.log_prob(x)
        assert bool(jnp.all(jnp.isfinite(lp)))

    def test_default_prior_period_override(self):
        """QD(LogGridDensity) drops into StandardRV().default_prior(period=...)."""
        d = LogGridDensity(
            jnp.log(jnp.array([20.0, 80.0, 300.0])), jnp.array([0.0, 3.0, 0.0])
        )
        prior = hm.StandardRV().default_prior(
            period=QD(d, "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
        )
        samples = prior.sample_nonlinear(jr.key(5), n_samples=256)
        periods = samples["period"]
        assert periods.shape == (256,)
        assert bool(jnp.all((periods >= 20.0) & (periods <= 300.0)))

    def test_rejection_sampler_smoke(self):
        """A hand-made grid prior runs end-to-end through RejectionSampler."""
        data, _ = simulate_rv_sb1_data(
            seed=13,
            n_obs=30,
            period=Q(100.0, "day"),
            eccentricity=0.2,
            rv_semiamp=Q(8.0, "km/s"),
            rv_err=Q(0.1, "km/s"),
        )
        # Grid prior peaked near the true period, with broad support:
        ln_grid = jnp.log(jnp.geomspace(20.0, 500.0, 101))
        log_density = -0.5 * ((ln_grid - jnp.log(100.0)) / 0.1) ** 2
        prior = hm.StandardRV().default_prior(
            period=QD(LogGridDensity(ln_grid, log_density), "day"),
            sigma_K0=Q(30.0, "km/s"),
            sigma_v0=Q(30.0, "km/s"),
        )
        sampler = RejectionSampler(prior, hm.RVModel())
        samples = sampler.run(data, n_prior_samples=20_000, seed=1)
        assert samples.n_samples > 0
        med = ustrip("day", samples.median("period"))
        assert abs(float(med) - 100.0) < 20.0


class TestArgConstraints:
    """Zero-density knots are a documented input, so the constraint must pass them."""

    def test_minus_inf_knots_pass_validation(self):
        ln_grid = jnp.log(jnp.array([1.0, 2.0, 4.0, 8.0]))
        log_density = jnp.array([-jnp.inf, 0.0, 0.5, -jnp.inf])
        d = LogGridDensity(ln_grid, log_density, validate_args=True)
        # The zero-density knots really are zero density.
        assert np.isneginf(float(d.log_prob(1.0)))
        assert np.isfinite(float(d.log_prob(2.0)))

    @pytest.mark.parametrize("bad", [jnp.inf, jnp.nan])
    def test_constraint_rejects_plus_inf_and_nan(self, bad):
        constraint = LogGridDensity.arg_constraints["log_density"]
        assert not bool(constraint(jnp.array([0.0, bad, 0.5])))

    def test_constraint_accepts_minus_inf_and_keeps_event_dim(self):
        constraint = LogGridDensity.arg_constraints["log_density"]
        assert bool(constraint(jnp.array([0.0, -jnp.inf, 0.5])))
        assert constraint.event_dim == 1

    def test_builder_output_validates(self):
        """peak_period_prior with floor=0 produces -inf knots; they must validate."""
        ln_grid = jnp.log(jnp.geomspace(10.0, 1000.0, 64))
        density = np.zeros(64)
        density[20:30] = 1.0
        with np.errstate(divide="ignore"):
            log_density = jnp.asarray(np.log(density))
        d = LogGridDensity(ln_grid, log_density, validate_args=True)
        assert np.isfinite(float(d.log_prob(float(np.exp(ln_grid[25])))))
