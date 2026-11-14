"""Tests for ``harv.stats.KippingDisk``."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS
from scipy import stats

from harv.stats import KippingDisk

KIPPING = stats.beta(0.867, 3.03)


class TestKippingDisk:
    def test_sample_ecc_marginal_is_kipping_beta(self):
        x = KippingDisk().sample(jax.random.key(0), (20_000,))
        ecc = np.sum(np.asarray(x) ** 2, axis=-1)
        omega = np.arctan2(x[:, 1], x[:, 0])
        assert stats.kstest(ecc, KIPPING.cdf).pvalue > 0.01
        assert stats.kstest(omega, stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue > 0.01

    def test_log_prob_is_beta_over_pi(self):
        value = jnp.array([[0.3, 0.4], [0.0, 0.1], [0.8, 0.8], [1.0, 0.0]])
        lp = KippingDisk().log_prob(value)
        ecc = jnp.array([0.25, 0.01])
        expected = dist.Beta(0.867, 3.03).log_prob(ecc) - jnp.log(jnp.pi)
        np.testing.assert_allclose(lp[:2], expected, rtol=1e-12)
        assert jnp.all(jnp.isneginf(lp[2:]))

    def test_origin_value_and_grad_are_finite(self):
        assert jnp.isfinite(KippingDisk().log_prob(jnp.zeros(2)))
        assert jnp.all(jnp.isfinite(jax.grad(KippingDisk().log_prob)(jnp.zeros(2))))

    def test_log_prob_under_vmap(self):
        value = jnp.array([[0.3, 0.4], [-0.1, 0.2]])
        np.testing.assert_allclose(
            jax.vmap(KippingDisk().log_prob)(value), KippingDisk().log_prob(value)
        )

    def test_survives_jit_flatten(self):
        """Pytree round trip keeps the distribution usable under jit."""
        lp = jax.jit(lambda d, v: d.log_prob(v))(KippingDisk(), jnp.array([0.3, 0.4]))
        assert jnp.isfinite(lp)

    def test_bijector_maps_into_support_with_finite_grad(self):
        t = biject_to(KippingDisk().support)
        u = jax.random.normal(jax.random.key(1), (100, 2))

        def logp(u):
            x = t(u)
            return KippingDisk().log_prob(x) + t.log_abs_det_jacobian(u, x)

        assert jnp.all(KippingDisk().support(t(u)))
        assert jnp.all(jnp.isfinite(jax.vmap(jax.grad(logp))(u)))

    def test_nuts_recovers_low_e_tail(self):
        """NUTS reaches the low-e tail (it did not in (e cos w, e sin w))."""

        def model():
            numpyro.sample("x", KippingDisk())

        mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=4000, progress_bar=False)
        mcmc.run(jax.random.key(0))
        ecc = np.sum(np.asarray(mcmc.get_samples()["x"]) ** 2, axis=-1)
        q05 = np.quantile(ecc, 0.05)
        assert 0.5 * KIPPING.ppf(0.05) < q05 < 1.6 * KIPPING.ppf(0.05)
