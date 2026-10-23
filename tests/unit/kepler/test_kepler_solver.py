"""Tests for the trig-free Kepler solver in harv.kepler._kepler_solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from harv.kepler._kepler_solver import kepler

ECCS = (0.0, 0.1, 0.5, 0.78, 0.9, 0.99, 0.999, 0.9999)


def _mean_anomalies() -> np.ndarray:
    """Wide range, plus the singular corner (small M) and the M ~ 0, pi edges."""
    rng = np.random.default_rng(42)
    special = np.array([0.0, 1e-12, 1e-6, 1e-3, np.pi, np.pi - 1e-9, np.pi + 1e-9])
    return np.concatenate(
        [
            rng.uniform(-900, 300, 2000),
            rng.uniform(0, 0.3, 500),
            -rng.uniform(0, 0.3, 500),
            special,
            -special,
            2 * np.pi * np.arange(-3, 4),
        ]
    )


def _newton(M: np.ndarray, e: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent float64 reference: Newton from Danby's start, run to convergence."""
    M = np.remainder(M, 2 * np.pi)
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(100):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    den = 1 - e * np.cos(E)
    return np.sqrt(1 - e * e) * np.sin(E) / den, (np.cos(E) - e) / den


def _tolerance(e: float) -> float:
    # df/dM ~ (1+e)^2 / (1-e^2)^1.5 near pericenter amplifies the round-off in M.
    return 1e-13 if e <= 0.9 else 5e-15 / (1 - e) ** 1.5


@pytest.mark.parametrize("e", ECCS)
def test_matches_converged_newton(e: float) -> None:
    M = _mean_anomalies()
    sin_f, cos_f = kepler(jnp.asarray(M), e)
    sin_ref, cos_ref = _newton(M, e)
    np.testing.assert_allclose(sin_f, sin_ref, rtol=0, atol=_tolerance(e))
    np.testing.assert_allclose(cos_f, cos_ref, rtol=0, atol=_tolerance(e))


@pytest.mark.parametrize("e", ECCS)
def test_matches_jaxoplanet(e: float) -> None:
    jaxoplanet_kepler = pytest.importorskip("jaxoplanet.core.kepler").kepler
    M = jnp.asarray(_mean_anomalies())
    ours = kepler(M, e)
    theirs = jaxoplanet_kepler(M, e)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=_tolerance(e))


@pytest.mark.parametrize("e", ECCS)
def test_satisfies_keplers_equation(e: float) -> None:
    """Map f back to E and check E - e sin E = M (mod 2 pi)."""
    M = _mean_anomalies()
    sin_f, cos_f = (np.asarray(x) for x in kepler(jnp.asarray(M), e))
    half_f = 0.5 * np.arctan2(sin_f, cos_f)
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(half_f), np.sqrt(1 + e) * np.cos(half_f))
    x = E - e * np.sin(E) - M
    resid = np.arctan2(np.sin(x), np.cos(x))
    assert np.max(np.abs(resid)) < 5e-13


def test_unit_circle_and_circular_orbit() -> None:
    M = jnp.asarray(_mean_anomalies())
    sin_f, cos_f = kepler(M, 0.0)
    np.testing.assert_allclose(sin_f, jnp.sin(M), atol=1e-13)
    np.testing.assert_allclose(cos_f, jnp.cos(M), atol=1e-13)
    sin_f, cos_f = kepler(M, 0.7)
    np.testing.assert_allclose(sin_f**2 + cos_f**2, 1.0, atol=1e-14)


def test_scalar_mean_anomaly() -> None:
    sin_f, cos_f = kepler(1.0, 0.3)
    assert sin_f.shape == cos_f.shape == ()
    np.testing.assert_allclose((sin_f, cos_f), _newton(np.array(1.0), 0.3), atol=1e-14)


def test_rejects_batched_eccentricity() -> None:
    with pytest.raises(ValueError, match="scalar eccentricity"):
        kepler(jnp.zeros(3), jnp.array([0.1, 0.2, 0.3]))


def test_jit_and_vmap_over_eccentricity() -> None:
    M = jnp.linspace(-10, 10, 50)
    eccs = jnp.array([0.0, 0.3, 0.95])
    sin_f, cos_f = jax.jit(jax.vmap(kepler, in_axes=(None, 0)))(M, eccs)
    assert sin_f.shape == (3, 50)
    for i, e in enumerate(eccs):
        np.testing.assert_allclose(sin_f[i], kepler(M, e)[0], atol=1e-15)
        np.testing.assert_allclose(cos_f[i], kepler(M, e)[1], atol=1e-15)


@pytest.mark.parametrize("e", [0.0, 0.3, 0.9])
def test_grad_matches_finite_differences(e: float) -> None:
    M = jnp.array([0.05, 1.0, 2.5, 3.0, 4.0, 6.0])
    check_grads(kepler, (M, e), order=1, modes=("fwd", "rev"), eps=1e-6)


def test_integer_eccentricity_is_differentiable() -> None:
    M = jnp.linspace(0.1, 6.0, 5)
    jac = jax.jacfwd(lambda m: kepler(m, 0)[0])(M)
    np.testing.assert_allclose(jnp.diag(jac), jnp.cos(M), atol=1e-12)
