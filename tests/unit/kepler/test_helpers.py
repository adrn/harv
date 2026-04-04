"""Unit tests for :mod:`harv.kepler.helpers`."""

import quaxed.numpy as jnp
from unxt import Quantity

from harv.kepler.helpers import compute_true_anomaly_components


class TestComputeTrueAnomalyComponents:
    def test_circular_orbit_at_pericenter(self) -> None:
        """At t=t_peri with e=0, true anomaly = 0: sin_f=0, cos_f=1."""
        sin_f, cos_f = compute_true_anomaly_components(
            time=Quantity(0.0, "yr"),
            period=Quantity(1.0, "yr"),
            eccentricity=0.0,
            t_peri=Quantity(0.0, "yr"),
        )
        assert jnp.allclose(sin_f, 0.0, atol=1e-7)
        assert jnp.allclose(cos_f, 1.0, atol=1e-7)

    def test_circular_orbit_quarter_period(self) -> None:
        """At t=P/4 with e=0, true anomaly = π/2: sin_f=1, cos_f=0."""
        sin_f, cos_f = compute_true_anomaly_components(
            time=Quantity(0.25, "yr"),
            period=Quantity(1.0, "yr"),
            eccentricity=0.0,
            t_peri=Quantity(0.0, "yr"),
        )
        assert jnp.allclose(sin_f, 1.0, atol=1e-6)
        assert jnp.allclose(cos_f, 0.0, atol=1e-6)

    def test_sin_cos_identity(self) -> None:
        """sin²f + cos²f = 1 for eccentric orbits."""
        sin_f, cos_f = compute_true_anomaly_components(
            time=Quantity(0.3, "yr"),
            period=Quantity(1.0, "yr"),
            eccentricity=0.4,
            t_peri=Quantity(0.0, "yr"),
        )
        assert jnp.allclose(sin_f**2 + cos_f**2, 1.0, atol=1e-10)

    def test_at_pericenter_eccentric(self) -> None:
        """At t=t_peri, true anomaly = 0 regardless of eccentricity."""
        sin_f, cos_f = compute_true_anomaly_components(
            time=Quantity(0.0, "yr"),
            period=Quantity(1.0, "yr"),
            eccentricity=0.6,
            t_peri=Quantity(0.0, "yr"),
        )
        assert jnp.allclose(sin_f, 0.0, atol=1e-7)
        assert jnp.allclose(cos_f, 1.0, atol=1e-7)
