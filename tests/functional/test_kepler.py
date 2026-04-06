"""Functional tests for :mod:`harv.kepler` — end-to-end workflows."""

import jax
import quaxed.numpy as jnp
from unxt import Quantity, ustrip

from harv.kepler import KeplerianBody, KeplerianOrientation, TwoBodySystem


def test_full_orbit_workflow() -> None:
    """End-to-end: build system from masses, compute positions and velocities."""
    m_star = Quantity(1.0, "Msun")
    m_planet = Quantity(1.0, "Mjup")
    period = Quantity(1.0, "yr")

    orientation = KeplerianOrientation.from_angles(
        arg_peri=Quantity(0.5, "rad"),
        lon_asc_node=Quantity(1.2, "rad"),
        inclination=Quantity(0.3, "rad"),
    )

    companion = KeplerianBody.from_masses(
        period=period,
        eccentricity=0.1,
        m_companion=m_planet,
        m_primary=m_star,
        t_peri=Quantity(0.0, "yr"),
        orientation=orientation,
    )
    system = TwoBodySystem(m_primary=m_star, companion=companion)

    # Verify basic properties
    assert system.n_bodies == 2
    assert ustrip("Msun", system.m_total) > ustrip("Msun", m_star)

    # Compute positions/velocities at multiple times
    times = Quantity(jnp.linspace(0.0, 1.0, 20), "yr")

    @jax.vmap
    def compute_state(t):
        r0 = system.position_barycentric(t, 0)
        r1 = system.position_barycentric(t, 1)
        v0 = system.velocity_barycentric(t, 0)
        v1 = system.velocity_barycentric(t, 1)
        return r0, r1, v0, v1

    r0, _r1, v0, v1 = compute_state(times)
    assert r0.shape == (20, 3)

    # Check momentum conservation at every time step
    p_total = system.m_primary * v0 + system.m_companion * v1
    assert jnp.allclose(
        ustrip("Msun AU / yr", p_total),
        jnp.zeros((20, 3)),
        atol=1e-10,
    )


def test_jit_full_pipeline() -> None:
    """The full pipeline (build + compute) runs under jax.jit."""
    m_star = Quantity(1.0, "Msun")

    companion = KeplerianBody.from_masses(
        period=Quantity(1.0, "yr"),
        eccentricity=0.2,
        m_companion=Quantity(1.0, "Mjup"),
        m_primary=m_star,
        t_peri=Quantity(0.0, "yr"),
    )
    system = TwoBodySystem(m_primary=m_star, companion=companion)

    @jax.jit
    def get_rv(system, t):
        v = system.velocity_barycentric(t, 0)
        return v[2]  # radial (z) component

    rv = get_rv(system, Quantity(0.25, "yr"))
    assert rv.shape == ()
