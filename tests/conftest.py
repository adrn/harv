"""Shared test fixtures for the harv test suite.

Canonical data, priors, and parameter values used across multiple test modules.
Import by placing test files under ``tests/`` -- pytest auto-discovers conftest.py
at each directory level.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxt import Q

from harv.data import RVData
from harv.distributions import QuantityDistribution as QD

# RV data


@pytest.fixture
def rv_data():
    """Small RV dataset (5 observations)."""
    return RVData(
        time=Q.from_([0.0, 50.0, 100.0, 150.0, 200.0], "day"),
        rv=Q.from_([1.0, -2.0, 0.5, 3.0, -1.0], "km/s"),
        rv_err=Q.from_([0.5, 0.5, 0.5, 0.5, 0.5], "km/s"),
    )


@pytest.fixture
def rv_data_primary():
    """RV data for an SB2 primary component (4 observations)."""
    return RVData(
        time=Q.from_([0.0, 50.0, 100.0, 150.0], "day"),
        rv=Q.from_([5.0, -3.0, 4.0, -2.0], "km/s"),
        rv_err=Q.from_([0.5, 0.5, 0.5, 0.5], "km/s"),
    )


@pytest.fixture
def rv_data_secondary():
    """RV data for an SB2 secondary component (4 observations)."""
    return RVData(
        time=Q.from_([0.0, 50.0, 100.0, 150.0], "day"),
        rv=Q.from_([-4.0, 2.5, -3.5, 1.5], "km/s"),
        rv_err=Q.from_([0.3, 0.3, 0.3, 0.3], "km/s"),
    )


# RV priors and parameter values


@pytest.fixture
def rv_linear_prior():
    """Standard RV linear prior: rv_semiamp ~ N(5, 5), v_sys ~ N(0, 10)."""
    return {
        "rv_semiamp": QD(dist.Normal(5.0, 5.0), "km/s"),
        "v_sys": QD(dist.Normal(0.0, 10.0), "km/s"),
    }


@pytest.fixture
def rv_nonlinear_priors():
    """Standard RV nonlinear priors for numpyro models."""
    return {
        "period": QD(dist.Uniform(10.0, 500.0), "day"),
        "eccentricity": dist.Uniform(0.0, 0.9),
        "phase_peri": dist.Uniform(0.0, 1.0),
        "arg_peri": QD(dist.Uniform(0.0, 2 * jnp.pi), "rad"),
    }


@pytest.fixture
def rv_nl_values():
    """Concrete nonlinear parameter values for RV evaluation."""
    return {
        "period": Q.from_(100.0, "day"),
        "eccentricity": 0.3,
        "phase_peri": 0.1,
        "arg_peri": Q.from_(1.0, "rad"),
    }
