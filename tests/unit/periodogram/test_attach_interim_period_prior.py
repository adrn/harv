"""Unit tests for harv.periodogram.attach_interim_period_prior."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from unxt import Q, ustrip

import harv.periodogram as hp
from harv.distributions import QD
from harv.samplers import Samples
from harv.samplers.samples import pad_and_stack_samples
from harv.stats import LogGridDensity


def _make_samples(periods) -> Samples:
    n = len(periods)
    return Samples(
        nonlinear={
            "period": Q(jnp.asarray(periods), "day"),
            "eccentricity": Q(jnp.linspace(0.0, 0.5, n), ""),
            "phase_peri": Q(jnp.linspace(0.1, 0.9, n), ""),
        },
        linear={"rv_semiamp": Q(jnp.ones(n), "km/s")},
        data_type="RVModel",
        metadata={"t_ref": 0.0, "t_ref_unit": "day"},
    )


def _grid_prior() -> QD:
    ln_grid = jnp.log(jnp.geomspace(10.0, 1000.0, 32))
    log_density = -0.5 * ((ln_grid - jnp.log(100.0)) / 0.3) ** 2
    return QD(LogGridDensity(ln_grid, log_density), "day")


class TestAttach:
    def test_value_and_unit(self):
        samples = _make_samples([50.0, 100.0, 400.0])
        prior = _grid_prior()
        out = hp.attach_interim_period_prior(samples, prior)

        col = out[hp.LN_INTERIM_PERIOD_PRIOR_KEY]
        assert col.shape == (3,)
        assert str(col.unit) == ""
        p = ustrip("day", samples["period"])
        expected = prior.distribution.log_prob(p) + jnp.log(p)
        assert jnp.allclose(ustrip("", col), expected)

    def test_immutability(self):
        samples = _make_samples([50.0, 100.0])
        _ = hp.attach_interim_period_prior(samples, _grid_prior())
        assert hp.LN_INTERIM_PERIOD_PRIOR_KEY not in samples.nonlinear

    def test_loguniform_gives_constant(self):
        samples = _make_samples([50.0, 100.0, 400.0])
        prior = QD(dist.LogUniform(10.0, 1000.0), "day")
        out = hp.attach_interim_period_prior(samples, prior)
        vals = ustrip("", out[hp.LN_INTERIM_PERIOD_PRIOR_KEY])
        expected = -np.log(np.log(1000.0 / 10.0))
        assert jnp.allclose(vals, expected, atol=1e-6)

    def test_unit_invariance(self):
        """The stored ln-density (per unit ln P) is unit-independent."""
        samples = _make_samples([50.0, 100.0])
        prior_day = QD(dist.LogUniform(10.0, 1000.0), "day")
        prior_yr = QD(dist.LogUniform(10.0 / 365.25, 1000.0 / 365.25), "yr")
        v_day = ustrip(
            "",
            hp.attach_interim_period_prior(samples, prior_day)[
                hp.LN_INTERIM_PERIOD_PRIOR_KEY
            ],
        )
        v_yr = ustrip(
            "",
            hp.attach_interim_period_prior(samples, prior_yr)[
                hp.LN_INTERIM_PERIOD_PRIOR_KEY
            ],
        )
        assert jnp.allclose(v_day, v_yr, atol=1e-4)

    def test_survives_pad_and_stack(self):
        prior = _grid_prior()
        s1 = hp.attach_interim_period_prior(_make_samples([50.0, 100.0, 200.0]), prior)
        s2 = hp.attach_interim_period_prior(_make_samples([80.0]), prior)
        stacked, mask = pad_and_stack_samples([s1, s2])
        col = stacked[hp.LN_INTERIM_PERIOD_PRIOR_KEY]
        assert col.shape == (2, 3)
        assert mask.shape == (2, 3)
        assert bool(mask[0].all())
        assert bool(mask[1][0])
        assert not bool(mask[1][1])

    def test_hdf5_roundtrip(self, tmp_path):
        out = hp.attach_interim_period_prior(
            _make_samples([50.0, 100.0]), _grid_prior()
        )
        path = tmp_path / "s.h5"
        out.to_hdf5(path)
        loaded = Samples.from_hdf5(path)
        assert jnp.allclose(
            ustrip("", loaded[hp.LN_INTERIM_PERIOD_PRIOR_KEY]),
            ustrip("", out[hp.LN_INTERIM_PERIOD_PRIOR_KEY]),
        )
