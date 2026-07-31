"""Unit tests for interim-period-prior HDF5 persistence."""

import h5py
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from unxt import Q

import harv.periodogram as hp
from harv.distributions import QD
from harv.samplers import Samples


def _make_prior() -> QD:
    ln_grid = jnp.log(jnp.geomspace(10.0, 1000.0, 64))
    log_density = -0.5 * ((ln_grid - jnp.log(100.0)) / 0.1) ** 2
    return QD(hp.LogGridDensity(ln_grid, log_density), "day")


class TestRoundTrip:
    def test_log_prob_identity(self, tmp_path):
        prior = _make_prior()
        path = tmp_path / "prior.h5"
        hp.save_period_prior(path, prior, metadata={"builder": "test", "beta": 1.0})
        loaded = hp.load_period_prior(path)

        assert str(loaded.unit) == str(prior.unit)
        x = jnp.geomspace(10.5, 999.0, 101)
        assert jnp.allclose(
            loaded.distribution.log_prob(x), prior.distribution.log_prob(x)
        )

    def test_metadata_attrs(self, tmp_path):
        path = tmp_path / "prior.h5"
        hp.save_period_prior(path, _make_prior(), metadata={"floor": 0.1})
        with h5py.File(path, "r") as f:
            g = f["interim_period_prior"]
            assert g.attrs["unit"] == "day"
            assert g.attrs["format_version"] == 1
            assert g.attrs["floor"] == 0.1

    def test_overwrite_existing_group(self, tmp_path):
        path = tmp_path / "prior.h5"
        hp.save_period_prior(path, _make_prior())
        prior2 = QD(
            hp.LogGridDensity(jnp.log(jnp.geomspace(1.0, 10.0, 8)), jnp.zeros(8)),
            "yr",
        )
        hp.save_period_prior(path, prior2)
        loaded = hp.load_period_prior(path)
        assert str(loaded.unit) == "yr"
        assert loaded.distribution.ln_grid.shape == (8,)

    def test_group_object_and_custom_name(self, tmp_path):
        path = tmp_path / "prior.h5"
        with h5py.File(path, "w") as f:
            hp.save_period_prior(f, _make_prior(), group="my_prior")
        with h5py.File(path, "r") as f:
            loaded = hp.load_period_prior(f, group="my_prior")
        assert str(loaded.unit) == "day"

    def test_missing_group_raises(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w"):
            pass
        with pytest.raises(KeyError, match="interim_period_prior"):
            hp.load_period_prior(path)

    def test_wrong_prior_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="LogGridDensity"):
            hp.save_period_prior(
                tmp_path / "x.h5", QD(dist.LogUniform(1.0, 10.0), "day")
            )


class TestSamplesCoexistence:
    def test_shares_file_with_samples(self, tmp_path):
        samples = Samples(
            nonlinear={
                "period": Q([100.0, 101.0], "day"),
                "eccentricity": Q([0.1, 0.2], ""),
                "phase_peri": Q([0.3, 0.4], ""),
            },
            linear={"rv_semiamp": Q([10.0, 11.0], "km/s")},
            data_type="RVModel",
            metadata={"t_ref": 0.0, "t_ref_unit": "day"},
        )
        path = tmp_path / "star.h5"
        samples.to_hdf5(path)
        hp.save_period_prior(path, _make_prior())

        loaded_samples = Samples.from_hdf5(path)
        assert loaded_samples.n_samples == 2
        assert set(loaded_samples.nonlinear) == set(samples.nonlinear)

        loaded_prior = hp.load_period_prior(path)
        assert str(loaded_prior.unit) == "day"
