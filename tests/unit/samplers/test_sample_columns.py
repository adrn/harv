"""Tests for ``Samples.to_columns`` / ``Samples.from_columns`` and ``SampleColumns``."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from unxt import Q

from harv.samplers import SampleColumns, Samples, pad_and_stack_samples


def _samples(n=20, *, with_logprobs=True, metadata=None, extra_nonlinear=None, seed=0):
    rng = np.random.default_rng(seed)
    nonlinear = {
        "period": Q(rng.uniform(40.0, 60.0, n), "day"),
        "eccentricity": Q(rng.uniform(0.0, 0.3, n), ""),
        "phase_peri": Q(rng.uniform(0.0, 1.0, n), ""),
        "arg_peri": Q(rng.uniform(0.0, 2 * np.pi, n), "rad"),
        **(extra_nonlinear or {}),
    }
    linear = {
        "rv_semiamp": Q(rng.uniform(5.0, 15.0, n), "km/s"),
        "v_sys": Q(rng.uniform(-1.0, 1.0, n), "km/s"),
        "offset_keck": Q(rng.uniform(-1.0, 1.0, n), "km/s"),
    }
    kwargs = {}
    if with_logprobs:
        kwargs["ln_likelihood"] = jnp.asarray(rng.normal(size=n))
        kwargs["ln_prior"] = jnp.asarray(rng.normal(size=n))
    return Samples(
        nonlinear=nonlinear,
        linear=linear,
        model_type="RVModel",
        linear_extension_names=("offset_keck",),
        metadata=(
            {"time_ref": 0.5, "time_ref_unit": "day"} if metadata is None else metadata
        ),
        **kwargs,
    )


def _assert_samples_equal(a: Samples, b: Samples) -> None:
    assert list(a.nonlinear) == list(b.nonlinear)
    assert list(a.linear) == list(b.linear)
    for group_a, group_b in ((a.nonlinear, b.nonlinear), (a.linear, b.linear)):
        for name, qty in group_a.items():
            assert str(group_b[name].unit) == str(qty.unit)
            np.testing.assert_array_equal(
                np.asarray(group_b[name].value), np.asarray(qty.value)
            )
    for field in ("ln_likelihood", "ln_prior"):
        va, vb = getattr(a, field), getattr(b, field)
        assert (va is None) == (vb is None)
        if va is not None:
            np.testing.assert_array_equal(np.asarray(vb), np.asarray(va))
    assert a.model_type == b.model_type
    assert a.linear_extension_names == b.linear_extension_names
    assert a.metadata == b.metadata


class TestToColumns:
    def test_structure(self):
        s = _samples()
        cols = s.to_columns()

        assert isinstance(cols, SampleColumns)
        assert cols.nonlinear_names == (
            "period",
            "eccentricity",
            "phase_peri",
            "arg_peri",
        )
        assert cols.linear_names == ("rv_semiamp", "v_sys", "offset_keck")
        assert list(cols.columns) == [
            *cols.nonlinear_names,
            *cols.linear_names,
            "ln_likelihood",
            "ln_prior",
        ]
        assert cols.units == {
            "period": "d",
            "eccentricity": "",
            "phase_peri": "",
            "arg_peri": "rad",
            "rv_semiamp": "km / s",
            "v_sys": "km / s",
            "offset_keck": "km / s",
        }
        assert cols.model_type == "RVModel"
        assert cols.linear_extension_names == ("offset_keck",)
        assert cols.metadata == {"time_ref": 0.5, "time_ref_unit": "day"}

    def test_columns_are_numpy(self):
        cols = _samples().to_columns()
        for name, arr in cols.columns.items():
            assert type(arr) is np.ndarray, name
            assert arr.shape == (20,)

    def test_without_logprobs(self):
        cols = _samples(with_logprobs=False).to_columns()
        assert "ln_likelihood" not in cols.columns
        assert "ln_prior" not in cols.columns

    def test_metadata_is_a_copy(self):
        s = _samples()
        cols = s.to_columns()
        cols.metadata["time_ref"] = 99.0
        assert s.metadata["time_ref"] == 0.5

    @pytest.mark.parametrize("reserved", ["ln_likelihood", "ln_prior"])
    def test_reserved_parameter_name_raises(self, reserved):
        s = _samples(extra_nonlinear={reserved: Q(np.zeros(20), "")})
        with pytest.raises(ValueError, match=reserved):
            s.to_columns()

    def test_name_in_both_groups_raises(self):
        s = _samples()
        clash = Samples(
            nonlinear={**s.nonlinear, "v_sys": Q(np.zeros(20), "km/s")},
            linear=s.linear,
            model_type=s.model_type,
        )
        with pytest.raises(ValueError, match="v_sys"):
            clash.to_columns()


class TestRoundTrip:
    def test_rejection_like_with_evidence_metadata(self):
        s = _samples(
            metadata={
                "time_ref": 0.5,
                "time_ref_unit": "day",
                "ln_Z_int": -12.5,
                "ln_Z_int_mcse": 0.1,
                "ln_Z_int_ess": 42.0,
                "max_ln_likelihood": -3.0,
                "n_prior_samples": 100_000,
                "weight_captured": 0.97,
            }
        )
        _assert_samples_equal(Samples.from_columns(s.to_columns()), s)

    def test_mcmc_like(self):
        s = _samples(
            metadata={"time_ref": 0.5, "time_ref_unit": "day", "num_chains": 4}
        )
        _assert_samples_equal(Samples.from_columns(s.to_columns()), s)

    def test_without_logprobs(self):
        s = _samples(with_logprobs=False)
        loaded = Samples.from_columns(s.to_columns())
        assert loaded.ln_likelihood is None
        assert loaded.ln_prior is None
        _assert_samples_equal(loaded, s)

    def test_extra_nonlinear_column(self):
        """Extra dimensionless columns round-trip like parameters."""
        s = _samples(extra_nonlinear={"ln_interim_period_prior": Q(np.ones(20), "")})
        loaded = Samples.from_columns(s.to_columns())
        assert "ln_interim_period_prior" in loaded.nonlinear
        _assert_samples_equal(loaded, s)

    def test_batched_samples_keep_their_shape(self):
        stacked, _ = pad_and_stack_samples([_samples(n=5), _samples(n=3, seed=1)])
        cols = stacked.to_columns()
        assert cols.columns["period"].shape == (2, 5)
        _assert_samples_equal(Samples.from_columns(cols), stacked)


class TestFromColumns:
    def test_numpy_metadata_scalars_become_python(self):
        """h5py and pyarrow return numpy scalars; static metadata needs Python ones."""
        cols = _samples().to_columns()
        cols = dataclasses.replace(
            cols,
            metadata={
                "time_ref": np.float64(0.5),
                "time_ref_unit": "day",
                "num_chains": np.int64(4),
            },
        )
        loaded = Samples.from_columns(cols)
        assert type(loaded.metadata["time_ref"]) is float
        assert type(loaded.metadata["num_chains"]) is int

    def test_columns_become_jax_arrays(self):
        loaded = Samples.from_columns(_samples().to_columns())
        assert not isinstance(loaded.nonlinear["period"].value, np.ndarray)
        assert not isinstance(loaded.ln_likelihood, np.ndarray)
