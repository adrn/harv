import jax.numpy as jnp
import pytest
from unxt import Q

from harv.data import GaiaAstrometryData, RVData, SourceData, SystemData
from harv.data.helpers import build_indicator_matrix, stack_datasets


def _make_rv_data(
    start: float = 0.0,
    values: tuple[float, float] = (1.0, -2.0),
    errs: tuple[float, float] = (0.5, 0.5),
) -> RVData:
    return RVData(
        time=Q(jnp.array([start, start + 50.0]), "day"),
        rv=Q(jnp.array(values), "km/s"),
        rv_err=Q(jnp.array(errs), "km/s"),
    )


def _make_astrometry_data(start: float = 0.0) -> GaiaAstrometryData:
    return GaiaAstrometryData(
        time=Q(jnp.array([start, start + 100.0]), "day"),
        al_position=Q(jnp.array([0.1, -0.2]), "mas"),
        al_position_err=Q(jnp.array([0.01, 0.01]), "mas"),
        scan_angle=Q(jnp.array([0.5, 1.2]), "rad"),
        parallax_factor=jnp.array([0.3, -0.1]),
    )


class TestSystemData:
    def test_requires_homogeneous_concrete_type(self):
        with pytest.raises(TypeError, match="same concrete data class"):
            SystemData(
                primary=_make_rv_data(),
                secondary=_make_astrometry_data(),
            )

    def test_dataset_type_records_shared_concrete_type(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        assert data.dataset_type is RVData

    def test_stacked_matches_helper(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        stacked = data.stacked()
        expected = stack_datasets(
            {"primary": data["primary"], "secondary": data["secondary"]}
        )
        assert isinstance(stacked, RVData)
        assert stacked.n_times == expected.n_times
        assert jnp.allclose(stacked.time.value, expected.time.value)
        assert jnp.allclose(stacked.rv.value, expected.rv.value)
        assert jnp.allclose(stacked.rv_err.value, expected.rv_err.value)

    def test_indicator_data_matches_helper(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        stacked, indicator, names = data.indicator_data(reference="primary")
        expected_stacked, expected_indicator, expected_names = build_indicator_matrix(
            {"primary": data["primary"], "secondary": data["secondary"]},
            reference="primary",
        )
        assert isinstance(stacked, RVData)
        assert names == expected_names
        assert indicator is not None
        assert expected_indicator is not None
        assert jnp.allclose(stacked.time.value, expected_stacked.time.value)
        assert jnp.allclose(stacked.rv.value, expected_stacked.rv.value)
        assert jnp.allclose(indicator, expected_indicator)


class TestSourceData:
    def test_stacked_by_type_returns_rv_subset(self):
        source = SourceData(
            survey1=_make_rv_data(0.0),
            survey2=_make_rv_data(10.0),
            gaia=_make_astrometry_data(),
        )
        stacked = source.stacked_by_type(RVData)
        expected = stack_datasets(
            {
                "survey1": source["survey1"],
                "survey2": source["survey2"],
            }
        )
        assert isinstance(stacked, RVData)
        assert stacked.n_times == expected.n_times
        assert jnp.allclose(stacked.time.value, expected.time.value)
        assert jnp.allclose(stacked.rv.value, expected.rv.value)

    def test_indicator_data_by_type_matches_helper(self):
        source = SourceData(
            keck=_make_rv_data(0.0),
            espresso=_make_rv_data(10.0),
            gaia=_make_astrometry_data(),
        )
        stacked, indicator, names = source.indicator_data_by_type(
            RVData,
            reference="keck",
        )
        expected_stacked, expected_indicator, expected_names = build_indicator_matrix(
            source.get_datasets_by_type(RVData),
            reference="keck",
        )
        assert isinstance(stacked, RVData)
        assert names == expected_names
        assert indicator is not None
        assert expected_indicator is not None
        assert jnp.allclose(stacked.time.value, expected_stacked.time.value)
        assert jnp.allclose(stacked.rv.value, expected_stacked.rv.value)
        assert jnp.allclose(indicator, expected_indicator)

    def test_stacked_by_type_raises_when_type_absent(self):
        source = SourceData(
            survey1=_make_rv_data(0.0),
            survey2=_make_rv_data(10.0),
        )
        with pytest.raises(ValueError, match="No datasets of type"):
            source.stacked_by_type(GaiaAstrometryData)
