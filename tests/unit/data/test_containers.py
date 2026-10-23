import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from unxt import Q

try:
    import matplotlib.pyplot as plt
    from cycler import cycler

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

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
        assert stacked.n_obs == expected.n_obs
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

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_returns_axes_with_legend(self):
        data = SystemData(
            primary=_make_rv_data(0.0, values=(10.0, -10.0)),
            secondary=_make_rv_data(10.0, values=(-10.0, 10.0)),
        )
        ax = data.plot()
        try:
            legend = ax.get_legend()
            assert legend is not None
            labels = [t.get_text() for t in legend.get_texts()]
            assert "primary" in labels
            assert "secondary" in labels
            assert len(ax.lines) + len(ax.collections) > 0
        finally:
            plt.close("all")

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_accepts_existing_axes(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        _, ax_in = plt.subplots()
        try:
            ax_out = data.plot(ax=ax_in)
            assert ax_out is ax_in
        finally:
            plt.close("all")

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_add_legend_false(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        ax = data.plot(add_legend=False)
        try:
            assert ax.get_legend() is None
        finally:
            plt.close("all")

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_uses_distinct_colors_by_default(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        ax = data.plot()
        try:
            # ax.containers holds ErrorbarContainer objects; [0] is the data Line2D
            colors = [c[0].get_color() for c in ax.containers]
            assert len(colors) >= 2
            assert colors[0] != colors[1]
        finally:
            plt.close("all")

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_custom_color_cycler(self):
        custom = cycler(color=["#ff0000", "#0000ff"])
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        ax = data.plot(color_cycler=custom)
        try:
            colors = [c[0].get_color() for c in ax.containers]
            assert len(colors) >= 2
            assert colors[0] != colors[1]
        finally:
            plt.close("all")

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_explicit_color_kwarg_overrides_cycler(self):
        data = SystemData(
            primary=_make_rv_data(0.0),
            secondary=_make_rv_data(10.0),
        )
        ax = data.plot(color="green")
        try:
            colors = [c[0].get_color() for c in ax.containers]
            assert all(c == "green" for c in colors)
        finally:
            plt.close("all")


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
        assert stacked.n_obs == expected.n_obs
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

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_homogeneous_delegates_to_super(self):
        source = SourceData(
            survey1=_make_rv_data(0.0, values=(1.0, -1.0)),
            survey2=_make_rv_data(20.0, values=(0.5, -0.5)),
        )
        ax = source.plot()
        try:
            legend = ax.get_legend()
            assert legend is not None
            labels = [t.get_text() for t in legend.get_texts()]
            assert "survey1" in labels
            assert "survey2" in labels
        finally:
            plt.close("all")

    @pytest.mark.skipif(not HAS_MPL, reason="matplotlib is required for plotting")
    def test_plot_raises_on_mixed_types(self):
        source = SourceData(
            keck=_make_rv_data(0.0),
            gaia=_make_astrometry_data(),
        )
        with pytest.raises(TypeError, match="same concrete"):
            source.plot()


def _with_time_ref(ds, time_ref):
    return eqx.tree_at(lambda d: d.time_ref, ds, time_ref)


def _time_ref_days(data) -> float:
    return float(data.time_ref.to("day").value)


class TestSharedTimeRef:
    """Spec §Shared reference epoch: explicit, else already shared, else mean."""

    @pytest.mark.parametrize("container", [SourceData, SystemData])
    def test_differing_time_refs_use_mean_of_all_times(self, container):
        data = container(a=_make_rv_data(0.0), b=_make_rv_data(100.0))
        # times are 0, 50, 100, 150
        assert _time_ref_days(data["a"]) == pytest.approx(75.0)
        assert _time_ref_days(data["b"]) == pytest.approx(75.0)

    @pytest.mark.parametrize("container", [SourceData, SystemData])
    def test_shared_time_ref_is_kept(self, container):
        t0 = Q(-500.0, "day")
        data = container(
            a=_with_time_ref(_make_rv_data(0.0), t0),
            b=_with_time_ref(_make_rv_data(100.0), t0),
        )
        assert _time_ref_days(data["a"]) == -500.0
        assert _time_ref_days(data["b"]) == -500.0

    def test_shared_time_ref_compared_across_units(self):
        data = SourceData(
            a=_with_time_ref(_make_rv_data(0.0), Q(1.0, "yr")),
            b=_with_time_ref(_make_rv_data(100.0), Q(365.25, "day")),
        )
        assert _time_ref_days(data) == pytest.approx(365.25)

    @pytest.mark.parametrize("container", [SourceData, SystemData])
    def test_explicit_time_ref_overrides(self, container):
        t0 = Q(-500.0, "day")
        data = container(
            a=_with_time_ref(_make_rv_data(0.0), Q(3.0, "day")),
            b=_make_rv_data(100.0),
            time_ref=t0,
        )
        assert _time_ref_days(data["a"]) == -500.0
        assert _time_ref_days(data["b"]) == -500.0

    def test_single_dataset_unchanged_without_explicit_time_ref(self):
        rv = _make_rv_data(0.0)
        assert _time_ref_days(SourceData(a=rv)) == _time_ref_days(rv)

    def test_single_dataset_takes_explicit_time_ref(self):
        data = SourceData(a=_make_rv_data(0.0), time_ref=Q(-500.0, "day"))
        assert _time_ref_days(data) == -500.0

    def test_mixed_types_share_explicit_time_ref(self):
        data = SourceData(
            rv=_make_rv_data(0.0), gaia=_make_astrometry_data(), time_ref=Q(7.0, "day")
        )
        assert _time_ref_days(data["rv"]) == 7.0
        assert _time_ref_days(data["gaia"]) == 7.0

    def test_stack_keeps_shared_time_ref(self):
        t0 = Q(-500.0, "day")
        stacked = stack_datasets(
            {
                "a": _with_time_ref(_make_rv_data(0.0), t0),
                "b": _with_time_ref(_make_rv_data(100.0), t0),
            }
        )
        assert _time_ref_days(stacked) == -500.0

    def test_stack_differing_time_refs_use_mean(self):
        stacked = stack_datasets({"a": _make_rv_data(0.0), "b": _make_rv_data(100.0)})
        assert _time_ref_days(stacked) == pytest.approx(75.0)

    def test_indicator_data_by_type_keeps_container_time_ref(self):
        # An RV-only container with an epoch chosen to match other (e.g. astrometry)
        # data: stacking it for a MultiSurveyOffset model must not reset that epoch.
        source = SourceData(
            keck=_make_rv_data(0.0),
            espresso=_make_rv_data(100.0),
            time_ref=Q(-500.0, "day"),
        )
        stacked, _, _ = source.indicator_data_by_type(RVData, reference="keck")
        assert _time_ref_days(stacked) == -500.0
        joint = SourceData(
            rv=stacked, gaia=_with_time_ref(_make_astrometry_data(), Q(-500.0, "day"))
        )
        assert _time_ref_days(joint["rv"]) == -500.0

    def test_under_jit(self):
        a = _with_time_ref(_make_rv_data(0.0), Q(-500.0, "day"))
        b = _make_rv_data(100.0)

        @jax.jit
        def shared(a, b, b_time_ref):
            return SourceData(a=a, b=_with_time_ref(b, b_time_ref)).time_ref.value

        assert float(shared(a, b, Q(-500.0, "day"))) == -500.0
        assert float(shared(a, b, Q(3.0, "day"))) == pytest.approx(75.0)

    def test_under_vmap(self):
        a = _make_rv_data(0.0)
        b = _make_rv_data(100.0)

        def shared(t_a, t_b):
            return SourceData(
                a=_with_time_ref(a, Q(t_a, "day")), b=_with_time_ref(b, Q(t_b, "day"))
            ).time_ref.value

        t_a = jnp.array([-500.0, 1.0, 2.0])
        t_b = jnp.array([-500.0, 1.0, 9.0])
        out = jax.vmap(shared)(t_a, t_b)
        assert jnp.allclose(out, jnp.array([-500.0, 1.0, 75.0]))
