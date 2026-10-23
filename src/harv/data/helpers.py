"""Helper functions for stacking and combining datasets."""

__all__ = (
    "build_indicator_matrix",
    "stack_datasets",
)

from dataclasses import fields
from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import Q, ustrip

from harv.custom_types import ScalarQTime

from .datasets import AbstractData

if TYPE_CHECKING:
    from unxt import AbstractQuantity


def _common_time_ref(datasets: dict[str, AbstractData]) -> ScalarQTime:
    """The ``time_ref`` a set of datasets should share.

    The datasets' own ``time_ref`` when they all agree, otherwise the mean of all
    their observation times (rules 2-3 of spec §Shared reference epoch). Selected
    with ``jnp.where`` so that tracers are never concretized.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.data import RVData
    >>> from harv.data.helpers import _common_time_ref
    >>> rv1 = RVData(time=Q([0.0, 10.0], "day"), rv=Q([1.0, 2.0], "km/s"),
    ...              rv_err=Q([0.1, 0.1], "km/s"), time_ref=Q(100.0, "day"))
    >>> rv2 = RVData(time=Q([20.0, 30.0], "day"), rv=Q([1.0, 2.0], "km/s"),
    ...              rv_err=Q([0.1, 0.1], "km/s"), time_ref=Q(100.0, "day"))
    >>> _common_time_ref({"a": rv1, "b": rv2}).value.item()
    100.0
    >>> rv3 = RVData(time=Q([20.0, 30.0], "day"), rv=Q([1.0, 2.0], "km/s"),
    ...              rv_err=Q([0.1, 0.1], "km/s"))
    >>> _common_time_ref({"a": rv1, "b": rv3}).value.item()
    15.0
    """
    # __check_init__ has already filled in every dataset's time_ref
    time_refs = [cast("ScalarQTime", ds.time_ref) for ds in datasets.values()]
    unit = time_refs[0].unit
    ref_values = jnp.stack([ustrip(unit, t) for t in time_refs])
    time_values = jnp.concatenate([ustrip(unit, ds.time) for ds in datasets.values()])
    shared = jnp.where(
        jnp.all(ref_values == ref_values[0]), ref_values[0], jnp.mean(time_values)
    )
    return Q(shared, unit)


def _synchronize_time_refs(
    datasets: dict[str, AbstractData],
    time_ref: ScalarQTime | None = None,
) -> dict[str, AbstractData]:
    """Return datasets that all carry the same ``time_ref``.

    When multiple datasets are combined (e.g., in SourceData, SystemData, or a
    JointModel), the shared ``phase_peri`` parameter is interpreted as a fraction of
    the orbit relative to ``time_ref``.  All component datasets must therefore use the
    same reference epoch. See spec §Shared reference epoch.

    Parameters
    ----------
    datasets
        Named datasets, possibly with different ``time_ref`` values.
    time_ref
        Explicit reference epoch to set on every dataset. When ``None``, the
        datasets keep their ``time_ref`` if they already agree, and otherwise use
        the mean of all observation times.

    Returns
    -------
        Same datasets with a shared ``time_ref``.
    """
    if time_ref is None:
        if len(datasets) <= 1:
            return dict(datasets)
        time_ref = _common_time_ref(datasets)

    return {
        name: eqx.tree_at(lambda d: d.time_ref, ds, time_ref)
        for name, ds in datasets.items()
    }


def stack_datasets[DT: AbstractData](
    datasets: dict[str, DT],
) -> DT:
    """Concatenate multiple datasets in dict order into a single one.

    Parameters
    ----------
    datasets
        Ordered mapping of instrument name -> dataset.  Dict order determines
        the row order in the stacked output; it must match the order used when
        building the indicator matrix (see :func:`build_indicator_matrix`).

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.data import RVData
    >>> from harv.data.helpers import stack_datasets
    >>> rv1 = RVData(
    ...     time=Q([0.0, 50.0], "day"),
    ...     rv=Q([1.0, -2.0], "km/s"),
    ...     rv_err=Q([0.5, 0.5], "km/s"),
    ... )
    >>> rv2 = RVData(
    ...     time=Q([10.0, 60.0], "day"),
    ...     rv=Q([0.5, -1.5], "km/s"),
    ...     rv_err=Q([0.3, 0.3], "km/s"),
    ... )
    >>> stacked = stack_datasets({"instr1": rv1, "instr2": rv2})
    >>> stacked.n_obs
    4

    The two datasets have different ``time_ref`` values (their own mean times), so
    the stacked ``time_ref`` is the mean of all four times:

    >>> stacked.time_ref.value.item()
    30.0
    """
    # first make sure that all datasets have the same type:
    dset_types = {type(ds) for ds in datasets.values()}
    if len(dset_types) != 1:
        msg = f"All datasets must have the same type to stack (got: {dset_types})"
        raise ValueError(msg)

    # the reference dataset, which we use to get the field names and units for the
    # output dataset
    ref = next(iter(datasets.values()))

    # units for each field:
    all_fields = {
        field.name
        for field in fields(ref)
        if field.name != "time_ref"  # scalar, not array -- skip and set below
    }

    # NOTE: we assume that all datasets have the same fields and units, and we assume
    # that all fields are present in all datasets and are array-valued (so they can be
    # concatenated). That's true for current datasets, but we might want to relax these
    # assumptions in the future.
    all_data: dict[str, AbstractQuantity | jax.Array] = {
        name: jnp.concatenate([getattr(ds, name) for ds in datasets.values()])
        for name in all_fields
    }
    # Datasets that share a time_ref keep it; otherwise it is the mean stacked time.
    return type(ref)(**all_data, time_ref=_common_time_ref(datasets))


def build_indicator_matrix[DT: AbstractData](
    datasets: dict[str, DT], reference: str
) -> tuple[DT, jax.Array | None, tuple[str, ...] | None]:
    """Build indicator matrix for multi-survey data of the same type.

    Parameters
    ----------
    datasets
        Ordered mapping of instrument name -> dataset.  Dict order must match the order
        used when stacking (see :func:`stack_datasets`).
    reference
        Name of the reference instrument (its observations get no offset
        column).

    Returns
    -------
    stacked : DT
        Stacked dataset containing all observations.
    indicator : jax.Array | None
        Shape ``(n_obs_total, n_non_ref)``.  ``indicator[i, j] = 1`` when
        observation ``i`` belongs to non-reference instrument ``j``.
    instrument_names : tuple[str, ...] | None
        Names of the non-reference instruments, in column order.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.data import RVData
    >>> from harv.data.helpers import build_indicator_matrix
    >>> rv1 = RVData(
    ...     time=Q([0.0, 50.0], "day"),
    ...     rv=Q([1.0, -2.0], "km/s"),
    ...     rv_err=Q([0.5, 0.5], "km/s"),
    ... )
    >>> rv2 = RVData(
    ...     time=Q([10.0, 60.0], "day"),
    ...     rv=Q([0.5, -1.5], "km/s"),
    ...     rv_err=Q([0.3, 0.3], "km/s"),
    ... )
    >>> stacked, indicator, names = build_indicator_matrix(
    ...     {"survey1": rv1, "survey2": rv2}, reference="survey1",
    ... )
    >>> stacked.n_obs
    4
    >>> names
    ('survey2',)
    >>> indicator.shape
    (4, 1)
    """
    if reference not in datasets:
        msg = f"Reference instrument {reference!r} not in {list(datasets)}"
        raise ValueError(msg)

    non_ref_names = [k for k in datasets if k != reference]
    n_non_ref = len(non_ref_names)
    rows = []
    for name, ds in datasets.items():
        n_obs = len(ds.time)
        row = jnp.zeros((n_obs, n_non_ref))
        if name != reference:
            j = non_ref_names.index(name)
            row = row.at[:, j].set(1.0)
        rows.append(row)
    return (
        stack_datasets(datasets),
        jnp.concatenate(rows, axis=0),
        tuple(non_ref_names),
    )
