"""Helper functions for stacking and combining datasets."""

__all__ = (
    "build_indicator_matrix",
    "stack_datasets",
)

from dataclasses import fields

import equinox as eqx
import jax
import numpy as np
import quaxed.numpy as jnp
from unxt import AbstractQuantity, Q, ustrip
from unxt.quantity import AllowValue

from .datasets import AbstractData


def _synchronize_t_refs(
    datasets: dict[str, AbstractData],
) -> dict[str, AbstractData]:
    """Return datasets with a shared t_ref equal to the mean of all observation times.

    When multiple datasets are combined (e.g. in SourceData, SystemData, or a
    JointModel), the shared ``phase_peri`` parameter is interpreted as a fraction of
    the orbit relative to ``t_ref``.  All component datasets must therefore use the
    same reference epoch.  This function computes the global mean time and rebuilds
    each dataset with that shared value.

    Parameters
    ----------
    datasets : dict[str, AbstractData]
        Named datasets, possibly with different ``t_ref`` values.

    Returns
    -------
    dict[str, AbstractData]
        Same datasets with ``t_ref`` replaced by the global mean time.
    """
    if len(datasets) <= 1:
        return dict(datasets)

    first = next(iter(datasets.values()))
    time_unit = str(first.time.unit)

    all_times = np.concatenate(
        [np.asarray(ustrip(time_unit, ds.time)) for ds in datasets.values()]
    )
    shared_t_ref = Q(float(np.mean(all_times)), time_unit)

    return {
        name: eqx.tree_at(lambda d: d.t_ref, ds, shared_t_ref)
        for name, ds in datasets.items()
    }


def stack_datasets(
    datasets: dict[str, AbstractData],
) -> AbstractData:
    """Concatenate multiple datasets in dict order into a single one.

    Parameters
    ----------
    datasets : dict[str, AbstractData]
        Ordered mapping of instrument name -> dataset.  Dict order determines
        the row order in the stacked output; it must match the order used when
        building the indicator matrix (see :func:`build_rv_indicator_matrix`).

    Returns
    -------
    data
        Single dataset containing all observations stacked in dict order.

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
    >>> stacked.n_times
    4
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
    all_units = {
        field.name: str(getattr(ref, field.name).unit)
        if hasattr(getattr(ref, field.name), "unit")
        else ""
        for field in fields(ref)
        if field.name != "t_ref"  # scalar, not array -- skip and recompute below
    }

    # NOTE: we assume that all datasets have the same fields and units, and we assume
    # that all fields are present in all datasets and are array-valued (so they can be
    # concatenated). That's true for current datasets, but we might want to relax these
    # assumptions in the future.
    all_data: dict[str, AbstractQuantity] = {
        name: Q(
            jnp.concatenate(
                [
                    ustrip(AllowValue, unit, getattr(ds, name))
                    for ds in datasets.values()
                ]
            ),
            unit,
        )
        for name, unit in all_units.items()
    }
    # NOTE: t_ref is recomputed from the stacked time by __check_init__
    # TODO: we need to add a note somewhere (probably SourceData or all of the *Data
    # class docstrings) about how t_ref is handled when stacking datasets, since it's
    # not just concatenated but recomputed from the mean time. A potentially better
    # thing to do would be to check if one t_ref is set (use that), else throw an error.
    return type(ref)(**all_data)


def build_indicator_matrix(
    datasets: dict[str, AbstractData], reference: str
) -> tuple[AbstractData, jax.Array | None, tuple[str, ...] | None]:
    """Build indicator matrix for multi-survey data of the same type.

    Parameters
    ----------
    datasets : dict[str, AbstractData]
        Ordered mapping of instrument name -> dataset.  Dict order must match
        the order used when stacking (see :func:`stack_datasets`).
    reference : str
        Name of the reference instrument (its observations get no offset
        column).

    Returns
    -------
    indicator_matrix : jax.Array
        Shape ``(n_obs_total, n_non_ref)``.  ``indicator[i, j] = 1`` when
        observation ``i`` belongs to non-reference instrument ``j``.
    instrument_names : tuple[str, ...]
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
    >>> stacked.n_times
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
