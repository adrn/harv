"""Container for rejection sampler posterior samples.

This module provides the Samples class which stores posterior samples from
rejection sampling with dict-like access, unit handling, and analysis tools.
"""

import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final, overload

import equinox as eqx
import h5py
import jax
import numpy as np
import quaxed.numpy as jnp
from unxt import AbstractQuantity, Q, ustrip

from harv._optional_deps import get_arviz
from harv.data.datasets import AbstractData
from harv.kepler import masses
from harv.models.parameterizations.base import AbstractParameterization
from harv.models.parameterizations.gaia import (
    StandardGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.samplers.conversion import convert_parameterization

__all__ = ("SampleColumns", "Samples", "pad_and_stack_samples")

# Column names reserved for the per-sample log-probabilities in
# :meth:`Samples.to_columns`; no parameter may use them.
_LOGPROB_COLUMNS = ("ln_likelihood", "ln_prior")

# Default minimum evidence effective sample size (ln_Z_int_ess) below which a
# rejection run is reported as under-resolved: the marginal-likelihood integral
# is then dominated by a handful of prior draws, so max_ln_likelihood may not
# have converged and the accepted-sample count is not a reliable posterior size.
#
# There is no sharp transition to calibrate against, so this is a convention,
# not a derived quantity: ESS = 3 is where the delta-method MC error on
# ln_Z_int (``sqrt(1/ESS - 1/M)``, reported as ``ln_Z_int_mcse``) reaches ~0.6
# nats, i.e. the log-evidence is uncertain at the factor-of-two level. Callers
# who want a different bar set ``min_evidence_ess`` on
# :class:`~harv.samplers.RejectionSampler` or pass it to
# :meth:`Samples.acceptance_diagnostics`; ``0.0`` silences the check and
# ``float("inf")`` always flags.
MIN_EVIDENCE_ESS = 3.0

_EVIDENCE_KEYS = ("ln_Z_int", "ln_Z_int_ess", "max_ln_likelihood", "n_prior_samples")


def _assess_resolution(
    *,
    n_prior: int,
    n_accepted: int,
    evidence_ess: float,
    max_ln_likelihood: float,
    min_evidence_ess: float = MIN_EVIDENCE_ESS,
) -> tuple[bool, str]:
    """Judge whether a rejection run resolved the posterior; return a message.

    ``evidence_ess`` is the evidence effective sample size
    (``(sum L)^2 / sum L^2``). When it is O(1) the evidence integral --- and
    hence the ``max``-normalization used by the rejection step --- is dominated
    by a single lucky draw, so the accepted count is not a reliable posterior
    size (see ``docs/spec.md``, "Interpreting acceptance"). Used by both
    :meth:`Samples.acceptance_diagnostics` and the sampler's under-resolution
    warning.

    ``min_evidence_ess`` is the bar the run is held to (default
    :data:`MIN_EVIDENCE_ESS`).
    """
    well_resolved = evidence_ess >= min_evidence_ess
    if well_resolved:
        msg = (
            f"Resolved: ~{evidence_ess:.0f} effective prior samples (of {n_prior}) "
            f"contribute to the evidence integral, so max_ln_likelihood="
            f"{max_ln_likelihood:.1f} is likely converged. Confirm across seeds "
            "if it matters."
        )
    else:
        msg = (
            f"Under-resolved rejection run: the evidence integral is dominated by "
            f"~{evidence_ess:.1f} effective prior sample(s) of {n_prior} "
            f"(below min_evidence_ess={min_evidence_ess:g}), so "
            f"max_ln_likelihood={max_ln_likelihood:.1f} may not have converged and "
            f"the accepted-sample count ({n_accepted}) is not a reliable posterior "
            "size. Increase n_prior_samples (compare max_ln_likelihood across runs "
            "to check it stops rising) and/or continue with "
            "NumpyroSampler(prior, model).run(data, init_samples=...) to draw the "
            "posterior."
        )
    return well_resolved, msg


def _find_namespaced_keys(d: dict[str, Any], param_name: str) -> list[str]:
    """Return all keys in ``d`` that match ``param_name`` (bare or namespaced).

    Matches the bare name (e.g. ``"rv_semiamp"``) and any
    ``"component_name.param_name"`` form (e.g. ``"primary.rv_semiamp"``)
    used by :class:`~harv.models.JointModel` to namespace per-component
    parameters.  Keys are returned in dict insertion order so that
    callers picking the "first" match are deterministic.
    """
    return [k for k in d if k == param_name or k.endswith(f".{param_name}")]


def _overlay_corner_truths(  # noqa: C901
    plot_matrix: Any,
    var_names: list[str],
    reference_values: dict[str, float],
    *,
    marginal: bool,
    triangle: str,
) -> None:
    """Overlay truth markers on an ArviZ pair plot.

    This is an ugly workaround because arviz's new API is not backwards compatible with
    the old plot_pair for displaying reference values / "truths" - ARGH!

    ``arviz_plots.plot_pair`` treats ``stats['point_estimate']`` as keyword
    arguments for computing a summary statistic or as a precomputed xarray
    object, not as literal truth coordinates. Overlay truths after plotting
    rather than routing them through the stats API.
    """
    if not reference_values or plot_matrix is None:
        return

    if getattr(plot_matrix, "backend", None) != "matplotlib":
        warnings.warn(
            "plot_corner truths are only overplotted for the matplotlib backend.",
            stacklevel=3,
        )
        return

    axes = np.asarray(plot_matrix.viz["plot"].values)
    line_kwargs = {"color": "C1", "linestyle": "--", "linewidth": 1.2, "alpha": 0.9}
    point_kwargs = {"color": "C1", "s": 24, "zorder": 5}

    for row_idx, y_name in enumerate(var_names):
        y_truth = reference_values.get(y_name)
        for col_idx, x_name in enumerate(var_names):
            ax = axes[row_idx, col_idx]
            if ax is None:
                continue

            x_truth = reference_values.get(x_name)

            if marginal and row_idx == col_idx:
                if x_truth is not None:
                    ax.axvline(x_truth, **line_kwargs)
                continue

            if triangle == "lower" and row_idx <= col_idx:
                continue
            if triangle == "upper" and row_idx >= col_idx:
                continue
            if triangle not in {"lower", "upper", "both"} and row_idx == col_idx:
                continue

            if x_truth is not None:
                ax.axvline(x_truth, **line_kwargs)
            if y_truth is not None:
                ax.axhline(y_truth, **line_kwargs)
            if x_truth is not None and y_truth is not None:
                ax.scatter([x_truth], [y_truth], **point_kwargs)


def _assemble_sample_params(
    samples: "Samples",
    model: Any,
    data: AbstractData,
    *,
    i: int | None = None,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    """Build ``(nonlinear_values, linear_values)`` from ``samples``, as models expect.

    Matches the convention used by the sampler's ``log_prob`` calls:
    dimensioned base nonlinear parameters stay as Quantities, dimensionless
    base parameters and all extension parameters are unit-stripped, and every
    linear parameter is unit-stripped to the model's linear-parameter units
    (see :meth:`~harv.models.component.AbstractComponentModel._linear_param_units`).

    Parameters
    ----------
    samples
        Single-component :class:`Samples`.
    model
        The component model whose ``parameterization`` and ``_linear_param_units``
        define the expected param form.
    data
        Data instance — used to resolve linear-parameter units.
    i
        If ``None`` (default), returns batched dicts (one array per parameter,
        shape ``(n_samples,)``) — suitable for use under ``jax.vmap`` (e.g.
        :meth:`Samples.chi2`).  If an integer, extracts sample ``i`` so each
        dict value is a scalar — suitable for the plot functions, which always
        plot one sample at a time.
    """
    base_nl_units = {
        p.name: p.unit for p in model.parameterization.params() if not p.linear
    }
    nonlinear_names = set(model._all_nonlinear_names())
    linear_units = model._linear_param_units(data)
    linear_names = set(linear_units)

    def _pick(value: Any) -> Any:
        return value if i is None else value[i]

    nonlinear_for_model: dict[str, Any] = {
        name: _pick(value)
        if base_nl_units.get(name, "")
        else ustrip(str(value.unit), _pick(value))
        for name, value in samples.nonlinear.items()
        if name in nonlinear_names
    }
    linear_stripped: dict[str, jax.Array] = {
        name: ustrip(linear_units[name], _pick(value))
        for name, value in samples.linear.items()
        if name in linear_names
    }
    return nonlinear_for_model, linear_stripped


class _MetadataView(Mapping[str, Any]):
    """Q-aware read-only view over :attr:`Samples.metadata`.

    The underlying ``metadata`` dict holds the *split* form for quantity-valued
    entries: a value (``float`` / ``int`` / ``str`` / ``bool``) under ``name``
    and a unit string under ``f"{name}_unit"`` (e.g. ``{"time_ref": 0.0,
    "time_ref_unit": "day"}``).  This split keeps the static field free of JAX
    arrays so equinox doesn't warn about JAX arrays being marked static.

    The view papers over that split: ``view[name]`` returns a :class:`~unxt.Q`
    whenever a ``f"{name}_unit"`` companion exists, and the bare value
    otherwise.  Iteration yields *logical* names — ``_unit`` companions of an
    existing base key are hidden.  ``get``, ``keys``, ``items``, ``values``,
    and ``__eq__`` come from :class:`collections.abc.Mapping` for free.

    Examples
    --------
    >>> view = _MetadataView({"time_ref": 0.0, "time_ref_unit": "day", "num_chains": 2})
    >>> view["time_ref"]  # doctest: +ELLIPSIS
    Quantity(..., unit='d')
    >>> view["num_chains"]
    2
    >>> sorted(view)
    ['num_chains', 'time_ref']
    >>> "time_ref" in view
    True
    >>> "time_ref_unit" in view  # the _unit companion is hidden
    False
    >>> view.get("missing", 5)
    5
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict[str, Any]) -> None:
        self._d = d

    def __getitem__(self, name: str) -> Any:
        if name not in self._d or self._is_hidden(name):
            raise KeyError(name)
        value = self._d[name]
        unit_key = f"{name}_unit"
        if unit_key in self._d:
            return Q(value, self._d[unit_key])
        return value

    def __iter__(self) -> Iterator[str]:
        for k in self._d:
            if not self._is_hidden(k):
                yield k

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def _is_hidden(self, key: str) -> bool:
        """True for ``_unit`` companion keys of an existing base."""
        return key.endswith("_unit") and key[: -len("_unit")] in self._d

    def __repr__(self) -> str:
        return f"_MetadataView({dict(self.items())!r})"


@final
@dataclass(frozen=True)
class SampleColumns:
    """A :class:`Samples` as plain NumPy columns, plus what is needed to rebuild it.

    Produced by :meth:`Samples.to_columns` and consumed by
    :meth:`Samples.from_columns`. It is the format-neutral mapping between a
    ``Samples`` and a table: every file format harv or a downstream package
    writes (HDF5 here, Parquet in hq) is built on it, so the structure of a
    ``Samples`` is defined in one place.

    Parameters
    ----------
    columns
        Every nonlinear and linear parameter, keyed by name, plus
        ``"ln_likelihood"`` and ``"ln_prior"`` when the ``Samples`` carries
        them. Arrays keep the ``Samples``' full shape: 1-D for a flat
        ``Samples``, ``(*batch_shape, n_samples)`` for a batched one.
    units
        Unit string for every parameter (``""`` for dimensionless). The
        log-probability columns are dimensionless and have no entry.
    nonlinear_names, linear_names
        Which columns are nonlinear and which linear parameters, in order.
    model_type, linear_extension_names, metadata
        The ``Samples`` fields of the same names. ``metadata`` holds only
        JSON-friendly scalars.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers import Samples
    >>> samples = Samples(
    ...     nonlinear={"period": Q([100.0, 101.0], "day")},
    ...     linear={"v_sys": Q([5.0, 5.1], "km/s")},
    ...     model_type="RVModel",
    ... )
    >>> cols = samples.to_columns()
    >>> cols.nonlinear_names, cols.linear_names
    (('period',), ('v_sys',))
    >>> cols.columns["period"]
    array([100., 101.])
    >>> Samples.from_columns(cols).n_samples
    2
    """

    columns: dict[str, np.ndarray]
    units: dict[str, str]
    nonlinear_names: tuple[str, ...]
    linear_names: tuple[str, ...]
    model_type: str
    linear_extension_names: tuple[str, ...]
    metadata: dict[str, int | float | str | bool]


def _python_scalar(value: Any) -> Any:
    """Coerce a numpy scalar (as h5py or pyarrow return) to the Python scalar.

    ``Samples.metadata`` is ``eqx.field(static=True)``, and equinox treats a numpy
    scalar as an array: storing one there breaks hashing and jit-cache keys, and
    warns. Anything else passes through unchanged.
    """
    return value.item() if isinstance(value, np.generic) else value


class Samples(eqx.Module):
    """Container for posterior samples.

    Stores both nonlinear and linear parameter samples as :class:`~unxt.Q` objects with
    units baked in. Provides dict-like access, statistical summaries, and visualization
    tools.

    Parameters
    ----------
    nonlinear
        Nonlinear parameter samples, one Q per parameter. Keys: ``"period"``,
        ``"eccentricity"``, ``"phase_peri"``, and optionally ``"arg_peri"``,
        ``"cos_i"``, ``"lon_asc_node"``. Units: period has time units; angles have
        ``"rad"``; dimensionless parameters have unit ``""``.
    linear
        Linear parameter samples, one Q per parameter. Keys: e.g. ``"rv_semiamp"``,
        ``"v_sys"`` for RV; ``"ra0"``, ``"dec0"``, ``"pmra"``, ``"pmdec"``,
        ``"parallax"``, ``"semi_major_axis"`` for astrometry.  Units are data-driven
        (e.g. ``"km/s"`` for RV).
    model_type
        Informational label identifying the model that produced these samples (e.g.
        ``"RVModel"``, ``"GaiaAstrometryModel"``, ``"JointModel"``). Stored in HDF5 for
        round-tripping.
    metadata
        Additional metadata (``time_ref``, ``num_chains``, acceptance rate, etc.).
    linear_extension_names
        Names of linear parameters introduced by extensions (instrument offsets,
        polynomial trends, etc.) beyond the base linear set.

    Examples
    --------
    ``Samples`` is normally produced by :meth:`~harv.RejectionSampler.run`, but can be
    constructed directly for testing or manual use:

    >>> from unxt import Q
    >>> from harv.samplers.samples import Samples
    >>> samples = Samples(
    ...     nonlinear={
    ...         "period": Q([100.0, 101.0, 99.5], "day"),
    ...         "eccentricity": Q([0.1, 0.15, 0.12], ""),
    ...         "phase_peri": Q([0.3, 0.31, 0.29], ""),
    ...         "arg_peri": Q([1.0, 1.1, 0.9], "rad"),
    ...     },
    ...     linear={
    ...         "rv_semiamp": Q([10.0, 11.0, 9.5], "km/s"),
    ...         "v_sys": Q([5.0, 5.1, 4.9], "km/s"),
    ...     },
    ...     model_type="RVModel",
    ...     metadata={"time_ref": 0.0, "time_ref_unit": "day"},
    ... )
    >>> samples.n_samples
    3
    >>> "period" in samples
    True
    >>> samples.keys()[:4]
    ['period', 'eccentricity', 'phase_peri', 'arg_peri']
    """

    # Pytree leaves -- Q arrays with units baked in
    nonlinear: dict[str, Q]
    linear: dict[str, Q]

    # Static fields -- not JAX leaves
    model_type: str = eqx.field(static=True, default="")
    metadata: dict[str, Any] = eqx.field(static=True, default_factory=dict)
    # Names of linear params introduced by extensions (offsets, trends, etc.).
    linear_extension_names: tuple[str, ...] = eqx.field(static=True, default=())

    # Optional per-sample log-probabilities from the sampling run.  Pytree
    # leaves (or None) so slicing and tree transforms carry them through.
    # Populated only when a sampler is run with ``return_logprobs=True``.
    ln_likelihood: jax.Array | None = None
    ln_prior: jax.Array | None = None

    @property
    def n_samples(self) -> int:
        """Number of posterior samples per batch entry.

        For a flat ``Samples`` (each parameter array 1-D) this is the total
        number of samples. For a batched ``Samples`` -- e.g. shape
        ``(N_stars, K_max)`` after :func:`pad_and_stack_samples` -- this is
        the trailing-axis length (samples per entity); the leading batch
        dimensions are exposed via :attr:`batch_shape`.
        """
        return int(next(iter(self.nonlinear.values())).shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading batch dimensions of the parameter arrays (``shape[:-1]``).

        Empty tuple for a flat ``Samples``; ``(N_stars,)`` after
        :func:`pad_and_stack_samples`.
        """
        return tuple(next(iter(self.nonlinear.values())).shape[:-1])

    @property
    def meta(self) -> _MetadataView:
        """Q-aware read-only view over :attr:`metadata`.

        ``metadata`` itself stores quantity-valued entries in split form
        (``<name>`` value + ``<name>_unit`` string) so the static field never
        holds a JAX array.  Use ``samples.meta[name]`` to read those entries
        back as :class:`~unxt.Q` instances without manually reassembling the
        two halves; plain scalar entries (e.g. ``num_chains``) round-trip
        unchanged.  See :class:`_MetadataView` for the full read API.
        """
        return _MetadataView(self.metadata)

    @property
    def ln_posterior(self) -> jax.Array:
        """Per-sample log-posterior density, ``ln_prior + ln_likelihood``.

        Both ``ln_prior`` and ``ln_likelihood`` must have been stored (run the
        sampler with ``return_logprobs=True``); otherwise :class:`ValueError`
        is raised.
        """
        if self.ln_prior is None or self.ln_likelihood is None:
            msg = (
                "ln_posterior requires both ln_prior and ln_likelihood to be "
                "stored; re-run the sampler with return_logprobs=True."
            )
            raise ValueError(msg)
        return self.ln_prior + self.ln_likelihood

    @property
    def weight(self) -> jax.Array:
        """Per-sample importance weight, normalized over the full prior library.

        ``w_i = exp(ln L_i - logsumexp(ln L))`` where the ``logsumexp`` runs
        over *every* prior draw the sampler evaluated, not just the ones
        returned.  It is reconstructed rather than stored: the normalization is
        ``ln_Z_int + ln(n_prior_samples)``, and both come from the evidence
        metadata that ``top_k`` / ``return_evidence_stats=True`` writes.

        Because the normalization spans the whole library, ``weight.sum()`` is
        the fraction of total posterior mass these samples capture -- reported
        directly as ``metadata["weight_captured"]`` on the ``top_k`` path.  It
        is **less than 1** whenever samples were truncated away, so posterior
        expectations need ``w / w.sum()``, and they are biased unless
        ``weight_captured`` is close to 1.  See ``docs/sharp-bits.md``.

        Also reachable as ``samples["weight"]``.  Deliberately *not* listed in
        :meth:`keys`, which enumerates model parameters and drives the default
        axes of :meth:`plot_corner`, :meth:`to_arviz` and :meth:`median`.

        Returns
        -------
        jax.Array
            Weights in the same order as the samples, shape ``(n_samples,)``.
            All-zero when every evaluated likelihood was non-finite.

        Raises
        ------
        ValueError
            If ``ln_likelihood`` or the evidence metadata is missing, or if
            this is a batched ``Samples``.

        Examples
        --------
        A library of four equally likely draws, of which two were returned:
        each carries weight ``1/4``, and half the posterior mass survived.

        >>> import jax.numpy as jnp
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 101.0], "day")},
        ...     linear={},
        ...     metadata={"ln_Z_int": 0.0, "n_prior_samples": 4},
        ...     ln_likelihood=jnp.zeros(2),
        ... )
        >>> samples.weight
        Array([0.25, 0.25], dtype=float64)
        >>> float(samples["weight"].sum())
        0.5
        """
        if self.batch_shape:
            msg = (
                "weight is not defined for a batched Samples: "
                "pad_and_stack_samples inherits metadata from the first entry, "
                "so its normalization does not apply to the other entries. "
                "Read weight from the per-entity Samples before stacking."
            )
            raise ValueError(msg)
        if self.ln_likelihood is None:
            msg = (
                "weight requires ln_likelihood to be stored; re-run the "
                "sampler with top_k=... or return_logprobs=True."
            )
            raise ValueError(msg)
        missing = [k for k in ("ln_Z_int", "n_prior_samples") if k not in self.metadata]
        if missing:
            msg = (
                f"weight requires the evidence metadata {missing} to normalize "
                "over the full prior library; re-run the sampler with "
                "top_k=... or return_evidence_stats=True."
            )
            raise ValueError(msg)

        ln_norm = float(self.metadata["ln_Z_int"]) + np.log(
            float(self.metadata["n_prior_samples"])
        )
        if not np.isfinite(ln_norm):
            # Every evaluated likelihood was non-finite; -inf - -inf is NaN,
            # and zero weight is the honest answer.
            return jnp.zeros_like(self.ln_likelihood)
        return jnp.exp(self.ln_likelihood - ln_norm)

    def keys(self) -> list[str]:
        """All available parameter names (nonlinear + linear + derived)."""
        base_keys = list(self.nonlinear.keys()) + list(self.linear.keys())
        derived_keys = ["log10_period"]
        # Kepler-free samples (e.g. Fourier parameterizations) have no
        # periastron phase, so time_peri is only derivable when phase_peri exists.
        if "phase_peri" in self.nonlinear:
            derived_keys.append("time_peri")
        if "cos_i" in self.nonlinear:
            derived_keys.append("inclination")
        if "rv_semiamp" in self.linear:
            derived_keys.append("binary_mass_function")
        if "semi_major_axis" in self.linear and "parallax" in self.linear:
            derived_keys.append("semi_major_axis_AU")
        return base_keys + derived_keys

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    @overload
    def __getitem__(self, key: str) -> "Q": ...

    @overload
    def __getitem__(self, key: int | slice | np.ndarray) -> "Samples": ...

    def __getitem__(  # noqa: C901
        self, key: str | int | slice | np.ndarray
    ) -> "Q | Samples":
        """Get parameter samples by name, or return a sliced ``Samples``.

        Parameters
        ----------
        key
            If ``str``, returns the named parameter array (with units).
            If ``int``, ``slice``, or boolean/integer array, returns a new
            ``Samples`` with all parameter arrays sliced along the sample axis.
            Integer keys are converted to length-1 slices to preserve 1-d shape.

        Returns
        -------
        values
            When ``key`` is a string.
        sliced
            When ``key`` is an int, slice, or array index.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 101.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 11.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.1], "km/s")},
        ...     model_type="RVModel",
        ...     metadata={"time_ref": 0.0, "time_ref_unit": "day"},
        ... )
        >>> samples["period"].unit
        Unit("d")
        >>> samples["rv_semiamp"].shape
        (2,)
        >>> samples[:1].n_samples
        1
        >>> samples[0].n_samples
        1
        >>> samples[-1].n_samples
        1
        >>> float(samples[-1]["period"].value[0])
        101.0
        """
        if not isinstance(key, str):
            # int/slice/array index — return a sliced Samples preserving 1-d shape
            if isinstance(key, int):
                n = self.n_samples
                if not -n <= key < n:
                    msg = (
                        f"Sample index {key} out of range for Samples with {n} samples"
                    )
                    raise IndexError(msg)
                if key < 0:
                    key += n
                idx = slice(key, key + 1)
            else:
                idx = key
            sliced_nl = {k: v[idx] for k, v in self.nonlinear.items()}
            sliced_lin = {k: v[idx] for k, v in self.linear.items()}
            return Samples(
                nonlinear=sliced_nl,
                linear=sliced_lin,
                model_type=self.model_type,
                metadata=self.metadata,
                linear_extension_names=self.linear_extension_names,
                ln_likelihood=(
                    None if self.ln_likelihood is None else self.ln_likelihood[idx]
                ),
                ln_prior=None if self.ln_prior is None else self.ln_prior[idx],
            )

        if key in self.nonlinear:
            return self.nonlinear[key]

        if key in self.linear:
            return self.linear[key]

        if key == "log10_period":
            period = self.nonlinear["period"]
            return jnp.log10(  # ty: ignore[invalid-return-type]
                ustrip(str(period.unit), period)
            )

        if key == "time_peri":
            # Express time_peri in absolute time: time_ref + phase_peri * period.
            # phase_peri encodes the fractional orbital phase at t=0, so
            # phase_peri * period is the periastron time relative to t=0, and
            # adding time_ref converts it to the same absolute coordinate as data.time.
            period = self.nonlinear["period"]
            time_unit = str(period.unit)
            time_ref = self.meta.get("time_ref")
            if time_ref is None:
                time_ref_val = 0.0
            elif isinstance(time_ref, AbstractQuantity):
                time_ref_val = float(ustrip(time_unit, time_ref))
            else:
                time_ref_val = float(time_ref)
            phase_peri = ustrip("", self.nonlinear["phase_peri"])
            period_val = ustrip(time_unit, period)
            return Q(time_ref_val + phase_peri * period_val, time_unit)

        if key == "inclination":
            if "cos_i" in self.nonlinear:
                cos_i = ustrip("", self.nonlinear["cos_i"])
                return Q(jnp.arccos(cos_i), "rad")
            msg = "Inclination only available for astrometry/combined data"
            raise KeyError(msg)

        if key == "weight":
            # Not a model parameter, so it stays out of keys() -- see the
            # ``weight`` property.
            return self.weight  # ty: ignore[invalid-return-type]

        if key == "binary_mass_function":
            return self.binary_mass_function()

        if key == "semi_major_axis_AU":
            return self.semi_major_axis_AU()

        msg = f"Parameter '{key}' not found"
        raise KeyError(msg)

    def __len__(self) -> int:
        """Number of samples."""
        return self.n_samples

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Samples(n_samples={self.n_samples}, "
            f"model_type='{self.model_type}', "
            f"parameters={len(self.keys())})"
        )

    def wrap_angles(self) -> "Samples":  # noqa: C901 -- two-step angle wrapping
        """Wrap negative ``rv_semiamp`` / ``semi_major_axis`` to positive.

        Negative ``rv_semiamp`` (``K``) or ``semi_major_axis`` (``a``) describes
        an orbit that is physically identical to the all-positive case under two
        symmetries of the model:

        * ``arg_peri -> arg_peri + pi`` flips the sign of *both* ``K`` and ``a``
          (the ``(omega, K, a) -> (omega + pi, -K, -a)`` symmetry);
        * ``lon_asc_node -> lon_asc_node + pi`` flips the sign of ``a`` *alone*
          (the astrometric ``(Omega, a) -> (Omega + pi, -a)`` symmetry;
          ``lon_asc_node`` does not enter the RV model).

        This method returns a new :class:`Samples` enforcing ``K >= 0`` and
        ``a >= 0`` in two steps:

        1. shift ``arg_peri`` by ``pi`` (flipping every ``rv_semiamp`` and
           ``semi_major_axis``) on the samples where the trigger amplitude is
           negative — enforcing ``K >= 0``;
        2. shift ``lon_asc_node`` by ``pi`` (flipping every ``semi_major_axis``)
           on the samples where ``a`` is *still* negative — enforcing ``a >= 0``.

        Both shifted angles are wrapped to ``[0, 2*pi)``.  A single ``arg_peri``
        shift cannot make both ``K`` and ``a`` positive when their signs disagree
        (which routinely happens in joint RV + astrometry posteriors), so the
        second ``lon_asc_node`` shift is required.

        Joint models (e.g. SB2) namespace per-component linear parameters as
        ``"component.param_name"``; this method discovers every ``rv_semiamp``-
        and ``semi_major_axis``-suffixed key.  The first ``rv_semiamp``-suffixed
        key (insertion order) triggers step 1; ``semi_major_axis`` triggers
        step 1 only when no ``rv_semiamp`` is present (astrometry-only fits).

        No-op when ``arg_peri`` is absent from ``nonlinear``, when neither
        ``rv_semiamp`` nor ``semi_major_axis`` is in ``linear``, or when no
        entries are negative.

        Raises
        ------
        NotImplementedError
            If the model has multiple per-component ``arg_peri`` or
            ``lon_asc_node`` keys (e.g. ``"primary.arg_peri"`` and
            ``"secondary.arg_peri"``).  All current harv joint factories share
            both, so this is not expected to arise in practice.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 100.0], "day"),
        ...                "eccentricity": Q([0.1, 0.1], ""),
        ...                "phase_peri": Q([0.3, 0.3], ""),
        ...                "arg_peri": Q([1.0, 1.0], "rad")},
        ...     linear={"rv_semiamp": Q([-10.0, 10.0], "km/s"),
        ...             "v_sys": Q([0.0, 0.0], "km/s")},
        ...     model_type="RVModel",
        ...     metadata={"time_ref": 0.0, "time_ref_unit": "day"},
        ... )
        >>> wrapped = samples.wrap_angles()
        >>> bool((wrapped["rv_semiamp"].value >= 0).all())
        True
        """
        # `arg_peri` must exist; otherwise there's no shift to apply.  We
        # support the typical case of a single shared `arg_peri` (either bare
        # or under a single component's namespace).  Per-component arg_peri is
        # not the default for any joint factory, so out-of-scope here.
        omega_keys = _find_namespaced_keys(self.nonlinear, "arg_peri")
        if not omega_keys:
            return self
        if len(omega_keys) > 1:
            msg = (
                "wrap_angles does not yet support multiple per-component "
                f"arg_peri keys; got {omega_keys!r}.  All current harv joint "
                "models share arg_peri, so this should not arise in practice."
            )
            raise NotImplementedError(msg)
        omega_key = omega_keys[0]

        # Discover every rv_semiamp- and semi_major_axis-suffixed key.  This
        # matches both bare names (single-component) and dot-namespaced names
        # like "primary.rv_semiamp" (joint / SB2).
        K_keys = _find_namespaced_keys(self.linear, "rv_semiamp")
        a_keys = _find_namespaced_keys(self.linear, "semi_major_axis")

        # Trigger for the `arg_peri` shift: prefer `rv_semiamp` (RV pins the
        # sign convention).  For astrometry-only fits there is no `rv_semiamp`,
        # so `semi_major_axis` triggers the shift instead.
        if K_keys:
            omega_trigger = self.linear[K_keys[0]]
        elif a_keys:
            omega_trigger = self.linear[a_keys[0]]
        else:
            return self

        # Step 1 — `arg_peri` shift.  `arg_peri -> arg_peri + pi` flips the
        # sign of *both* `rv_semiamp` and `semi_major_axis` (the
        # `(omega, K, a) -> (omega + pi, -K, -a)` model symmetry), enforcing
        # `K >= 0`.
        omega_flip = ustrip(str(omega_trigger.unit), omega_trigger) < 0

        # Step 2 — `lon_asc_node` shift.  `lon_asc_node -> lon_asc_node + pi`
        # flips `semi_major_axis` *alone* (the astrometric
        # `(Omega, a) -> (Omega + pi, -a)` symmetry; `Omega` does not enter the
        # RV model).  A single `arg_peri` shift cannot make both `K` and `a`
        # positive when their signs disagree, so this second shift fixes any
        # `semi_major_axis` still negative after step 1.
        node_flip = None
        node_key = None
        if a_keys:
            node_keys = _find_namespaced_keys(self.nonlinear, "lon_asc_node")
            if len(node_keys) > 1:
                msg = (
                    "wrap_angles does not yet support multiple per-component "
                    f"lon_asc_node keys; got {node_keys!r}.  All current harv "
                    "joint models share lon_asc_node, so this should not arise "
                    "in practice."
                )
                raise NotImplementedError(msg)
            if node_keys:
                node_key = node_keys[0]
                # `semi_major_axis` sign *after* the `arg_peri` shift.
                a0 = self.linear[a_keys[0]]
                a0_val = ustrip(str(a0.unit), a0)
                node_flip = jnp.where(omega_flip, -a0_val, a0_val) < 0

        if not jnp.any(omega_flip) and (node_flip is None or not jnp.any(node_flip)):
            return self

        new_lin = dict(self.linear)
        new_nl = dict(self.nonlinear)

        # Apply step 1: flip every K and every a, and shift `arg_peri`.
        for k in (*K_keys, *a_keys):
            v = new_lin[k]
            v_val = ustrip(str(v.unit), v)
            new_lin[k] = Q(jnp.where(omega_flip, -v_val, v_val), v.unit)

        arg_peri = self.nonlinear[omega_key]
        arg_val = ustrip(str(arg_peri.unit), arg_peri)
        new_nl[omega_key] = Q(
            jnp.where(omega_flip, jnp.mod(arg_val + jnp.pi, 2.0 * jnp.pi), arg_val),
            arg_peri.unit,
        )

        # Apply step 2: flip every still-negative a, and shift `lon_asc_node`.
        if node_flip is not None and node_key is not None:
            for k in a_keys:
                v = new_lin[k]
                v_val = ustrip(str(v.unit), v)
                new_lin[k] = Q(jnp.where(node_flip, -v_val, v_val), v.unit)

            node = self.nonlinear[node_key]
            node_val = ustrip(str(node.unit), node)
            new_nl[node_key] = Q(
                jnp.where(
                    node_flip, jnp.mod(node_val + jnp.pi, 2.0 * jnp.pi), node_val
                ),
                node.unit,
            )

        return Samples(
            nonlinear=new_nl,
            linear=new_lin,
            model_type=self.model_type,
            metadata=self.metadata,
            linear_extension_names=self.linear_extension_names,
            ln_likelihood=self.ln_likelihood,
            ln_prior=self.ln_prior,
        )

    def thiele_innes_to_campbell(self) -> "Samples":
        """Convert Thiele-Innes linear parameters to Campbell orbital elements.

        See :func:`~harv.kepler.orbits.campbell_from_thiele_innes` for the mathematical
        details of the conversion.

        Returns
        -------
        Samples
            New :class:`Samples` with ``semi_major_axis``, ``arg_peri``,
            ``lon_asc_node``, ``cos_i`` (replacing the four TI constants).

        """
        ti_names = ("ti_A", "ti_B", "ti_F", "ti_G")
        if not any(n in self.linear for n in ti_names):
            return self
        if not all(n in self.linear for n in ti_names):
            msg = "TI to Campbell conversion requires linear parameters: " + ", ".join(
                ti_names
            )
            raise RuntimeError(msg)

        return self.convert_parameterization(
            source=ThieleInnesGaiaAstrometry(),
            target=StandardGaiaAstrometry(),
        )

    def convert_parameterization(
        self,
        *,
        source: AbstractParameterization,
        target: AbstractParameterization,
    ) -> "Samples":
        """Convert stored values between supported parameterizations.

        Wraps :func:`harv.samplers.convert_parameterization`, returning a new
        :class:`Samples` with ``metadata``, ``model_type``, and
        ``linear_extension_names`` preserved.  The initial implementation
        supports single-component RV and Gaia astrometry parameterizations only.
        """
        new_nonlinear, new_linear = convert_parameterization(
            self.nonlinear,
            self.linear,
            source=source,
            target=target,
        )
        return Samples(
            nonlinear=new_nonlinear,
            linear=new_linear,
            model_type=self.model_type,
            metadata=self.metadata,
            linear_extension_names=self.linear_extension_names,
            ln_likelihood=self.ln_likelihood,
            ln_prior=self.ln_prior,
        )

    def median(
        self, param: str | None = None
    ) -> dict[str, AbstractQuantity | jnp.ndarray] | AbstractQuantity | jnp.ndarray:
        """Compute median values for parameters.

        Parameters
        ----------
        param
            If provided, return the median for this parameter only.
            If None, return a dict of medians for all parameters.

        Returns
        -------
            Median value(s).

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 102.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 12.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.2], "km/s")},
        ...     model_type="RVModel",
        ...     metadata={"time_ref": 0.0, "time_ref_unit": "day"},
        ... )
        >>> med = samples.median("period")
        >>> med.unit
        Unit("d")
        >>> all_medians = samples.median()
        >>> "period" in all_medians
        True
        """
        if param is not None:
            return jnp.median(self[param])

        result: dict[str, AbstractQuantity | jnp.ndarray] = {}
        for name in self.keys():
            try:
                result[name] = jnp.median(self[name])
            except (KeyError, ValueError):
                continue
        return result

    def percentile(
        self, param: str, percentiles: list[float] | tuple[float, ...] = (16, 50, 84)
    ) -> list[AbstractQuantity | jnp.ndarray]:
        """Compute percentiles for a parameter.

        Parameters
        ----------
        param
            Parameter name.
        percentiles
            Percentile values to compute (0-100). Default: (16, 50, 84)
            which corresponds to the 16th, 50th, 84th percentiles for Gaussian.

        Returns
        -------
            Percentile values with appropriate units.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 102.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 12.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.2], "km/s")},
        ...     model_type="RVModel",
        ...     metadata={"time_ref": 0.0, "time_ref_unit": "day"},
        ... )
        >>> p16, p50, p84 = samples.percentile("eccentricity")
        >>> len(samples.percentile("period", [5, 50, 95]))
        3
        """
        values = self[param]
        return [jnp.percentile(values, p) for p in percentiles]

    def summary(self, params: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Compute summary statistics for parameters.

        For each parameter, computes:
        - median
        - mean
        - std (standard deviation)
        - percentiles (16th, 84th) for +/-1-sigma equivalent

        Parameters
        ----------
        params
            List of parameter names to summarize. If None, summarizes all.

        Returns
        -------
            Dictionary mapping parameter names to their statistics.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.samplers.samples import Samples
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 102.0], "day"),
        ...                "eccentricity": Q([0.1, 0.15], ""),
        ...                "phase_peri": Q([0.3, 0.31], ""),
        ...                "arg_peri": Q([1.0, 1.1], "rad")},
        ...     linear={"rv_semiamp": Q([10.0, 12.0], "km/s"),
        ...             "v_sys": Q([5.0, 5.2], "km/s")},
        ...     model_type="RVModel",
        ...     metadata={"time_ref": 0.0, "time_ref_unit": "day"},
        ... )
        >>> summary = samples.summary(["period", "eccentricity"])
        >>> sorted(summary.keys())
        ['eccentricity', 'period']
        >>> sorted(summary["period"].keys())
        ['mean', 'median', 'p16', 'p84', 'std']
        """
        if params is None:
            params = self.keys()

        result: dict[str, dict[str, Any]] = {}
        for key in params:
            try:
                values = self[key]
                result[key] = {
                    "median": jnp.median(values),
                    "mean": jnp.mean(values),
                    "std": jnp.std(values),
                    "p16": jnp.percentile(values, 16),
                    "p84": jnp.percentile(values, 84),
                }
            except (KeyError, ValueError):
                continue

        return result

    # ========================================================================
    # Sample analysis (ported from thejoker.samples_analysis)
    #

    def _require_single_component(self, method: str) -> None:
        """Raise if the samples carry namespaced (joint-model) parameters."""
        namespaced = sorted(k for k in (*self.nonlinear, *self.linear) if "." in k)
        if namespaced:
            msg = (
                f"{method}() supports single-component samples only; namespaced "
                f"(joint-model) parameters are not supported: {namespaced!r}"
            )
            raise NotImplementedError(msg)

    def _phases(self, data: AbstractData) -> np.ndarray:
        """Return ``(n_samples, n_obs)`` orbital phases in ``[0, 1)``.

        Phase is ``((time - time_ref) / period) mod 1`` evaluated at each sample's
        period.
        """
        period = self.nonlinear["period"]
        time_unit = str(period.unit)
        time = np.asarray(ustrip(time_unit, data.time))
        time_ref = (
            0.0 if data.time_ref is None else float(ustrip(time_unit, data.time_ref))
        )
        period_val = np.asarray(ustrip(time_unit, period))
        return ((time[None, :] - time_ref) / period_val[:, None]) % 1.0

    def map_sample(
        self, *, return_index: bool = False
    ) -> "Samples | tuple[Samples, int]":
        """Return the maximum a posteriori (MAP) sample.

        Selects the single sample with the highest :attr:`ln_posterior`.

        Parameters
        ----------
        return_index
            If ``True``, also return the integer index of the MAP sample.

        Returns
        -------
            A length-1 :class:`Samples` with the MAP sample, or
            ``(Samples, index)`` when ``return_index`` is ``True``.

        Raises
        ------
        ValueError
            If per-sample log-probabilities were not stored (run the sampler
            with ``return_logprobs=True``).
        """
        idx = int(jnp.argmax(self.ln_posterior))
        map_sample = self[idx]
        if return_index:
            return map_sample, idx
        return map_sample

    def acceptance_diagnostics(
        self, *, min_evidence_ess: float = MIN_EVIDENCE_ESS
    ) -> dict[str, Any]:
        """Assess whether the rejection run resolved the posterior.

        The rejection step accepts each prior draw with probability
        ``exp(L - max L)``, so the accepted-sample count is only a meaningful
        posterior size once ``max_ln_likelihood`` has converged to the true
        peak. When the evidence effective sample size (``ln_Z_int_ess``) is
        O(1), the evidence integral is dominated by a single lucky draw:
        ``max_ln_likelihood`` is likely under-resolved and the count is
        misleading (a broad prior can "accept" a poor fit simply because it
        never sampled a good one). See ``docs/spec.md``, "Interpreting
        acceptance".

        Requires the sampler to have been run with
        ``return_evidence_stats=True``.

        Parameters
        ----------
        min_evidence_ess
            Evidence ESS at or above which the run counts as resolved. Defaults
            to :data:`MIN_EVIDENCE_ESS` (3.0), the same bar the sampler warns
            at; see that constant for what the number means and how to pick
            another. ``0.0`` always reports resolved, ``float("inf")`` never
            does.

        Returns
        -------
            A dict with ``n_prior_samples``, ``n_accepted``, ``evidence_ess``,
            ``min_evidence_ess``, ``max_ln_likelihood``, ``ln_Z_int``, a
            boolean ``well_resolved``, and a human-readable ``message``.

        Raises
        ------
        ValueError
            If evidence statistics were not stored (re-run with
            ``return_evidence_stats=True``).
        """
        missing = [k for k in _EVIDENCE_KEYS if k not in self.metadata]
        if missing:
            msg = (
                "acceptance_diagnostics requires evidence statistics; re-run the "
                f"sampler with return_evidence_stats=True (missing: {missing})."
            )
            raise ValueError(msg)
        n_prior = int(self.metadata["n_prior_samples"])
        ess = float(self.metadata["ln_Z_int_ess"])
        max_ll = float(self.metadata["max_ln_likelihood"])
        n_accepted = self.n_samples
        well_resolved, message = _assess_resolution(
            n_prior=n_prior,
            n_accepted=n_accepted,
            evidence_ess=ess,
            max_ln_likelihood=max_ll,
            min_evidence_ess=min_evidence_ess,
        )
        return {
            "n_prior_samples": n_prior,
            "n_accepted": n_accepted,
            "evidence_ess": ess,
            "min_evidence_ess": min_evidence_ess,
            "max_ln_likelihood": max_ll,
            "ln_Z_int": float(self.metadata["ln_Z_int"]),
            "well_resolved": well_resolved,
            "message": message,
        }

    def period_unimodal(self, data: AbstractData) -> bool:
        """Whether the period samples lie within a single mode.

        Uses the criterion ``ptp(P) < 4 * P_min**2 / (2*pi*T)``, where ``T`` is
        the data time span -- the period spacing at which adjacent aliases
        become resolvable (see thejoker's ``is_P_unimodal``).

        Parameters
        ----------
        data
            The data the samples were fit to (for the observed time span).
        """
        self._require_single_component("period_unimodal")
        period = self.nonlinear["period"]
        time_unit = str(period.unit)
        period_val = np.asarray(ustrip(time_unit, period))
        time = np.asarray(ustrip(time_unit, data.time))
        span = float(np.ptp(time))
        p_min = float(np.min(period_val))
        delta = 4.0 * p_min**2 / (2.0 * np.pi * span)
        return bool(np.ptp(period_val) < delta)

    def period_modes(
        self, data: AbstractData, n_clusters: int = 2
    ) -> tuple[bool, Q, np.ndarray]:
        """Cluster the period samples and test each mode for unimodality.

        Runs K-means on ``log(period)`` (experimental; see thejoker's
        ``is_P_Kmodal``).  Requires the optional ``scikit-learn`` dependency.

        Parameters
        ----------
        data
            The data the samples were fit to.
        n_clusters
            Number of period modes to cluster into. Default 2.

        Returns
        -------
            ``(all_unimodal, mode_periods, n_per_mode)`` -- whether every mode
            is individually unimodal, the median period of each mode (a ``Q``),
            and the sample count per mode.
        """
        self._require_single_component("period_modes")
        try:
            from sklearn.cluster import KMeans  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            msg = "period_modes() requires the optional 'scikit-learn' dependency."
            raise ImportError(msg) from exc

        period = self.nonlinear["period"]
        time_unit = str(period.unit)
        period_val = np.asarray(ustrip(time_unit, period))
        labels = KMeans(n_clusters=n_clusters).fit_predict(
            np.log(period_val).reshape(-1, 1)
        )

        unimodal: list[bool] = []
        mode_periods: list[float] = []
        n_per_mode: list[int] = []
        for j in np.unique(labels):
            mask = labels == j
            sub = self[mask]
            unimodal.append(True if sub.n_samples == 1 else sub.period_unimodal(data))
            mode_periods.append(float(np.median(period_val[mask])))
            n_per_mode.append(int(mask.sum()))

        return all(unimodal), Q(np.array(mode_periods), time_unit), np.array(n_per_mode)

    def max_phase_gap(self, data: AbstractData) -> np.ndarray:
        """Largest gap in orbital-phase coverage, per sample.

        The maximum gap between consecutive observations on the (circular)
        phase axis -- the ESA Gaia "maximum phase gap" statistic.

        Parameters
        ----------
        data
            The data the samples were fit to.

        Returns
        -------
            Array of shape ``(n_samples,)``; values in ``[0, 1]``.
        """
        self._require_single_component("max_phase_gap")
        phases = np.sort(self._phases(data), axis=1)
        if phases.shape[1] < 2:
            return np.ones(phases.shape[0])
        gaps = np.diff(phases, axis=1)
        wrap = 1.0 - (phases[:, -1] - phases[:, 0])
        return np.maximum(gaps.max(axis=1), wrap)

    def phase_coverage(self, data: AbstractData, n_bins: int = 10) -> np.ndarray:
        """Fraction of phase bins containing at least one observation.

        The ESA Gaia "phase coverage" statistic, per sample.

        Parameters
        ----------
        data
            The data the samples were fit to.
        n_bins
            Number of equal-width phase bins. Default 10.

        Returns
        -------
            Array of shape ``(n_samples,)``; values in ``[0, 1]``.
        """
        self._require_single_component("phase_coverage")
        phases = self._phases(data)  # (n_samples, n_obs)
        bin_idx = np.clip((phases * n_bins).astype(int), 0, n_bins - 1)
        # Count occupied bins per sample with bincount, so the only allocations
        # are bin_idx and per-row (n_bins,) temporaries -- never an
        # (n_samples, n_obs, n_bins) broadcast tensor.
        occupied = np.array(
            [np.count_nonzero(np.bincount(row, minlength=n_bins)) for row in bin_idx]
        )
        return occupied / n_bins

    def periods_spanned(self, data: AbstractData) -> np.ndarray:
        """Number of orbital periods spanned by the data, per sample.

        Parameters
        ----------
        data
            The data the samples were fit to.

        Returns
        -------
            Array of shape ``(n_samples,)``.
        """
        self._require_single_component("periods_spanned")
        period = self.nonlinear["period"]
        time_unit = str(period.unit)
        time = np.asarray(ustrip(time_unit, data.time))
        period_val = np.asarray(ustrip(time_unit, period))
        return float(np.ptp(time)) / period_val

    def phase_coverage_per_period(self, data: AbstractData) -> np.ndarray:
        """Maximum number of observations within any single period, per sample.

        Parameters
        ----------
        data
            The data the samples were fit to.

        Returns
        -------
            Array of shape ``(n_samples,)``.
        """
        self._require_single_component("phase_coverage_per_period")
        period = self.nonlinear["period"]
        time_unit = str(period.unit)
        time = np.asarray(ustrip(time_unit, data.time))
        time_ref = (
            0.0 if data.time_ref is None else float(ustrip(time_unit, data.time_ref))
        )
        period_val = np.asarray(ustrip(time_unit, period))
        n_per = (time - time_ref) / period_val[:, None]  # (n_samples, n_obs)

        out = np.empty(n_per.shape[0], dtype=int)
        for s, row in enumerate(n_per):
            # Two integer-period binnings offset by half a period catch windows
            # that straddle a bin edge; take the larger maximum count.
            base = np.bincount((row - row.min()).astype(int))
            offset = np.bincount((row - row.min() + 0.5).astype(int))
            out[s] = max(int(base.max()), int(offset.max()))
        return out

    def chi2(self, data: AbstractData, model: Any) -> jax.Array:
        r"""Per-sample goodness-of-fit :math:`\chi^2` against the data.

        For each posterior sample, evaluates the model prediction at the stored
        parameter values and returns
        :math:`\chi^2 = r^\top C^{-1} r` (see
        :meth:`~harv.models.component.AbstractComponentModel.chi_squared`).
        Jitter and other extension noise terms are included via ``C``.

        Parameters
        ----------
        data
            The data the samples were fit to.
        model
            The component model used for the fit (``RVModel`` or
            ``GaiaAstrometryModel``); provides the prediction and noise model.

        Returns
        -------
            Array of shape ``(n_samples,)``.
        """
        self._require_single_component("chi2")

        nonlinear_for_model, linear_stripped = _assemble_sample_params(
            self,
            model,
            data,
            i=None,
        )

        def _one(nonlinear_i: dict[str, Any], linear_i: dict[str, Any]) -> jax.Array:
            return model.chi_squared(nonlinear_i, linear_i, data)

        return jax.vmap(_one)(nonlinear_for_model, linear_stripped)

    def reduced_chi2(
        self, data: AbstractData, model: Any, *, dof: int | None = None
    ) -> jax.Array:
        r"""Per-sample reduced :math:`\chi^2` (:math:`\chi^2 / \mathrm{dof}`).

        Parameters
        ----------
        data
            The data the samples were fit to.
        model
            The component model used for the fit.
        dof
            Degrees of freedom. Defaults to ``n_obs - n_params``, where
            ``n_params`` counts every fitted parameter (orbital + linear +
            extension). A well-fitting model gives reduced :math:`\chi^2` near 1.

        Returns
        -------
            Array of shape ``(n_samples,)``.

        Raises
        ------
        ValueError
            If ``dof`` (default or supplied) is not positive.
        """
        if dof is None:
            n_params = len(model._all_nonlinear_names()) + len(
                model._all_linear_names()
            )
            dof = int(data.n_obs) - n_params
        if dof <= 0:
            msg = (
                f"Degrees of freedom must be positive, got dof={dof} "
                f"(n_obs={int(data.n_obs)}). Pass an explicit dof= if needed."
            )
            raise ValueError(msg)
        return self.chi2(data, model) / dof

    # ========================================================================
    # Derived physical quantities (masses, physical orbit size)
    #

    def binary_mass_function(self) -> Q:
        """Binary mass function from the RV orbital elements.

        See :func:`harv.kepler.masses.binary_mass_function`.

        Returns
        -------
            The binary mass function (a ``Q`` in solar masses), one per sample.

        Raises
        ------
        KeyError
            If the samples do not contain ``rv_semiamp`` (not an RV fit).
        """
        self._require_single_component("binary_mass_function")
        if "rv_semiamp" not in self.linear:
            msg = "binary_mass_function() requires RV samples with 'rv_semiamp'."
            raise KeyError(msg)
        return masses.binary_mass_function(
            self.nonlinear["period"],
            self.linear["rv_semiamp"],
            self.nonlinear["eccentricity"],
        )

    def semi_major_axis_AU(self) -> Q:
        """Physical semi-major axis (AU) from the angular orbit size + parallax.

        See :func:`harv.kepler.masses.semi_major_axis_physical`.

        Returns
        -------
            The physical semi-major axis (a ``Q`` in AU), one per sample.

        Raises
        ------
        KeyError
            If the samples lack ``semi_major_axis`` or ``parallax``.
        """
        self._require_single_component("semi_major_axis_AU")
        if "semi_major_axis" not in self.linear or "parallax" not in self.linear:
            msg = (
                "semi_major_axis_AU() requires astrometry samples with "
                "'semi_major_axis' and 'parallax'."
            )
            raise KeyError(msg)
        return masses.semi_major_axis_physical(
            self.linear["semi_major_axis"], self.linear["parallax"]
        )

    def companion_mass(self, m_primary: Q, *, sin_i: float | None = None) -> Q:
        """Companion mass :math:`m_2` given the primary mass.

        For RV samples the mass function is
        :func:`~harv.kepler.masses.binary_mass_function`; ``sin_i`` defaults to 1
        (the *minimum* companion mass).  For astrometry samples the dark-companion
        astrometric mass function is used and ``sin_i`` is ignored (the inclination
        is already encoded in the physical orbit size).

        Parameters
        ----------
        m_primary
            Primary mass (a ``Q``).
        sin_i
            Sine of the inclination, for RV samples only. Default 1 (edge-on).

        Returns
        -------
            The companion mass (a ``Q`` in solar masses), one per sample.
        """
        self._require_single_component("companion_mass")
        is_astrometry = "semi_major_axis" in self.linear and "parallax" in self.linear
        if "rv_semiamp" in self.linear and not is_astrometry:
            mass_function = self.binary_mass_function()
            return masses.companion_mass_from_mass_function(
                mass_function, m_primary, 1.0 if sin_i is None else sin_i
            )
        if is_astrometry:
            mass_function = masses.astrometric_mass_function(
                self.semi_major_axis_AU(), self.nonlinear["period"]
            )
            return masses.companion_mass_from_mass_function(
                mass_function, m_primary, 1.0
            )
        msg = (
            "companion_mass() needs either RV ('rv_semiamp') or astrometry "
            "('semi_major_axis' + 'parallax') samples."
        )
        raise KeyError(msg)

    def minimum_companion_mass(self, m_primary: Q) -> Q:
        """Minimum companion mass (edge-on, ``sin i = 1``).

        Convenience wrapper for :meth:`companion_mass` with ``sin_i=1``.

        Parameters
        ----------
        m_primary
            Primary mass (a ``Q``).
        """
        return self.companion_mass(m_primary, sin_i=1.0)

    def to_columns(self) -> SampleColumns:
        """Flatten to plain NumPy columns plus the structure needed to rebuild.

        This is the format-neutral mapping between a ``Samples`` and a table;
        :meth:`to_hdf5` is built on it, and so is any downstream writer (e.g.
        hq's Parquet results). :meth:`from_columns` inverts it exactly.

        Returns
        -------
            A :class:`SampleColumns`. Arrays keep the full shape of the
            parameter arrays, so a batched ``Samples`` round-trips too.

        Raises
        ------
        ValueError
            If a parameter is named ``"ln_likelihood"`` or ``"ln_prior"``
            (reserved for the log-probability columns), or if a name appears
            in both ``nonlinear`` and ``linear``.

        Examples
        --------
        >>> from unxt import Q
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 101.0], "day")},
        ...     linear={"v_sys": Q([5.0, 5.1], "km/s")},
        ...     model_type="RVModel",
        ... )
        >>> cols = samples.to_columns()
        >>> list(cols.columns), cols.units["v_sys"]
        (['period', 'v_sys'], 'km / s')
        """
        reserved = [k for k in (*self.nonlinear, *self.linear) if k in _LOGPROB_COLUMNS]
        if reserved:
            msg = (
                f"Parameter name(s) {reserved} are reserved for the per-sample "
                f"log-probability columns {_LOGPROB_COLUMNS}."
            )
            raise ValueError(msg)
        overlap = sorted(set(self.nonlinear) & set(self.linear))
        if overlap:
            msg = f"Parameter name(s) {overlap} appear in both nonlinear and linear."
            raise ValueError(msg)

        columns: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        for group in (self.nonlinear, self.linear):
            for name, qty in group.items():
                columns[name] = np.asarray(qty.value)
                units[name] = str(qty.unit)
        for name in _LOGPROB_COLUMNS:
            value = getattr(self, name)
            if value is not None:
                columns[name] = np.asarray(value)

        # The metadata invariant says only JSON-friendly scalars live here
        # (Q-valued entries in split form); anything else is out of contract
        # and is dropped, as to_hdf5 always has.
        metadata = {
            k: v for k, v in self.metadata.items() if isinstance(v, int | float | str)
        }
        return SampleColumns(
            columns=columns,
            units=units,
            nonlinear_names=tuple(self.nonlinear),
            linear_names=tuple(self.linear),
            model_type=self.model_type,
            linear_extension_names=tuple(self.linear_extension_names),
            metadata=metadata,
        )

    @classmethod
    def from_columns(cls, columns: SampleColumns) -> "Samples":
        """Rebuild a ``Samples`` from :meth:`to_columns` output.

        Columns may be any array-like (NumPy arrays from a file reader, for
        instance); they are converted to JAX arrays. Numpy scalars in
        ``metadata`` (as h5py and pyarrow return them) are converted to Python
        scalars, as the static ``metadata`` field requires.

        Parameters
        ----------
        columns
            The flattened samples.

        Returns
        -------
            The reconstructed ``Samples``.

        Raises
        ------
        ValueError
            If a named parameter has no column or no unit.

        Examples
        --------
        >>> from unxt import Q
        >>> samples = Samples(
        ...     nonlinear={"period": Q([100.0, 101.0], "day")},
        ...     linear={"v_sys": Q([5.0, 5.1], "km/s")},
        ...     model_type="RVModel",
        ... )
        >>> Samples.from_columns(samples.to_columns())["v_sys"]
        Quantity(Array([5. , 5.1], dtype=float64), unit='km / s')
        """
        names = (*columns.nonlinear_names, *columns.linear_names)
        missing_columns = [n for n in names if n not in columns.columns]
        missing_units = [n for n in names if n not in columns.units]
        if missing_columns or missing_units:
            msg = (
                "SampleColumns is missing parameter data: columns for "
                f"{missing_columns}, units for {missing_units}."
            )
            raise ValueError(msg)

        def _quantities(group: tuple[str, ...]) -> dict[str, Q]:
            return {
                name: Q(jnp.asarray(columns.columns[name]), columns.units[name])
                for name in group
            }

        logprobs = {
            name: jnp.asarray(columns.columns[name])
            if name in columns.columns
            else None
            for name in _LOGPROB_COLUMNS
        }
        return cls(
            nonlinear=_quantities(columns.nonlinear_names),
            linear=_quantities(columns.linear_names),
            model_type=columns.model_type,
            linear_extension_names=tuple(columns.linear_extension_names),
            metadata={k: _python_scalar(v) for k, v in columns.metadata.items()},
            **logprobs,
        )

    def to_hdf5(self, filename: str | Path) -> None:
        """Save samples to HDF5 file.

        Built on :meth:`to_columns`: parameters go to ``nonlinear/`` and
        ``linear/`` datasets with a ``unit`` attribute, the log-probabilities to
        top-level datasets, and the rest to attributes of ``metadata/``.

        Parameters
        ----------
        filename
            Output HDF5 filename.

        Examples
        --------
        Save posterior samples and reload them:

        >>> samples.to_hdf5("posterior_samples.h5")  # doctest: +SKIP
        >>> reloaded = Samples.from_hdf5("posterior_samples.h5")  # doctest: +SKIP
        >>> reloaded.n_samples == samples.n_samples  # doctest: +SKIP
        True
        """
        cols = self.to_columns()

        with h5py.File(Path(filename), "w") as f:
            for group_name, names in (
                ("nonlinear", cols.nonlinear_names),
                ("linear", cols.linear_names),
            ):
                group = f.create_group(group_name)
                for name in names:
                    ds = group.create_dataset(name, data=cols.columns[name])
                    ds.attrs["unit"] = cols.units[name]

            for name in _LOGPROB_COLUMNS:
                if name in cols.columns:
                    f.create_dataset(name, data=cols.columns[name])

            meta_group = f.create_group("metadata")
            meta_group.attrs["model_type"] = cols.model_type
            meta_group.attrs["linear_extension_names"] = ",".join(
                cols.linear_extension_names
            )
            meta_group.attrs["n_samples"] = self.n_samples
            for key, value in cols.metadata.items():
                meta_group.attrs[key] = value

    @classmethod
    def from_hdf5(cls, filename: str | Path) -> "Samples":
        """Load samples from HDF5 file.

        Reads the layout :meth:`to_hdf5` writes into a :class:`SampleColumns`
        and rebuilds with :meth:`from_columns`.

        Parameters
        ----------
        filename
            Input HDF5 filename.

        Returns
        -------
            Loaded samples object.

        Examples
        --------
        >>> samples = Samples.from_hdf5("posterior_samples.h5")  # doctest: +SKIP
        >>> samples.n_samples  # doctest: +SKIP
        42
        >>> samples.model_type  # doctest: +SKIP
        'rv'
        """
        with h5py.File(Path(filename), "r") as f:
            meta = f["metadata"]

            # ``offset_names`` is the pre-rename spelling of
            # ``linear_extension_names`` in older files.
            raw_extra = meta.attrs.get("linear_extension_names", "") or meta.attrs.get(
                "offset_names", ""
            )
            structural = {
                "linear_extension_names",
                "offset_names",
                "n_samples",
                "model_type",
            }
            metadata = {k: v for k, v in meta.attrs.items() if k not in structural}

            columns: dict[str, np.ndarray] = {}
            units: dict[str, str] = {}
            names: dict[str, tuple[str, ...]] = {}
            for group_name in ("nonlinear", "linear"):
                group = f[group_name]
                names[group_name] = tuple(group)
                for key in group:
                    columns[key] = group[key][:]
                    units[key] = group[key].attrs.get("unit", "")

            # Optional per-sample log-probabilities (absent in older files).
            for name in _LOGPROB_COLUMNS:
                if name in f:
                    columns[name] = f[name][:]

            cols = SampleColumns(
                columns=columns,
                units=units,
                nonlinear_names=names["nonlinear"],
                linear_names=names["linear"],
                model_type=meta.attrs.get("model_type", ""),
                linear_extension_names=tuple(raw_extra.split(",")) if raw_extra else (),
                metadata=metadata,
            )
        return cls.from_columns(cols)

    def to_arviz(
        self, params: list[str] | None = None, labels: dict[str, str] | None = None
    ) -> Any:
        """Export samples to an ``arviz.InferenceData`` object.

        Parameters
        ----------
        params
            Parameters to include.  If ``None``, all parameters returned by
            :meth:`keys` are included.
        labels
            Override display names for specific parameters, e.g. ``{"period": "period
            [day]", "rv_semiamp": "K [km/s]"}``. Parameters not listed use their plain
            parameter name as the label.

        Returns
        -------
            Inference data suitable for ``arviz.plot_pair``, ``arviz.summary``, etc.

        Raises
        ------
        ImportError
            If ``arviz`` is not installed.

        Examples
        --------
        >>> idata = samples.to_arviz(["period", "eccentricity"])  # doctest: +SKIP
        """
        az, _ = get_arviz("to_arviz()")

        if params is None:
            params = self.keys()

        num_chains: int = int(self.metadata.get("num_chains", 1))
        n_per_chain, remainder = divmod(self.n_samples, num_chains)
        if remainder != 0:
            # Fall back to treating all samples as a single chain.
            num_chains = 1
            n_per_chain = self.n_samples
            warnings.warn(
                "Number of samples is not divisible by num_chains. "
                "Falling back to a single chain.",
                category=UserWarning,
                stacklevel=1,
            )

        data_dict: dict[str, Any] = {}
        for param in params:
            try:
                values = self[param]
                var_name = (labels or {}).get(param, param)
                if isinstance(values, Q):
                    arr = np.asarray(values.value).reshape(num_chains, n_per_chain)
                else:
                    arr = np.asarray(values).reshape(num_chains, n_per_chain)
                data_dict[var_name] = arr
            except (KeyError, ValueError):
                continue

        if len(data_dict) == 0:
            msg = "No valid parameters found"
            raise ValueError(msg)

        return az.from_dict({"posterior": data_dict})

    def plot_corner(
        self,
        params: list[str] | None = None,
        truths: dict[str, Any] | None = None,
        labels: dict[str, str] | None = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Create corner plot of posterior samples using arviz.

        Parameters
        ----------
        params
            Parameters to include in corner plot. If None, selects a default
            set based on model_type.
        truths
            Dictionary of true parameter values to overplot as reference values.
        labels
            Override display names for specific parameters, e.g. ``{"period": "period
            [day]", "rv_semiamp": "K [km/s]"}``. Parameters not listed use their plain
            parameter name as the label.
        **plot_kwargs
            Additional keyword arguments passed to arviz.plot_pair().

        Returns
        -------
            Array of axes from ``arviz.plot_pair``.

        Examples
        --------
        Default corner plot (selects parameters based on data type):

        >>> axes = samples.plot_corner()  # doctest: +SKIP

        Specify parameters and overplot true values:

        >>> from unxt import Q  # doctest: +SKIP
        >>> axes = samples.plot_corner(  # doctest: +SKIP
        ...     params=["period", "eccentricity"],
        ...     truths={"period": Q(100, "day"), "eccentricity": 0.3},
        ... )
        """
        az, MapLabeller = get_arviz("corner plots")

        # Select default parameters based on available params
        if params is None:
            params = self.keys()

        # Build data dictionary for arviz InferenceData
        data_dict = {}
        var_names = []
        reference_values = {}

        param_units = {}
        for param in params:
            try:
                values = self[param]
                if isinstance(values, Q):
                    param_units[param] = str(values.unit)
                    values = values.value
                data_dict[param] = np.asarray(values)[None, :]
                var_names.append(param)
                if truths is not None and param in truths:
                    truth_val = truths[param]
                    # Strip a Q truth into the plotted unit so the reference
                    # marker lines up with the (already-unitless) sample axis.
                    # ``values`` has already been unwrapped above, so the
                    # recorded ``param_units[param]`` is the source of truth.
                    if isinstance(truth_val, Q):
                        target_unit = param_units.get(param, "")
                        reference_values[param] = float(ustrip(target_unit, truth_val))
                    else:
                        reference_values[param] = (
                            float(truth_val) if truth_val is not None else None
                        )
            except (KeyError, ValueError):
                continue

        if len(data_dict) == 0:
            msg = "No valid parameters found for plotting"
            raise ValueError(msg)

        # Create arviz InferenceData object
        idata = az.from_dict({"posterior": data_dict})

        # Set default plot kwargs
        default_kwargs: dict[str, Any] = {
            "var_names": var_names,
            "marginal": True,
            "triangle": "lower",
            "marginal_kind": "hist",
            "figure_kwargs": {},
        }

        user_labels = labels or {}
        resolved_labels = {
            k: user_labels[k]
            if k in user_labels
            else f"{k} [{param_units[k]}]"
            if param_units.get(k)
            else k
            for k in params
        }
        if any(resolved_labels[k] != k for k in params):
            default_kwargs["labeller"] = MapLabeller(
                var_name_map=resolved_labels  # ty: ignore[invalid-argument-type]
            )

        # Merge with user kwargs (user kwargs take precedence)
        default_kwargs.update(plot_kwargs)

        default_kwargs["figure_kwargs"].setdefault(
            "figsize", (3 * len(var_names) + 1.5, 3 * len(var_names))
        )

        plot_matrix = az.plot_pair(idata, **default_kwargs)
        _overlay_corner_truths(
            plot_matrix,
            var_names,
            reference_values,
            marginal=bool(default_kwargs.get("marginal", True)),
            triangle=default_kwargs.get("triangle", "lower"),
        )
        return plot_matrix


def _check_stack_consistency(samples_list: Sequence[Samples]) -> None:
    """Verify all entries share schema (model_type, keys, units, ext names)."""
    first = samples_list[0]
    first_nl_units = {k: str(v.unit) for k, v in first.nonlinear.items()}
    first_lin_units = {k: str(v.unit) for k, v in first.linear.items()}
    for i, s in enumerate(samples_list[1:], start=1):
        if s.model_type != first.model_type:
            msg = (
                f"Samples[{i}].model_type={s.model_type!r} does not match "
                f"Samples[0].model_type={first.model_type!r}"
            )
            raise ValueError(msg)
        if s.linear_extension_names != first.linear_extension_names:
            msg = (
                f"Samples[{i}].linear_extension_names="
                f"{s.linear_extension_names!r} does not match "
                f"Samples[0].linear_extension_names="
                f"{first.linear_extension_names!r}"
            )
            raise ValueError(msg)
        if set(s.nonlinear) != set(first.nonlinear):
            msg = (
                f"Samples[{i}].nonlinear keys {set(s.nonlinear)!r} differ from "
                f"Samples[0].nonlinear keys {set(first.nonlinear)!r}"
            )
            raise ValueError(msg)
        if set(s.linear) != set(first.linear):
            msg = (
                f"Samples[{i}].linear keys {set(s.linear)!r} differ from "
                f"Samples[0].linear keys {set(first.linear)!r}"
            )
            raise ValueError(msg)
        for k, expected in first_nl_units.items():
            got = str(s.nonlinear[k].unit)
            if got != expected:
                msg = (
                    f"Samples[{i}].nonlinear[{k!r}] has unit {got!r}, "
                    f"expected {expected!r} (from Samples[0])"
                )
                raise ValueError(msg)
        for k, expected in first_lin_units.items():
            got = str(s.linear[k].unit)
            if got != expected:
                msg = (
                    f"Samples[{i}].linear[{k!r}] has unit {got!r}, "
                    f"expected {expected!r} (from Samples[0])"
                )
                raise ValueError(msg)


def _pad_quantity_dict(
    qdicts: Sequence[dict[str, Q]],
    sample_counts: Sequence[int],
    K_max: int,
    pad_value: float,
) -> dict[str, Q]:
    """Pad a list of ``{key: Q}`` dicts to a single ``{key: Q[N, K_max]}`` dict."""
    out: dict[str, Q] = {}
    N = len(qdicts)
    for key, ref_q in qdicts[0].items():
        unit = str(ref_q.unit)
        dtype = np.asarray(ref_q.value).dtype
        padded = np.full((N, K_max), pad_value, dtype=dtype)
        for n, (qd, k_n) in enumerate(zip(qdicts, sample_counts, strict=True)):
            padded[n, :k_n] = np.asarray(ustrip(unit, qd[key]))
        out[key] = Q(jnp.asarray(padded), unit)
    return out


def pad_and_stack_samples(
    samples_list: Sequence[Samples],
    *,
    pad_value: float = float("nan"),
) -> tuple[Samples, jax.Array]:
    """Stack a list of per-entity ``Samples`` into one batched ``Samples`` + mask.

    All inputs must share ``model_type``, ``linear_extension_names``, and the
    set of nonlinear / linear keys with matching units. Per-entity sample
    counts may differ; the trailing axis is padded to
    ``K_max = max(s.n_samples for s in samples_list)`` with ``pad_value``.

    ``ln_likelihood`` and ``ln_prior`` are stacked iff every input carries
    them, with ``-inf`` as the log-space padding sentinel; otherwise the
    stacked ``Samples`` has ``None`` for those fields. ``metadata`` is
    inherited from ``samples_list[0]``.

    Parameters
    ----------
    samples_list
        Sequence of per-entity ``Samples`` to stack. Must be non-empty.
    pad_value
        Fill value for unused trailing-axis positions in the nonlinear and
        linear arrays. Default ``NaN``: a quiet stand-in that propagates
        loudly if a consumer forgets to apply the returned mask.

    Returns
    -------
    stacked : Samples
        Batched ``Samples`` with each parameter array of shape
        ``(N, K_max)`` where ``N = len(samples_list)``. ``stacked.n_samples``
        is ``K_max`` and ``stacked.batch_shape`` is ``(N,)``.
    mask : jax.Array
        Boolean array of shape ``(N, K_max)``; ``True`` where the entry
        carries a real sample.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.samplers.samples import Samples, pad_and_stack_samples
    >>> def _mk(periods, K):
    ...     return Samples(
    ...         nonlinear={
    ...             "period": Q(periods, "day"),
    ...             "eccentricity": Q([0.1] * len(periods), ""),
    ...             "phase_peri": Q([0.0] * len(periods), ""),
    ...         },
    ...         linear={
    ...             "rv_semiamp": Q([K] * len(periods), "km/s"),
    ...             "v_sys": Q([0.0] * len(periods), "km/s"),
    ...         },
    ...         model_type="RVModel",
    ...     )
    >>> stacked, mask = pad_and_stack_samples([_mk([10.0, 20.0], 5.0),
    ...                                        _mk([30.0, 40.0, 50.0], 7.0)])
    >>> stacked.batch_shape, stacked.n_samples
    ((2,), 3)
    >>> bool(mask[0, 2]), bool(mask[1, 2])
    (False, True)
    """
    if len(samples_list) == 0:
        msg = "pad_and_stack_samples requires at least one Samples instance"
        raise ValueError(msg)

    _check_stack_consistency(samples_list)

    first = samples_list[0]
    N = len(samples_list)
    sample_counts = [s.n_samples for s in samples_list]
    K_max = max(sample_counts)

    nonlinear = _pad_quantity_dict(
        [s.nonlinear for s in samples_list], sample_counts, K_max, pad_value
    )
    linear = _pad_quantity_dict(
        [s.linear for s in samples_list], sample_counts, K_max, pad_value
    )

    mask_np = np.zeros((N, K_max), dtype=bool)
    for n, k_n in enumerate(sample_counts):
        mask_np[n, :k_n] = True
    mask = jnp.asarray(mask_np)

    def _stack_logprob(getter: str) -> jax.Array | None:
        if any(getattr(s, getter) is None for s in samples_list):
            return None
        ref = getattr(first, getter)
        dtype = np.asarray(ref).dtype
        padded = np.full((N, K_max), -np.inf, dtype=dtype)
        for n, s in enumerate(samples_list):
            padded[n, : sample_counts[n]] = np.asarray(getattr(s, getter))
        return jnp.asarray(padded)

    ln_likelihood = _stack_logprob("ln_likelihood")
    ln_prior = _stack_logprob("ln_prior")

    stacked = Samples(
        nonlinear=nonlinear,
        linear=linear,
        model_type=first.model_type,
        metadata=first.metadata,
        linear_extension_names=first.linear_extension_names,
        ln_likelihood=ln_likelihood,
        ln_prior=ln_prior,
    )
    return stacked, mask
