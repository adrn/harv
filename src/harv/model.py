"""Model class combining prior and data for Keplerian orbit inference.

The :class:`Model` pre-computes the likelihood at construction time and
provides a clean :meth:`log_prob` method for evaluating the log-probability
at arbitrary parameter values.

Examples
--------
>>> from unxt import Q
>>> from harv import Model
>>> from harv.samplers import RejectionPrior

Create a model from a prior and data::

    prior = RejectionPrior.default_rv(
        period_min=Q(2.0, "day"), period_max=Q(1000.0, "day"),
        sigma_K0=Q(30.0, "km/s"), sigma_v0=Q(50.0, "km/s"),
    )
    model = Model(prior, rv_data)

Evaluate log-probability (marginalizing over linear parameters by default)::

    lp = model.log_prob({
        "period": Q(100.0, "day"),
        "eccentricity": Q(0.3, ""),
        "phase_peri": Q(0.1, ""),
        "arg_peri": Q(1.5, "rad"),
    })

Evaluate full (non-marginalized) log-probability::

    lp = model.log_prob(
        {
            "period": Q(100.0, "day"),
            "eccentricity": Q(0.3, ""),
            "phase_peri": Q(0.1, ""),
            "arg_peri": Q(1.5, "rad"),
            "rv_semiamp": Q(10.0, "km/s"),
            "v_sys": Q(5.0, "km/s"),
        },
        marginalize=False,
    )

Evaluate over a batch of parameter samples with ``jax.vmap``::

    import jax
    import jax.numpy as jnp

    # Build a dict of batched Quantity arrays (e.g. 1000 samples)
    values = {
        "period": Q(jnp.linspace(50.0, 500.0, 1000), "day"),
        "eccentricity": Q(jnp.full(1000, 0.3), ""),
        "phase_peri": Q(jnp.full(1000, 0.1), ""),
        "arg_peri": Q(jnp.full(1000, 1.5), "rad"),
    }
    log_probs = jax.vmap(model.log_prob)(values)  # shape (1000,)

JIT-compile for repeated evaluation::

    import equinox as eqx

    fast_log_prob = eqx.filter_jit(jax.vmap(model.log_prob))
    log_probs = fast_log_prob(values)
"""

from typing import Any

import equinox as eqx
import jax
import jax.random as jr
import numpyro.distributions as dist
from unxt import AbstractQuantity, Q, ustrip

from harv.data import (
    AbstractAstrometryData,
    GaiaAstrometryData,
    InputData,
    RVData,
    SourceData,
    SystemData,
    build_indicator_matrix,
    stack_datasets,
)
from harv.distributions import QuantityDistribution
from harv.likelihood.composite import CompositeLikelihood
from harv.likelihood.gaia_astrometry import GaiaAstrometryLikelihood
from harv.likelihood.params import (
    AbstractParameters,
    GaiaAstrometryParameters,
    MarginalizedParameters,
    RVParameters,
    SB2RVParameters,
)
from harv.likelihood.rv import RVLikelihood, SB2RVLikelihood
from harv.samplers.rejection_prior import RejectionPrior

__all__ = ("Model",)

# Type for the per-component tuple: (component_name, param_class, data_type_label)
_ComponentInfo = tuple[str, type[AbstractParameters], str]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _infer_data_type(data: InputData) -> str:
    """Infer the data-type string from the input data."""
    if isinstance(data, SourceData):
        has_rv = data._n_rv() > 0
        has_astro = data._n_astrometry() > 0
        if has_astro and has_rv:
            return "combined"
        if has_astro:
            return "astrometry"
        if has_rv:
            return "rv"
        msg = "SourceData must contain at least one dataset"
        raise ValueError(msg)
    if isinstance(data, SystemData):
        return "sb2"
    if isinstance(data, AbstractAstrometryData):
        return "astrometry"
    if isinstance(data, RVData):
        return "rv"
    msg = f"Unsupported data type: {type(data)}"
    raise TypeError(msg)


def _data_type_to_components(
    data_type: str,
) -> tuple[_ComponentInfo, ...]:
    """Map *data_type* to component descriptors."""
    if data_type == "rv":
        return (("rv", RVParameters, "rv"),)
    if data_type == "astrometry":
        return (("astro", GaiaAstrometryParameters, "astrometry"),)
    if data_type == "combined":
        return (
            ("astro", GaiaAstrometryParameters, "astrometry"),
            ("rv", RVParameters, "rv"),
        )
    if data_type == "sb2":
        return (("sb2", SB2RVParameters, "sb2"),)
    msg = f"Unknown data type: {data_type}"
    raise ValueError(msg)


def _extract_datasets(data: InputData) -> dict[str, Any]:
    """Extract concrete data objects from the input."""
    if isinstance(data, RVData):
        return {"rv": data}
    if isinstance(data, GaiaAstrometryData):
        return {"astro": data}
    if isinstance(data, SystemData):
        return {"sb2": data}
    if isinstance(data, SourceData):
        _data: dict[str, Any] = {}
        rv_datasets = data.get_datasets_by_type(RVData)
        if len(rv_datasets) == 1:
            _data["rv"] = next(iter(rv_datasets.values()))
        elif len(rv_datasets) > 1:
            _data["rv"] = rv_datasets
        astro_datasets = data.get_datasets_by_type(GaiaAstrometryData)
        if len(astro_datasets) == 1:
            _data["astro"] = next(iter(astro_datasets.values()))
        elif len(astro_datasets) > 1:
            msg = "Multiple astrometry datasets not supported yet"
            raise NotImplementedError(msg)
        return _data
    msg = f"Expected AbstractData or SourceData/SystemData, got {type(data)}"
    raise TypeError(msg)


def _jitter_units_from_prior(prior: RejectionPrior) -> dict[str, str]:
    """Extract jitter units from the prior's ``jitter_priors`` dict."""
    if prior.jitter_priors is None:
        return {}
    return {
        dt_label: str(d.unit) if isinstance(d, QuantityDistribution) else ""
        for dt_label, d in prior.jitter_priors.items()
    }


def _build_rv_likelihood(
    datasets: dict[str, Any],
    prior: RejectionPrior,
) -> RVLikelihood:
    """Build an RV likelihood from datasets and prior."""
    rv_raw = datasets["rv"]
    rv_offsets = prior.offsets.get("rv") if prior.offsets is not None else None

    indicator = None
    instrument_names = None
    if isinstance(rv_raw, dict) and rv_offsets is not None:
        reference = next(name for name, v in rv_offsets.items() if v is None)
        rv_data, indicator, instrument_names = build_indicator_matrix(
            rv_raw, reference=reference
        )
    elif isinstance(rv_raw, dict):
        rv_data = stack_datasets(rv_raw)
    else:
        rv_data = rv_raw

    linear_prior = None
    if isinstance(prior.linear_prior, dict):
        linear_prior = {
            name: prior.linear_prior[name]
            for name in RVParameters.linear_param_names
            if name in prior.linear_prior
        }

    offsets_prior = None
    if rv_offsets is not None:
        offsets_prior = {name: v for name, v in rv_offsets.items() if v is not None}

    return RVLikelihood(
        data=rv_data,
        linear_marginalized_prior=linear_prior or None,
        offsets_marginalized_prior=offsets_prior,
        trend_marginalized_prior=(
            dict(prior.trend_priors) if prior.trend_priors is not None else None
        ),
        trend_order=prior.trend_order,
        indicator_matrix=indicator,
        instrument_names=instrument_names,
    )


def _build_astrometry_likelihood(
    datasets: dict[str, Any],
    prior: RejectionPrior,
) -> GaiaAstrometryLikelihood:
    """Build a Gaia astrometry likelihood from datasets and prior."""
    linear_prior = None
    if isinstance(prior.linear_prior, dict):
        linear_prior = {
            name: prior.linear_prior[name]
            for name in GaiaAstrometryParameters.linear_param_names
            if name in prior.linear_prior
        }

    return GaiaAstrometryLikelihood(
        data=datasets["astro"],
        linear_marginalized_prior=linear_prior or None,
        trend_marginalized_prior=(
            dict(prior.trend_priors) if prior.trend_priors is not None else None
        ),
        trend_order=prior.trend_order,
    )


def _build_sb2_likelihood(
    datasets: dict[str, Any],
    prior: RejectionPrior,
) -> SB2RVLikelihood:
    """Build an SB2 RV likelihood from datasets and prior."""
    linear_prior = None
    if isinstance(prior.linear_prior, dict):
        linear_prior = {
            name: prior.linear_prior[name]
            for name in SB2RVParameters.linear_param_names
            if name in prior.linear_prior
        }

    return SB2RVLikelihood(
        data=datasets["sb2"],
        linear_marginalized_prior=linear_prior or None,
        trend_marginalized_prior=(
            dict(prior.trend_priors) if prior.trend_priors is not None else None
        ),
        trend_order=prior.trend_order,
    )


def _build_likelihood(
    datasets: dict[str, Any],
    data_type: str,
    prior: RejectionPrior,
) -> Any:
    """Build the appropriate likelihood for the given data type."""
    if data_type == "rv":
        return _build_rv_likelihood(datasets, prior)
    if data_type == "astrometry":
        return _build_astrometry_likelihood(datasets, prior)
    if data_type == "sb2":
        return _build_sb2_likelihood(datasets, prior)
    if data_type == "combined":
        rv_data = datasets.get("rv")
        if "astro" in datasets and isinstance(rv_data, dict) and len(rv_data) > 1:
            msg = "Combined astrometry + multi-survey RV is not yet implemented."
            raise NotImplementedError(msg)

        components: dict[str, Any] = {}
        if "astro" in datasets:
            components["astro"] = _build_astrometry_likelihood(datasets, prior)
        if "rv" in datasets:
            components["rv"] = _build_rv_likelihood(datasets, prior)
        return CompositeLikelihood(**components)
    msg = f"Unknown data type: {data_type}"
    raise ValueError(msg)


def _validate_prior(
    prior: RejectionPrior,
    data_type: str,
    components: tuple[_ComponentInfo, ...],
) -> None:
    """Validate that the prior has all required parameters for the data type.

    Raises ``ValueError`` for missing parameters and ``TypeError`` for
    dimensioned parameters that lack a ``QuantityDistribution`` wrapper.
    """
    all_prior_keys: set[str] = set(prior.nonlinear_priors)
    if isinstance(prior.linear_prior, dict):
        all_prior_keys |= set(prior.linear_prior)

    # Required params: nonlinear + explicitly-sampled linear
    required: list[str] = []
    seen: set[str] = set()
    for _, cls, _ in components:
        for name in cls.nonlinear_param_names:
            if name not in seen:
                seen.add(name)
                required.append(name)

    if isinstance(prior.linear_prior, dict):
        marg_names = prior.marginalize_names
        if marg_names is not None:
            marg_set = set(marg_names)
            for name in prior.linear_prior:
                if name not in marg_set and name not in seen:
                    seen.add(name)
                    required.append(name)

    missing = [p for p in required if p not in all_prior_keys]
    if missing:
        msg = (
            f"Prior missing required parameters for {data_type} data: {missing}. "
            f"Use RejectionPrior.default_{data_type}() or provide these parameters."
        )
        raise ValueError(msg)

    # Dimensioned parameters must use QuantityDistribution.
    dimensioned: set[str] = set()
    for _, cls, _ in components:
        dimensioned.update(cls._dimensioned_param_names)

    bad: list[str] = []
    for name in dimensioned:
        d = prior.nonlinear_priors.get(name)
        if d is None and isinstance(prior.linear_prior, dict):
            d = prior.linear_prior.get(name)
        if d is None:
            continue
        if isinstance(d, QuantityDistribution):
            continue
        if callable(d) and not isinstance(d, dist.Distribution):
            continue
        bad.append(name)

    if bad:
        msg = (
            f"Parameters {sorted(bad)} have physical dimensions and require "
            f"a QuantityDistribution prior, but received bare "
            f"numpyro distributions. Wrap each in "
            f"QuantityDistribution(dist, unit_str)."
        )
        raise TypeError(msg)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Model(eqx.Module):
    """A model combining a prior and data for Keplerian orbit inference.

    Construct a ``Model`` to pre-compute the likelihood and get a clean
    interface for evaluating log-probabilities and sampling linear parameters.

    Parameters
    ----------
    prior : RejectionPrior
        Prior distribution for nonlinear and linear parameters.
    data : InputData
        Observational data (``RVData``, ``GaiaAstrometryData``,
        ``SourceData``, or ``SystemData``).

    Attributes
    ----------
    likelihood
        The pre-built likelihood object.
    data_type : str
        Inferred data type (``"rv"``, ``"astrometry"``, ``"combined"``,
        ``"sb2"``).
    time_unit : str
        Unit string for the time axis (e.g. ``"day"``).
    t_ref : float or None
        Reference epoch as a scalar in ``time_unit``.
    full_cls : tuple[type, ...]
        Parameter class(es) for the inferred data type.

    Examples
    --------
    >>> from unxt import Q
    >>> from harv import Model
    >>> from harv.samplers import RejectionPrior
    >>> from harv.data import RVData
    >>> rv_data = RVData(
    ...     time=Q([0.0, 50.0, 100.0], "day"),
    ...     rv=Q([10.0, -5.0, 8.0], "km/s"),
    ...     rv_err=Q([1.0, 1.0, 1.0], "km/s"),
    ... )
    >>> prior = RejectionPrior.default_rv(
    ...     period_min=Q(2.0, "day"), period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"), sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> model = Model(prior, rv_data)
    >>> model.data_type
    'rv'
    >>> model.time_unit
    'd'
    """

    prior: RejectionPrior
    data: InputData

    # -- Computed at construction time (do not pass manually) --
    likelihood: Any = eqx.field(repr=False, default=None)
    data_type: str = eqx.field(static=True, default="")
    time_unit: str = eqx.field(static=True, default="")
    t_ref: float | None = eqx.field(static=True, default=None)
    full_cls: tuple[type[AbstractParameters], ...] = eqx.field(static=True, default=())
    _components: tuple[_ComponentInfo, ...] = eqx.field(static=True, default=())
    _jitter_units: dict[str, str] | None = eqx.field(static=True, default=None)

    def __check_init__(self) -> None:
        # Skip recomputation during pytree unflatten.
        if self.likelihood is not None:
            return

        data_type = _infer_data_type(self.data)
        components = _data_type_to_components(data_type)
        _validate_prior(self.prior, data_type, components)

        datasets = _extract_datasets(self.data)
        lik = _build_likelihood(datasets, data_type, self.prior)

        # Extract time metadata from the reference dataset.
        if isinstance(self.data, (SourceData, SystemData)):
            _ref = next(iter(self.data.values()))
        else:
            _ref = self.data
        time_unit = str(_ref.time.unit)
        t_ref = _ref.t_ref
        if isinstance(t_ref, Q):
            t_ref_scalar: float | None = float(ustrip(time_unit, t_ref))
        elif t_ref is not None:
            t_ref_scalar = float(t_ref)
        else:
            t_ref_scalar = None

        full_cls = tuple(cls for _, cls, _ in components)

        object.__setattr__(self, "likelihood", lik)
        object.__setattr__(self, "data_type", data_type)
        object.__setattr__(self, "time_unit", time_unit)
        object.__setattr__(self, "t_ref", t_ref_scalar)
        object.__setattr__(self, "full_cls", full_cls)
        object.__setattr__(self, "_components", components)
        object.__setattr__(self, "_jitter_units", _jitter_units_from_prior(self.prior))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def linear_param_units(self) -> dict[str, str]:
        """Map from linear parameter name to its physical unit string.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv import Model
        >>> from harv.data import RVData
        >>> from harv.samplers import RejectionPrior
        >>> rv_data = RVData(
        ...     time=Q([0.0, 50.0, 100.0], "day"),
        ...     rv=Q([10.0, -5.0, 8.0], "km/s"),
        ...     rv_err=Q([1.0, 1.0, 1.0], "km/s"),
        ... )
        >>> prior = RejectionPrior.default_rv(
        ...     period_min=Q(2.0, "day"), period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"), sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> model = Model(prior, rv_data)
        >>> model.linear_param_units
        {'rv_semiamp': 'km / s', 'v_sys': 'km / s'}
        """
        return self.likelihood.linear_param_units

    @property
    def all_linear_names(self) -> tuple[str, ...]:
        """All linear parameter names including trends and offsets.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv import Model
        >>> from harv.data import RVData
        >>> from harv.samplers import RejectionPrior
        >>> rv_data = RVData(
        ...     time=Q([0.0, 50.0, 100.0], "day"),
        ...     rv=Q([10.0, -5.0, 8.0], "km/s"),
        ...     rv_err=Q([1.0, 1.0, 1.0], "km/s"),
        ... )
        >>> prior = RejectionPrior.default_rv(
        ...     period_min=Q(2.0, "day"), period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(30.0, "km/s"), sigma_v0=Q(50.0, "km/s"),
        ... )
        >>> model = Model(prior, rv_data)
        >>> model.all_linear_names
        ('rv_semiamp', 'v_sys')
        """
        names: tuple[str, ...] = sum(
            (cls.linear_param_names for cls in self.full_cls), ()
        )
        if self.prior.trend_priors is not None:
            names = names + tuple(self.prior.trend_priors.keys())
        if (
            self.prior.offsets is not None
            and isinstance(self.data, SourceData)
            and self.data._n_rv() > 1
        ):
            names = names + tuple(
                k for k, v in self.prior.offsets.items() if v is not None
            )
        return names

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def log_prob(
        self,
        values: dict[str, AbstractQuantity],
        *,
        marginalize: bool = True,
    ) -> jax.Array:
        """Evaluate the log-probability at the given parameter values.

        When ``marginalize=True`` (default), linear parameters are
        analytically integrated out using the Gaussian prior from
        ``self.prior``.  The result is a *marginal likelihood*:

        .. math::

            \\log p(d \\mid \\theta_{\\rm nl}) =
            \\log \\int p(d \\mid \\theta_{\\rm nl}, \\theta_{\\rm lin})
            \\, p(\\theta_{\\rm lin}) \\, d\\theta_{\\rm lin}

        so changing the linear prior changes the output even for identical
        nonlinear parameter values.

        Parameters
        ----------
        values : dict[str, Quantity]
            Parameter values keyed by name.  When ``marginalize=True``, only
            nonlinear parameters (plus any explicitly-sampled linear parameters
            such as parallax) are required.  When ``marginalize=False``, all
            parameters must be provided.
        marginalize : bool
            If ``True`` (default), analytically marginalize over linear
            parameters.  If ``False``, evaluate the full likelihood at all
            provided parameter values.

        Returns
        -------
        jax.Array
            Scalar log-probability value.

        Examples
        --------
        >>> import jax  # doctest: +SKIP
        >>> lp = model.log_prob({  # doctest: +SKIP
        ...     "period": Q(100.0, "day"),
        ...     "eccentricity": Q(0.3, ""),
        ...     "phase_peri": Q(0.1, ""),
        ...     "arg_peri": Q(1.5, "rad"),
        ... })
        """
        params = self.build_params(values, marginalize=marginalize)
        return self.likelihood.log_prob(params)

    def build_params(
        self,
        values: dict[str, AbstractQuantity],
        *,
        marginalize: bool = True,
    ) -> MarginalizedParameters | dict[str, MarginalizedParameters]:
        """Build parameter struct(s) from a values dictionary.

        Parameters
        ----------
        values : dict[str, Quantity]
            Parameter values keyed by name.
        marginalize : bool
            If ``True``, build ``MarginalizedParameters`` that analytically
            integrate over the linear subset.  If ``False``, all linear
            parameters must appear in *values*.

        Returns
        -------
        MarginalizedParameters or dict[str, MarginalizedParameters]
            For single-component models (RV or astrometry), a single struct.
            For combined models, a dict keyed by component name
            (e.g. ``{"astro": ..., "rv": ...}``).
        """
        if self.data_type == "combined":
            return {
                name: self._build_single_params(
                    values, cls, dt_label, marginalize=marginalize
                )
                for name, cls, dt_label in self._components
            }
        _, cls, dt_label = self._components[0]
        return self._build_single_params(values, cls, dt_label, marginalize=marginalize)

    def sample_conditional_linear(
        self,
        values: dict[str, AbstractQuantity],
        key: jax.Array,
    ) -> dict[str, AbstractQuantity]:
        """Sample linear parameters conditioned on nonlinear values and data.

        Draws one sample from the conditional posterior over marginalizable
        linear parameters given the provided nonlinear parameter values.

        Parameters
        ----------
        values : dict[str, Quantity]
            Nonlinear parameter values (same format as :meth:`log_prob` with
            ``marginalize=True``).
        key : jax.Array
            JAX random key.

        Returns
        -------
        dict[str, Quantity]
            Sampled linear parameter values with units.
        """
        params = self.build_params(values, marginalize=True)
        if self.data_type == "combined":
            keys = jr.split(key, len(self._components))
            result: dict[str, AbstractQuantity] = {}
            for (name, _, _), k in zip(self._components, keys, strict=True):
                sub_lik = self.likelihood[name]
                sub_params = params[name]
                result.update(sub_lik.sample_conditional_linear(sub_params, k))
            return result
        return self.likelihood.sample_conditional_linear(params, key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_single_params(
        self,
        values: dict[str, AbstractQuantity],
        cls: type[AbstractParameters],
        dt_label: str,
        *,
        marginalize: bool = True,
    ) -> MarginalizedParameters:
        """Build ``MarginalizedParameters`` for a single component.

        Parameters
        ----------
        values
            User-supplied parameter values (``Quantity`` objects).
        cls
            The full parameter class (e.g. ``RVParameters``).
        dt_label
            Data-type label used for jitter key namespacing.
        marginalize
            Whether to indicate linear params for analytic marginalization.
        """
        kw: dict[str, Any] = {}

        # Nonlinear parameters: strip all to raw scalars. The existing
        # likelihood code expects plain floats for nonlinear params (the
        # strategy's sample_nonlinear returns bare JAX arrays).
        for name in cls.nonlinear_param_names:
            if name not in values:
                msg = f"Missing required nonlinear parameter: {name!r}"
                raise ValueError(msg)
            val = values[name]
            if isinstance(val, AbstractQuantity):
                kw[name] = ustrip(str(val.unit), val)
            else:
                kw[name] = val

        # Re-wrap period with the model's time unit, converting if necessary.
        if "period" in kw:
            period_val = values["period"]
            if isinstance(period_val, AbstractQuantity):
                kw["period"] = Q(ustrip(self.time_unit, period_val), self.time_unit)
            else:
                kw["period"] = Q(period_val, self.time_unit)

        # Optional nonlinear parameters (e.g. jitter).
        # Jitter uses a namespaced key "jitter_{dt_label}" to avoid collision
        # in combined fits, but the parameter field on the struct is "jitter".
        jitter_units = self._jitter_units or {}
        for name in cls._optional_nonlinear_param_names:
            values_key = f"jitter_{dt_label}" if name == "jitter" else name
            if values_key in values:
                val = values[values_key]
                unit = jitter_units.get(dt_label, "")
                if isinstance(val, AbstractQuantity) and unit:
                    kw[name] = Q(ustrip(unit, val), unit)
                elif isinstance(val, AbstractQuantity):
                    kw[name] = ustrip(str(val.unit), val)
                else:
                    kw[name] = Q(val, unit) if unit else val

        # Determine which linear parameters to marginalize.
        if marginalize:
            marginalize_names = self.prior.marginalize_names
            if marginalize_names is not None:
                marg = tuple(
                    n for n in marginalize_names if n in cls.linear_param_names
                )
            else:
                marg = cls.linear_param_names
        else:
            marg = ()

        # Include non-marginalized linear parameter values.
        # Linear params are kept as Quantity objects — the likelihood
        # strips units internally via ustrip().
        units = self.linear_param_units
        for name in cls.linear_param_names:
            if name not in marg:
                if name in values:
                    val = values[name]
                    if isinstance(val, AbstractQuantity):
                        kw[name] = val
                    else:
                        u = units.get(name, "")
                        kw[name] = Q(val, u) if u else val
                elif not marginalize:
                    msg = (
                        f"Missing required linear parameter: {name!r} "
                        f"(required when marginalize=False)"
                    )
                    raise ValueError(msg)

        if marg:
            return cls.marginalized(*marg, **kw)
        return MarginalizedParameters(values=kw, marginalized_names=(), source_cls=cls)

    # ------------------------------------------------------------------
    # Raw-array helpers (used by samplers inside JIT/vmap loops)
    # ------------------------------------------------------------------

    @property
    def required_prior_params(self) -> tuple[str, ...]:
        """Parameter names that must appear in the raw sampled-values dict.

        Includes nonlinear params (always sampled) plus any linear params
        that are *not* in ``prior.marginalize_names`` and therefore sampled
        explicitly.
        """
        seen: set[str] = set()
        result: list[str] = []
        for cls in self.full_cls:
            for name in cls.nonlinear_param_names:
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        if isinstance(self.prior.linear_prior, dict):
            marg_names = self.prior.marginalize_names
            if marg_names is not None:
                marg_set = set(marg_names)
                for name in self.prior.linear_prior:
                    if name not in marg_set and name not in seen:
                        seen.add(name)
                        result.append(name)
        return tuple(result)

    def _build_params_raw(
        self,
        values: dict[str, Any],
    ) -> MarginalizedParameters | dict[str, MarginalizedParameters]:
        """Build parameter structs from raw JAX arrays.

        Like :meth:`build_params` but operates on plain JAX arrays (no
        ``Quantity`` wrappers). Period must already be in ``self.time_unit``.
        Used by the sampler's JIT-compiled inner loops.
        """
        if self.data_type == "combined":
            return {
                name: self._build_single_params_raw(values, cls, dt_label)
                for name, cls, dt_label in self._components
            }
        _, cls, dt_label = self._components[0]
        return self._build_single_params_raw(values, cls, dt_label)

    def _build_single_params_raw(
        self,
        values: dict[str, Any],
        cls: type[AbstractParameters],
        dt_label: str,
    ) -> MarginalizedParameters:
        """Build ``MarginalizedParameters`` for one component from raw arrays."""
        kw: dict[str, Any] = {name: values[name] for name in cls.nonlinear_param_names}
        kw["period"] = Q(kw["period"], self.time_unit)

        # Optional nonlinear params (e.g. jitter).
        _ju = self._jitter_units or {}
        for name in cls._optional_nonlinear_param_names:
            values_key = f"_{name}_{dt_label}" if name == "jitter" else name
            if values_key in values:
                unit = _ju.get(dt_label, "")
                kw[name] = Q(values[values_key], unit) if unit else values[values_key]

        # Determine which linear params to marginalize.
        marg_names = self.prior.marginalize_names
        if marg_names is not None:
            marg = tuple(n for n in marg_names if n in cls.linear_param_names)
        else:
            marg = cls.linear_param_names

        # Explicit linear values (those not being marginalized).
        units = self.linear_param_units
        for name in cls.linear_param_names:
            if name not in marg and name in values:
                u = units.get(name, "")
                kw[name] = Q(values[name], u) if u else values[name]

        if marg:
            return cls.marginalized(*marg, **kw)
        return MarginalizedParameters(values=kw, marginalized_names=(), source_cls=cls)

    def _build_params_with_fixed_linear_raw(
        self,
        values: dict[str, Any],
        fixed_linear: dict[str, Any],
    ) -> MarginalizedParameters | dict[str, MarginalizedParameters]:
        """Build params with some linear parameters fixed to explicit values.

        Like :meth:`_build_params_raw` but overrides certain linear parameters
        with fixed values (e.g. from an ``extra_model``).  The remaining
        linear parameters are analytically marginalized.
        """
        if self.data_type == "combined":
            return {
                name: self._apply_fixed_linear(
                    self._build_single_params_raw(values, cls, dt_label),
                    fixed_linear,
                    cls,
                )
                for name, cls, dt_label in self._components
            }
        _, cls, dt_label = self._components[0]
        base = self._build_single_params_raw(values, cls, dt_label)
        return self._apply_fixed_linear(base, fixed_linear, cls)

    def _apply_fixed_linear(
        self,
        base: MarginalizedParameters,
        fixed_linear: dict[str, Any],
        cls: type[AbstractParameters],
    ) -> MarginalizedParameters:
        """Override some linear params with fixed values, adjusting marginalization."""
        _lin = cls.linear_param_names
        free = tuple(n for n in _lin if n not in fixed_linear)
        kw = dict(base.values)
        units = self.linear_param_units
        for name in _lin:
            if name in fixed_linear:
                kw[name] = Q(fixed_linear[name], units.get(name, ""))
        if free:
            return cls.marginalized(*free, **kw)
        return MarginalizedParameters(values=kw, marginalized_names=(), source_cls=cls)

    def _sample_conditional_linear_raw(
        self,
        values: dict[str, Any],
        key: jax.Array,
    ) -> dict[str, AbstractQuantity]:
        """Sample linear params from raw array values (no Quantity wrapping)."""
        params = self._build_params_raw(values)
        if self.data_type == "combined":
            keys = jr.split(key, len(self._components))
            result: dict[str, AbstractQuantity] = {}
            for (name, _, _), k in zip(self._components, keys, strict=True):
                sub_lik = self.likelihood[name]
                sub_params = params[name]
                result.update(sub_lik.sample_conditional_linear(sub_params, k))
            return result
        return self.likelihood.sample_conditional_linear(params, key)
