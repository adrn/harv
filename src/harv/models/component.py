"""Abstract base for component models.

A *component model* generates predictions for a single data type (RV or
astrometry) and evaluates the log-likelihood -- either by analytically
marginalizing over the linear parameters or by evaluating at explicit values.
"""

__all__ = ("AbstractComponentModel",)

import types
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import numpyro
import numpyro.distributions as dist
import quaxed.numpy as jnp
from numpyro_ext.distributions import MarginalizedLinear
from unxt import Q
from unxt.quantity import AllowValue, ustrip

from harv.distributions import QuantityDistribution
from harv.extensions.base import AbstractExtension, ParamInfo
from harv.models._helpers import (
    LinearPriorCallable,
    PriorDist,
    _needs_explicit_sampling,
    _unwrap_dist,
)


def _resolve_prior_to_mvn(
    prior_dict: dict[str, PriorDist | LinearPriorCallable],
    nl_values: dict[str, Any],
    unit_dict: dict[str, str],
    extra_values: dict[str, Any] | None = None,
) -> dist.MultivariateNormal:
    """Build diagonal MVN from per-parameter priors."""
    locs: list[Any] = []
    scales: list[Any] = []
    # Build a namespace proxy for any LinearPriorCallable that needs it.
    # Include explicit linear values so that callables depending on
    # explicitly-sampled linear params (e.g. parallax) can resolve.
    proxy_values = dict(nl_values)
    if extra_values:
        proxy_values.update(extra_values)
    params_proxy = types.SimpleNamespace(**proxy_values)
    for name, prior in prior_dict.items():
        target_u = unit_dict.get(name, "")
        resolved = None

        if isinstance(prior, (dist.Distribution, QuantityDistribution)):
            resolved = prior
        elif callable(prior):
            resolved = prior(params_proxy)

        expected_msg = (
            f"Expected Normal inside QuantityDistribution for {name}, "
            f"got {type(resolved)}"
        )
        if isinstance(resolved, QuantityDistribution):
            prior_unit = cast("str", resolved.unit)
            inner = resolved.distribution
            if not isinstance(inner, dist.Normal):
                raise TypeError(expected_msg)
            loc = ustrip(AllowValue, target_u, Q(inner.loc, prior_unit))
            scale = ustrip(AllowValue, target_u, Q(inner.scale, prior_unit))
        elif isinstance(resolved, dist.Normal):
            loc = resolved.loc
            scale = resolved.scale
        else:
            raise TypeError(expected_msg)
        locs.append(loc)
        scales.append(scale)

    return dist.MultivariateNormal(
        loc=jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in locs]),
        scale_tril=jnp.diag(jnp.stack([jnp.squeeze(jnp.asarray(x)) for x in scales])),
    )


class _MargComponents(NamedTuple):
    """Return type of ``_build_marginalized_linear``."""

    dist: MarginalizedLinear
    obs: jax.Array
    marg_names: tuple[str, ...]
    explicit_names: tuple[str, ...]
    linear_params: dict[str, jax.Array]


class AbstractComponentModel(eqx.Module):
    """Abstract base for single-data-type component models.

    Concrete subclasses must implement:

    - ``_param_infos``: all parameter descriptors.
    - ``_base_design_matrix``: the base design matrix from data + nonlinear values.
    - ``_strip_obs``: return (obs, obs_err) as unit-stripped JAX arrays.
    - ``_obs_unit``: the physical unit string of the observations.

    Concrete subclasses must also declare:

    - ``linear_prior``: dict or None (priors for marginalization).
    - ``extensions``: tuple of AbstractExtension (model modifiers).
    """

    # Concrete subclasses must declare these fields:
    data: eqx.AbstractVar[Any]
    linear_prior: eqx.AbstractVar[dict[str, Any] | None]
    extensions: eqx.AbstractVar[tuple[AbstractExtension, ...]]

    # Subclass hooks

    @abstractmethod
    def _param_infos(self) -> tuple[ParamInfo, ...]:
        """All parameter descriptors (base + extensions, nonlinear first)."""

    @abstractmethod
    def _base_design_matrix(self, nl_values: dict[str, Any]) -> jax.Array:
        """Build the base design matrix from data and nonlinear values.

        Columns correspond to the *base* linear parameters only (no
        extensions). Extensions append columns via ``modify_design_matrix``.
        """

    @abstractmethod
    def _strip_obs(self) -> tuple[jax.Array, jax.Array]:
        """Return (observations, observation_errors) as unit-stripped arrays.

        Both arrays have shape ``(n_obs,)`` and share an implicit common unit
        (e.g. ``km/s`` for RV, ``mas`` for astrometry).
        """

    @abstractmethod
    def _obs_unit(self) -> str:
        """The physical unit string of the observations (e.g. ``'km/s'``)."""

    def _linear_param_units(self) -> dict[str, str]:
        """Map from linear parameter name to its concrete unit string.

        The default implementation returns ``obs_unit`` for every linear
        parameter.  Subclasses where different linear parameters have
        different units (e.g. astrometry: mas vs mas/yr) must override.
        """
        obs_unit = self._obs_unit()
        return dict.fromkeys(self._all_linear_names(), obs_unit)

    # Derived values:

    def _all_linear_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._param_infos() if p.linear)

    def _all_nonlinear_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self._param_infos() if not p.linear)

    def _base_nonlinear_names(self) -> frozenset[str]:
        """Base (non-extension) nonlinear parameter names.

        Extension parameters (jitter, etc.) are excluded because they
        need different unit-wrapping logic.
        """
        parameterization = getattr(self, "parameterization", None)
        if parameterization is not None and hasattr(parameterization, "params"):
            return frozenset(p.name for p in parameterization.params() if not p.linear)
        return frozenset()

    def _extract_explicit_linear_values(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...],
    ) -> tuple[dict[str, Any], dict[str, jax.Array]]:
        """Split explicit linear values out of a flat parameter dict."""
        explicit_name_set = set(self._all_linear_names()) - set(marginalized_names)
        explicit_linear = {
            key: nl_values[key] for key in list(nl_values) if key in explicit_name_set
        }
        stripped_nonlinear = {
            key: value for key, value in nl_values.items() if key not in explicit_linear
        }
        return stripped_nonlinear, explicit_linear

    def _auto_marginalized_names(self) -> tuple[str, ...]:
        """Classify linear priors: Gaussian -> marginalize, non-Gaussian -> explicit.

        Returns the tuple of linear parameter names whose priors are
        Gaussian (Normal, callable returning Normal, or Delta) and can be
        analytically marginalized.  Non-Gaussian priors (HalfNormal, etc.)
        are excluded -- their values must be passed alongside the nonlinear
        parameters.
        """
        if self.linear_prior is None:
            return self._all_linear_names()
        return tuple(
            n for n, d in self.linear_prior.items() if not _needs_explicit_sampling(d)
        )

    @property
    def params_explicit(self) -> tuple[str, ...]:
        """Names of parameters that must be explicitly sampled.

        Includes all nonlinear parameters (orbital + extension, e.g. jitter)
        and any linear parameters with non-Gaussian priors (e.g. parallax with
        a HalfNormal prior) that cannot be analytically marginalized.

        These are the keys that must appear in the ``values`` dict passed to
        :meth:`log_prob`.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.data import RVData
        >>> from harv.models.factories import rv_model
        >>> from harv.samplers import RejectionPrior
        >>> data = RVData(
        ...     time=Q([0.0, 50.0], "day"),
        ...     rv=Q([1.0, -2.0], "km/s"),
        ...     rv_err=Q([0.5, 0.5], "km/s"),
        ... )
        >>> prior = RejectionPrior.default_rv(
        ...     period_min=Q(10.0, "day"), period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(5.0, "km/s"), sigma_v0=Q(5.0, "km/s"),
        ... )
        >>> model = rv_model(data, linear_prior=prior.linear_prior)
        >>> model.params_explicit
        ('period', 'eccentricity', 'phase_peri', 'arg_peri')
        """
        marg = set(self._auto_marginalized_names())
        explicit_linear = tuple(n for n in self._all_linear_names() if n not in marg)
        return self._all_nonlinear_names() + explicit_linear

    @property
    def params_marginalized(self) -> tuple[str, ...]:
        """Names of linear parameters analytically marginalized in log_prob.

        These have Gaussian (or callable-returning-Normal) priors and are
        integrated out analytically via the Woodbury identity rather than
        sampled.  Their values are NOT required in the ``values`` dict; they
        are recovered afterward via :meth:`sample_conditional_linear`.

        Examples
        --------
        >>> from unxt import Q
        >>> from harv.data import RVData
        >>> from harv.models.factories import rv_model
        >>> from harv.samplers import RejectionPrior
        >>> data = RVData(
        ...     time=Q([0.0, 50.0], "day"),
        ...     rv=Q([1.0, -2.0], "km/s"),
        ...     rv_err=Q([0.5, 0.5], "km/s"),
        ... )
        >>> prior = RejectionPrior.default_rv(
        ...     period_min=Q(10.0, "day"), period_max=Q(1000.0, "day"),
        ...     sigma_K0=Q(5.0, "km/s"), sigma_v0=Q(5.0, "km/s"),
        ... )
        >>> model = rv_model(data, linear_prior=prior.linear_prior)
        >>> model.params_marginalized
        ('rv_semiamp', 'v_sys')
        """
        return self._auto_marginalized_names()

    def _full_design_matrix(
        self,
        nl_values: dict[str, Any],
        data: Any,
    ) -> jax.Array:
        """Base design matrix + extension columns."""
        X = self._base_design_matrix(nl_values)
        for ext in self.extensions:
            X = ext.modify_design_matrix(X, data, nl_values)
        return X

    def _full_obs_err(
        self,
        obs_err: jax.Array,
        nl_values: dict[str, Any],
        data: Any,
    ) -> jax.Array:
        """Observation errors modified by extensions (jitter, GP, ...).

        Returns a 1-d diagonal or a 2-d covariance matrix depending on
        whether any extension promotes the shape.
        """
        cov = obs_err**2  # start with variances (diagonal)
        for ext in self.extensions:
            cov = ext.modify_covariance(cov, data, nl_values)
        return cov

    # Marginalization internals

    def _classify_columns(
        self,
        marginalized_names: tuple[str, ...],
        explicit_param_values: dict[str, jax.Array],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Split design-matrix columns into marginalized vs explicit.

        Returns (all_cols, explicit_names, marg_names).
        """
        all_cols = self._all_linear_names()
        explicit_names = tuple(
            n
            for n in all_cols
            if n not in marginalized_names or n in explicit_param_values
        )
        marg_names = tuple(n for n in all_cols if n not in explicit_names)
        return all_cols, explicit_names, marg_names

    def _assemble_prior(
        self,
        marg_names: tuple[str, ...],
        _: dict[str, Any],
    ) -> tuple[dict[str, PriorDist | LinearPriorCallable], dict[str, str]]:
        """Gather priors and units for marginalized columns.

        Returns (prior_dict, unit_dict).
        """
        if self.linear_prior is None:
            msg = "Cannot marginalize without linear_prior"
            raise ValueError(msg)

        obs_unit = self._obs_unit()
        param_units = self._linear_param_units()

        prior_dict: dict[str, PriorDist | LinearPriorCallable] = {}
        unit_dict: dict[str, str] = {}
        for name in marg_names:
            if name in self.linear_prior:
                prior_dict[name] = self.linear_prior[name]
                unit_dict[name] = param_units.get(name, obs_unit)

        return prior_dict, unit_dict

    @staticmethod
    def _handle_delta_priors(
        prior_dict: dict[str, PriorDist | LinearPriorCallable],
        unit_dict: dict[str, str],
        obs_unit: str,
        explicit_names: tuple[str, ...],
        marg_names: tuple[str, ...],
        linear_params: dict[str, jax.Array],
    ) -> tuple[
        dict[str, PriorDist | LinearPriorCallable],
        dict[str, str],
        tuple[str, ...],
        tuple[str, ...],
        dict[str, jax.Array],
    ]:
        """Reclassify Delta priors as explicit parameters."""
        delta_fixed: dict[str, jax.Array] = {}
        for name in list(prior_dict.keys()):
            prior_dist = prior_dict[name]
            if isinstance(prior_dist, QuantityDistribution) and isinstance(
                prior_dist.distribution, dist.Delta
            ):
                unit = unit_dict.get(name, obs_unit)

                delta_fixed[name] = jnp.array(
                    ustrip(
                        unit,
                        Q(prior_dist.distribution.v, cast("str", prior_dist.unit)),
                    )
                )
                del prior_dict[name]
            elif isinstance(prior_dist, dist.Delta):
                delta_fixed[name] = jnp.array(prior_dist.v)
                del prior_dict[name]

        if delta_fixed:
            explicit_names = (*explicit_names, *delta_fixed)
            marg_names = tuple(n for n in marg_names if n not in delta_fixed)
            linear_params = {**linear_params, **delta_fixed}
            unit_dict = {n: v for n, v in unit_dict.items() if n not in delta_fixed}

        return prior_dict, unit_dict, explicit_names, marg_names, linear_params

    def _build_marginalized_linear(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...],
        explicit_linear: dict[str, jax.Array],
        data: Any,
    ) -> _MargComponents:
        """Assemble the MarginalizedLinear distribution.

        Parameters
        ----------
        nl_values : dict
            Nonlinear parameter values (unit-stripped scalars).
        marginalized_names : tuple of str
            Which linear params to marginalize.
        explicit_linear : dict
            Values for any linear params evaluated explicitly (unit-stripped).
        data
            Raw data object (passed to extensions).

        Returns
        -------
        _MargComponents
        """
        X = self._full_design_matrix(nl_values, data)
        arr_obs, arr_obs_err = self._strip_obs()
        obs_unit = self._obs_unit()

        # Apply extension covariance modifications
        cov = self._full_obs_err(arr_obs_err, nl_values, data)

        all_cols, explicit_names, marg_names = self._classify_columns(
            marginalized_names, explicit_linear
        )

        prior_dict, unit_dict = self._assemble_prior(marg_names, nl_values)

        # Handle Delta priors
        prior_dict, unit_dict, explicit_names, marg_names, linear_params = (
            self._handle_delta_priors(
                prior_dict,
                unit_dict,
                obs_unit,
                explicit_names,
                marg_names,
                dict(explicit_linear),
            )
        )

        if len(marg_names) == 0:
            msg = (
                "No marginalized parameters remain after classification -- "
                "cannot build MarginalizedLinear"
            )
            raise ValueError(msg)

        # Subtract explicit linear contributions from observations
        if linear_params:
            idx = jnp.array([all_cols.index(n) for n in linear_params])
            y = jnp.array([linear_params[n] for n in linear_params])
            arr_obs = arr_obs - X[:, idx] @ y

        # Slice to marginalized columns
        marg_idx = jnp.array([all_cols.index(n) for n in marg_names])
        X_marg = X[:, marg_idx]

        # Resolve to MVN prior.
        # Wrap explicit linear values in Quantity so that callable priors
        # (e.g. ParallaxDependentProperMotionPrior) can read their units.
        param_units = self._linear_param_units()
        extra_q: dict[str, Any] = {}
        for name, val in linear_params.items():
            u = param_units.get(name, "")
            extra_q[name] = Q(val, u) if u else val
        lp = _resolve_prior_to_mvn(
            prior_dict, nl_values, unit_dict, extra_values=extra_q
        )

        # Build data distribution from covariance
        if cov.ndim == 1:
            data_dist = dist.Normal(0.0, jnp.sqrt(cov))
        else:
            data_dist = dist.MultivariateNormal(
                loc=jnp.zeros(cov.shape[0]), covariance_matrix=cov
            )

        marg_dist = MarginalizedLinear(
            design_matrix=X_marg,
            prior_distribution=lp,
            data_distribution=data_dist,
        )
        return _MargComponents(
            marg_dist, arr_obs, marg_names, explicit_names, linear_params
        )

    # Public API

    def log_prob(
        self,
        nl_values: dict[str, Any],
        linear_values: dict[str, jax.Array] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
        data: Any = None,
    ) -> jax.Array:
        """Compute the log-likelihood.

        Three calling conventions are supported:

        1. **Auto mode** (recommended): ``model.log_prob(values)`` where
           *values* may contain both nonlinear and explicit-linear entries.
           The model auto-classifies which linear params to marginalize
           from its ``linear_prior``.  Non-Gaussian linear priors are
           expected as entries in *values*.
        2. **Manual marginalization**: pass ``marginalized_names`` (and
           optionally ``linear_values``) to control exactly which linear
           params are marginalized.
        3. **Explicit evaluation**: pass ``linear_values`` without
           ``marginalized_names`` to evaluate the Gaussian log-likelihood
           at fixed linear parameter values.

        Parameters
        ----------
        nl_values : dict
            Parameter values.  In auto mode this may contain explicit
            linear parameter values alongside the nonlinear ones.
        linear_values : dict or None
            Explicit linear parameter values (unit-stripped).  When given
            without ``marginalized_names``, triggers explicit evaluation.
        marginalized_names : tuple of str or None
            Which linear params to marginalize (manual mode).
        data
            Raw data object.  Defaults to ``self.data`` when ``None``.

        Returns
        -------
        jax.Array
            Scalar log-likelihood.
        """
        if data is None:
            data = getattr(self, "data", None)

        # Auto mode: classify from linear_prior, extract explicit from values
        if (
            self.linear_prior is not None
            and linear_values is None
            and marginalized_names is None
        ):
            marginalized_names = self._auto_marginalized_names()
            all_linear = set(self._all_linear_names())
            linear_values = {
                k: nl_values[k] for k in list(nl_values) if k in all_linear
            }
            nl_values = {k: v for k, v in nl_values.items() if k not in all_linear}
            return self._log_prob_marginalized(
                nl_values, marginalized_names, linear_values, data
            )

        # Manual marginalization mode
        if marginalized_names is not None and self.linear_prior is not None:
            if linear_values is None:
                nl_values, linear_values = self._extract_explicit_linear_values(
                    nl_values,
                    marginalized_names,
                )
            if len(marginalized_names) == 0:
                return self._log_prob_explicit(nl_values, linear_values or {}, data)
            return self._log_prob_marginalized(
                nl_values,
                marginalized_names,
                linear_values or {},
                data,
            )

        # Explicit evaluation
        return self._log_prob_explicit(nl_values, linear_values or {}, data)

    def _log_prob_marginalized(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...],
        explicit_linear: dict[str, jax.Array],
        data: Any,
    ) -> jax.Array:
        c = self._build_marginalized_linear(
            nl_values, marginalized_names, explicit_linear, data
        )
        return c.dist.log_prob(c.obs)

    def _log_prob_explicit(
        self,
        nl_values: dict[str, Any],
        linear_values: dict[str, jax.Array],
        data: Any,
    ) -> jax.Array:
        """Explicit Gaussian log-likelihood (no marginalization)."""
        X = self._full_design_matrix(nl_values, data)
        arr_obs, arr_obs_err = self._strip_obs()

        # Apply extension covariance modifications
        cov = self._full_obs_err(arr_obs_err, nl_values, data)

        all_cols = self._all_linear_names()
        y = jnp.array([linear_values.get(name, 0.0) for name in all_cols])
        y_pred = X @ y

        if cov.ndim == 1:
            return dist.Normal(y_pred, jnp.sqrt(cov)).log_prob(arr_obs).sum()
        return dist.MultivariateNormal(loc=y_pred, covariance_matrix=cov).log_prob(
            arr_obs
        )

    def sample_conditional_linear(
        self,
        nl_values: dict[str, Any],
        key: jax.Array,
        marginalized_names: tuple[str, ...] | None = None,
        explicit_linear: dict[str, jax.Array] | None = None,
        data: Any = None,
    ) -> dict[str, jax.Array]:
        """Sample linear parameters from the conditional posterior.

        In auto mode (both ``marginalized_names`` and ``explicit_linear``
        are ``None``), the method classifies from ``linear_prior`` and
        extracts explicit linear values from *nl_values*.

        Returns all linear parameter values (both sampled and explicit),
        unit-stripped.
        """
        if data is None:
            data = getattr(self, "data", None)
        # Auto-classify when no explicit arguments given
        if (
            marginalized_names is None
            and explicit_linear is None
            and self.linear_prior is not None
        ):
            marginalized_names = self._auto_marginalized_names()
            all_linear = set(self._all_linear_names())
            explicit_lin_names = all_linear - set(marginalized_names)
            explicit_linear = {
                k: nl_values[k] for k in list(nl_values) if k in explicit_lin_names
            }
        elif marginalized_names is not None and explicit_linear is None:
            nl_values, explicit_linear = self._extract_explicit_linear_values(
                nl_values,
                marginalized_names,
            )

        if marginalized_names is not None and len(marginalized_names) == 0:
            return explicit_linear or {}

        marg_names = marginalized_names or self._all_linear_names()
        c = self._build_marginalized_linear(
            nl_values, marg_names, explicit_linear or {}, data
        )
        sample = c.dist.conditional(c.obs).sample(key)

        result: dict[str, jax.Array] = {}
        for i, name in enumerate(c.marg_names):
            result[name] = sample[i]
        for name in c.explicit_names:
            result[name] = c.linear_params[name]
        return result

    def numpyro_model(
        self,
        nonlinear_priors: dict[str, PriorDist],
        *,
        marginalized: bool = True,
        marginalized_names: tuple[str, ...] | None = None,
        data: Any = None,
    ) -> Callable[[], None]:
        """Build a numpyro model function for MCMC sampling.

        Parameters
        ----------
        nonlinear_priors : dict[str, PriorDist]
            Prior distributions for nonlinear parameters. Keys are parameter
            names (e.g. ``"period"``, ``"eccentricity"``). Values are
            :class:`~numpyro.distributions.Distribution` or
            :class:`~harv.distributions.QuantityDistribution`.
        marginalized : bool
            If ``True`` (default), linear parameters are analytically
            marginalized and only nonlinear parameters are sampled. If
            ``False``, all parameters are sampled explicitly.
        marginalized_names : tuple of str or None
            Optional subset of linear parameter names to analytically
            marginalize when ``marginalized=True``. ``None`` means use the
            model's automatic prior-based classification.
        data
            Raw data object.  Defaults to ``self.data`` when ``None``.

        Returns
        -------
        model_fn : callable
            A no-argument callable suitable for ``numpyro.infer.MCMC``.
        """
        if data is None:
            data = getattr(self, "data", None)
        if not marginalized and marginalized_names is not None:
            msg = "marginalized_names cannot be set when marginalized=False"
            raise ValueError(msg)
        if marginalized:
            return _build_marginalized_component_model(
                self,
                nonlinear_priors,
                data,
                marginalized_names=marginalized_names,
            )
        return _build_full_component_model(self, nonlinear_priors, data)


# Numpyro model builder helpers (module-level for pickling)


def _sample_nonlinear_params(
    nonlinear_priors: dict[str, PriorDist],
) -> dict[str, Any]:
    """Sample nonlinear parameters inside a numpyro model context.

    Returns a dict of sampled values. QuantityDistribution values are
    sampled from the underlying distribution; the unit is stored in the
    prior for later conversion.
    """
    values: dict[str, Any] = {}
    for name, d in nonlinear_priors.items():
        values[name] = numpyro.sample(
            name,
            cast("dist.Distribution", _unwrap_dist(d)),
        )
    return values


def _apply_unit_conversions(
    values: dict[str, Any],
    nonlinear_priors: dict[str, PriorDist],
    component: AbstractComponentModel,
) -> dict[str, Any]:
    """Convert sampled values from prior units to model units where needed.

    For QuantityDistribution priors on *base* (non-extension) nonlinear
    parameters, the sampled value is wrapped in a Quantity with the prior's
    unit. The model's Kepler solver handles stripping to internal units.

    Extension nonlinear parameters (e.g. jitter) are left as plain scalars
    because extension hooks operate on unit-stripped arrays.
    """
    base_params = component._base_nonlinear_names()
    result = dict(values)
    for name, d in nonlinear_priors.items():
        if isinstance(d, QuantityDistribution) and name in base_params:
            prior_unit = cast("str", d.unit)
            result[name] = Q(result[name], prior_unit)
    return result


def _build_marginalized_component_model(
    component: AbstractComponentModel,
    nonlinear_priors: dict[str, PriorDist],
    data: Any,
    *,
    marginalized_names: tuple[str, ...] | None = None,
) -> Callable[[], None]:
    """Build a marginalized numpyro model for a single component.

    Linear priors outside the marginalized subset are sampled explicitly via
    ``numpyro.sample`` and passed to
    ``component.log_prob(..., marginalized_names=...)``.
    """
    linear_prior = component.linear_prior or {}
    requested_marginalized_names = (
        component._auto_marginalized_names()
        if marginalized_names is None
        else marginalized_names
    )
    explicit_linear_prior = {
        name: prior_dist
        for name, prior_dist in linear_prior.items()
        if name not in set(requested_marginalized_names)
    }
    explicit_direct_prior = {
        name: prior_dist
        for name, prior_dist in explicit_linear_prior.items()
        if not callable(prior_dist)
        or isinstance(prior_dist, (dist.Distribution, QuantityDistribution))
    }
    explicit_callable_prior = {
        name: prior_dist
        for name, prior_dist in explicit_linear_prior.items()
        if callable(prior_dist)
        and not isinstance(prior_dist, (dist.Distribution, QuantityDistribution))
    }
    param_units = component._linear_param_units()

    def model_fn() -> None:
        values = _sample_nonlinear_params(nonlinear_priors)
        nl_values = _apply_unit_conversions(values, nonlinear_priors, component)
        explicit_linear_values: dict[str, Any] = {}
        explicit_linear_proxy: dict[str, Any] = {}

        for name, prior_dist in explicit_direct_prior.items():
            raw = numpyro.sample(
                name,
                cast("dist.Distribution", _unwrap_dist(prior_dist)),
            )
            target_unit = param_units.get(name, "")
            if isinstance(prior_dist, QuantityDistribution) and target_unit:
                raw = ustrip(target_unit, Q(raw, cast("str", prior_dist.unit)))
            explicit_linear_values[name] = raw
            explicit_linear_proxy[name] = Q(raw, target_unit) if target_unit else raw

        for name, prior_dist in explicit_callable_prior.items():
            target_unit = param_units.get(name, "")
            resolved_prior = _resolve_prior_to_mvn(
                {name: prior_dist},
                nl_values,
                {name: target_unit},
                extra_values=explicit_linear_proxy,
            )
            raw = numpyro.sample(
                name,
                dist.Normal(resolved_prior.loc[0], resolved_prior.scale_tril[0, 0]),
            )
            explicit_linear_values[name] = raw
            explicit_linear_proxy[name] = Q(raw, target_unit) if target_unit else raw

        numpyro.factor(
            "log_lik",
            component.log_prob(
                nl_values,
                linear_values=explicit_linear_values,
                marginalized_names=requested_marginalized_names,
                data=data,
            ),
        )

    return model_fn


def _build_full_component_model(  # noqa: C901
    component: AbstractComponentModel,
    nonlinear_priors: dict[str, PriorDist],
    data: Any,
) -> Callable[[], None]:
    """Build an explicit (non-marginalized) numpyro model.

    Both nonlinear and linear parameters are sampled. Linear parameters
    that have Gaussian priors are sampled jointly from their MVN; those
    with non-Gaussian priors (e.g. HalfNormal) are sampled individually.
    """
    linear_prior = component.linear_prior
    if linear_prior is None:
        msg = "Cannot build full numpyro model without linear_prior"
        raise ValueError(msg)

    # Classify linear priors: Gaussian (can go into joint MVN) vs
    # non-Gaussian (must be sampled individually).
    gaussian_lp: dict[str, PriorDist | LinearPriorCallable] = {}
    explicit_lp: dict[str, PriorDist] = {}
    for name, d in linear_prior.items():
        if _needs_explicit_sampling(d):
            explicit_lp[name] = d
        else:
            gaussian_lp[name] = d

    param_units = component._linear_param_units()
    gaussian_names = list(gaussian_lp.keys())

    def model_fn() -> None:
        values = _sample_nonlinear_params(nonlinear_priors)
        nl_values = _apply_unit_conversions(values, nonlinear_priors, component)

        # Sample non-Gaussian linear params individually
        linear_values: dict[str, Any] = {}
        for name, d in explicit_lp.items():
            raw = numpyro.sample(
                name,
                cast("dist.Distribution", _unwrap_dist(d)),
            )
            target_u = param_units.get(name, "")
            if isinstance(d, QuantityDistribution) and target_u:
                raw = ustrip(target_u, Q(raw, cast("str", d.unit)))
            linear_values[name] = raw

        # Sample Gaussian linear params jointly
        if gaussian_names:
            # Resolve callable priors (e.g. parallax-dependent proper motion)
            resolved_lp: dict[str, PriorDist | LinearPriorCallable] = {}
            proxy = types.SimpleNamespace(**nl_values)
            for name, d in gaussian_lp.items():
                if callable(d) and not isinstance(
                    d, dist.Distribution | QuantityDistribution
                ):
                    resolved_lp[name] = d(proxy)
                else:
                    resolved_lp[name] = d
            gaussian_units = {n: param_units.get(n, "") for n in gaussian_names}
            mvn = _resolve_prior_to_mvn(resolved_lp, nl_values, gaussian_units)
            linear_vec = jnp.atleast_1d(numpyro.sample("_linear", mvn))
            for i, lname in enumerate(gaussian_names):
                numpyro.deterministic(lname, linear_vec[i])
                linear_values[lname] = linear_vec[i]

        # Evaluate explicit log-likelihood
        numpyro.factor(
            "log_lik",
            component._log_prob_explicit(nl_values, linear_values, data),
        )

    return model_fn
