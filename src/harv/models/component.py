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
from unxt import Q
from unxt.quantity import ustrip

from harv.data.datasets import AbstractData
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    LinearPriorCallable,
    PriorDist,
    _needs_explicit_sampling,
    _resolve_prior_to_mvn,
    _unwrap_dist,
)
from harv.models.extensions.base import AbstractExtension, ParamInfo
from harv.stats import MarginalizedLinear


class _MargComponents(NamedTuple):
    """Internal return type of ``_build_marginalized_linear``."""

    dist: MarginalizedLinear
    obs: jax.Array
    marg_names: tuple[str, ...]
    explicit_names: tuple[str, ...]
    linear_params: dict[str, jax.Array]


class _MargBuildingBlocks(NamedTuple):
    """Intermediate building blocks before constructing ``MarginalizedLinear``.

    Returned by ``_build_marg_blocks``. The ``_build_marginalized_linear`` method
    assembles these into a full ``_MargComponents``. This exists so we can support
    shared linear parameters in joint models.
    """

    X: jax.Array  # (n, k_marg) — marginalized design matrix
    y: jax.Array  # (n,) — residualized observations
    cov: jax.Array  # (n,) diagonal or (n, n) full noise covariance
    marg_names: tuple[str, ...]  # length k_marg
    prior_mu: jax.Array  # (k_marg,) prior mean
    prior_scale_tril: jax.Array  # (k_marg, k_marg) prior Cholesky factor
    explicit_linear: dict[str, Any]  # explicit linear param values (unit-stripped)


class AbstractComponentModel(eqx.Module):
    """Abstract base for single-data-type component models.

    Component models are *templates*: they carry only ``parameterization`` and
    ``extensions`` (config). ``data`` and ``linear_prior`` are passed at
    evaluation time to the methods that need them. This means the same model
    instance can be re-used across multiple datasets without rebuilding.

    Concrete subclasses must implement:

    - ``_param_infos``: all parameter descriptors.
    - ``_base_design_matrix(nl_values, data)``: the base design matrix from
      runtime data + nonlinear values.
    - ``_strip_obs(data)``: return (obs, obs_err) as unit-stripped JAX arrays.
    - ``_obs_unit(data)``: the physical unit string of the observations.

    Concrete subclasses must also declare:

    - ``extensions``: tuple of AbstractExtension (model modifiers).
    """

    # Concrete subclasses must declare:
    extensions: eqx.AbstractVar[tuple[AbstractExtension, ...]]

    # Subclass hooks

    @abstractmethod
    def _param_infos(self) -> tuple[ParamInfo, ...]:
        """All parameter descriptors (base + extensions, nonlinear first)."""

    # NOTE: these abstract hooks take ``data: Any`` rather than ``AbstractData``
    # because concrete subclasses narrow it to their own dataset type (e.g.
    # ``RVModel._strip_obs(data: RVData)``) and access type-specific fields.
    # A narrowed override of an ``AbstractData`` param would be an LSP violation;
    # ``Any`` lets each model declare its concrete data contract.
    @abstractmethod
    def _base_design_matrix(self, nl_values: dict[str, Any], data: Any) -> jax.Array:
        """Build the base design matrix from data and nonlinear values.

        Columns correspond to the *base* linear parameters only (no
        extensions). Extensions append columns via ``modify_design_matrix``.
        """

    @abstractmethod
    def _strip_obs(self, data: Any) -> tuple[jax.Array, jax.Array]:
        """Return (observations, observation_errors) as unit-stripped arrays.

        Both arrays have shape ``(n_obs,)`` and share an implicit common unit
        (e.g. ``km/s`` for RV, ``mas`` for astrometry).
        """

    @abstractmethod
    def _obs_unit(self, data: Any) -> str:
        """The physical unit string of the observations (e.g. ``'km/s'``)."""

    def _linear_param_units(self, data: Any) -> dict[str, str]:
        """Map from linear parameter name to its concrete unit string.

        The default implementation returns ``obs_unit`` for every linear
        parameter.  Subclasses where different linear parameters have
        different units (e.g. astrometry: mas vs mas/yr) must override.
        """
        obs_unit = self._obs_unit(data)
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

    def _auto_marginalized_names(
        self, linear_priors: dict[str, Any] | None
    ) -> tuple[str, ...]:
        """Classify linear priors: Gaussian -> marginalize, non-Gaussian -> explicit.

        Returns the tuple of linear parameter names whose priors are
        Gaussian (Normal, callable returning Normal, or Delta) and can be
        analytically marginalized.  Non-Gaussian priors (HalfNormal, etc.)
        are excluded -- their values must be passed alongside the nonlinear
        parameters.
        """
        if linear_priors is None:
            return self._all_linear_names()
        return tuple(
            n for n, d in linear_priors.items() if not _needs_explicit_sampling(d)
        )

    def params_explicit(self, linear_priors: dict[str, Any] | None) -> tuple[str, ...]:
        """Names of parameters that must be explicitly sampled.

        Given a (resolved) ``linear_prior`` dict, returns all nonlinear
        parameters (orbital + extension, e.g. jitter) plus any linear
        parameters with non-Gaussian priors (e.g. parallax with a HalfNormal
        prior) that cannot be analytically marginalized.

        These are the keys that must appear in the ``values`` dict passed to
        :meth:`log_prob`.
        """
        marg = set(self._auto_marginalized_names(linear_priors))
        explicit_linear = tuple(n for n in self._all_linear_names() if n not in marg)
        return self._all_nonlinear_names() + explicit_linear

    def params_marginalized(
        self, linear_priors: dict[str, Any] | None
    ) -> tuple[str, ...]:
        """Names of linear parameters analytically marginalized in log_prob.

        Given a (resolved) ``linear_prior`` dict, returns the linear
        parameter names whose priors are Gaussian (or callable-returning-Normal)
        and so are integrated out analytically via the Woodbury identity
        rather than sampled. Their values are NOT required in the ``values``
        dict; they are recovered afterward via :meth:`sample_conditional_linear`.
        """
        return self._auto_marginalized_names(linear_priors)

    def _full_design_matrix(
        self,
        nl_values: dict[str, Any],
        data: AbstractData,
    ) -> jax.Array:
        """Base design matrix + extension columns."""
        X = self._base_design_matrix(nl_values, data)
        for ext in self.extensions:
            X = ext.modify_design_matrix(X, data, nl_values)
        return X

    def _full_obs_err(
        self,
        obs_err: jax.Array,
        nl_values: dict[str, Any],
        data: AbstractData,
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
        data: AbstractData,
        linear_priors: dict[str, Any] | None,
    ) -> tuple[dict[str, PriorDist | LinearPriorCallable], dict[str, str]]:
        """Gather priors and units for marginalized columns.

        Returns (prior_dict, unit_dict).
        """
        if linear_priors is None:
            msg = "Cannot marginalize without linear_priors"
            raise ValueError(msg)

        obs_unit = self._obs_unit(data)
        param_units = self._linear_param_units(data)

        prior_dict: dict[str, PriorDist | LinearPriorCallable] = {}
        unit_dict: dict[str, str] = {}
        for name in marg_names:
            if name in linear_priors:
                prior_dict[name] = linear_priors[name]
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

    def _build_marg_blocks(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...],
        explicit_linear: dict[str, jax.Array],
        data: AbstractData,
        linear_priors: dict[str, Any] | None,
    ) -> _MargBuildingBlocks:
        """Extract the building blocks needed to construct a MarginalizedLinear.

        This method performs all classification, prior resolution, and
        design-matrix slicing, but stops short of constructing the final
        ``MarginalizedLinear`` and ``_MargComponents``.  It is used by
        :meth:`_build_marginalized_linear` (single-component path) and by
        :meth:`~harv.models.joint.JointModel._build_joint_marginalized_linear`
        (joint-marginalization path).

        Parameters
        ----------
        nl_values
            Nonlinear parameter values (unit-stripped scalars).
        marginalized_names
            Which linear params to marginalize.
        explicit_linear
            Values for any linear params evaluated explicitly (unit-stripped).
        data
            Raw data object (passed to extensions and to ``_strip_obs``).
        linear_priors
            Per-parameter priors for analytic marginalization.

        Returns
        -------
            Marginalization building blocks for this component.
        """
        X = self._full_design_matrix(nl_values, data)
        arr_obs, arr_obs_err = self._strip_obs(data)
        obs_unit = self._obs_unit(data)

        # Apply extension covariance modifications
        cov = self._full_obs_err(arr_obs_err, nl_values, data)

        all_cols, explicit_names, marg_names = self._classify_columns(
            marginalized_names, explicit_linear
        )

        prior_dict, unit_dict = self._assemble_prior(
            marg_names, nl_values, data, linear_priors
        )

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
        param_units = self._linear_param_units(data)
        extra_q: dict[str, Any] = {}
        for name, val in linear_params.items():
            u = param_units.get(name, "")
            extra_q[name] = Q(val, u) if u else val
        lp = _resolve_prior_to_mvn(
            prior_dict, nl_values, unit_dict, extra_values=extra_q
        )

        return _MargBuildingBlocks(
            X=X_marg,
            y=arr_obs,
            cov=cov,
            marg_names=marg_names,
            prior_mu=lp.loc,  # ty: ignore[invalid-argument-type]
            prior_scale_tril=lp.scale_tril,
            explicit_linear=linear_params,
        )

    def _build_marginalized_linear(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...],
        explicit_linear: dict[str, jax.Array],
        data: AbstractData,
        linear_priors: dict[str, Any] | None,
    ) -> _MargComponents:
        """Assemble the MarginalizedLinear distribution.

        Parameters
        ----------
        nl_values
            Nonlinear parameter values (unit-stripped scalars).
        marginalized_names
            Which linear params to marginalize.
        explicit_linear
            Values for any linear params evaluated explicitly (unit-stripped).
        data
            Raw data object (passed to extensions).
        linear_priors
            Per-parameter priors for analytic marginalization.

        Returns
        -------
            Per-component marginalization components ready to be summed
            into a joint marginalization.
        """
        blocks = self._build_marg_blocks(
            nl_values, marginalized_names, explicit_linear, data, linear_priors
        )

        # Build data distribution from covariance
        if blocks.cov.ndim == 1:
            data_dist = dist.Normal(0.0, jnp.sqrt(blocks.cov))
        else:
            data_dist = dist.MultivariateNormal(
                loc=jnp.zeros(blocks.cov.shape[0]), covariance_matrix=blocks.cov
            )

        prior = dist.MultivariateNormal(
            loc=blocks.prior_mu, scale_tril=blocks.prior_scale_tril
        )
        marg_dist = MarginalizedLinear(
            design_matrix=blocks.X,
            prior_distribution=prior,
            data_distribution=data_dist,
        )
        explicit_names = tuple(blocks.explicit_linear.keys())
        return _MargComponents(
            marg_dist,
            blocks.y,
            blocks.marg_names,
            explicit_names,
            blocks.explicit_linear,
        )

    # Public API

    def log_prob(
        self,
        nl_values: dict[str, Any],
        data: AbstractData,
        *,
        linear_priors: dict[str, Any] | None = None,
        linear_values: dict[str, jax.Array] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute the log-likelihood.

        Three calling conventions are supported:

        1. **Auto mode** (recommended): pass ``linear_prior`` and let the
           model classify which linear params to marginalize. Non-Gaussian
           linear priors are expected as entries in ``nl_values``.
        2. **Manual marginalization**: pass ``marginalized_names`` (and
           optionally ``linear_values``) to control exactly which linear
           params are marginalized.
        3. **Explicit evaluation**: pass ``linear_values`` without
           ``marginalized_names`` (and ``linear_priors=None``) to evaluate
           the Gaussian log-likelihood at fixed linear parameter values.

        Parameters
        ----------
        nl_values
            Parameter values.  In auto mode this may contain explicit
            linear parameter values alongside the nonlinear ones.
        data
            Runtime observation data (RVData / GaiaAstrometryData / SystemData).
        linear_priors
            Per-parameter priors for analytic marginalization. Required for
            auto and manual-marginalization modes.
        linear_values
            Explicit linear parameter values (unit-stripped).  When given
            without ``marginalized_names``, triggers explicit evaluation.
        marginalized_names
            Which linear params to marginalize (manual mode).

        Returns
        -------
            Scalar log-likelihood.
        """
        # Auto mode: classify from linear_priors, extract explicit from values
        if (
            linear_priors is not None
            and linear_values is None
            and marginalized_names is None
        ):
            marginalized_names = self._auto_marginalized_names(linear_priors)
            all_linear = set(self._all_linear_names())
            linear_values = {
                k: nl_values[k] for k in list(nl_values) if k in all_linear
            }
            nl_values = {k: v for k, v in nl_values.items() if k not in all_linear}
            return self._log_prob_marginalized(
                nl_values, marginalized_names, linear_values, data, linear_priors
            )

        # Manual marginalization mode
        if marginalized_names is not None and linear_priors is not None:
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
                linear_priors,
            )

        # Explicit evaluation
        return self._log_prob_explicit(nl_values, linear_values or {}, data)

    def _log_prob_marginalized(
        self,
        nl_values: dict[str, Any],
        marginalized_names: tuple[str, ...],
        explicit_linear: dict[str, jax.Array],
        data: AbstractData,
        linear_priors: dict[str, Any] | None,
    ) -> jax.Array:
        c = self._build_marginalized_linear(
            nl_values, marginalized_names, explicit_linear, data, linear_priors
        )
        base_lp: jax.Array = c.dist.log_prob(c.obs)

        # Apply optional Jacobian correction from the parameterization.
        # This is non-trivial for e.g. ThieleInnesGaiaAstrometry, where the
        # TI linear params (A,B,F,G) are not the natural physical params
        # (a0, ω, Ω, cos i) and a Jacobian correction is needed to recover
        # the correct posterior.  The conditional mean is a zeroth-order
        # approximation following Hsieh et al.
        parameterization = getattr(self, "parameterization", None)
        if parameterization is not None:
            cond_mean = c.dist.conditional(c.obs).mean  # (n_marg_params,)
            linear_map = dict(zip(c.marg_names, cond_mean, strict=True))
            correction = parameterization.linear_log_prior_correction(linear_map)
            if correction is not None:
                return base_lp + correction
        return base_lp

    def predict(
        self,
        nl_values: dict[str, Any],
        linear_values: dict[str, jax.Array],
        data: AbstractData,
    ) -> jax.Array:
        """Full predicted observable ``y_pred = X @ y`` at the data times.

        ``X`` is the extension-augmented design matrix
        (:meth:`_full_design_matrix`) and ``y`` is the ordered linear-parameter
        vector built from ``linear_values`` (missing entries default to ``0``).
        Used by :meth:`_log_prob_explicit`, :meth:`chi_squared`, and the plot
        functions so that every prediction path shares the same construction.
        """
        X = self._full_design_matrix(nl_values, data)
        y = jnp.array(
            [linear_values.get(name, 0.0) for name in self._all_linear_names()]
        )
        return X @ y

    def _log_prob_explicit(
        self,
        nl_values: dict[str, Any],
        linear_values: dict[str, jax.Array],
        data: AbstractData,
    ) -> jax.Array:
        """Explicit Gaussian log-likelihood (no marginalization)."""
        arr_obs, arr_obs_err = self._strip_obs(data)
        cov = self._full_obs_err(arr_obs_err, nl_values, data)
        y_pred = self.predict(nl_values, linear_values, data)

        if cov.ndim == 1:
            return dist.Normal(y_pred, jnp.sqrt(cov)).log_prob(arr_obs).sum()
        return dist.MultivariateNormal(loc=y_pred, covariance_matrix=cov).log_prob(
            arr_obs
        )

    def chi_squared(
        self,
        nl_values: dict[str, Any],
        linear_values: dict[str, jax.Array],
        data: AbstractData,
    ) -> jax.Array:
        r"""Goodness-of-fit :math:`\chi^2` for one fully-specified parameter set.

        Unlike :meth:`log_prob` (which marginalizes the linear parameters and
        returns a *marginal* log-likelihood), this evaluates the model at the
        given linear-parameter values and returns the residual statistic

        .. math::

            \chi^2 = r^\top C^{-1} r, \qquad r = y_\mathrm{obs} - X\,y,

        where :math:`C` is the (extension-modified) observation covariance.  For
        a diagonal covariance this is :math:`\sum_i r_i^2 / \sigma_i^2`; a
        Gaussian-process extension promotes :math:`C` to a full matrix and the
        Mahalanobis form is used.  Jitter is included via the inflated :math:`C`.

        Parameters
        ----------
        nl_values
            Nonlinear parameter values (orbital + any extension parameters), in
            the same form accepted by :meth:`log_prob`.
        linear_values
            Linear parameter values, unit-stripped to the model's linear
            parameter units (see :meth:`_linear_param_units`).
        data
            Runtime observation data.

        Returns
        -------
            Scalar :math:`\chi^2`.
        """
        arr_obs, arr_obs_err = self._strip_obs(data)
        cov = self._full_obs_err(arr_obs_err, nl_values, data)
        resid = arr_obs - self.predict(nl_values, linear_values, data)

        if cov.ndim == 1:
            return jnp.sum(resid**2 / cov)
        return resid @ jnp.linalg.solve(cov, resid)

    def sample_conditional_linear(
        self,
        nl_values: dict[str, Any],
        key: jax.Array,
        data: AbstractData,
        *,
        linear_priors: dict[str, Any] | None = None,
        marginalized_names: tuple[str, ...] | None = None,
        explicit_linear: dict[str, jax.Array] | None = None,
        use_mean: bool = False,
    ) -> dict[str, jax.Array]:
        """Sample linear parameters from the conditional posterior.

        In auto mode (both ``marginalized_names`` and ``explicit_linear``
        are ``None``), the method classifies from ``linear_prior`` and
        extracts explicit linear values from ``nl_values``.

        Returns all linear parameter values (both sampled and explicit),
        unit-stripped.

        When ``use_mean=True``, the conditional posterior **mean** is returned
        for the marginalized linear parameters instead of a random draw. For
        a Gaussian conditional this is also the conditional MAP. This is the
        appropriate choice when completing a MAP estimate (see
        :meth:`~harv.samplers.NumpyroSampler.optimize`); MCMC paths should keep
        the default ``use_mean=False``.
        """
        # Auto-classify when no explicit arguments given
        if (
            marginalized_names is None
            and explicit_linear is None
            and linear_priors is not None
        ):
            marginalized_names = self._auto_marginalized_names(linear_priors)
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
            nl_values, marg_names, explicit_linear or {}, data, linear_priors
        )
        cond = c.dist.conditional(c.obs)
        sample = cond.mean if use_mean else cond.sample(key)

        result: dict[str, jax.Array] = {}
        for i, name in enumerate(c.marg_names):
            result[name] = sample[i]
        for name in c.explicit_names:
            result[name] = c.linear_params[name]
        return result

    def numpyro_model(
        self,
        nonlinear_priors: dict[str, PriorDist],
        data: AbstractData,
        linear_priors: dict[str, Any] | None,
        *,
        marginalized: bool = True,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> Callable[[], None]:
        """Build a numpyro model function for MCMC sampling.

        Parameters
        ----------
        nonlinear_priors
            Prior distributions for nonlinear parameters. Keys are parameter
            names (e.g. ``"period"``, ``"eccentricity"``). Values are
            :class:`~numpyro.distributions.distribution.Distribution` or
            :class:`~harv.distributions.QuantityDistribution`.
        data
            Runtime observation data (RVData / GaiaAstrometryData).
        linear_priors
            Per-parameter priors for the linear parameters. Required when
            any marginalization happens (``marginalized=True`` or full
            non-marginalized mode that still needs explicit linear priors).
        marginalized
            If ``True`` (default), linear parameters are analytically
            marginalized and only nonlinear parameters are sampled. If
            ``False``, all parameters are sampled explicitly.
        marginalized_names
            Optional subset of linear parameter names to analytically
            marginalize when ``marginalized=True``. ``None`` means use the
            model's automatic prior-based classification.

        Returns
        -------
            A no-argument callable suitable for ``numpyro.infer.MCMC``.
        """
        if not marginalized and marginalized_names is not None:
            msg = "marginalized_names cannot be set when marginalized=False"
            raise ValueError(msg)
        if marginalized:
            return _build_marginalized_component_model(
                self,
                nonlinear_priors,
                data,
                linear_priors,
                marginalized_names=marginalized_names,
            )
        return _build_full_component_model(self, nonlinear_priors, data, linear_priors)


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
            _unwrap_dist(d),
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
    data: AbstractData,
    linear_priors: dict[str, Any] | None,
    *,
    marginalized_names: tuple[str, ...] | None = None,
) -> Callable[[], None]:
    """Build a marginalized numpyro model for a single component.

    Linear priors outside the marginalized subset are sampled explicitly via
    ``numpyro.sample`` and passed to
    ``component.log_prob(..., marginalized_names=...)``.
    """
    lp_dict = linear_priors or {}
    requested_marginalized_names = (
        component._auto_marginalized_names(linear_priors)
        if marginalized_names is None
        else marginalized_names
    )
    explicit_linear_prior = {
        name: prior_dist
        for name, prior_dist in lp_dict.items()
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
    param_units = component._linear_param_units(data)

    def model_fn() -> None:
        values = _sample_nonlinear_params(nonlinear_priors)
        nl_values = _apply_unit_conversions(values, nonlinear_priors, component)
        explicit_linear_values: dict[str, Any] = {}
        explicit_linear_proxy: dict[str, Any] = {}

        for name, prior_dist in explicit_direct_prior.items():
            raw = numpyro.sample(name, _unwrap_dist(prior_dist))
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
                data,
                linear_priors=linear_priors,
                linear_values=explicit_linear_values,
                marginalized_names=requested_marginalized_names,
            ),
        )

    return model_fn


def _build_full_component_model(  # noqa: C901
    component: AbstractComponentModel,
    nonlinear_priors: dict[str, PriorDist],
    data: AbstractData,
    linear_priors: dict[str, Any] | None,
) -> Callable[[], None]:
    """Build an explicit (non-marginalized) numpyro model.

    Both nonlinear and linear parameters are sampled. Linear parameters
    that have Gaussian priors are sampled jointly from their MVN; those
    with non-Gaussian priors (e.g. HalfNormal) are sampled individually.
    """
    if linear_priors is None:
        msg = "Cannot build full numpyro model without linear_priors"
        raise ValueError(msg)

    # Classify linear priors: Gaussian (can go into joint MVN) vs
    # non-Gaussian (must be sampled individually).
    gaussian_lp: dict[str, PriorDist | LinearPriorCallable] = {}
    explicit_lp: dict[str, PriorDist] = {}
    for name, d in linear_priors.items():
        if _needs_explicit_sampling(d):
            explicit_lp[name] = d
        else:
            gaussian_lp[name] = d

    param_units = component._linear_param_units(data)
    gaussian_names = list(gaussian_lp.keys())

    def model_fn() -> None:
        values = _sample_nonlinear_params(nonlinear_priors)
        nl_values = _apply_unit_conversions(values, nonlinear_priors, component)

        # Sample non-Gaussian linear params individually
        linear_values: dict[str, Any] = {}
        for name, d in explicit_lp.items():
            target_u = param_units.get(name, "")
            raw = numpyro.sample(
                name,
                _unwrap_dist(d),
            )
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
