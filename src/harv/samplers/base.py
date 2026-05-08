"""Abstract base for harv samplers.

Carries the shared field set and the :meth:`from_prior` ergonomic entry
point used by both :class:`~harv.RejectionSampler` and
:class:`~harv.NumpyroSampler`. Algorithm-specific surface (``run()`` and
its private helpers) lives on each concrete subclass, since the rejection
and MCMC algorithms have fundamentally different ``run()`` signatures and
internal state.
"""

from typing import Any, cast

import equinox as eqx

from harv.data import GaiaAstrometryData, RVData
from harv.extensions.base import AbstractExtension
from harv.models.component import AbstractComponentModel
from harv.models.factories import _build_model
from harv.models.joint import JointModel
from harv.models.parameterizations import AbstractParameterization
from harv.samplers.rejection_prior import RejectionPrior

__all__ = ("AbstractSampler",)


class AbstractSampler(eqx.Module):
    """Abstract base for harv samplers.

    Every sampler holds a fully-built model. Two construction paths are
    supported:

    - **Bare constructor** (``Sampler(prior, model)``): expert path. Hand
      in a pre-built :class:`~harv.AbstractComponentModel` or
      :class:`~harv.JointModel`. Required for joint / multi-survey /
      custom-parameterization workflows where the user constructs the
      model explicitly.
    - :meth:`from_prior` classmethod (``Sampler.from_prior(prior, data, ...)``):
      ergonomic shortcut for the typical single-component case. Builds a
      default model from data + (optional) extensions + parameterization
      under the hood, wires linear extension priors from ``prior``, and
      forwards to the bare constructor. Default and intermediate users
      need not touch the model classes directly.

    Concrete subclasses are marked ``@final`` per the project's
    abstract-final pattern.
    """

    prior: RejectionPrior
    model: AbstractComponentModel | JointModel
    marginalized_names: tuple[str, ...] | None = None

    @classmethod
    def from_prior(
        cls,
        prior: RejectionPrior,
        data: RVData | GaiaAstrometryData,
        *,
        extensions: tuple[AbstractExtension, ...] = (),
        parameterization: AbstractParameterization | None = None,
        marginalized_names: tuple[str, ...] | None = None,
    ) -> "AbstractSampler":
        """Build a default single-component model and wrap it in a sampler.

        Recommended entry point for default and intermediate users. The
        method calls :func:`harv.rv_model` (or
        :func:`harv.gaia_astrometry_model`) under the hood to build the
        model from ``data``, then attaches it to a new sampler instance.
        Linear extension priors from ``prior.extension_priors`` are
        merged into the model's ``linear_prior`` automatically.

        For joint, multi-survey, or custom-parameterization workflows,
        construct the model yourself and call the bare constructor:
        ``Sampler(prior, model)``.

        Parameters
        ----------
        prior : RejectionPrior
            Prior distributions for nonlinear, linear, and extension
            parameters.
        data : RVData or GaiaAstrometryData
            Observed data (single component).
        extensions : tuple of AbstractExtension, optional
            Extensions to attach to the built model (jitter, GP, trends,
            etc.).
        parameterization : AbstractParameterization, optional
            Custom parameterization (RV only).
        marginalized_names : tuple of str, optional
            Subset of linear parameters to analytically marginalize. If
            ``None``, the model auto-classifies based on its prior types.

        Returns
        -------
        sampler
            A new instance of the concrete subclass on which this method
            was called.
        """
        model = _build_model(
            data,
            prior=prior,
            extensions=extensions,
            parameterization=cast("Any", parameterization),
        )
        return cls(prior=prior, model=model, marginalized_names=marginalized_names)

    def get_extensions(
        self,
    ) -> tuple[AbstractExtension, ...] | dict[str, tuple[AbstractExtension, ...]]:
        """Return the extensions in effect for this sampler.

        Walks the attached model: returns ``model.extensions`` for a
        single component model, or
        ``dict[component_name, tuple[Extension, ...]]`` for a
        :class:`~harv.JointModel` so the per-component association used
        for namespaced parameter names like ``"primary.jitter"`` is
        preserved. Use this in any downstream consumer (e.g.
        :func:`harv.plot.plot_rv`) that needs to know which extensions
        are in play.
        """
        if isinstance(self.model, JointModel):
            return {
                name: comp.extensions for name, comp in self.model.components.items()
            }
        return self.model.extensions
