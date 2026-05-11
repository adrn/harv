"""Abstract base for harv samplers.

Carries the shared field set (``prior``, ``model``, ``marginalized_names``).
Algorithm-specific surface (``run()`` and its private helpers) lives on each
concrete subclass, since the rejection and MCMC algorithms have fundamentally
different ``run()`` signatures and internal state.
"""

import equinox as eqx

from harv.extensions.base import AbstractExtension
from harv.models.component import AbstractComponentModel
from harv.models.joint import JointModel
from harv.samplers.rejection_prior import RejectionPrior

__all__ = ("AbstractSampler",)


class AbstractSampler(eqx.Module):
    """Abstract base for harv samplers.

    Every sampler holds a prior and a pre-built model.  Use the bare
    constructor to combine them:

    .. code-block:: python

        sampler = RejectionSampler(prior, RVModel(extensions=...))
        samples = sampler.run(data, n_prior_samples=100_000)

    Concrete subclasses are marked ``@final`` per the project's
    abstract-final pattern.
    """

    prior: RejectionPrior
    model: AbstractComponentModel | JointModel
    marginalized_names: tuple[str, ...] | None = None

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
