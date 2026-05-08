"""Pre-configured model factories (convenience recipes).

These functions construct :class:`~harv.models.rv.RVModel` and
:class:`~harv.models.astrometry.GaiaAstrometryModel` with sensible default
priors, reducing boilerplate for common use-cases.
"""

__all__ = ("gaia_astrometry_model", "rv_model")

from typing import TYPE_CHECKING, Any, Literal

import numpyro.distributions as dist

from harv.data import GaiaAstrometryData, RVData
from harv.extensions.base import AbstractExtension
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV
from harv.models.rv import RVModel

if TYPE_CHECKING:
    from harv.samplers.rejection_prior import RejectionPrior


def _merge_linear_extension_priors(
    base_linear_prior: dict[str, Any],
    extensions: tuple[AbstractExtension, ...],
    extension_priors: dict[str, Any],
) -> dict[str, Any]:
    """Merge linear-extension priors into a base linear-prior dict.

    For each extension parameter declared as ``linear=True``, look up its
    prior in ``extension_priors`` and add to the merged dict. Raises
    :class:`ValueError` if a linear extension parameter has no matching
    entry in ``extension_priors``.

    The nonlinear extension priors are not merged here — they're consumed
    by the sampler at run time, not stored on the model.
    """
    merged = dict(base_linear_prior)
    for ext in extensions:
        for p in ext.extra_params():
            if not p.linear:
                continue
            if p.name not in extension_priors:
                raise ValueError(
                    f"Extension '{type(ext).__name__}' requires a prior for "
                    f"parameter '{p.name}'. Pass it as a keyword argument: "
                    f"RejectionPrior.default_rv(..., {p.name}=QD(...))."
                )
            merged[p.name] = extension_priors[p.name]
    return merged


def rv_model(
    data: RVData,
    *,
    prior: "RejectionPrior | None" = None,
    parameterization: StandardRV | EcoswEsinwRV | None = None,
    extensions: tuple[AbstractExtension, ...] = (),
    linear_prior: dict[str, Any] | Literal[False] | None = None,
) -> RVModel:
    """Build an RV model with sensible defaults.

    Parameters
    ----------
    data : RVData
        Observed radial velocities.
    prior : RejectionPrior, optional
        When supplied (and ``linear_prior is None``), the model's
        ``linear_prior`` is built from ``prior.linear_prior`` plus the
        linear extension priors looked up in ``prior.extension_priors``.
        This is the typical wiring used by
        :meth:`~harv.RejectionSampler.from_prior`.
    parameterization : StandardRV or EcoswEsinwRV or None
        RV parameterization. Defaults to :class:`StandardRV`.
    extensions : tuple of AbstractExtension
        Model extensions (jitter, trends, offsets).
    linear_prior : dict, False, or None
        If ``None`` (default) and ``prior`` is also ``None``, uses wide
        Normal priors on ``rv_semiamp`` and ``v_sys`` (sigma=1000 km/s).
        If ``None`` and ``prior`` is supplied, the linear prior is taken
        from ``prior``. Pass ``False`` for explicit (non-marginalized)
        mode, or a custom dict to override.

    Returns
    -------
    RVModel

    Examples
    --------
    >>> from unxt import Q
    >>> from harv.data import RVData
    >>> from harv.models.factories import rv_model
    >>> data = RVData(
    ...     time=Q([0.0, 50.0, 100.0], "day"),
    ...     rv=Q([1.0, -2.0, 0.5], "km/s"),
    ...     rv_err=Q([0.5, 0.5, 0.5], "km/s"),
    ... )
    >>> model = rv_model(data)
    >>> sorted(model._all_linear_names())
    ['rv_semiamp', 'v_sys']
    """
    if parameterization is None:
        parameterization = StandardRV()

    if linear_prior is None:
        if prior is not None:
            base = (
                dict(prior.linear_prior) if isinstance(prior.linear_prior, dict) else {}
            )
            linear_prior = _merge_linear_extension_priors(
                base, extensions, prior.extension_priors
            )
        else:
            linear_prior = {
                "rv_semiamp": dist.Normal(0.0, 1000.0),
                "v_sys": dist.Normal(0.0, 1000.0),
            }
    elif linear_prior is False:
        linear_prior = None

    return RVModel(
        data=data,
        parameterization=parameterization,
        extensions=extensions,
        linear_prior=linear_prior,
    )


def gaia_astrometry_model(
    data: GaiaAstrometryData,
    *,
    prior: "RejectionPrior | None" = None,
    extensions: tuple[AbstractExtension, ...] = (),
    linear_prior: dict[str, Any] | Literal[False] | None = None,
) -> GaiaAstrometryModel:
    """Build a Gaia astrometry model with sensible defaults.

    Parameters
    ----------
    data : GaiaAstrometricData
        Gaia epoch astrometric data.
    prior : RejectionPrior, optional
        When supplied (and ``linear_prior is None``), the model's
        ``linear_prior`` is built from ``prior.linear_prior`` plus the
        linear extension priors looked up in ``prior.extension_priors``.
    extensions : tuple of AbstractExtension
        Model extensions.
    linear_prior : dict, False, or None
        If ``None`` (default) and ``prior`` is also ``None``, uses wide
        Normal priors on all 6 linear astrometric parameters. If ``None``
        and ``prior`` is supplied, the linear prior is taken from
        ``prior``. Pass ``False`` for explicit mode, or a custom dict.

    Returns
    -------
    GaiaAstrometryModel

    Examples
    --------
    >>> from harv.models.factories import gaia_astrometry_model
    >>> # gaia_astrometry_model(data)  # requires GaiaAstrometricData
    """
    if linear_prior is None:
        if prior is not None:
            base = (
                dict(prior.linear_prior) if isinstance(prior.linear_prior, dict) else {}
            )
            linear_prior = _merge_linear_extension_priors(
                base, extensions, prior.extension_priors
            )
        else:
            linear_prior = {
                "ra0": dist.Normal(0.0, 1e6),
                "dec0": dist.Normal(0.0, 1e6),
                "pmra": dist.Normal(0.0, 1e6),
                "pmdec": dist.Normal(0.0, 1e6),
                "parallax": dist.Normal(0.0, 1e6),
                "semi_major_axis": dist.Normal(0.0, 1e6),
            }
    elif linear_prior is False:
        linear_prior = None

    return GaiaAstrometryModel(
        data=data,
        extensions=extensions,
        linear_prior=linear_prior,
    )


def _build_model(
    data: RVData | GaiaAstrometryData,
    *,
    prior: "RejectionPrior | None" = None,
    extensions: tuple[AbstractExtension, ...] = (),
    parameterization: StandardRV | EcoswEsinwRV | None = None,
    linear_prior: dict[str, Any] | None = None,
) -> RVModel | GaiaAstrometryModel:
    """Build a component model by dispatching on data type.

    Parameters
    ----------
    data : RVData or GaiaAstrometryData
        Observed data. Dispatches to :func:`rv_model` or
        :func:`gaia_astrometry_model` based on type.
    prior : RejectionPrior, optional
        When supplied, the linear prior of the resulting model is built
        from ``prior.linear_prior`` plus the linear extension priors
        looked up in ``prior.extension_priors``. Mutually exclusive with
        a non-``None`` ``linear_prior``.
    extensions : tuple of AbstractExtension
        Extensions to include.
    parameterization : StandardRV, EcoswEsinwRV, or None
        RV parameterization. Only used when ``data`` is :class:`RVData`.
        Ignored for :class:`GaiaAstrometryData`.
    linear_prior : dict or None
        Explicit per-parameter linear prior. ``None`` means use the
        ``prior``-derived merge (if ``prior`` is supplied) or the wide
        Normal defaults from the public factories.

    Returns
    -------
    RVModel or GaiaAstrometryModel

    Raises
    ------
    TypeError
        If ``data`` is not :class:`RVData` or :class:`GaiaAstrometryData`.
    """
    # The public factories use ``False`` for "explicit non-marginalized
    # mode" and ``None`` for "build defaults"; this internal helper takes
    # ``None`` directly and forwards ``prior`` so the factory can do its
    # own default-building.
    lp_arg: dict[str, Any] | Literal[False] | None = linear_prior

    if isinstance(data, RVData):
        return rv_model(
            data,
            prior=prior,
            parameterization=parameterization,
            extensions=extensions,
            linear_prior=lp_arg,
        )
    if isinstance(data, GaiaAstrometryData):
        return gaia_astrometry_model(
            data,
            prior=prior,
            extensions=extensions,
            linear_prior=lp_arg,
        )
    msg = (
        f"_build_model() received unsupported data type {type(data).__name__}. "
        "Supported types: RVData, GaiaAstrometryData."
    )
    raise TypeError(msg)
