"""Pre-configured model factories (convenience recipes).

These functions construct :class:`~harv.models.rv.RVModel` and
:class:`~harv.models.astrometry.GaiaAstrometryModel` with sensible default
priors, reducing boilerplate for common use-cases.
"""

__all__ = ("gaia_astrometry_model", "rv_model")

from typing import Any, Literal

import numpyro.distributions as dist

from harv.data import GaiaAstrometryData, RVData
from harv.extensions.base import AbstractExtension
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV
from harv.models.rv import RVModel


def rv_model(
    data: RVData,
    *,
    parameterization: StandardRV | EcoswEsinwRV | None = None,
    extensions: tuple[AbstractExtension, ...] = (),
    linear_prior: dict[str, Any] | Literal[False] | None = None,
) -> RVModel:
    """Build an RV model with sensible defaults.

    Parameters
    ----------
    data : RVData
        Observed radial velocities.
    parameterization : StandardRV or EcoswEsinwRV or None
        RV parameterization. Defaults to :class:`StandardRV`.
    extensions : tuple of AbstractExtension
        Model extensions (jitter, trends, offsets).
    linear_prior : dict, False, or None
        If ``None`` (default), uses wide Normal priors on ``rv_semiamp``
        and ``v_sys`` (sigma=1000 km/s). Pass ``False`` for explicit
        (non-marginalized) mode, or a custom dict.

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
    extensions: tuple[AbstractExtension, ...] = (),
    linear_prior: dict[str, Any] | Literal[False] | None = None,
) -> GaiaAstrometryModel:
    """Build a Gaia astrometry model with sensible defaults.

    Parameters
    ----------
    data : GaiaAstrometricData
        Gaia epoch astrometric data.
    extensions : tuple of AbstractExtension
        Model extensions.
    linear_prior : dict, False, or None
        If ``None`` (default), uses wide Normal priors on all 6 linear
        astrometric parameters (positions, proper motions, parallax,
        semi-major axis). Pass ``False`` for explicit mode, or a custom
        dict.

    Returns
    -------
    GaiaAstrometryModel

    Examples
    --------
    >>> from harv.models.factories import gaia_astrometry_model
    >>> # gaia_astrometry_model(data)  # requires GaiaAstrometricData
    """
    if linear_prior is None:
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
    linear_prior: dict[str, Any] | None,
    extensions: tuple[AbstractExtension, ...],
    parameterization: StandardRV | EcoswEsinwRV | None,
) -> RVModel | GaiaAstrometryModel:
    """Build a component model from data, prior components, and extensions.

    Parameters
    ----------
    data : RVData or GaiaAstrometryData
        Observed data. Dispatches to :func:`rv_model` or
        :func:`gaia_astrometry_model` based on type.
    linear_prior : dict or None
        Per-parameter priors for linear parameters, or ``None`` for explicit
        (non-marginalized) mode.
    extensions : tuple of AbstractExtension
        Extensions to include.
    parameterization : StandardRV, EcoswEsinwRV, or None
        RV parameterization. Only used when ``data`` is :class:`RVData`.
        Ignored for :class:`GaiaAstrometryData`.

    Returns
    -------
    RVModel or GaiaAstrometryModel

    Raises
    ------
    TypeError
        If ``data`` is not :class:`RVData` or :class:`GaiaAstrometryData`.
    """
    # When linear_prior is None it means explicit (non-marginalized) mode.
    # The model factories use False for that and None for "use wide defaults".
    lp_arg: dict[str, Any] | Literal[False] | None = (
        False if linear_prior is None else linear_prior
    )

    if isinstance(data, RVData):
        return rv_model(
            data,
            parameterization=parameterization,
            extensions=extensions,
            linear_prior=lp_arg,
        )
    if isinstance(data, GaiaAstrometryData):
        return gaia_astrometry_model(
            data,
            extensions=extensions,
            linear_prior=lp_arg,
        )
    msg = (
        f"_build_model() received unsupported data type {type(data).__name__}. "
        "Supported types: RVData, GaiaAstrometryData."
    )
    raise TypeError(msg)
