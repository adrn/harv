"""Convert sampled parameter values between parameterizations.

This first implementation supports single-component RV and Gaia astrometry
parameterizations only:

- RV: ``StandardRV <-> EcoswEsinwRV``
- Gaia astrometry: ``StandardGaiaAstrometry <-> ThieleInnesGaiaAstrometry``

Parameters not declared by the source parameterization (for example, extension
parameters such as jitter or polynomial-trend coefficients) are preserved
unchanged.  See ``docs/spec.md`` ("Parameterization conversion").
"""

__all__ = ("convert_parameterization",)

from unxt import Q

from harv.kepler.orbits import (
    campbell_from_thiele_innes,
    ecc_omega_from_ecosw_esinw,
    ecosw_esinw_from_ecc_omega,
    thiele_innes_from_campbell,
)
from harv.models.parameterizations._base import AbstractParameterization
from harv.models.parameterizations.gaia import (
    StandardGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.models.parameterizations.rv import EcoswEsinwRV, StandardRV

_RV = (StandardRV, EcoswEsinwRV)
_GAIA = (StandardGaiaAstrometry, ThieleInnesGaiaAstrometry)


def convert_parameterization(
    nonlinear: dict[str, Q],
    linear: dict[str, Q],
    *,
    source: AbstractParameterization,
    target: AbstractParameterization,
) -> tuple[dict[str, Q], dict[str, Q]]:
    """Convert parameter values between supported parameterizations.

    Both ``source`` and ``target`` must belong to the same family (both RV, or
    both Gaia astrometry).  Parameters not declared by ``source`` are preserved
    unchanged.

    Parameters
    ----------
    nonlinear
        Nonlinear parameter samples keyed by parameter name.
    linear
        Linear parameter samples keyed by parameter name.
    source
        Parameterization that the supplied values are currently in.
    target
        Parameterization to convert into.

    Returns
    -------
    tuple[dict[str, Q], dict[str, Q]]
        Converted ``(nonlinear, linear)`` parameter dictionaries.

    Raises
    ------
    NotImplementedError
        If the samples contain namespaced (joint-model) keys, or ``source`` and
        ``target`` are not a supported same-family pair.
    ValueError
        If a parameter declared by ``source`` is missing from the inputs.
    """
    namespaced = sorted(k for k in (*nonlinear, *linear) if "." in k)
    if namespaced:
        msg = (
            "Parameterization conversion supports single-component samples only; "
            f"namespaced (joint-model) parameters are not supported: {namespaced!r}"
        )
        raise NotImplementedError(msg)

    nl_names = tuple(p.name for p in source.nonlinear_params())
    lin_names = tuple(p.name for p in source.linear_params())
    missing = [n for n in nl_names if n not in nonlinear]
    missing += [n for n in lin_names if n not in linear]
    if missing:
        msg = f"Missing required source parameters for conversion: {missing!r}"
        raise ValueError(msg)

    # Anything not declared by the source parameterization (e.g. extension
    # parameters) is carried through to the converted output unchanged.
    extra_nl = {k: v for k, v in nonlinear.items() if k not in nl_names}
    extra_lin = {k: v for k, v in linear.items() if k not in lin_names}

    if type(source) is type(target):
        base_nl, base_lin = dict(nonlinear), dict(linear)
    elif isinstance(source, _RV) and isinstance(target, _RV):
        base_nl, base_lin = _convert_rv(nonlinear, linear, source, target)
    elif isinstance(source, _GAIA) and isinstance(target, _GAIA):
        base_nl, base_lin = _convert_gaia(nonlinear, linear, source, target)
    else:
        msg = (
            "Unsupported parameterization conversion: "
            f"{type(source).__name__} -> {type(target).__name__}. Source and "
            "target must both be RV parameterizations, or both Gaia astrometry "
            "parameterizations."
        )
        raise NotImplementedError(msg)

    return {**base_nl, **extra_nl}, {**base_lin, **extra_lin}


def _convert_rv(
    nonlinear: dict[str, Q],
    linear: dict[str, Q],
    source: AbstractParameterization,
    target: AbstractParameterization,
) -> tuple[dict[str, Q], dict[str, Q]]:
    """Convert between RV parameterizations via the canonical ``(e, omega)`` pair.

    ``period`` and ``phase_peri`` (nonlinear) and the linear parameters
    (``rv_semiamp``, ``v_sys``) are shared by both parameterizations.
    """
    # source -> canonical (eccentricity, arg_peri)
    if isinstance(source, StandardRV):
        eccentricity = nonlinear["eccentricity"]
        arg_peri = nonlinear["arg_peri"]
    else:
        eccentricity, arg_peri = ecc_omega_from_ecosw_esinw(
            nonlinear["ecosw"], nonlinear["esinw"]
        )

    # canonical -> target, built in the target's declared parameter order
    if isinstance(target, StandardRV):
        new_nonlinear = {
            "period": nonlinear["period"],
            "eccentricity": eccentricity,
            "phase_peri": nonlinear["phase_peri"],
            "arg_peri": arg_peri,
        }
    else:
        ecosw, esinw = ecosw_esinw_from_ecc_omega(eccentricity, arg_peri)
        new_nonlinear = {
            "period": nonlinear["period"],
            "ecosw": ecosw,
            "esinw": esinw,
            "phase_peri": nonlinear["phase_peri"],
        }

    new_linear = {"rv_semiamp": linear["rv_semiamp"], "v_sys": linear["v_sys"]}
    return new_nonlinear, new_linear


def _convert_gaia(
    nonlinear: dict[str, Q],
    linear: dict[str, Q],
    source: AbstractParameterization,
    target: AbstractParameterization,
) -> tuple[dict[str, Q], dict[str, Q]]:
    """Convert between Gaia parameterizations via the canonical Campbell elements.

    ``period``, ``eccentricity``, ``phase_peri`` (nonlinear) and the astrometric
    linear parameters (``ra0``, ``dec0``, ``pmra``, ``pmdec``, ``parallax``) are
    shared by both parameterizations.
    """
    shared_linear = {k: linear[k] for k in ("ra0", "dec0", "pmra", "pmdec", "parallax")}

    # source -> canonical Campbell elements
    if isinstance(source, StandardGaiaAstrometry):
        semi_major_axis = linear["semi_major_axis"]
        arg_peri = nonlinear["arg_peri"]
        lon_asc_node = nonlinear["lon_asc_node"]
        cos_i = nonlinear["cos_i"]
    else:
        campbell = campbell_from_thiele_innes(
            A=linear["ti_A"], B=linear["ti_B"], F=linear["ti_F"], G=linear["ti_G"]
        )
        semi_major_axis = campbell["semi_major_axis"]
        arg_peri = campbell["arg_peri"]
        lon_asc_node = campbell["lon_asc_node"]
        cos_i = campbell["cos_i"]

    # canonical -> target, built in the target's declared parameter order
    if isinstance(target, StandardGaiaAstrometry):
        new_nonlinear = {
            "period": nonlinear["period"],
            "eccentricity": nonlinear["eccentricity"],
            "phase_peri": nonlinear["phase_peri"],
            "arg_peri": arg_peri,
            "lon_asc_node": lon_asc_node,
            "cos_i": cos_i,
        }
        new_linear = {**shared_linear, "semi_major_axis": semi_major_axis}
    else:
        ti_A, ti_B, ti_F, ti_G = thiele_innes_from_campbell(
            semi_major_axis, arg_peri, lon_asc_node, cos_i
        )
        new_nonlinear = {
            "period": nonlinear["period"],
            "eccentricity": nonlinear["eccentricity"],
            "phase_peri": nonlinear["phase_peri"],
        }
        new_linear = {
            **shared_linear,
            "ti_A": ti_A,
            "ti_B": ti_B,
            "ti_F": ti_F,
            "ti_G": ti_G,
        }

    return new_nonlinear, new_linear
