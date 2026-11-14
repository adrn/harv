"""Convert sampled parameter values between parameterizations.

This first implementation supports single-component RV and Gaia astrometry
parameterizations only:

- RV: ``StandardRV <-> SqrtEccRV``
- Gaia astrometry: ``StandardGaiaAstrometry <-> ThieleInnesGaiaAstrometry``

Parameters not declared by the source parameterization (for example, extension
parameters such as jitter or polynomial-trend coefficients) are preserved
unchanged.  See ``docs/spec.md`` ("Parameterization conversion").
"""

__all__ = ("convert_parameterization",)

from typing import cast

import quaxed.numpy as jnp
from unxt import Q

from harv.kepler.orbits import (
    campbell_from_thiele_innes,
    ecc_omega_from_sqrte_cosw_sinw,
    sqrte_cosw_sinw_from_ecc_omega,
    thiele_innes_from_campbell,
)
from harv.models.parameterizations.base import AbstractParameterization
from harv.models.parameterizations.gaia import (
    StandardGaiaAstrometry,
    ThieleInnesGaiaAstrometry,
)
from harv.models.parameterizations.rv import SqrtEccRV, StandardRV

_RV = (StandardRV, SqrtEccRV)
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

    nonlinear_names = tuple(p.name for p in source.nonlinear_params())
    linear_names = tuple(p.name for p in source.linear_params())
    missing = [n for n in nonlinear_names if n not in nonlinear]
    missing += [n for n in linear_names if n not in linear]
    if missing:
        msg = f"Missing required source parameters for conversion: {missing!r}"
        raise ValueError(msg)

    # Anything not declared by the source parameterization (e.g. extension
    # parameters) is carried through to the converted output unchanged.
    extra_nl = {k: v for k, v in nonlinear.items() if k not in nonlinear_names}
    extra_lin = {k: v for k, v in linear.items() if k not in linear_names}

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
    """Convert between RV parameterizations via canonical ``(e, omega, phase_peri)``.

    ``period`` and the linear parameters (``rv_semiamp``, ``v_sys``) are shared by
    both parameterizations. ``SqrtEccRV`` carries the mean longitude
    ``omega - 2*pi*phase_peri`` instead of ``phase_peri``. Angles come out wrapped
    into ``[0, 2*pi)`` and ``phase_peri`` into ``[0, 1)``, matching the default priors.
    """
    two_pi = Q(2 * jnp.pi, "rad")
    # source -> canonical (eccentricity, arg_peri, phase_peri)
    if isinstance(source, StandardRV):
        eccentricity = nonlinear["eccentricity"]
        arg_peri = nonlinear["arg_peri"]
        phase_peri = nonlinear["phase_peri"]
    else:
        eccentricity, arg_peri = ecc_omega_from_sqrte_cosw_sinw(
            nonlinear["sqrte_cosw"], nonlinear["sqrte_sinw"]
        )
        arg_peri = cast("Q", jnp.mod(arg_peri, two_pi))
        phase_peri = Q(jnp.mod(SqrtEccRV().phase_peri(nonlinear), 1.0), "")

    # canonical -> target, built in the target's declared parameter order
    if isinstance(target, StandardRV):
        new_nonlinear = {
            "period": nonlinear["period"],
            "eccentricity": eccentricity,
            "phase_peri": phase_peri,
            "arg_peri": arg_peri,
        }
    else:
        sqrte_cosw, sqrte_sinw = sqrte_cosw_sinw_from_ecc_omega(eccentricity, arg_peri)
        mean_longitude = arg_peri - two_pi * phase_peri
        new_nonlinear = {
            "period": nonlinear["period"],
            "sqrte_cosw": sqrte_cosw,
            "sqrte_sinw": sqrte_sinw,
            "mean_longitude": cast("Q", jnp.mod(mean_longitude, two_pi)),
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
