"""Trig-free solver for Kepler's equation, returning the true anomaly.

A JAX port of the eccentric-anomaly solver in orvara (Brandt et al. 2021, AJ 162,
186, Section 3.1; ``orvara/orbit.pyx`` ``calc_EA_RPP`` and ``orvara/calcEA.h``),
which in turn follows Raposo-Pulido & Pelaez (2017, MNRAS 467, 1702):

1. reduce the mean anomaly ``M`` to ``[0, pi]``;
2. start from a piecewise quintic in ``M`` whose 12 pieces match ``E``, ``E'``
   and ``E''`` at ``E = k pi / 12`` (or, in the singular corner
   ``2M + 1 - e <= 0.2``, from RPP's series in ``sqrt(1 - e)``);
3. get ``sin E`` and ``cos E`` from a 15th-order Taylor series plus a square
   root, never a trig call;
4. take one Householder (third-order) step and update ``sin E`` / ``cos E`` by
   the angle-addition identities, to third order in the step.

orvara takes Halley's (second-order) step below ``e = 0.78``; here every element
takes the Householder step, since a vectorized ``where`` evaluates both branches
anyway. Against a converged Newton solve, ``E - e sin E - M`` stays at the
float64 round-off of the range reduction (~1e-14) for every ``e <= 0.9999``.

The only transcendental is the ``cbrt`` of the corner starter; everything else is
arithmetic and ``sqrt``. That matters beyond the call count. As observed with
jax/jaxlib 0.8.1 on CPU, XLA splits a long elementwise chain into several loop
fusions and recomputes the cheap-looking producers in each one rather than
reading them back (``jax.lax.optimization_barrier`` did not prevent it). With
jaxoplanet's ``sin``/``cos``/``tan`` solver that multiplied the transcendental
count inside the Gaia likelihood several-fold; here a recomputed producer is a
few multiplies, and the in-model cost matches the standalone cost.

The quintic piece is picked with a chain of ``where`` selects, not by indexing
the coefficient table: XLA lowers ``coeffs[:, piece]`` to a gather that it
materializes as a separate ``(..., 6)`` array, which measured slower. Both claims
are tracked by ``benchmarks/test_gaia_likelihood.py`` (results in
``benchmarks/results/likelihood/``).

orvara is distributed under the following license, which applies to the parts of
this module derived from it:

    BSD 3-Clause License

    Copyright (c) 2021, Timothy Brandt, Trent Dupuy, Yiting Li, G. Mirek Brandt,
    Yunlin Zeng, Daniel Michalik, and Virginia Raposo-Pulido.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = ("kepler",)

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

# Nodes E_k = k pi / 12 of the piecewise quintic, and their sines and cosines.
_E_NODES = np.arange(13) * np.pi / 12
_SIN_NODES = np.sin(_E_NODES)
_COS_NODES = np.cos(_E_NODES)
# 1/3!, 1/5!, ..., 1/15!: the odd Taylor coefficients of sin, innermost last.
_SIN_COEFFS = tuple(1 / math.factorial(n) for n in range(3, 16, 2))


def _quintic_table(ecc: Array) -> tuple[Array, Array]:
    """Interval bounds in ``M`` and quintic coefficients for one eccentricity.

    Piece ``k`` covers ``M`` in ``[bounds[k], bounds[k+1]]`` and evaluates
    ``E = sum_j coeffs[j, k] (M - bounds[k])^j``. The value and first two
    derivatives of ``E(M)`` are pinned at both ends of every piece (orvara's
    ``getbounds``).
    """
    bounds = _E_NODES - ecc * _SIN_NODES
    d1 = 1 / (1 - ecc * _COS_NODES)  # dE/dM
    d2 = -0.5 * ecc * _SIN_NODES * d1**3  # (d^2E/dM^2) / 2
    a0, a1, a2 = _E_NODES[:12], d1[:12], d2[:12]
    idx = 1 / (bounds[1:] - bounds[:-1])
    b0 = idx * (-a2 - idx * (a1 - idx * np.pi / 12))
    b1 = idx * (-2 * a2 - idx * (a1 - d1[1:]))
    b2 = idx * (d2[1:] - a2)
    a3 = b2 - 4 * b1 + 10 * b0
    a4 = (-2 * b2 + 7 * b1 - 15 * b0) * idx
    a5 = (b2 - 3 * b1 + 6 * b0) * idx * idx
    return bounds, jnp.stack([jnp.broadcast_to(a0, a3.shape), a1, a2, a3, a4, a5])


def _corner_start(M: Array, ecc: Array) -> Array:
    """RPP (2017) Eq. 19: series start for ``E`` when ``e -> 1`` and ``M -> 0``."""
    ome = 1 - ecc
    sqrt_ome = jnp.sqrt(ome)
    chi = M / (sqrt_ome * ome)
    lam = jnp.sqrt(8 + 9 * chi * chi)
    s = jnp.cbrt(lam + 3 * chi)
    s2 = s * s
    sigma = 6 * chi / (2 + s2 + 4 / s2)
    sig2 = sigma * sigma
    den = sig2 + 2
    poly = sig2 * (sig2 * (sig2 + 25) + 340) + 840
    series = (sig2 + 20) / (60 * den) + sig2 * ome * poly / (1400 * den**3)
    return sigma * (1 + sig2 * ome * series) * sqrt_ome


def _short_sin(x: Array) -> Array:
    """15th-order Taylor series for ``sin x``; float64-accurate for ``|x| <= pi/4``."""
    x2 = x * x
    poly = _SIN_COEFFS[-1]
    for c in _SIN_COEFFS[-2::-1]:
        poly = c - x2 * poly
    return x * (1 - x2 * poly)


def _sin_cos_ecc_anomaly(M: Array, ecc: Array) -> tuple[Array, Array]:
    """``(sin E, cos E)`` for mean anomaly ``M`` and scalar eccentricity ``ecc``."""
    pi = np.pi
    M = jnp.remainder(M, 2 * pi)
    lower_half = pi < M
    M = jnp.where(lower_half, 2 * pi - M, M)

    # Starting guess: the quintic piece containing M, picked by selects (see module
    # docstring for why not a gather).
    bounds, coeffs = _quintic_table(ecc)
    lo, a = bounds[0], list(coeffs[:, 0])
    for k in range(1, 12):
        above = bounds[k] < M
        lo = jnp.where(above, bounds[k], lo)
        a = [jnp.where(above, coeffs[j, k], a[j]) for j in range(6)]
    x = M - lo
    E = a[0] + x * (a[1] + x * (a[2] + x * (a[3] + x * (a[4] + x * a[5]))))
    # The corner starter runs for every element (a vectorized where evaluates both
    # sides; a lax.cond on ecc becomes the same select under vmap). ~1 ns/element.
    E = jnp.where(2 * M + (1 - ecc) > 0.2, E, _corner_start(M, ecc))

    # sin E and cos E from the series on an argument in [0, pi/4], plus a sqrt.
    near_0, near_pi = pi / 4 >= E, 3 * pi / 4 <= E
    s = _short_sin(jnp.where(near_0, E, jnp.where(near_pi, pi - E, pi / 2 - E)))
    c = jnp.sqrt(jnp.maximum(1 - s * s, 0.0))
    sin_E = jnp.where(near_0 | near_pi, s, c)
    cos_E = jnp.where(near_0, c, jnp.where(near_pi, -c, s))

    # One Householder step on f(E) = E - e sin E - M, multiplied through by e so
    # e = 0 needs no special case (orvara divides by e instead).
    num = M - E + ecc * sin_E
    den = 1 - ecc * cos_E
    e_sin, e_cos = ecc * sin_E, ecc * cos_E
    dE = num * (den * den + 0.5 * num * e_sin)
    dE = dE / (den * den * den + num * (den * e_sin + num * e_cos / 6))

    # Rotate (sin E, cos E) by dE, to third order in dE.
    q = dE * dE / 6
    cos_dE, sin_dE = 1 - 3 * q, dE * (1 - q)
    sin_E, cos_E = sin_E * cos_dE + cos_E * sin_dE, cos_E * cos_dE - sin_E * sin_dE
    return jnp.where(lower_half, -sin_E, sin_E), cos_E


# Jitted so eager callers (plotting, simulation) make one dispatch instead of ~100.
@jax.jit
def kepler(M: ArrayLike, ecc: ArrayLike) -> tuple[Array, Array]:
    """Solve Kepler's equation and return ``(sin f, cos f)`` of the true anomaly.

    Parameters
    ----------
    M
        Mean anomaly in radians, any shape and any range.
    ecc
        Eccentricity, a scalar in ``[0, 1)``. Use ``jax.vmap`` to batch over
        eccentricities: the quintic table is built once per eccentricity, which is
        what makes the per-epoch cost a handful of multiplies. ``ecc >= 1`` returns
        NaN (not an error), which the samplers already treat as a rejected draw.

    Returns
    -------
    tuple[Array, Array]
        ``(sin f, cos f)``, each with the shape of ``M``.

    Raises
    ------
    ValueError
        If ``ecc`` is not a scalar.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from harv.kepler._kepler_solver import kepler
    >>> sin_f, cos_f = kepler(jnp.array([0.0, jnp.pi]), 0.5)
    >>> print(jnp.round(cos_f, 12))
    [ 1. -1.]
    """
    # Cast outside the custom_jvp: an integer eccentricity would otherwise reach
    # the JVP rule with a float0 tangent.
    M = jnp.asarray(M, dtype=jnp.result_type(M, float))
    ecc = jnp.asarray(ecc, dtype=M.dtype)
    if ecc.ndim != 0:
        msg = f"kepler() takes a scalar eccentricity; got shape {ecc.shape}. "
        msg += "Use jax.vmap to batch over eccentricities."
        raise ValueError(msg)
    return _kepler(M, ecc)


@jax.custom_jvp
def _kepler(M: Array, ecc: Array) -> tuple[Array, Array]:
    sin_E, cos_E = _sin_cos_ecc_anomaly(M, ecc)
    den = 1 / (1 - ecc * cos_E)
    sin_f = jnp.sqrt((1 - ecc) * (1 + ecc)) * sin_E * den
    cos_f = (cos_E - ecc) * den
    return sin_f, cos_f


@_kepler.defjvp
def _kepler_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    # Differentiate the exact solution, not the iteration: df/dM and df/de follow
    # from f(E(M, e)) and Kepler's equation.
    M, ecc = primals
    M_dot, ecc_dot = tangents
    sin_f, cos_f = _kepler(M, ecc)
    e_cos_f = ecc * cos_f
    one_m_e2 = 1 - ecc * ecc
    f_dot = M_dot * (1 + e_cos_f) ** 2 / one_m_e2**1.5
    f_dot = f_dot + ecc_dot * (2 + e_cos_f) * sin_f / one_m_e2
    return (sin_f, cos_f), (cos_f * f_dot, -sin_f * f_dot)
