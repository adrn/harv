"""Tests for support-constrained and mixture linear marginalization.

Every numeric claim here is checked against an *independent* oracle -- scipy
quadrature, ``scipy.stats`` CDFs, or rejection sampling -- never against another
part of :mod:`harv.stats.marginalized`.
"""

import itertools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy import integrate
from scipy.stats import multivariate_normal as smvn
from scipy.stats import norm, truncnorm

from harv.stats import MarginalizedLinear
from harv.stats.marginalized import (
    GeneralizedMarginalizedLinear,
    ResolvedLinearPrior,
    _ln_bvn_rect,
    _ln_ndtr_interval,
    _ln_post_normalizer,
    build_marginalized,
)

INF = np.inf

# harv runs in JAX's default float32 unless the user enables x64, so that is the
# precision these tolerances are set from (measured, not guessed: agreement with
# float64 quadrature is ~3.4e-6 absolute on log-probs of order 10).
LN_PROB_TOL = 1e-5

# ``_ln_bvn_rect`` is trustworthy while the alternating corner sum stays within
# the working precision of its largest term. Beyond that it returns -inf by
# design; this is the region where it is expected to be *accurate*.
LN_RECT_ACCURATE_ABOVE = -8.0


def _prior(low, high, *, loc=None, scale=None, ln_weights=None, k=2):
    """Build a :class:`ResolvedLinearPrior`, inferring the static bound flags."""
    low, high = np.asarray(low, float), np.asarray(high, float)
    n_comp = 1 if ln_weights is None else len(ln_weights)
    loc = np.zeros((n_comp, k)) if loc is None else np.atleast_2d(loc)
    scale = np.full((n_comp, k), 3.0) if scale is None else np.atleast_2d(scale)
    ln_weights = np.zeros(1) if ln_weights is None else np.asarray(ln_weights, float)
    return ResolvedLinearPrior(
        jnp.asarray(loc),
        jnp.asarray(scale),
        jnp.asarray(ln_weights),
        jnp.asarray(low),
        jnp.asarray(high),
        tuple(bool(v) for v in np.isfinite(low)),
        tuple(bool(v) for v in np.isfinite(high)),
        tuple(f"b{i}" for i in range(k)),
    )


@pytest.fixture
def problem():
    """A small (n_obs=6, k=2) linear problem with heteroscedastic noise."""
    rng = np.random.default_rng(3)
    n, k = 6, 2
    design = rng.normal(size=(n, k))
    err = 0.3 + 0.2 * rng.random(n)
    obs = design @ np.array([1.2, -0.8]) + err * rng.normal(size=n)
    return design, err, obs, dist.Normal(0.0, jnp.asarray(err))


# ---------------------------------------------------------------------------
# Numerics, against scipy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-INF, INF), (-INF, 0.5), (-1.0, INF), (-1.0, 2.0), (2.0, 6.0), (-8.0, -6.0)],
)
def test_ln_ndtr_interval_matches_scipy(lo, hi):
    """The univariate normalizer is exact, including at infinite bounds."""
    got = float(_ln_ndtr_interval(jnp.asarray(lo), jnp.asarray(hi)))
    ref = (
        np.log(norm.cdf(hi) - norm.cdf(lo))
        if hi < 5
        else np.log(norm.sf(lo) - norm.sf(hi))
    )
    assert got == pytest.approx(ref, abs=1e-4)


def test_ln_ndtr_interval_zero_width_is_neg_inf():
    """A degenerate interval holds no mass."""
    assert _ln_ndtr_interval(jnp.asarray(1.0), jnp.asarray(1.0)) == -INF


@pytest.mark.parametrize("rho", [-0.999, -0.95, -0.5, 0.0, 0.5, 0.95, 0.999])
def test_ln_bvn_rect_matches_scipy(rho):
    """The bivariate rectangle probability, against scipy, in nats.

    Two separate claims, because they matter for different reasons:

    1. *Accuracy* where the value carries weight (``ln P`` above
       ``LN_RECT_ACCURATE_ABOVE``).
    2. *Safety* everywhere else -- a finite answer is never badly wrong. Below
       the working precision the alternating corner sum cancels, and without the
       dtype-aware guard in ``_ln_bvn_rect`` it returns finite values tens of
       nats too high, which would over-weight a draw the truncation forbids.
       Collapsing to -inf instead is the conservative direction, so this asserts
       the guard is doing its job rather than asserting a magic floor value.
       Note the guard must hold back a margin, because it can only test the
       *computed* total -- the very quantity the cancellation corrupts.
    """
    ref_dist = smvn(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    worst_accurate = 0.0
    worst_finite = 0.0
    for a, b in itertools.product([-3.0, -1.0, 0.0, 1.0, 3.0], repeat=2):
        ref = ref_dist.cdf([a, b])
        if ref <= 0:
            continue
        ln_ref = np.log(ref)
        got = float(
            _ln_bvn_rect(
                jnp.asarray([-INF, -INF]),
                jnp.asarray([a, b]),
                jnp.asarray(rho),
                lo_finite=(False, False),
                hi_finite=(True, True),
            )
        )
        if np.isfinite(got):
            worst_finite = max(worst_finite, abs(got - ln_ref))
        if ln_ref > LN_RECT_ACCURATE_ABOVE:
            assert np.isfinite(got)
            worst_accurate = max(worst_accurate, abs(got - ln_ref))
    assert worst_accurate < 1e-3
    # Pins the guard: measured worst finite error across the sweep is 0.004 nats.
    assert worst_finite < 0.05


def test_ln_bvn_rect_upper_orthant_matches_scipy():
    """The orthant form the positivity use case actually hits."""
    for rho in (-0.7, 0.0, 0.7):
        ref = smvn(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(
            [1e10, 1e10], lower_limit=[0.3, -0.4]
        )
        got = float(
            _ln_bvn_rect(
                jnp.asarray([0.3, -0.4]),
                jnp.asarray([INF, INF]),
                jnp.asarray(rho),
                lo_finite=(True, True),
                hi_finite=(False, False),
            )
        )
        assert got == pytest.approx(np.log(ref), abs=1e-4)


# ---------------------------------------------------------------------------
# log_prob against brute-force integration
# ---------------------------------------------------------------------------


def _brute_ln_prob(design, err, obs, prior):
    """Numerically integrate the marginal likelihood over the linear params."""
    loc = np.asarray(prior.loc)
    scale = np.asarray(prior.scale)
    weights = np.exp(np.asarray(prior.ln_weights))
    low, high = np.asarray(prior.low), np.asarray(prior.high)
    k = loc.shape[-1]

    z_norm = sum(
        weights[c]
        * np.prod(
            [
                norm.cdf((high[i] - loc[c, i]) / scale[c, i])
                - norm.cdf((low[i] - loc[c, i]) / scale[c, i])
                for i in range(k)
            ]
        )
        for c in range(len(weights))
    )

    def integrand(b1, b0):
        beta = np.array([b0, b1])
        if not (np.all(beta >= low) and np.all(beta <= high)):
            return 0.0
        prior_pdf = (
            sum(
                weights[c] * np.prod(norm.pdf(beta, loc[c], scale[c]))
                for c in range(len(weights))
            )
            / z_norm
        )
        resid = obs - design @ beta
        like = np.exp(-0.5 * np.sum((resid / err) ** 2)) / np.prod(
            err * np.sqrt(2 * np.pi)
        )
        return like * prior_pdf

    span = [
        (
            max(low[i], loc[:, i].min() - 12 * scale[:, i].max()),
            min(high[i], loc[:, i].max() + 12 * scale[:, i].max()),
        )
        for i in range(k)
    ]
    # ``dblquad`` reports slow convergence on the truncated integrands (the
    # indicator makes them non-smooth). Suppressed locally rather than added to
    # the project-wide ``filterwarnings``: it is chatter from this file's own
    # oracle, and silencing it globally would hide it everywhere else too.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        val, _ = integrate.dblquad(
            integrand, *span[0], *span[1], epsabs=1e-13, epsrel=1e-11
        )
    return np.log(val)


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("untruncated", [-INF, -INF], [INF, INF]),
        ("one positive", [0.0, -INF], [INF, INF]),
        ("one negative", [-INF, -INF], [INF, 0.0]),
        ("one interval", [-1.0, -INF], [2.0, INF]),
        ("two: quadrant (+,-)", [0.0, -INF], [INF, 0.0]),
        ("two: both intervals", [-0.5, -2.0], [3.0, 1.0]),
    ],
)
def test_log_prob_matches_quadrature(problem, name, low, high):
    """The truncation correction reproduces a direct numerical integral."""
    design, err, obs, data_dist = problem
    prior = _prior(low, high)
    got = float(
        build_marginalized(jnp.asarray(design), prior, data_dist).log_prob(
            jnp.asarray(obs)
        )
    )
    assert got == pytest.approx(
        _brute_ln_prob(design, err, obs, prior), abs=LN_PROB_TOL
    )


@pytest.mark.parametrize(
    ("low", "high"),
    [([-INF, -INF], [INF, INF]), ([0.0, -INF], [INF, INF]), ([0.0, -INF], [INF, 0.0])],
)
def test_mixture_log_prob_matches_quadrature(problem, low, high):
    """Mixture priors, with and without a shared truncation.

    The truncated cases are what catch getting the mixture normalizer wrong:
    ``Z_mix = sum_c w_c Z_prior_c`` must be subtracted once, not per component.
    """
    design, err, obs, data_dist = problem
    prior = _prior(
        low,
        high,
        loc=[[-2.0, 1.0], [1.5, -1.0]],
        scale=[[1.0, 2.0], [3.0, 1.0]],
        ln_weights=np.log([0.3, 0.7]),
    )
    got = float(
        build_marginalized(jnp.asarray(design), prior, data_dist).log_prob(
            jnp.asarray(obs)
        )
    )
    assert got == pytest.approx(
        _brute_ln_prob(design, err, obs, prior), abs=LN_PROB_TOL
    )


def test_mixture_log_prob_equals_explicit_logsumexp(problem):
    """Without truncation a mixture is a weighted logsumexp of its components."""
    design, _, obs, data_dist = problem
    ln_w = np.log([0.3, 0.7])
    loc = np.array([[-2.0, 1.0], [1.5, -1.0]])
    scale = np.array([[1.0, 2.0], [3.0, 1.0]])
    prior = _prior([-INF, -INF], [INF, INF], loc=loc, scale=scale, ln_weights=ln_w)
    got = float(
        build_marginalized(jnp.asarray(design), prior, data_dist).log_prob(
            jnp.asarray(obs)
        )
    )
    per_component = [
        float(
            MarginalizedLinear(
                design_matrix=jnp.asarray(design),
                prior_distribution=dist.MultivariateNormal(
                    loc=jnp.asarray(loc[c]), scale_tril=jnp.diag(jnp.asarray(scale[c]))
                ),
                data_distribution=data_dist,
            ).log_prob(jnp.asarray(obs))
        )
        for c in range(2)
    ]
    expected = float(jax.scipy.special.logsumexp(jnp.asarray(ln_w + per_component)))
    assert got == pytest.approx(expected, abs=LN_PROB_TOL)


def test_three_constrained_params_raises(problem):
    """Three constrained params need an orthant probability we do not compute."""
    _, _, _, data_dist = problem
    rng = np.random.default_rng(0)
    wide = rng.normal(size=(6, 3))
    prior = _prior([0.0, 0.0, 0.0], [INF, INF, INF], k=3)
    with pytest.raises(NotImplementedError, match="orthant probability"):
        build_marginalized(jnp.asarray(wide), prior, data_dist).log_prob(jnp.zeros(6))


def test_error_names_the_offending_params(problem):
    """The message must say *which* params to unconstrain, not just that it failed."""
    prior = _prior([0.0, 0.0, 0.0], [INF, INF, INF], k=3)
    with pytest.raises(NotImplementedError, match=r"b0.*b1.*b2"):
        _ln_post_normalizer(jnp.zeros(3), jnp.eye(3), prior)


# ---------------------------------------------------------------------------
# Fast path is untouched
# ---------------------------------------------------------------------------


def test_untruncated_single_gaussian_takes_the_bare_fast_path(problem):
    """An ordinary Gaussian prior must not pay for this feature.

    Asserts both that no wrapper is constructed and that ``log_prob`` is
    *bitwise* identical to the pre-existing construction.
    """
    design, _, obs, data_dist = problem
    prior = _prior([-INF, -INF], [INF, INF])
    built = build_marginalized(jnp.asarray(design), prior, data_dist)
    assert type(built) is MarginalizedLinear

    direct = MarginalizedLinear(
        design_matrix=jnp.asarray(design),
        prior_distribution=dist.MultivariateNormal(
            loc=jnp.zeros(2), scale_tril=jnp.diag(jnp.full((2,), 3.0))
        ),
        data_distribution=data_dist,
    )
    assert float(built.log_prob(jnp.asarray(obs))) == float(
        direct.log_prob(jnp.asarray(obs))
    )


def test_truncated_prior_takes_the_wrapper(problem):
    design, _, _, data_dist = problem
    built = build_marginalized(
        jnp.asarray(design), _prior([0.0, -INF], [INF, INF]), data_dist
    )
    assert isinstance(built, GeneralizedMarginalizedLinear)


# ---------------------------------------------------------------------------
# Truncated conditional sampling
# ---------------------------------------------------------------------------


@pytest.fixture
def sampling_problem():
    """An (n_obs=8, k=3) problem: two params constrainable, one always free."""
    rng = np.random.default_rng(11)
    n, k = 8, 3
    design = rng.normal(size=(n, k))
    err = 0.4 + 0.2 * rng.random(n)
    obs = design @ np.array([1.5, -1.0, 0.4]) + err * rng.normal(size=n)
    return design, obs, dist.Normal(0.0, jnp.asarray(err))


# Bounds are chosen so the constraint is *active* -- it has to cut real mass for
# the test to mean anything -- but within a couple of sigma of the conditional
# mean, which is the regime float32 resolves. The conditional mean of this
# fixture is about [1.78, -0.96, 0.23] with sigmas near 0.24. The deep-tail
# regime gets its own test below, asserting the guarantee that still holds
# there.
SAMPLING_CASES = {
    "one active lower bound": ([1.7, -INF, -INF], [INF, INF, INF]),
    "one active interval": ([-INF, -1.0, -INF], [INF, -0.9, INF]),
    "two: lower and upper": ([1.7, -INF, -INF], [INF, -0.9, INF]),
    "two: both lower": ([1.7, -1.0, -INF], [INF, INF, INF]),
}


@pytest.mark.parametrize("case", list(SAMPLING_CASES))
def test_conditional_draws_respect_the_support(sampling_problem, case):
    """Not one draw may fall outside the box -- that is the whole point."""
    design, obs, data_dist = sampling_problem
    low, high = SAMPLING_CASES[case]
    prior = _prior(low, high, k=3)
    cond = build_marginalized(jnp.asarray(design), prior, data_dist).conditional(
        jnp.asarray(obs)
    )
    draws = np.asarray(
        jax.vmap(cond.sample)(jax.random.split(jax.random.key(0), 20_000))
    )
    assert np.all(draws >= np.asarray(low, float) - 1e-6)
    assert np.all(draws <= np.asarray(high, float) + 1e-6)


@pytest.mark.parametrize("case", list(SAMPLING_CASES))
def test_conditional_draws_reproduce_the_analytic_truncated_mean(
    sampling_problem, case
):
    """Ties the sampler to ``.mean``, which is derived independently.

    ``.mean`` comes from differentiating ``ln Z_post``; the draws come from
    inverse-CDF sampling. They share no code, so agreement checks both.
    """
    design, obs, data_dist = sampling_problem
    low, high = SAMPLING_CASES[case]
    prior = _prior(low, high, k=3)
    cond = build_marginalized(jnp.asarray(design), prior, data_dist).conditional(
        jnp.asarray(obs)
    )
    n_draws = 40_000
    draws = np.asarray(
        jax.vmap(cond.sample)(jax.random.split(jax.random.key(0), n_draws))
    )
    mc_err = draws.std(axis=0) / np.sqrt(n_draws)
    assert np.all(np.abs(draws.mean(axis=0) - np.asarray(cond.mean)) < 5 * mc_err)


def test_deep_tail_truncation_still_returns_finite_in_box_draws(sampling_problem):
    """The guarantee that survives even where float32 cannot resolve the tail.

    ``b1 >= 0`` sits about 4 sigma above this fixture's conditional mean, so the
    inverse CDF saturates and draws pile up near the bound (documented on
    ``_trunc_normal_draw``). Distributional accuracy is gone, but the two
    properties callers actually depend on must hold: every draw is inside the
    box, and nothing is NaN -- a single NaN would propagate into the returned
    posterior samples for the whole batch.
    """
    design, obs, data_dist = sampling_problem
    prior = _prior([0.0, 0.0, -INF], [INF, INF, INF], k=3)
    cond = build_marginalized(jnp.asarray(design), prior, data_dist).conditional(
        jnp.asarray(obs)
    )
    draws = np.asarray(
        jax.vmap(cond.sample)(jax.random.split(jax.random.key(0), 20_000))
    )
    assert not np.any(np.isnan(draws))
    assert np.all(draws[:, :2] >= -1e-6)


def test_exact_sampler_beats_rejection_in_the_forbidden_corner(sampling_problem):
    """The case that rules out a rejection- or Gibbs-based sampler.

    In the ``(+,+)`` quadrant the untruncated conditional puts so little mass
    inside the box that plain rejection sampling is hopeless -- asserted here so
    the reason for the inverse-CDF machinery is recorded rather than assumed.
    The exact sampler still returns a full set of in-box draws.
    """
    design, obs, data_dist = sampling_problem
    prior = _prior([0.0, 0.0, -INF], [INF, INF, INF], k=3)
    cond = build_marginalized(jnp.asarray(design), prior, data_dist).conditional(
        jnp.asarray(obs)
    )
    rng = np.random.default_rng(5)
    naive = rng.multivariate_normal(
        np.asarray(cond.mean_untruncated)[0], np.asarray(cond.cov)[0], size=200_000
    )
    kept = naive[(naive[:, 0] >= 0) & (naive[:, 1] >= 0)]
    assert len(kept) < 200  # < 0.1% acceptance: rejection sampling is unusable

    draws = np.asarray(
        jax.vmap(cond.sample)(jax.random.split(jax.random.key(3), 5_000))
    )
    assert np.all(draws[:, 0] >= -1e-6)
    assert np.all(draws[:, 1] >= -1e-6)


def test_one_constrained_param_matches_closed_form_truncated_moments(
    sampling_problem,
):
    """For a single constrained param the marginal is a plain truncated normal."""
    design, obs, data_dist = sampling_problem
    prior = _prior([0.0, -INF, -INF], [INF, INF, INF], k=3)
    cond = build_marginalized(jnp.asarray(design), prior, data_dist).conditional(
        jnp.asarray(obs)
    )
    mean0 = float(np.asarray(cond.mean_untruncated)[0, 0])
    sd0 = float(np.sqrt(np.asarray(cond.cov)[0, 0, 0]))
    expected = truncnorm.mean(a=(0.0 - mean0) / sd0, b=np.inf, loc=mean0, scale=sd0)
    assert float(np.asarray(cond.mean)[0]) == pytest.approx(expected, rel=1e-4)


def test_untruncated_mean_is_the_ordinary_conditional_mean(sampling_problem):
    """With no truncation the autodiff correction must vanish exactly."""
    design, obs, data_dist = sampling_problem
    prior = _prior([-INF] * 3, [INF] * 3, k=3, ln_weights=np.log([0.5, 0.5]))
    built = build_marginalized(jnp.asarray(design), prior, data_dist)
    cond = built.conditional(jnp.asarray(obs))
    mixed = np.asarray(
        jnp.einsum(
            "c,ci->i",
            jnp.exp(cond.ln_component_weights),
            cond.mean_untruncated,
        )
    )
    assert np.allclose(np.asarray(cond.mean), mixed, atol=1e-5)


# ---------------------------------------------------------------------------
# JAX transformation contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high"),
    [
        ([-INF, -INF], [INF, INF]),
        ([0.0, -INF], [INF, INF]),
        ([0.0, -INF], [INF, 0.0]),
    ],
)
def test_log_prob_is_jit_vmap_and_grad_safe(problem, low, high):
    """Required of anything on the sampler's hot path (spec Agent Checklist)."""
    design, _, obs, data_dist = problem
    static = _prior(low, high)

    def ln_prob(scale, obs_):
        prior = ResolvedLinearPrior(
            static.loc,
            scale,
            static.ln_weights,
            static.low,
            static.high,
            static.low_finite,
            static.high_finite,
            static.names,
        )
        return build_marginalized(jnp.asarray(design), prior, data_dist).log_prob(obs_)

    scale = jnp.full((1, 2), 3.0)
    assert np.isfinite(float(jax.jit(ln_prob)(scale, jnp.asarray(obs))))

    batched = jax.vmap(ln_prob, in_axes=(0, None))(
        jnp.stack([scale, 2.0 * scale, 5.0 * scale]), jnp.asarray(obs)
    )
    assert batched.shape == (3,)
    assert np.all(np.isfinite(np.asarray(batched)))

    grad = jax.grad(lambda s: ln_prob(s, jnp.asarray(obs)))(scale)
    assert np.all(np.isfinite(np.asarray(grad)))
    # A truncated prior's log-prob must actually respond to the prior width.
    assert np.any(np.abs(np.asarray(grad)) > 1e-8)
