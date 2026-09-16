"""Signed semi-amplitudes for SB2 systems.

The motivating case for support-constrained linear marginalization. See
``docs/spec.md`` -> ``default_sb2_prior`` and :mod:`harv.stats.marginalized`.

Because ``rv_shape`` is exactly odd in ``arg_peri`` (pinned by
``tests/unit/kepler/test_orbit_math.py::TestRVShapeAntisymmetry``) and
``JointModel.for_sb2`` *shares* ``arg_peri`` and ``phase_peri``, the secondary's
antiphase motion is carried by a negative ``rv_semiamp``. Sign-free priors on
both semi-amplitudes therefore admit an exact two-fold degeneracy. These tests
pin the degeneracy, then pin that the signed default removes it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm
from unxt import Q

import harv.models.joint
from harv.data import RVData, SystemData
from harv.kepler.orbits import rv_at_times
from harv.models import JointModel, default_sb2_prior
from harv.samplers import RejectionSampler

TRUE = {
    "period": Q(137.0, "day"),
    "eccentricity": 0.25,
    "arg_peri": Q(1.1, "rad"),
    "time_peri": Q(11.0, "day"),
    "v_sys": Q(-8.0, "km/s"),
    "K_primary": Q(28.0, "km/s"),
    "K_secondary": Q(41.0, "km/s"),
}


def _sb2_data(
    seed: int = 7, n_obs: int = 24, span: float = 600.0, err: float = 0.7
) -> SystemData:
    """Synthetic SB2 built the *physical* way: the secondary sits at omega + pi.

    Constructing it this way rather than by negating K keeps the test honest --
    nothing here presumes the sign convention the model uses.
    """
    rng = np.random.default_rng(seed)
    times = Q(np.sort(rng.uniform(0.0, span, n_obs)), "day")
    err_1, err_2 = Q(err * 0.86, "km/s"), Q(err * 1.14, "km/s")

    def curve(semiamp: Q, arg_peri: Q) -> Q:
        return rv_at_times(
            times,
            TRUE["period"],
            TRUE["eccentricity"],
            TRUE["time_peri"],
            arg_peri,
            semiamp,
            TRUE["v_sys"],
        )

    rv_1 = curve(TRUE["K_primary"], TRUE["arg_peri"])
    rv_2 = curve(TRUE["K_secondary"], TRUE["arg_peri"] + Q(np.pi, "rad"))
    return SystemData(
        primary=RVData(
            time=times,
            rv=rv_1 + err_1 * rng.normal(size=n_obs),
            rv_err=Q(np.full(n_obs, err_1.value), "km/s"),
        ),
        secondary=RVData(
            time=times,
            rv=rv_2 + err_2 * rng.normal(size=n_obs),
            rv_err=Q(np.full(n_obs, err_2.value), "km/s"),
        ),
    )


def _phase_peri(data: SystemData) -> float:
    """``phase_peri`` for the injected orbit, measured from the data's own epoch.

    ``phase_peri`` is defined relative to ``time_ref`` (the mean observation
    time), not relative to zero -- see ``docs/spec.md`` -> "phase_peri vs
    time_peri". Getting this wrong shifts the orbital phase and can flip which
    sign branch the data prefer, so it is computed once here rather than inline.
    """
    period = TRUE["period"]
    offset = (TRUE["time_peri"] - data["primary"].time_ref) / period
    return float(offset.value % 1.0)


def _prior(*, signed: bool):
    return default_sb2_prior(
        period_min=Q(100.0, "day"),
        period_max=Q(200.0, "day"),
        sigma_K0=Q(40.0, "km/s"),
        sigma_v0=Q(30.0, "km/s"),
        signed_semiamp=signed,
    )


@pytest.fixture
def data():
    return _sb2_data()


@pytest.fixture
def weakly_constrained_data():
    """Four low-precision epochs inside a fraction of one period.

    Chosen so the positivity constraint is genuinely *active* and the two
    semi-amplitudes are genuinely *coupled* -- measured on this fixture,
    ``rho(K_1, K_2) = +0.41`` and ``ln Z_post = -0.77`` (the box holds about half
    the conditional mass). With well-sampled, precise data neither is true: the
    conditional posterior sits many sigma inside the allowed quadrant, so
    ``Z_post == 1`` and ``rho`` falls to ~0.001. See the exactness test for why
    that matters.
    """
    return _sb2_data(n_obs=4, span=15.0, err=40.0)


# ---------------------------------------------------------------------------
# The degeneracy being removed
# ---------------------------------------------------------------------------


def test_sign_flip_plus_pi_shift_is_an_exact_degeneracy(data):
    """``(K1, K2, omega)`` and ``(-K1, -K2, omega + pi)`` predict identical RVs.

    Documents *why* a sign constraint is needed at all. This holds before and
    after the change -- it is a property of the model, not of the prior.
    """
    joint = JointModel.for_sb2(prior=_prior(signed=True))
    nonlinear = {
        "period": TRUE["period"],
        "eccentricity": TRUE["eccentricity"],
        "phase_peri": _phase_peri(data),
    }
    linear = {
        "primary": {"rv_semiamp": jnp.asarray(28.0), "v_sys": jnp.asarray(-8.0)},
        "secondary": {"rv_semiamp": jnp.asarray(-41.0), "v_sys": jnp.asarray(-8.0)},
    }
    flipped_linear = {
        "primary": {"rv_semiamp": jnp.asarray(-28.0), "v_sys": jnp.asarray(-8.0)},
        "secondary": {"rv_semiamp": jnp.asarray(41.0), "v_sys": jnp.asarray(-8.0)},
    }
    for comp in ("primary", "secondary"):
        base = joint.components[comp].predict(
            {**nonlinear, "arg_peri": TRUE["arg_peri"]},
            linear[comp],
            data[comp],
        )
        flipped = joint.components[comp].predict(
            {**nonlinear, "arg_peri": TRUE["arg_peri"] + Q(np.pi, "rad")},
            flipped_linear[comp],
            data[comp],
        )
        assert jnp.allclose(base, flipped, atol=1e-4)


# ---------------------------------------------------------------------------
# The signed default
# ---------------------------------------------------------------------------


def test_signed_priors_are_the_default():
    """A behaviour change, so pin it explicitly in both directions."""
    signed = _prior(signed=True).linear_priors
    assert signed["primary.rv_semiamp"].support == "positive"
    assert signed["secondary.rv_semiamp"].support == "negative"

    unsigned = _prior(signed=False).linear_priors
    assert unsigned["primary.rv_semiamp"].support == "real"
    assert unsigned["secondary.rv_semiamp"].support == "real"


def test_signed_semiamps_are_still_marginalized(data):
    """The regression that matters most.

    A truncated prior must remain *analytically marginalized*; silently demoting
    it to explicit sampling would cost the rejection sampler two dimensions of
    acceptance volume, which is the opposite of the point.
    """
    prior = _prior(signed=True)
    joint = JointModel.for_sb2(prior=prior)
    sampler = RejectionSampler(prior, joint)

    text = sampler.summary()
    for line in text.splitlines():
        if "rv_semiamp" in line:
            assert "marginalized" in line, line
            assert "sampled" not in line, line

    samples = sampler.run(data, key=jax.random.key(0), n_prior_samples=2000, top_k=32)
    assert "primary.rv_semiamp" in samples.linear
    assert "primary.rv_semiamp" not in samples.nonlinear


def test_signed_priors_enforce_the_sign_of_every_draw(data):
    """Support constraints are worthless if any drawn value violates them."""
    prior = _prior(signed=True)
    samples = RejectionSampler(prior, JointModel.for_sb2(prior=prior)).run(
        data, key=jax.random.key(1), n_prior_samples=4000, top_k=64
    )
    assert np.all(np.asarray(samples["primary.rv_semiamp"].value) > 0.0)
    assert np.all(np.asarray(samples["secondary.rv_semiamp"].value) < 0.0)


def test_unsigned_priors_give_a_bimodal_posterior(data):
    """The degeneracy, demonstrated rather than asserted from theory.

    With sign-free priors the posterior populates *both* branches:
    ``(K_1 > 0, K_2 < 0, omega)`` and ``(K_1 < 0, K_2 > 0, omega + pi)``. This is
    what ``signed_semiamp=True`` removes.
    """
    prior = _prior(signed=False)
    samples = RejectionSampler(prior, JointModel.for_sb2(prior=prior)).run(
        data, key=jax.random.key(4), n_prior_samples=20_000, top_k=10
    )
    k_1 = np.asarray(samples["primary.rv_semiamp"].value)
    omega = np.asarray(samples["arg_peri"].value)
    assert np.any(k_1 > 0), "expected the K_1 > 0 branch"
    assert np.any(k_1 < 0), "expected the mirror K_1 < 0 branch"
    # The two branches are separated by pi in arg_peri.
    gap = abs(omega[k_1 > 0].mean() - omega[k_1 < 0].mean())
    assert abs(gap - np.pi) < 0.6, gap


def test_wrap_angles_canonicalizes_unsigned_sb2_but_is_a_no_op_when_signed(data):
    """How the sign prior relates to the post-hoc fix, measured both ways.

    ``Samples.wrap_angles()`` *does* repair the unsigned SB2 sign pattern: the
    shared ``arg_peri`` shift flips both semi-amplitudes together, so
    ``(K_1 < 0, K_2 > 0, omega)`` maps onto ``(K_1 > 0, K_2 < 0, omega + pi)``.
    The sign prior is therefore not the only way to get a canonical sign pattern
    -- it is the way that does not spend prior volume and acceptance on the
    mirror branch first, and it additionally excludes the unphysical *same-sign*
    region, which the ``arg_peri`` symmetry cannot fix because it flips both
    semi-amplitudes at once.

    Under the signed prior ``wrap_angles`` has nothing left to do, which is the
    property worth pinning: the two mechanisms do not fight each other.
    """
    unsigned = _prior(signed=False)
    raw = RejectionSampler(unsigned, JointModel.for_sb2(prior=unsigned)).run(
        data, key=jax.random.key(4), n_prior_samples=20_000, top_k=10
    )
    wrapped = raw.wrap_angles()
    assert np.all(np.asarray(wrapped["primary.rv_semiamp"].value) > 0)
    assert np.all(np.asarray(wrapped["secondary.rv_semiamp"].value) < 0)

    signed = _prior(signed=True)
    already = RejectionSampler(signed, JointModel.for_sb2(prior=signed)).run(
        data, key=jax.random.key(0), n_prior_samples=4000, top_k=16
    )
    after = already.wrap_angles()
    for key in ("primary.rv_semiamp", "secondary.rv_semiamp", "arg_peri"):
        assert np.allclose(
            np.asarray(already[key].value), np.asarray(after[key].value)
        ), key


# ---------------------------------------------------------------------------
# Exactness of the coupled two-parameter case
# ---------------------------------------------------------------------------


def test_joint_log_prob_with_two_constrained_params_matches_monte_carlo(
    weakly_constrained_data,
):
    """``k_c = 2`` in the joint path, against prior Monte-Carlo integration.

    This is the test a conditional-independence shortcut would fail, so the
    fixture has to make the two semi-amplitudes genuinely dependent. They are
    coupled only through the *shared* ``v_sys`` column, and how strongly depends
    on how well the data constrain the orbit: measured on this model,
    ``rho(K_1, K_2)`` is 0.0008 for 24 precise epochs spanning 4.4 periods --
    where the constraint is also inactive, ``Z_post == 1``, so the truncation
    costs nothing -- rising to 0.41 on four low-precision epochs inside a
    fraction of one period, where ``ln Z_post = -0.77``.

    Both the coupling and the constraint therefore bite in the same regime, the
    weakly-constrained one, which is where a sign prior earns its keep. The
    correlation is asserted before the value is, so this cannot pass for the
    wrong reason.
    """
    prior = _prior(signed=True)
    joint = JointModel.for_sb2(prior=prior)
    nonlinear = {
        "period": TRUE["period"],
        "eccentricity": TRUE["eccentricity"],
        "phase_peri": _phase_peri(weakly_constrained_data),
        "arg_peri": TRUE["arg_peri"],
    }

    comp_nl = harv.models.joint._split_nl_values(
        nonlinear,
        joint._shared_param_names(),
        joint.component_names,
        joint._per_component_nonlinear_names(),
    )
    per_comp_lp = joint._per_component_linear_prior(prior.linear_priors)
    per_comp_marg = {
        name: comp._auto_marginalized_names(per_comp_lp[name])
        for name, comp in joint.components.items()
    }
    marg, y_joint, global_cols, _ = joint._build_joint_marginalized_linear(
        comp_nl, per_comp_marg, weakly_constrained_data, prior.linear_priors
    )
    cov = np.asarray(marg.conditional(y_joint).cov)[0]
    names = [nm for nm, _ in global_cols]
    i = names.index("rv_semiamp")
    j = names.index("rv_semiamp", i + 1)
    rho = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    assert abs(rho) > 0.25, f"fixture does not couple the two K's (rho={rho})"

    got = float(
        joint.log_prob(
            nonlinear, weakly_constrained_data, linear_priors=prior.linear_priors
        )
    )
    assert got == pytest.approx(
        _monte_carlo_ln_prob(joint, prior, nonlinear, weakly_constrained_data), abs=0.02
    )


def _monte_carlo_ln_prob(joint, prior, nonlinear, data, n_draws=400_000, seed=0):
    """Prior Monte-Carlo estimate of the marginal likelihood.

    An unbiased estimate of ``E_prior[L(y | K_1, K_2, v_sys)]``, drawing the two
    semi-amplitudes from their *truncated* priors and ``v_sys`` from its Normal.
    Independent of every line of the analytic path, and dimension-agnostic where
    nested quadrature would not be -- what matters is that it would not
    reproduce the analytic value if the ``Z_post / Z_prior`` correction were
    wrong, since that factor is O(1) nats here.
    """
    rng = np.random.default_rng(seed)
    k_prior = prior.linear_priors["primary.rv_semiamp"]
    sigma_k = float(
        k_prior(
            {
                "period": nonlinear["period"],
                "eccentricity": nonlinear["eccentricity"],
            }
        ).distribution.base_dist.scale
    )
    sigma_v = float(prior.linear_priors["v_sys"].distribution.scale)

    k_1 = truncnorm.rvs(0.0, np.inf, scale=sigma_k, size=n_draws, random_state=rng)
    k_2 = truncnorm.rvs(-np.inf, 0.0, scale=sigma_k, size=n_draws, random_state=rng)
    v_sys = rng.normal(0.0, sigma_v, size=n_draws)

    ln_like = np.zeros(n_draws)
    for comp_name, semiamp in (("primary", k_1), ("secondary", k_2)):
        obs = np.asarray(data[comp_name].rv.value)
        err = np.asarray(data[comp_name].rv_err.value)
        design = np.asarray(
            joint.components[comp_name]._full_design_matrix(nonlinear, data[comp_name])
        )
        # Columns are [rv_shape, 1] -- see StandardRV.design_matrix.
        model = semiamp[:, None] * design[:, 0][None, :] + v_sys[:, None]
        resid = obs[None, :] - model
        ln_like += -0.5 * np.sum((resid / err[None, :]) ** 2, axis=1) - np.sum(
            np.log(err * np.sqrt(2 * np.pi))
        )

    top = ln_like.max()
    return top + np.log(np.mean(np.exp(ln_like - top)))
