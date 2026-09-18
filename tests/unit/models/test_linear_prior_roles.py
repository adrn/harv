"""The single classifier that decides how each linear parameter is treated.

``classify_linear_prior`` owns the *precedence* between the four roles, which
is the part that is easy to get wrong: the two introspection surfaces
(``RejectionSampler.summary()`` and the ``verbose=True`` advisory) once
disagreed about a pinned-and-unmarginalizable prior, and each pointed the user
at a fix that would not have worked. Both now render this enum.

See ``docs/spec.md`` -> Linear prior classification (auto mode).
"""

import numpyro.distributions as dist
import pytest
from unxt import Q

from harv.distributions import QD
from harv.models._helpers import (
    LinearRole,
    _can_marginalize,
    _linear_sampling_order,
    _needs_explicit_sampling,
    classify_linear_prior,
)
from harv.models.priors.callables import (
    ParallaxDependentProperMotionPrior,
    PeriodDependentKPrior,
)

PINNED = frozenset({"parallax"})


@pytest.mark.parametrize(
    ("label", "prior", "expected"),
    [
        ("normal", QD(dist.Normal(0.0, 1.0), "km/s"), LinearRole.MARGINALIZED),
        ("half normal", QD(dist.HalfNormal(1.0), "mas"), LinearRole.MARGINALIZED),
        (
            "truncated normal",
            QD(dist.TruncatedNormal(0.0, 1.0, low=0.0), "km/s"),
            LinearRole.MARGINALIZED,
        ),
        (
            "callable",
            PeriodDependentKPrior(Q(1.0, "km/s"), Q(1.0, "yr")),
            LinearRole.MARGINALIZED,
        ),
        ("delta", QD(dist.Delta(3.0), "km/s"), LinearRole.FIXED),
        ("uniform", QD(dist.Uniform(0.0, 1.0), "km/s"), LinearRole.EXPLICIT),
        ("gamma", QD(dist.Gamma(2.0, 1.0), "mas"), LinearRole.EXPLICIT),
    ],
)
def test_roles_without_pinning(label, prior, expected):
    """With nothing pinned, the role is a statement about the prior family."""
    assert classify_linear_prior(prior, name="x") is expected, label


class TestPrecedence:
    def test_unmarginalizable_beats_pinned(self):
        """The ordering the review found inverted.

        A ``Gamma`` on a pinned name is ``EXPLICIT``, not ``PINNED``: the fix the
        pinned label points at -- drop the dependency -- would leave a ``Gamma``
        prior that still cannot be marginalized.
        """
        role = classify_linear_prior(
            QD(dist.Gamma(2.0, 1.0), "mas"), name="parallax", pinned_names=PINNED
        )
        assert role is LinearRole.EXPLICIT

    def test_marginalizable_and_pinned_is_pinned(self):
        """The Gaia default: a HalfNormal parallax two callables read."""
        role = classify_linear_prior(
            QD(dist.HalfNormal(10.0), "mas"), name="parallax", pinned_names=PINNED
        )
        assert role is LinearRole.PINNED

    def test_pinning_needs_the_name(self):
        """Omitting the name asks only whether the math permits it."""
        prior = QD(dist.HalfNormal(10.0), "mas")
        assert classify_linear_prior(prior, pinned_names=PINNED) is (
            LinearRole.MARGINALIZED
        )


class TestDerivedPredicates:
    """The two booleans callers still use must stay consistent with the roles."""

    @pytest.mark.parametrize(
        ("prior", "expected"),
        [
            (QD(dist.Normal(0.0, 1.0), "km/s"), True),
            (QD(dist.HalfNormal(1.0), "mas"), True),
            (QD(dist.Delta(3.0), "km/s"), True),
            (QD(dist.Uniform(0.0, 1.0), "km/s"), False),
        ],
    )
    def test_can_marginalize_ignores_pinning(self, prior, expected):
        assert _can_marginalize(prior) is expected

    def test_needs_explicit_sampling_covers_explicit_and_pinned(self):
        assert _needs_explicit_sampling(QD(dist.Uniform(0.0, 1.0), "km/s"))
        assert _needs_explicit_sampling(
            QD(dist.HalfNormal(1.0), "mas"), name="parallax", pinned_names=PINNED
        )
        assert not _needs_explicit_sampling(QD(dist.HalfNormal(1.0), "mas"))

    def test_delta_is_not_explicit_here(self):
        """``FIXED`` joins the marginalized set, then has its value extracted.

        Short-circuiting it to explicit would skip ``_handle_delta_priors`` and
        leave the value unset.
        """
        assert not _needs_explicit_sampling(QD(dist.Delta(3.0), "km/s"))


def test_sampling_order_puts_plain_priors_before_callables():
    """A callable must be resolved after the values its ``requires`` names."""
    priors = {
        "pmra": ParallaxDependentProperMotionPrior(Q(50.0, "km/s")),
        "parallax": QD(dist.HalfNormal(10.0), "mas"),
        "ra0": QD(dist.Normal(0.0, 100.0), "mas"),
    }
    order = _linear_sampling_order(priors)
    assert order.index("parallax") < order.index("pmra")
    assert set(order) == set(priors)
