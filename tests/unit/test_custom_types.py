"""Tests for the type aliases in :mod:`harv.custom_types`.

The ``N*`` aliases all describe 1-d arrays of ``n`` observations; the ``Scalar*``
aliases describe 0-d values.  These tests pin that distinction down so an alias
cannot silently drift to the wrong shape (see ``docs/spec.md``, "Annotation
conventions").
"""

import jax.numpy as jnp
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import TypeCheckError, jaxtyped
from unxt import Q

from harv.custom_types import (
    NAngle,
    NFloatArray,
    NQAny,
    NSpeed,
    NTime,
    ScalarQAny,
    ScalarQTime,
)

# beartype raises its own violation for a failed jaxtyping check under some
# versions; accept either so the test pins shape behavior, not the reporting path.
SHAPE_ERRORS = (TypeCheckError, BeartypeCallHintParamViolation)


@jaxtyped(typechecker=beartype)
def _takes_n_quantity(x: NQAny) -> NQAny:
    return x


@jaxtyped(typechecker=beartype)
def _takes_scalar_quantity(x: ScalarQAny) -> ScalarQAny:
    return x


def test_nqany_accepts_1d_quantity():
    """``NQAny`` is the 1-d observation-array alias, not a scalar one."""
    x = Q(jnp.array([1.0, 2.0, 3.0]), "km/s")
    assert _takes_n_quantity(x).shape == (3,)


def test_nqany_rejects_scalar_quantity():
    """A 0-d Quantity must not satisfy ``NQAny`` -- that was the original bug."""
    with pytest.raises(SHAPE_ERRORS):
        _takes_n_quantity(Q(1.0, "km/s"))


def test_scalar_qany_is_distinct_from_nqany():
    """``ScalarQAny`` and ``NQAny`` describe different shapes."""
    assert _takes_scalar_quantity(Q(1.0, "km/s")).shape == ()
    with pytest.raises(SHAPE_ERRORS):
        _takes_scalar_quantity(Q(jnp.array([1.0, 2.0]), "km/s"))


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        (NAngle, Q(jnp.array([0.1, 0.2]), "rad")),
        (NTime, Q(jnp.array([0.1, 0.2]), "day")),
        (NSpeed, Q(jnp.array([0.1, 0.2]), "km/s")),
        (NFloatArray, jnp.array([0.1, 0.2])),
        (NQAny, Q(jnp.array([0.1, 0.2]), "mas")),
    ],
)
def test_n_aliases_are_all_one_dimensional(alias, value):
    """Every ``N*`` alias accepts a length-2 1-d value and rejects a scalar."""

    @jaxtyped(typechecker=beartype)
    def f(x: alias):
        return x

    assert f(value).shape == (2,)
    with pytest.raises(SHAPE_ERRORS):
        f(value[0])


def test_scalar_qtime_rejects_arrays():
    """Sanity check on the ``Scalar*`` side of the convention."""

    @jaxtyped(typechecker=beartype)
    def f(x: ScalarQTime):
        return x

    assert f(Q(1.0, "day")).shape == ()
    with pytest.raises(SHAPE_ERRORS):
        f(Q(jnp.array([1.0, 2.0]), "day"))
