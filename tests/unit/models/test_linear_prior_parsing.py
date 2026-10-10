"""Shape and support contracts on ``_parse_linear_prior`` / ``_resolve_linear_priors``.

One linear parameter takes one scalar prior. The two ways that can go wrong are
both silent rather than loud: a *batched* prior gets read as an equal-weight
mixture whose weights do not sum to one, and a mixture whose components carry
different bounds breaks the shared-support assumption the whole normalizer rests
on. See ``docs/spec.md`` -> Support-constrained and mixture linear priors.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from harv.distributions import QD
from harv.models._helpers import _resolve_linear_priors


def _resolve(prior, unit=""):
    return _resolve_linear_priors({"a": prior}, {}, {"a": unit})


class TestBatchedPriorsAreRejected:
    def test_batched_normal_raises(self):
        """Not an unweighted 3-component mixture with total prior weight 3."""
        with pytest.raises(ValueError, match="is batched"):
            _resolve(QD(dist.Normal(jnp.zeros(3), 5.0), "km/s"), "km/s")

    def test_batched_truncated_normal_raises(self):
        with pytest.raises(ValueError, match="is batched"):
            _resolve(dist.TruncatedNormal(jnp.zeros(2), jnp.ones(2), low=0.0))

    def test_the_error_points_at_mixturesamefamily(self):
        with pytest.raises(ValueError, match="MixtureSameFamily"):
            _resolve(dist.Normal(jnp.zeros(4), 1.0))


class TestScalarAndMixturePriorsStillParse:
    def test_scalar_normal(self):
        resolved = _resolve(dist.Normal(1.0, 2.0))
        assert resolved.n_components == 1
        assert resolved.is_plain_gaussian

    def test_half_normal_becomes_a_lower_bound(self):
        resolved = _resolve(QD(dist.HalfNormal(2.0), "mas"), "mas")
        assert resolved.low_finite == (True,)
        assert float(resolved.low[0]) == 0.0

    def test_mixture_keeps_its_own_weights(self):
        """The component axis is legitimate here, and the weights are normalized."""
        mixture = dist.MixtureSameFamily(
            dist.Categorical(jnp.array([0.3, 0.7])),
            dist.Normal(jnp.array([0.0, 5.0]), jnp.array([1.0, 2.0])),
        )
        resolved = _resolve(mixture)
        assert resolved.n_components == 2
        assert float(jnp.sum(jnp.exp(resolved.ln_weights))) == pytest.approx(1.0)
