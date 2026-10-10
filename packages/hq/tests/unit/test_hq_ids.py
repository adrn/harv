"""Tests for ``harv_hq.ids``: the stable hash and per-source PRNG keys."""

import os
import subprocess
import sys

import jax
import numpy as np

from harv_hq import stable_hash
from harv_hq.ids import prior_cache_key, source_key


def same(a, b):
    return bool(np.array_equal(jax.random.key_data(a), jax.random.key_data(b)))


def test_stable_hash_known_values():
    # Hard-coded: a change here changes every run's per-source randomness.
    assert stable_hash("2M00000002+7417074") == 807116804
    assert stable_hash(4295806720) == 1127149370


def test_stable_hash_matches_across_processes():
    """Unlike hash(), the value does not depend on PYTHONHASHSEED."""
    code = "from harv_hq.ids import stable_hash; print(stable_hash('abc'))"
    out = subprocess.run(  # noqa: S603 -- fixed command, no untrusted input
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "123", "JAX_ENABLE_X64": "1"},
    )
    assert int(out.stdout) == stable_hash("abc")


def test_stable_hash_range_and_id_types():
    assert 0 <= stable_hash(2**62) < 2**32
    assert stable_hash(np.int64(12345)) == stable_hash(12345) == stable_hash("12345")


def test_source_key_is_deterministic_and_distinct():
    key = source_key(42, "2M00000002+7417074", "rejection")
    assert same(key, source_key(42, "2M00000002+7417074", "rejection"))
    assert not same(key, source_key(42, "2M00000002+7417074", "mcmc"))
    assert not same(key, source_key(42, "2M00000003+0000000", "rejection"))
    assert not same(key, source_key(43, "2M00000002+7417074", "rejection"))


def test_source_key_for_large_gaia_ids():
    gaia_id = 4295806720000000000
    assert same(source_key(1, gaia_id, "mcmc"), source_key(1, str(gaia_id), "mcmc"))


def test_prior_cache_key_is_distinct_from_source_keys():
    key = prior_cache_key(42)
    assert same(key, prior_cache_key(42))
    assert not same(key, source_key(42, 0, "rejection"))
    assert not same(key, jax.random.key(42))
