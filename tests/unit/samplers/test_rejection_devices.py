"""The rejection sampler shards each batch over local devices, without changing results.

Runs in subprocesses: the number of host CPU devices is fixed when JAX first
initializes, so a multi-device run cannot be staged inside the test process. Each
probe runs every rejection path (``run``, in-memory and HDF5 ``run_with_samples``)
at one device count; the test compares a sharded probe against the one-device
probe (spec §``batch_size`` and GPU support).
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

PROBE = textwrap.dedent("""
    import json
    import sys
    import warnings
    import jax
    jax.config.update("jax_num_cpu_devices", int(sys.argv[1]))
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from unxt import Q
    import harv
    import harv.models as hm
    from harv.samplers import rejection

    t = jnp.linspace(0.0, 300.0, 12)
    data = harv.RVData(
        time=Q(t, "day"),
        rv=Q(5.0 * jnp.sin(2 * jnp.pi * t / 100.0), "km/s"),
        rv_err=Q(jnp.full(12, 0.5), "km/s"),
    )
    prior = hm.StandardRV().default_prior(
        period_min=Q(50.0, "day"), period_max=Q(200.0, "day"),
        sigma_K0=Q(10.0, "km/s"), sigma_v0=Q(10.0, "km/s"),
    )
    # 1400 = 2 full batches + a padded tail, so padding is sharded and trimmed too.
    sampler = harv.RejectionSampler(prior, harv.RVModel(), batch_size=500)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_top_k = sampler.run(
            data, key=jax.random.key(0), n_prior_samples=1400, top_k=20
        )
        rejected = sampler.run(data, key=jax.random.key(0), n_prior_samples=1400)
        library = prior.sample(1400, key=jax.random.key(1), model=harv.RVModel())
        rws_memory = sampler.run_with_samples(
            data, library, key=jax.random.key(2), top_k=20
        )
        harv.make_prior_cache(
            prior, harv.RVModel(), 1400, sys.argv[2], key=jax.random.key(1)
        )
        rws_hdf5 = sampler.run_with_samples(
            data, sys.argv[2], key=jax.random.key(2), top_k=20
        )
        mesh = rejection._device_mesh(500)
        mesh7_is_none = rejection._device_mesh(7) is None

    results = {"run_top_k": run_top_k, "rws_memory": rws_memory, "rws_hdf5": rws_hdf5}
    arrays = [
        a
        for s in (*results.values(), rejected)
        for a in (
            *(q.value for q in s.nonlinear.values()),
            *(q.value for q in s.linear.values()),
            s.ln_likelihood,
        )
        if a is not None  # the plain rejection run returns no ln_likelihood
    ]
    ln_l = run_top_k.ln_likelihood
    try:
        # Raises "incompatible devices" if ln_l is committed to device 0.
        jax.jit(lambda a, b: a + b)(
            ln_l, jax.device_put(jnp.zeros_like(ln_l), jax.devices()[-1])
        )
        uncommitted = True
    except ValueError:
        uncommitted = False

    print(json.dumps({
        "n_devices": jax.device_count(),
        "n_shards": 1 if mesh is None else mesh.size,
        "mesh7_is_none": mesh7_is_none,
        **{
            name: {
                "period": s.nonlinear["period"].value.tolist(),
                "ln_likelihood": s.ln_likelihood.tolist(),
                "keys": [list(s.nonlinear), list(s.linear)],
            }
            for name, s in results.items()
        },
        "n_accepted": rejected.n_samples,
        "devices_per_array": sorted({len(a.sharding.device_set) for a in arrays}),
        "uncommitted": uncommitted,
        "warnings": [
            str(w.message) for w in caught if "local devices" in str(w.message)
        ],
    }))
""")

_RESULTS: dict[int, dict] = {}


def _probe(n: int, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run PROBE with ``n`` host CPU devices; cached, so each count runs once."""
    if n not in _RESULTS:
        path = tmp_path_factory.mktemp(f"devices{n}") / "prior.h5"
        # Pinned to CPU: jax_num_cpu_devices only multiplies the CPU backend.
        result = subprocess.run(  # noqa: S603  -- argv is this module's own constant
            [sys.executable, "-c", PROBE, str(n), str(path)],
            capture_output=True,
            text=True,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
            check=False,
        )
        assert result.returncode == 0, result.stderr
        _RESULTS[n] = json.loads(result.stdout.strip().splitlines()[-1])
    return _RESULTS[n]


@pytest.mark.parametrize(("n", "n_shards"), [(2, 2), (3, 2)])
def test_sharded_run_matches_one_device(n, n_shards, tmp_path_factory):
    base, out = _probe(1, tmp_path_factory), _probe(n, tmp_path_factory)
    assert out["n_devices"] == n
    assert out["n_shards"] == n_shards
    for path in ("run_top_k", "rws_memory", "rws_hdf5"):
        assert out[path]["keys"] == base[path]["keys"]  # declaration order kept
        assert out[path]["period"] == base[path]["period"]  # identical draws
        np.testing.assert_allclose(
            out[path]["ln_likelihood"], base[path]["ln_likelihood"], rtol=1e-10
        )
    assert out["n_accepted"] == base["n_accepted"]
    assert out["devices_per_array"] == [1]
    assert out["uncommitted"]
    # batch_size=500 splits over 2 of 3 devices; 7 has no divisor <= n but 1.
    assert any("uses 2 of them" in w for w in out["warnings"]) == (n == 3)
    assert out["mesh7_is_none"]
    assert any("uses 1 of them" in w for w in out["warnings"])
