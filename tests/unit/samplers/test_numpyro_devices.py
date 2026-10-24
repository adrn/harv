"""MCMC posteriors come back on one device, even when chains ran in parallel.

Runs in a subprocess: the number of host CPU devices is fixed when JAX first
initializes, so a multi-device run cannot be staged inside the test process.
Parallel chains return arrays sharded across devices, and every eager use of a
sharded array is a multi-device dispatch (``plot_rv`` was ~5x slower), so
``NumpyroSampler.run`` gathers the posterior onto one device.
"""

import json
import os
import subprocess
import sys
import textwrap

PROBE = textwrap.dedent("""
    import json
    import jax
    import numpyro
    numpyro.set_host_device_count(2)
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from unxt import Q
    import harv
    import harv.models as hm

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
    init = harv.RejectionSampler(prior, harv.RVModel()).run(
        data, key=jax.random.key(0), n_prior_samples=2000, top_k=2
    )
    post = harv.NumpyroSampler(prior, harv.RVModel()).run(
        data, init_samples=init, key=jax.random.key(1),
        num_chains=2, num_warmup=5, num_samples=5, chain_method="parallel",
        return_logprobs=True,
    )
    arrays = [q.value for q in (*post.nonlinear.values(), *post.linear.values())]
    arrays += [post.ln_likelihood, post.ln_prior]
    print(json.dumps({
        "n_devices": jax.device_count(),
        "devices_per_array": sorted({len(a.sharding.device_set) for a in arrays}),
    }))
""")


def test_parallel_chain_posterior_is_on_one_device():
    # Pinned to CPU: set_host_device_count only multiplies the CPU backend.
    result = subprocess.run(  # noqa: S603  -- argv is this module's own constant
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["n_devices"] == 2
    assert out["devices_per_array"] == [1]
