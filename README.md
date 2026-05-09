# harv

[![Tests](https://github.com/adrn/harv/actions/workflows/test.yml/badge.svg)](https://github.com/adrn/harv/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/harv/badge/?version=latest)](https://harv.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)

<br/>
<div align="center">
<img
    src="https://raw.githubusercontent.com/adrn/harv/refs/heads/main/docs/_static/logo-text-only.png"
    alt="harv logo"
    width="300"
>
</div>
<br/>

**harv** is a Python package for inferring Keplerian orbital parameters of binary-star
and star–exoplanet systems from time series data. Built on
[JAX](https://github.com/google/jax), [NumPyro](https://github.com/pyro-ppl/numpyro),
and [unxt](https://github.com/GalacticDynamics/unxt) for units-aware computation.

It's pronounced _harvey_.

## ⚠️ Warning! ⚠️

`harv` is in rapid development and is pre-alpha. Meaning the API is not stable or
guaranteed! Once the first version is released, we will have some guarantees about
backwards compatibility, but there are no guarantees for API stability with the current
development versions of this package. Sorry!

<!-- ### Key features

- **JAX-native** — JIT-compiled likelihoods and samplers; runs on CPU or GPU.
- **Units throughout** — all physical quantities carry explicit units via
  [unxt](https://github.com/GalacticDynamics/unxt), reducing unit-conversion errors
- **Analytical marginalization** — linear parameters (semi-amplitude, systemic velocity,
  astrometric offsets) are marginalized analytically for fast rejection sampling
- **Multi-instrument support** — fit RV data from multiple spectrographs with
  per-instrument offsets
- **Gaia DR4 ready** — native support for Gaia epoch astrometry (along-scan
  measurements) using the local plane coordinate convention
- **SB1 and SB2** — supports single- and double-lined spectroscopic binaries
- **Polynomial trends** — optional polynomial velocity trends for long-period companions -->

## Installation

Requires Python 3.12+. Install from GitHub:

```bash
pip install git+https://github.com/adrn/harv
```

Or, if using [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/adrn/harv
```

## Quickstart

TODO: move to a quickstart page in documentation and link here.

### Radial velocity modeling

```python
from unxt import Q
import harv

# Load or create RV data with explicit units - some sample RV data:
data = harv.RVData(
    time=Q([56000.0, 56100.0, 56250.0, 56400.0, 56600.0], "day"),
    rv=Q([12.3, -8.7, 5.1, -14.2, 10.8], "km/s"),
    rv_err=Q([1.2, 0.9, 1.1, 0.8, 1.0], "km/s"),
)

# Set up a prior with default structure (log-uniform in period, etc.)
prior = harv.RejectionPrior.default_rv(
    period_min=Q(10, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),   # RV semi-amplitude scale
    sigma_v0=Q(10, "km/s"),   # systemic velocity prior width
)

# Run the rejection sampler
sampler = harv.RejectionSampler(prior, harv.RVModel())
samples = sampler.run(data, n_prior_samples=1_000_000, seed=42)

# Inspect results — quantities carry units:
print(f"Accepted {samples.n_samples} posterior samples")
print(f"Period: {samples['period']}")
print(f"Eccentricity: {samples['eccentricity']}")
```

### Gaia epoch astrometry

<!-- TODO: custom parallax prior parallax=harv.QD(dist.TruncatedNormal(0.5, 0.5, low=0.0), "mas"),  # parallax prior -->

```python
import numpyro.distributions as dist
import quaxed.numpy as jnp

astro_data = harv.GaiaAstrometryData(
    time=Q([958.110978, 994.910525, 995.086642, 1010.091395, 1076.918577], "day"),
    al_position=Q([147.066, 379.996, 378.656, 74.666, -293.923], "mas"),
    al_position_err=Q([0.370, 0.446, 0.428, 0.270, 0.247], "mas"),
    scan_angle=Q([-59.047, -5.114, -5.783, -68.579, -134.155], "deg"),
    parallax_factor=jnp.array([0.70828, -0.46657, -0.45946, 0.19659, -0.21379])
)

prior = harv.RejectionPrior.default_gaia_astrometry(
    period_min=Q(50, "day"),
    period_max=Q(3000, "day"),
    sigma_a0=Q(1, "au"),   # astrometric semi-major axis prior scale
    sigma_parallax=Q(1.0, "mas"),  # parallax prior width
    sigma_pos=Q(1, "mas"),  # position offset prior width
    sigma_vtan=Q(30, "km/s"),  # tangential velocity prior width
)

sampler = harv.RejectionSampler(prior, harv.GaiaAstrometryModel())
samples = sampler.run(astro_data, n_prior_samples=1_000_000, seed=42)
```

### MCMC continuation

When the rejection sampler returns a small number of samples, you can refine with
NumPyro MCMC, started from the posterior samples:

```py
from harv.samplers import NumpyroSampler

samples = sampler.run(n_prior_samples=1_000_000, max_posterior_samples=128)
mcmc_sampler = NumpyroSampler(model=model, prior=prior)
mcmc_samples = mcmc_sampler.run(samples, seed=0)
```

## How it works

**harv** is built on the same tricks as [The Joker](https://github.com/adrn/thejoker)
(Price-Whelan et al. 2017) to make rejection sampling practical for Keplerian orbit
inference. It uses a two-level parameterization that separates orbital parameters into:

1. **Nonlinear parameters** (period, eccentricity, argument of pericenter, phase) —
   sampled directly from the prior via rejection sampling, and
1. **Linear parameters** (RV semi-amplitude, systemic velocity, astrometric offsets) —
   these can be analytically marginalized given each set of nonlinear parameters.

This approach makes rejection sampling possible for high-dimensional parameter spaces by
reducing the effective dimensionality of the sampling problem. The main utility of the
rejection sampler is to map out the multi-modal posterior distribution of orbital
parameters in the low signal-to-noise or low number of observations regime, which is
challenging or impossible for MCMC methods. The samples returned by the rejection
sampler are exact draws from the posterior distribution, and can be used to initialize
MCMC samplers for further refinement, if necessary.

All objects are [Equinox](https://github.com/patrick-kidger/equinox) modules (valid JAX
pytrees), so they work seamlessly with `jax.jit`, `jax.vmap`, and `jax.grad`.

## Documentation

TODO: Full documentation will be available at
[harv.readthedocs.io](https://harv.readthedocs.io).

## License

**harv** is free software released under the MIT License. See [LICENSE](LICENSE) for
details.
