# harv

[![Tests](https://github.com/adrn/harv/actions/workflows/test.yml/badge.svg)](https://github.com/adrn/harv/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/harv/badge/?version=latest)](https://harv.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)

<br/>
<div align="center">
<img
    src="https://raw.githubusercontent.com/adrn/harv/refs/heads/main/docs/_static/logo_med.png"
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

### Radial velocity fitting

```python
from unxt import Q
from harv import Model
from harv.data import RVData
from harv.samplers import RejectionPrior, RejectionSampler

# Load or create RV data with explicit units - some sample RV data:
data = RVData(
    time=Q([56000.0, 56100.0, 56250.0, 56400.0, 56600.0], "day"),
    rv=Q([12.3, -8.7, 5.1, -14.2, 10.8], "km/s"),
    rv_err=Q([1.2, 0.9, 1.1, 0.8, 1.0], "km/s"),
)

# Set up a prior with default structure (log-uniform in period, etc.)
prior = RejectionPrior.default_rv(
    period_min=Q(10, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),   # RV semi-amplitude scale
    sigma_v0=Q(10, "km/s"),   # systemic velocity prior width
)

# Build a model (combines prior + data, constructs likelihood)
model = Model(prior, data)

# Run the rejection sampler
sampler = RejectionSampler(model)
samples = sampler.run(n_prior_samples=1_000_000, seed=42)

# Inspect results — quantities carry units:
print(f"Accepted {samples.n_samples} posterior samples")
print(f"Period: {samples['period']}")
print(f"Eccentricity: {samples['eccentricity']}")
```

### Gaia epoch astrometry

```python
from harv.data import GaiaAstrometryData

astro_data = GaiaAstrometryData(
    time=times,                          # Quantity["time"]
    al_position=al_pos,                  # Quantity["angle"] (mas)
    al_position_err=al_err,              # Quantity["angle"] (mas)
    scan_angle=scan_angles,              # Quantity["angle"] (rad)
    parallax_factor=parallax_factors,    # dimensionless
)

prior = RejectionPrior.default_gaia_astrometry(
    period_min=Q(50, "day"),
    period_max=Q(3000, "day"),
    sigma_a0=Q(1, "au"),   # astrometric semi-major axis scale
    parallax=Q(5, "mas"),    # source parallax (for physical prior scaling)
)

model = Model(prior, astro_data)
sampler = RejectionSampler(model)
samples = sampler.run(n_prior_samples=1_000_000, seed=42)
```

### MCMC continuation

When the rejection sampler returns a small number of samples, you can refine with
NumPyro MCMC, started from the posterior samples:

```python
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
