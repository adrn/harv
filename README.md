# harv

[![Tests](https://github.com/adrn/harv/actions/workflows/test.yml/badge.svg)](https://github.com/adrn/harv/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/harvey/badge/?version=latest)](https://harvey.readthedocs.io/en/latest/)
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

Here are a few entry points to the documentation:

- [Documentation](https://harvey.readthedocs.io/en/latest)
- [Getting started with harv](https://harvey.readthedocs.io/en/latest/getting-started.html)
- [Tutorials](https://harvey.readthedocs.io/en/latest/tutorials/index.html)

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

The full documentation is available at:
[harvey.readthedocs.io](https://harvey.readthedocs.io).

## License

**harv** is free software released under the MIT License. See [LICENSE](LICENSE) for
details.
