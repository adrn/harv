# harv

## Introduction

:::{toctree}
:maxdepth: 2
:hidden:

Home <self>
concepts
sharp-bits
tutorials/index
api/index
math-notes/index
at-scale
benchmarks
running-benchmarks
:::

**harv** (pronounced _harvey_) is a Python package for inferring Keplerian orbital
parameters of binary-star and star–exoplanet systems from time series data.
We currently support modeling radial velocities, *Gaia* epoch astrometry, or both
jointly.
harv is designed for population inferences of binary star, exoplanet, and compact object
systems with large spectroscopic surveys and *Gaia* DR4.

harv is built on [JAX](https://github.com/google/jax) for JIT and multi-device (e.g.,
GPU) support, [NumPyro](https://github.com/pyro-ppl/numpyro) for representing
probability distributions and running MCMC, and
[unxt](https://github.com/GalacticDynamics/unxt) for units-aware computation throughout.


## Overview

A star with an unseen companion produces two observable signals: an astrometric wobble
as the photocenter of the system (the visible star) traces an ellipse on the sky, and
radial velocity variations.
Modeling either of these data constrains the orbit of the companion and modeling both
jointly breaks degeneracies between inclination, parallax, and semi-major axis.

When the data are sparse or have low signal-to-noise, the posterior pdf over the orbital
parameters is often very multimodal, and MCMC and most other sampling methods struggle
to generate useful samples from the posterior.
harv is designed for these cases, so that population inferences can include systems
whose orbits are not well constrained.

### Where to go next

- {doc}`getting-started`: installation and first examples.
- {doc}`tutorials/index`: worked examples for radial velocities, *Gaia* astrometry, and
  case studies.
- {doc}`Key concepts <concepts>`: the main modeling ideas behind harv.
- {doc}`sharp-bits`: conventions and pitfalls for users new to JAX, NumPyro, or unxt.
- {doc}`api/index`: the documentation for each class and function.
- {doc}`Math notes <math-notes/index>`: longer derivations behind parts of the
  implementation.


## Installation

[![PyPI version][pypi-version]][pypi-link] [![PyPI platforms][pypi-platforms]][pypi-link]

::::{tab-set}

:::{tab-item} pip

```bash
pip install harv
```

:::

:::{tab-item} uv

```bash
uv add harv
```

:::

:::{tab-item} source, via pip

```bash
pip install git+https://github.com/adrn/harv.git
```

:::

:::{tab-item} building from source

```bash
cd /path/to/parent
git clone https://github.com/adrn/harv.git
cd harv
pip install -e .  # editable mode
```

:::

::::

<!-- LINKS -->

[equinox]: https://docs.kidger.site/equinox/
[jax]: https://jax.readthedocs.io/en/latest/
<!-- [joss]: https://joss.theoj.org/papers/10.21105/joss.07771/status.svg
[joss-link]: https://doi.org/10.21105/joss.07771 -->
<!-- [quax]: https://github.com/patrick-kidger/quax
[quaxed]: https://quaxed.readthedocs.io/en/latest/ -->
[pypi-link]: https://pypi.org/project/harv/
[pypi-platforms]: https://img.shields.io/pypi/pyversions/harv
[pypi-version]: https://img.shields.io/pypi/v/harv
<!-- [zenodo-badge]: https://zenodo.org/badge/734877295.svg
[zenodo-link]: https://zenodo.org/doi/10.5281/zenodo.10850455 -->
