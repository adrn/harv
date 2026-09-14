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
at-scale
benchmarks
running-benchmarks
:::

**harv** (pronounced _harvey_) is a Python package for inferring Keplerian orbital
parameters of binary-star and star–exoplanet systems from time series data.
We currently support modeling radial velocities, *Gaia* epoch astrometry, or both
jointly.
It is designed to be a computational backbone for binary-star and exoplanet population
science with large spectroscopic surveys, Gaia DR4, and beyond.

harv is built on [JAX](https://github.com/google/jax) for JIT and multi-device (e.g.,
GPU) support, [NumPyro](https://github.com/pyro-ppl/numpyro) for representing
probability distributions and running MCMC, and
[unxt](https://github.com/GalacticDynamics/unxt) for units-aware computation throughout.

:::{warning}
harv is in rapid development! The public API is not yet stable.
:::


## Overview

A star with a (unseen) companion produces two observable signals on top of ordinary
stellar astrophysics: an astrometric wobble as the system photocenter (the visible star)
traces an ellipse on the sky, and radial velocity variations.
Modeling either of these data (or both, jointly) constrains the companion's orbit and
breaks degeneracies between inclination, parallax, and semi-major axis.

When the data are sparse or low signal-to-noise, orbital solutions are often strongly
multimodal, and inference can be challenging because standard methods like MCMC will
struggle to explore the posterior pdf effectively.
harv is designed to handle these cases, to enable population inferences even when the
individual systems are not well-constrained.

### Where to go next

- New to harv? Start with **{doc}`getting-started`**.
- Check out the **{doc}`tutorials/index`**.
- Read about some  **{doc}`key concepts <concepts>`**.
- New to JAX, NumPyro, or unxt? Check **{doc}`sharp-bits`**.
- Looking up a specific class or function? Use the **{doc}`api/index`**.


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
pip install git+https://https://github.com/adrn/harv.git
```

:::

:::{tab-item} building from source

```bash
cd /path/to/parent
git clone https://https://github.com/adrn/harv.git
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
