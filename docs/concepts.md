# Key concepts in harv

`harv` is built around a set of modeling ideas that are important to understand when working with the package.

## Rejection sampling

The primary new inference method provided by `harv` is rejection sampling. The
implementation follows the same broad idea as [The
Joker](https://github.com/adrn/thejoker): draw many samples from the prior, evaluate the
likelihood for each sample, and retain samples with probability proportional to the
posterior weight.

In the simplest form, the posterior pdf over parameters {math}`\theta` given data
{math}`D` is

```{math}
p(\theta \mid D) \propto p(D \mid \theta) \, p(\theta) \quad ,
```

so if you draw candidate parameter values {math}`\theta^{(k)}` from the prior
{math}`p(\theta)`, the only remaining weight is the likelihood. In practice, `harv`
works with log weights and accepts samples with probability based on the likelihood
relative to the maximum weight in the current batch.

This is especially useful for orbital inference problems with sparse, noisy, or highly
multimodal data, where all MCMC methods will struggle to mix across well-separated but
narrow modes.

The main tradeoff is straightforward:

- if rejection sampling returns enough samples, those samples are independent posterior
  draws and do not require MCMC convergence diagnostics,
- but if the likelihood occupies a tiny fraction of prior volume, rejection sampling can
  become expensive.

In practice, `harv` is designed so that rejection sampling handles the hard global
geometry first, and MCMC can optionally be used afterward to continue from those
accepted samples.

## Marginalized likelihoods

One of the central design ideas in `harv` is the separation between nonlinear orbital
parameters, such as `period`, `eccentricity`, and `arg_peri`, and linear parameters,
such as `rv_semiamp`, `v_sys`, astrometric offsets, proper motions, parallax, and some
extension parameters.

This matters because linear parameters can often be handled much more efficiently than
fully nonlinear ones. In many cases, Gaussian linear parameters are analytically
marginalized out of the likelihood, leaving a lower-dimensional problem for sampling.

Schematically, `harv` treats the parameters as

```{math}
\theta = (\theta_{\mathrm{nl}}, \theta_{\mathrm{lin}}),
```

and then works with the marginalized likelihood

```{math}
p(D \mid \theta_{\mathrm{nl}}) =
\int p(D \mid \theta_{\mathrm{nl}}, \theta_{\mathrm{lin}})
\, p(\theta_{\mathrm{lin}} \mid \theta_{\mathrm{nl}})
\, d\theta_{\mathrm{lin}} \quad .
```

This integral removes the linear parameters from the expensive outer sampling loop when
their prior family permits analytic marginalization.

In `harv`, linear parameters are defined by how they enter the design matrix, not by how
they must be sampled. If a linear parameter has a Gaussian prior, it can usually be
marginalized analytically. If it has a non-Gaussian prior, it is sampled explicitly
instead.

## JAX

`harv` is built on JAX. That has a few practical consequences.

- Many objects in `harv` are pytrees via [Equinox](https://docs.kidger.site/equinox/)
  modules.
- Core computations are designed to work under `jax.jit` and `jax.vmap`.
- Numerical code should be written in a trace-friendly way.

The payoff is that likelihood evaluation can be compiled and vectorized over large
batches of prior samples, and reused efficiently. This is a big part of why the
rejection sampler is practical. The downside is that some ordinary Python paradigms
don't always transfer cleanly to JAX-compatible code. For example, code that depends on
Pythonic mutation, hidden global state, or value-dependent control flow is usually
invalid or leads to bad performance in JAX-based inference.

`harv` is also unit-aware throughout the public API. Times, angles, velocities, lengths,
and other physical quantities should usually be passed around as `unxt.Quantity`
objects, not bare floats.

## numpyro

`harv` uses `numpyro` for the probabilistic machinery around priors and for MCMC-based
continuation. The package is not built as a pure probabilistic-programming interface;
instead, it uses `numpyro` where it is most useful.

In particular, `numpyro` provides the probability distributions used to define priors,
the machinery for building MCMC models from `harv` model definitions, and a continuation
path after rejection sampling has already located posterior modes.

That last point is the most important in practice. Rejection sampling can behave like a
global search tool while `numpyro` can be used to refine and generate dense posterior
samplings of modes that are found.

For most analyses, the practical workflow is:

1. define your data,
1. define a physically sensible prior,
1. choose a model plus any needed extensions,
1. use rejection sampling to get posterior samples,
1. optionally continue with MCMC if you need denser local exploration.
