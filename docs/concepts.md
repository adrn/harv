# Key concepts in harv

This page describes the main modeling ideas behind `harv`.

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
works with log weights and accepts each sample with probability equal to its likelihood
relative to the maximum likelihood over all of the prior samples.

This is especially useful for orbital inference problems with sparse, noisy, or highly
multimodal data, where MCMC methods struggle to move between narrow, well-separated
modes.

The main tradeoff is:

- if rejection sampling returns enough samples, those samples are independent posterior
  draws and do not require MCMC convergence diagnostics,
- but if the likelihood occupies a tiny fraction of prior volume, rejection sampling can
  become expensive.

In practice, rejection sampling locates the posterior modes, and MCMC can optionally be
used afterward to continue from the accepted samples.

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
they must be sampled. The priors that can be marginalized analytically are the Gaussian
family read broadly: a `Normal`, a truncated Normal (including `HalfNormal`, and so any
positivity or sign constraint), or a mixture of those sharing one support. Anything
outside that family, such as a `Uniform` or a `Gamma`, is sampled explicitly instead.

Being marginalizable is not quite the same as being marginalized. A linear parameter
whose value is read by another parameter's prior has to stay explicitly sampled, because
a marginalized parameter has no sampled value to read. That is why `parallax` is sampled
explicitly under the Gaia defaults even though its `HalfNormal` prior could be
marginalized: the semi-major-axis and proper-motion priors both depend on it. For a
star with a well-measured parallax (say, better than 10%), passing
`scale_parallax=` (e.g. the catalog parallax) to the Gaia `default_prior` computes
those prior widths from that fixed value instead, so the parallax is marginalized
like the other linear parameters and the sampler has one less dimension to explore.
The posterior parallax should still be compared with the value passed in, and if the
two differ by more than a few percent, the fit can be rerun with the posterior value.

For how the truncated and mixture cases are actually computed, see
{doc}`math-notes/constrained-linear-priors`.

## JAX

`harv` is built on JAX. That has a few practical consequences.

- Many objects in `harv` are pytrees via [Equinox](https://docs.kidger.site/equinox/)
  modules.
- Core computations are designed to work under `jax.jit` and `jax.vmap`.
- Numerical code should be written in a trace-friendly way.

As a result, the likelihood evaluation can be compiled, vectorized over large batches of
prior samples, and reused, which is a large part of why the rejection sampler is
practical. However, some ordinary Python patterns do not transfer to JAX-compatible
code. For example, code that depends on mutation, hidden global state, or
value-dependent control flow is usually invalid or slow in JAX-based inference.

`harv` is also unit-aware throughout the public API. Times, angles, velocities, lengths,
and other physical quantities should usually be passed around as `unxt.Quantity`
objects, not bare floats.

## numpyro

`harv` uses `numpyro` for the probabilistic machinery around priors and for MCMC-based
continuation. In particular, `numpyro` provides the probability distributions used to
define priors, the machinery for building MCMC models from `harv` model definitions, and
a continuation path after rejection sampling has located the posterior modes. Rejection
sampling acts as a global search, and `numpyro` can then generate dense posterior
samplings of the modes that it finds.

For most analyses, the practical workflow is:

1. define your data,
1. define a physically sensible prior,
1. choose a model plus any needed extensions,
1. use rejection sampling to get posterior samples,
1. optionally continue with MCMC or nested sampling if you need more samples.
