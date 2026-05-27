# Sharp bits

This page lists some conventions, gotchas, and design choices that could be overlooked if you are new to `harv` or JAX.

See also the `unxt` sharp bits guide, since `harv` uses `unxt` extensively for unit handling: <https://unxt.readthedocs.io/en/latest/guides/sharp-bits.html>.

## Work in float64

`harv` is designed for orbital inference problems where the likelihood can span a very
large dynamic range. In practice, this means you should usually enable 64-bit JAX:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Using float32 can lead to unstable likelihood evaluations, especially in rejection
sampling and when working with broad priors or high-S/N data.

## Public APIs are unit-aware

`harv` uses `unxt.Quantity` throughout the public API. Times, velocities, angles,
lengths, and similar physical values should generally be passed in as quantities with
units, not bare floats.

This applies both to data objects and to priors. For example, orbital periods should be
quantities like `Q(100, "day")`, and unit-bearing priors should be wrapped in
`harv.QD(...)` so that the sampler knows what unit the samples live in.

Internally, `harv` strips units where necessary at the innermost numerical boundary. You
usually should not need to do that yourself.

## Prefer `quaxed.numpy` when units might be involved

Most public `harv` APIs accept or return `unxt.Quantity` objects. If you do array math
on those values yourself, use `quaxed.numpy`:

```python
import quaxed.numpy as jnp
```

instead of plain `jax.numpy`.

The `quaxed.numpy` functions dispatch correctly on quantities and preserve units. Plain
`jax.numpy` is only appropriate when you know you are working exclusively with unit-less
JAX arrays.

## "Linear" parameters do not have to be marginalized

`harv` splits parameters into nonlinear and linear pieces because Gaussian linear
parameters can be marginalized analytically. But just because a parameter is linear does
not mean it must be marginalized. You can often specify a list of `marginalized_names`
to control which linear parameters are marginalized and which are sampled explicitly.
Or, if you pass a custom prior for a linear parameter that is not Gaussian, it will be
sampled explicitly by default.

## Extensions require both a model extension and a prior entry

If you add an extension like jitter, survey offsets, trends, or a gaussian process, you
usually need two pieces:

1. The extension attached to the model, and
1. A matching prior entry for every extra parameter introduced by that extension.

For example, adding `Jitter(param_unit="km/s")` to the model is not enough on its own;
you also need a `jitter=...` prior in the corresponding `HarvPrior`.

For joint models, extension-parameter names are component-qualified, such as
`"rv.jitter"`.

## `SourceData` and `SystemData` mean different things

These two data containers are easy to confuse:

- `SourceData` is for one physical source observed by multiple instruments or data
  products.
- `SystemData` is for distinct physical components in the same gravitational system.

An SB2 should usually use `SystemData`, not `SourceData`, because the primary and
secondary RVs are measurements of different bodies, not multiple instruments observing
the same body. If you have both RV data and astrometry for the same source, those should
usually be combined into a single `SourceData` object.

## Sampled amplitudes may come back with sign degeneracies

Posterior samples can contain negative `rv_semiamp` or `semi_major_axis` values (which
are distances and you might think these cannot be negative). That is not necessarily
wrong; it is part of the orbital angle degeneracy. If you want a canonical
representation with positive amplitudes, call `samples.wrap_angles()`.

This rewrites the samples into the equivalent convention

- `rv_semiamp >= 0`
- `semi_major_axis >= 0`
- `arg_peri` wrapped into $\[0, 2\\pi)$

without changing the physical orbit.
