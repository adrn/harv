# Sharp bits

This page lists some conventions, gotchas, and design choices that could be overlooked if you are new to `harv` or JAX.

See also the `unxt` sharp bits guide, since `harv` uses `unxt` extensively for unit handling: <https://unxt.readthedocs.io/en/latest/guides/sharp-bits.html>.

## harv turns on float64 for you

`harv` is designed for orbital inference problems where the likelihood can span a very
large dynamic range, so importing it enables JAX's 64-bit mode and warns that it has
done so:

```python
import harv  # enables jax_enable_x64 for the whole process
```

This changes the default dtype of every JAX array created afterwards, including arrays
belonging to other libraries in the same process, which is why the warning exists. To
silence it, say so yourself before importing `harv`, with either the `JAX_ENABLE_X64=1`
environment variable or `jax.config.update("jax_enable_x64", True)`.

You can opt out with `JAX_ENABLE_X64=0`, but the results will be unreliable. float32 gives unstable
likelihood evaluations in rejection sampling and with broad priors or high-S/N data,
and it is not only the likelihood: a barycentric time in float32 has a resolution of
0.25 days, so a BJD of 2460123.125 silently becomes 2460123.0 the moment you build an
`RVData`. JAX's flag is global and cannot be set per array, so `harv` cannot protect
its own arrays from a float32 process, and switching it on later does not repair data
already built.

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
Or, if you pass a custom prior for a linear parameter from outside the Gaussian family,
it will be sampled explicitly by default. Truncated Normals and mixtures of Normals are
*inside* that family, so a positivity constraint does not by itself cost you the
marginalization. `RejectionSampler.summary()` prints the classification it will use,
including which parameters stay explicit only because another prior reads them.

## Extensions require both a model extension and a prior entry

If you add an extension like jitter, survey offsets, trends, or a gaussian process, you
usually need two pieces:

1. The extension attached to the model, and
1. A matching prior entry for every extra parameter introduced by that extension.

For example, adding `Jitter(obs_unit="km/s")` to the model is not enough on its own;
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

Posterior samples can contain negative `rv_semiamp` or `semi_major_axis` values, even
though these are amplitudes. This is expected with the default sign-free priors,
because changing the sign of an amplitude is equivalent to shifting an orbital angle by
$\pi$. If you want a canonical representation with positive amplitudes, call
`samples.wrap_angles()`.

This rewrites the samples into the equivalent convention

- `rv_semiamp >= 0`
- `semi_major_axis >= 0`
- every angle it shifts wrapped into $\[0, 2\\pi)$

without changing the physical orbit. For `SqrtEccRV` samples it negates `sqrte_cosw` and
`sqrte_sinw` and shifts `mean_longitude` by $\pi$ instead.

The degeneracy matters most when you start MCMC from rejection samples. Those contain
both signs, so with several chains some start in one mirror mode and some in the other.
NUTS cannot move between them, so r_hat and ESS for the angles are poor even though each
chain is fine. `NumpyroSampler.run` warns when this happens. Start from
`init_samples=rej_samples.wrap_angles()` so that every chain begins with non-negative
amplitudes (astrometry-only fits still keep the separate mirror with both `arg_peri` and
`lon_asc_node` shifted by $\pi$). The shifted angles land in $\[0, 2\\pi)$, the range of the
default angle priors; with a prior on another interval, wrap them into it yourself. We keep the
default amplitude priors sign-free rather than truncating them at zero because the
truncated prior halves the rejection sampler's acceptance rate (see `docs/spec.md`).
