# Known bugs

Deferred defects with enough analysis recorded that picking one up does not mean
re-deriving it. Fixing one should also remove its entry here.

______________________________________________________________________

## `EcoswEsinwRV`'s default prior support is a square, not the unit disk

**Status:** open, mitigated but not fixed.
**Found:** 2026-08-31, by the rejection-sampler benchmark grid (`benchmarks/`).
**Affects:** `EcoswEsinwRV.default_prior`, and any caller relying on it.

### The defect

`EcoswEsinwRV.default_prior` puts independent `Uniform(-1, 1)` priors on `ecosw` and
`esinw` (`src/harv/models/parameterizations/rv.py`). That support is a **square**, but a
bound orbit requires the **unit disk**:

```
e = sqrt(ecosw**2 + esinw**2) < 1
```

About 21% of draws (`1 - pi/4`) fall outside it with `e >= 1`, which is not an orbit.
Measured over 4000 draws: `e` ranged 0.015 to 1.404, with 21.3% at `e >= 1`.

Two consequences:

1. **Wasted prior library.** A fifth of every prior cache is unphysical and can never
   be accepted. At `M = 1e7` that is ~2.1M dead draws.
1. **Silent `NaN` evidence statistics.** The default `rv_semiamp` prior
   (`PeriodDependentKPrior`) scales as `(1 - e**2)**(-1/2)`, which is `NaN` for
   `e >= 1`. `NaN` propagates through the `max` reduction that the rejection step
   normalizes by, so `max_log_likelihood`, `logZ_int`, and `logZ_int_ess` all come
   back `NaN` and **no samples are accepted, with no error raised**.

### Current mitigation (not a fix)

`ignore_non_finite=True` converts the `NaN` draws into ordinary rejections. Measured:

|                           | `max_log_likelihood` |
| ------------------------- | -------------------- |
| `ignore_non_finite=False` | `nan`                |
| `ignore_non_finite=True`  | `-1709.60`           |

This is documented in `docs/sharp-bits.md` ("`EcoswEsinwRV`'s default prior admits
unbound orbits") and in the spec's `EcoswEsinwRV` section, and `benchmarks/` passes the
flag. It removes the silent-`NaN` trap but not the wasted 21%.

### Why this is not a small fix

`harv.stats.numpyro_ext.UnitDisk` is the right distribution — uniform on the disk, with
`log_prob = -log(pi)` — but it is **two-dimensional**:

```python
class UnitDisk(dist.Distribution):
    support = unit_disk                       # event_dim = 1, asserts shape (2,)
    def __init__(...):
        super().__init__(batch_shape=(), event_shape=(2,), ...)
    def sample(self, key, sample_shape=()):   # -> (*sample_shape, 2)
```

`HarvPrior.nonlinear_priors` is `dict[str, PriorDist]` — exactly one *scalar* prior per
parameter name — and `HarvPrior.sample_nonlinear` draws each independently at shape
`(n_samples,)`:

```python
return {
    name: _unwrap_dist(d).sample(k, (n_samples,))
    for (name, d), k in zip(self.nonlinear_priors.items(), keys, strict=True)
}
```

So `UnitDisk` cannot be dropped in. A single distribution spanning `ecosw` *and*
`esinw` needs a new concept: a **joint (multi-name) nonlinear prior**, e.g. a
tuple-keyed entry `{("ecosw", "esinw"): UnitDisk()}`, whose sample's trailing axis is
split across the named parameters and whose `log_prob` is counted once for the pair.

Sites that assume one scalar prior per name, and would need to handle the joint case:

- `HarvPrior.nonlinear_priors` type and any `__check_init__` validation
- `HarvPrior.sample_nonlinear` — the zip above
- `HarvPrior.sample` — `all_nonlinear_priors[name]` lookup by string
  (`src/harv/models/priors/prior.py`, around the nonlinear unit-restoration loop)
- `_sample_nonlinear_params` in `src/harv/models/component.py` — the numpyro path,
  one `numpyro.sample(name, ...)` per key
- `_expected_prior_keys` in `src/harv/samplers/rejection.py` — prior-cache key validation
- `ln_prior` accumulation, so the pair contributes one `-log(pi)` rather than two terms
- `EcoswEsinwRV.default_prior` itself

`nonlinear_priors` has ~70 references across 12 files; most are annotations or
docstrings, but the seven above are load-bearing.

### Alternatives considered

- **Conditional scalar priors** — `ecosw ~ semicircle`, then
  `esinw | ecosw ~ Uniform(±sqrt(1 - ecosw**2))`. Mathematically exact and keeps one
  prior per name, but the nonlinear prior machinery has no equivalent of
  `LinearPriorCallable`, so a nonlinear prior cannot depend on another nonlinear
  parameter. Would need its own new concept.
- **One 2-vector parameter** — declare a single nonlinear param of shape `(2,)`. Avoids
  the joint-prior concept but changes the parameterization's public parameter names,
  breaking `Samples["ecosw"]`, `convert_parameterization`, and the design matrix
  contract.
- **`UnitDiskTransform`** — not applicable. It is the `biject_to` transform for
  unconstrained MCMC, and pushing `Uniform(-1, 1)**2` through it gives density
  `∝ 1 / sqrt(1 - y0**2)`, not uniform on the disk.

### Spec implication

A joint nonlinear prior is new public API, so `docs/spec.md` must define it before
implementation: the tuple-key form, how `sample_nonlinear` splits the event axis, how
`ln_prior` counts it, and how prior caches key it on disk.
