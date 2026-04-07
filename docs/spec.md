# harv — Design Specification

**harv** is a JAX-native Python package for inferring Keplerian orbital parameters of
binary star or star-exoplanet systems from Gaia epoch astrometry and/or radial velocity
data.
In the future, harv will also support absolute and relative astrometry from other
instruments.
It is designed to be the computational backbone for binary-star and exoplanet population
science with Gaia DR4.

______________________________________________________________________

## Scientific context

A star with a companion --- a two-body system --- produces two observable signals
layered on top of ordinary stellar astrophysical astrometry and spectroscopy:

- **Astrometric wobble** — the photocenter of the system traces an ellipse on the sky
  as the companion orbits. Gaia's epoch astrometry measures the along-scan projection
  of this motion in units of milliarcseconds (mas), together with the 5-parameter
  astrometric solution (reference position, proper motion, parallax).

- **Radial velocity (RV) variation** — the line-of-sight velocity of the star (or its
  photocenter) oscillates with the orbital period. Spectrographs measure this directly
  in km/s.

Combining both datasets jointly constrains the orbit much more than either alone and
breaks degeneracies between inclination, parallax, and semi-major axis.

The target use case is **photocentric SB1 systems**: single-lined spectroscopic binaries
or stars with non-luminous companions whose combined light centroid traces the
photocenter orbit. The package also supports analysis of Gaia-only astrometry or RV-only
datasets.

______________________________________________________________________

## Core design principles

1. **JAX throughout.** All computation inside the likelihood and sampler is JAX code
   so that it JIT-compiles, vmaps, and can run on GPU/TPU. External boundaries (data
   loading, Numpy simulation helpers) may use NumPy.

1. **equinox Modules as pytrees.** All state-bearing objects (`KeplerianBody`,
   likelihood classes, parameter structs, the sampler itself) are `eqx.Module`
   subclasses, which makes them valid JAX pytrees. This allows `jax.vmap` and
   `jax.jit` to work on batches of parameters without any extra bookkeeping.

1. **Units via unxt.** Physical quantities carry units using the `unxt.Quantity` type,
   which is itself a JAX pytree. Units are stripped at the innermost computation
   boundary with `ustrip`, keeping all array math JAX-compatible. Type aliases like
   `Time = Literal["time"]` and `Speed = Literal["speed"]` make unit constraints
   readable at the type-annotation level.

1. **Two-level parameterization.** Every observable model has a clean split between
   *nonlinear* orbital parameters (which are awkward to marginalize over because they
   appear nonlinearly in the forward model) and *linear* parameters (which appear only
   in a linear design matrix and can be analytically marginalized out). The rejection
   sampler exploits this split directly.

1. **No global state.** Likelihoods close over data; samplers close over priors. All
   random state passes explicitly as JAX PRNGKey values.

______________________________________________________________________

## Type annotations and runtime checking

### Annotation conventions

All module fields and function signatures use **jaxtyping** shape-and-dtype annotations
built on top of **unxt.Quantity**. The canonical aliases live in `harv.custom_types`:

| Alias                 | Definition                                                              | Use for                                       |
| --------------------- | ----------------------------------------------------------------------- | --------------------------------------------- |
| `ScalarQTime`         | `Real[Quantity["time"], ""]`                                            | Scalar time quantities (period, t_peri, …)    |
| `ScalarQLength`       | `Real[Quantity["length"], ""]`                                          | Scalar length quantities (semi-major axis, …) |
| `ScalarQMass`         | `Real[Quantity["mass"], ""]`                                            | Scalar mass quantities                        |
| `ScalarQSpeed`        | `Real[Quantity["speed"], ""]`                                           | Scalar velocity quantities                    |
| `ScalarQAngle`        | `Real[Quantity["angle"], ""]`                                           | Scalar angle quantities                       |
| `ScalarQAngularSpeed` | `Real[Quantity["angular speed"], ""]`                                   | Scalar angular speed quantities               |
| `ScalarQDimless`      | `Real[Quantity["dimensionless"], ""]`                                   | Scalar dimensionless quantities               |
| `Vec3QLength`         | `Real[Quantity["length"], "3"]`                                         | 3-vector position returns                     |
| `Vec3QSpeed`          | `Real[Quantity["speed"], "3"]`                                          | 3-vector velocity returns                     |
| `NTime`, `NAngle`, …  | `Real[Quantity[dim], "n"]`                                              | 1-d arrays of observations                    |
| `NFloatArray`         | `Float[jax.Array, "n"]`                                                 | Plain JAX float arrays                        |
| `ScalarFloat`         | `Float[jax.Array, ""] \| np.floating \| float \| int \| ScalarQDimless` | Dimensionless scalar *inputs*                 |

Dimension literal aliases (`Time = Literal["time"]`, `Speed = Literal["speed"]`, etc.)
are also exported for use in `Quantity[Time]`-style annotations elsewhere.

### `ScalarFloat` and `float_converter`

Dimensionless scalar fields (e.g. eccentricity, sin/cos of angles) accept a wide union
of input types via `ScalarFloat` and normalize them to bare `Float[jax.Array, ""]` at
storage time using `float_converter`:

```python
class KeplerianBody(eqx.Module):
    eccentricity: ScalarFloat = eqx.field(converter=float_converter)
```

`float_converter` calls `ustrip(AllowValue, "", x)`, which strips units from a
dimensionless `Quantity` or passes through plain scalars, always producing a 0-d JAX
array.

### Annotation semantics

Field annotations describe the **accepted input type**, not necessarily the stored type.
When a field has a `converter`, the stored type is whatever the converter returns. For
example, `eccentricity: ScalarFloat` accepts `float`, `int`, `jax.Array`, or a
dimensionless `Quantity`, but after `float_converter` the stored value is always
`Float[jax.Array, ""]`.

### No `from __future__ import annotations`

Modules checked by beartype **must not** use `from __future__ import annotations`.
That directive turns all annotations into strings, which prevents jaxtyping and beartype
from inspecting them at runtime. Python 3.12+ provides native `X | Y` union syntax, so
the future import is unnecessary.

### Runtime checking with beartype

The jaxtyping import hook activates beartype checking for annotated functions and
methods:

```python
# conftest.py
from jaxtyping import install_import_hook
install_import_hook("harv.kepler", "beartype.beartype")
```

The first argument is the package prefix to instrument; the second is the **string**
path to the typechecker callable (not a direct import). This ensures that every public
method in `harv.kepler` validates its argument shapes and dtypes at call time during
tests.

### Trace-friendly validation

Runtime checks that guard field *values* (as opposed to types) inside `__check_init__`
must use `eqx.error_if` instead of Python `if … raise`. Plain conditionals attempt to
concretize JAX tracers, which breaks inside `jax.vmap` and `jax.jit`.

```python
def __check_init__(self):
    eqx.error_if(
        self.eccentricity,
        (self.eccentricity < 0) | (self.eccentricity >= 1),
        "eccentricity must be in [0, 1)",
    )
```

`eqx.error_if` inserts a runtime assertion into the traced computation graph that
raises `EquinoxRuntimeError` (a subclass of `RuntimeError`) when the condition is true.
This works correctly under all JAX transformations.

**Exception:** Checks on static metadata (e.g. verifying that unit dimensions are
consistent) can remain as plain `if … raise ValueError`, since those values are never
traced.

______________________________________________________________________

## Package structure

```
src/harv/
├── custom_types.py          # Unit-dimension Literal aliases
├── data.py                  # Observation data classes
├── kepler/                  # Orbit mechanics (JAX)
│   ├── orbits.py            # Low-level building blocks and orbit functions
│   ├── body.py              # KeplerianBody
│   ├── orientation.py       # KeplerianOrientation + Thiele-Innes
│   └── constants.py         # G
├── likelihood/              # Log-likelihood evaluators
│   ├── base.py              # AbstractLikelihood[ParamT]
│   ├── _params.py           # Parameter structs (eqx.Module pytrees)
│   ├── helpers.py           # _solve_kepler
│   ├── rv.py                # RVLikelihood (unified)
│   ├── gaia_astrometry.py   # GaiaAstrometryLikelihood (unified)
│   ├── combined.py          # CompositeLikelihood
│   └── astrometry.py        # Stub: future absolute/relative astrometry
├── priors/
│   └── rejection.py         # RejectionPrior
├── samplers/
│   ├── rejection.py         # RejectionSampler
│   ├── _strategies.py       # Data-type strategies + _ComponentSlice metadata
│   ├── _numpyro.py          # Numpyro model builders for MCMC (_ModelContext)
│   └── samples.py           # Samples container
└── simulate/                # Synthetic data generators
    ├── rv.py                # simulate_rv_sb1_data
    ├── astrometry.py        # simulate_gaia_epoch_astrometry
    └── scanlaw.py           # Gaia scanning law utilities
```

______________________________________________________________________

## Data layer (`harv.data`)

### `AbstractData`

The root base class for all observational datasets. Carries a `time: Quantity["time"]`
array (barycentric TCB) and an optional `t_ref` reference epoch. Subclasses add the
observed quantities and their uncertainties.

### `AbstractAstrometryData` / `GaiaAstrometryData`

`GaiaAstrometryData` stores the Gaia epoch astrometry for a single source:

| Field             | Units         | Description                                                      |
| ----------------- | ------------- | ---------------------------------------------------------------- |
| `time`            | time          | Barycentric observation times                                    |
| `al_position`     | angle (mas)   | Along-scan position residuals                                    |
| `al_position_err` | angle (mas)   | Per-observation 1σ uncertainties                                 |
| `scan_angle`      | angle (rad)   | Scan angle ψ of Gaia's field of view                             |
| `parallax_factor` | dimensionless | AL parallax factor H_ϖ(t)                                        |
| `t_ref`           | time          | Reference epoch for proper motion (required; see §discrepancy 3) |
| `transit_index`   | int           | Optional grouping of CCDs into transits                          |

The along-scan model is (see §Gaia astrometry likelihood):

```
y_AL(t) = α₀ cos(ψ) + δ₀ sin(ψ)
         + (μ_α cos(ψ) + μ_δ sin(ψ)) · (t − t_ref)
         + ϖ · H_ϖ(t)
         + a · [(A sin(ψ) + B cos(ψ)) cos(f) + (F sin(ψ) + G cos(ψ)) sin(f)]
```

where A, B, F, G are Thiele-Innes constants that encode the orbit orientation, and f
is the true anomaly.

### `AbstractRadialVelocityData` / `RadialVelocityData`

`RadialVelocityData` stores RV observations from a single instrument:

| Field    | Units        | Description                                         |
| -------- | ------------ | --------------------------------------------------- |
| `time`   | time         | Barycentric observation times                       |
| `rv`     | speed (km/s) | Measured radial velocities                          |
| `rv_err` | speed (km/s) | Per-observation 1σ uncertainties                    |
| `t_ref`  | time         | Reference epoch (defaults to mean observation time) |

The RV model is:

```
RV(t) = K · [cos(ω + f(t)) + e · cos(ω)] + v₀
```

where K is the semi-amplitude, ω is the argument of pericenter, and v₀ is the
systemic velocity.

### `SourceData`

`SourceData` is a named dictionary of datasets for a single source. It is the natural
container for multi-instrument observations and for combined astrometry + RV analyses:

```python
data = SourceData(
    gaia=GaiaAstrometryData(...),
    keck=RadialVelocityData(...),
    espresso=RadialVelocityData(...),
)
```

Each dataset is accessed by name. `SourceData` provides `get_datasets_by_type`,
`keys()`, `values()`, and `items()` for iteration.

**Important:** `SourceData` is for heterogeneous or multi-instrument data for a
*single stellar photocenter*. It is *not* the right container for SB2 systems (see
§Planned: `SystemData`).

**Known inconsistency (see §discrepancy 4):** `SourceData` currently inherits from
`AbstractData`, which requires a `time` field, but `SourceData.__init__` never sets
it. The inheritance relationship needs to be resolved.

### Planned: `SystemData`

The intended container for double-lined spectroscopic binaries (SB2), where separate
RV time series are measured for two distinct stellar components. The design sketch
from `api.py`:

```python
sb2_data = SystemData(
    RadialVelocityData(time1, rv1, rv_err1),  # primary (SB1 convention)
    RadialVelocityData(time2, rv2, rv_err2),  # secondary
)
# Optionally with astrometry:
sb2_data = SystemData(
    RadialVelocityData(...),    # primary
    RadialVelocityData(...),    # secondary
    photocenter=GaiaAstrometryData(...),
)
```

`SystemData` is explicitly *not* a `SourceData` — the two components measure different
stars' velocities (which move in anti-phase), not the same star through different
instruments. The SB2 model requires two separate semi-amplitudes K₁ and K₂ with
opposite signs in the design matrix.

In the future, we may want `SystemData` to also support hierarchical systems with more
than two components, but the immediate priority is SB2s.

**Until `SystemData` exists, SB2 support is not available.**  The previous heuristic
of detecting SB2 by `len(rv_datasets) > 1` inside `SourceData` was wrong: multiple
RV datasets in `SourceData` means multi-survey single-star RV, not SB2.

______________________________________________________________________

## Kepler mechanics (`harv.kepler`)

### Shared building blocks (`harv.kepler.orbits`)

Four functions that provide the canonical implementations of core orbit
computations. All three consumers (`harv.kepler`, `harv.likelihood`,
`harv.simulate`) call these building blocks instead of duplicating the math.

`mean_anomaly` and `true_anomaly_from_mean` accept and return `Quantity` objects
so callers never need to strip units themselves:

- `mean_anomaly(dt: BatchableQTime, period: ScalarQTime) -> BatchableQAngle` — `M = 2π · dt / period`
- `true_anomaly_from_mean(M: BatchableQAngle, eccentricity: ScalarFloat) -> (sin f, cos f)` — solve Kepler's equation

`rv_shape` and `thiele_innes_ABFG` remain pure functions on raw JAX arrays
because their inputs are always already dimensionless at every call site:

- `rv_shape(sin_f, cos_f, eccentricity, arg_peri)` — RV shape function: cos(ω+f) + e·cos(ω)
- `thiele_innes_ABFG(cos_ω, sin_ω, cos_Ω, sin_Ω, cos_i)` — unit Thiele-Innes constants (a=1)

The building blocks are shape-agnostic: they work for both scalar inputs
(`KeplerianBody`) and batched inputs (`jax.vmap` over parameter structs).

### `KeplerianOrientation`

Stores the three Euler angles (ω, Ω, i) that orient the orbital plane relative to the
observer, stored as sin/cos pairs for numerical stability under `jax.grad`. Provides:

- `from_angles(arg_peri, lon_asc_node, inclination)` — construct from angle Quantities
- `from_thiele_innes(A, B, F, G)` — invert Thiele-Innes constants to recover
  orientation + semi-major axis
- `rotation_matrix` — 3×3 rotation matrix (ZXZ Euler: R_z(Ω) @ R_x(i) @ R_z(ω))
- `thiele_innes_constants(semi_major_axis)` — compute (A, B, F, G)

The Thiele-Innes linearization is central to the astrometric model: by factoring out
the semi-major axis `a`, the orbital contribution to the sky plane is a linear
combination of `a · (A sin ψ + B cos ψ)` and `a · (F sin ψ + G cos ψ)`. This makes
`a` a linear parameter that can be analytically marginalized.

### `KeplerianBody`

A full Keplerian orbit: `period`, `eccentricity`, `semi_major_axis`, `t_peri`, and an
optional `KeplerianOrientation`. Provides `get_position(time)` and `get_velocity(time)`
in 3D, accounting for the orbit orientation. Alternative constructor:
`from_masses(period, e, m_total, m_body, t_peri)` — uses Kepler's 3rd law to
derive the barycentric semi-major axis from the total system mass and this body's mass.

`KeplerianBody` is the *physical* orbit model. The likelihood layer uses its own
lighter-weight parameter structs (see §Parameter structs) that are shaped to the
specific inference problem.

______________________________________________________________________

## Parameter structs (`harv.likelihood._params`)

These are the objects passed to `likelihood.log_prob(params)`. Each struct is an
`eqx.Module` and therefore a JAX pytree, which is what makes
`jax.vmap(lik.log_prob)(params_batch)` work with zero extra machinery — JAX
automatically vectorizes over all leaves simultaneously.

### Abstract-final hierarchy

The parameter classes follow the project-wide **abstract-final** pattern: a single
abstract base class `AbstractParameters(eqx.Module)` defines the interface, and each
concrete class is `@final` with all fields declared explicitly (no intermediate
abstract classes, no multi-level inheritance).

There are two levels for each data type: a **full-parameters** struct (used when all
parameters are specified explicitly, e.g. for forward modeling, MCMC, or plotting) and
a **marginalized wrapper** (created on-the-fly via `.marginalize()` or `.marginalized()`).

**Full structs** (all parameters specified explicitly — orbit + linear/observational):

| Struct                     | Additional fields                                                              | `linear_param_names`                                              |
| -------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `RVParameters`             | `K: Quantity["speed"]`, `v0: Quantity["speed"]`                                | `("K", "v0")`                                                     |
| `GaiaAstrometryParameters` | `ra0`, `dec0`, `pmra`, `pmdec`, `parallax`, `semi_major_axis` (all `Quantity`) | `("ra0", "dec0", "pmra", "pmdec", "parallax", "semi_major_axis")` |

`linear_param_names` is a `ClassVar[tuple[str, ...]]` on the full structs. It names
every linear parameter the struct holds, in design-matrix column order. It is *not* a
pytree leaf (ClassVar is excluded by equinox). The rejection sampler reads these
class attributes to name the output `Samples` columns, avoiding hardcoded strings.

**`MarginalizedParameters` wrapper** (nonlinear parameters only; linear parameters are
analytically marginalized out):

Instead of separate marginalized classes for each data type, a single
`MarginalizedParameters(eqx.Module)` wrapper is used. It stores non-marginalized field
values in a `_values: dict[str, Any]` (pytree leaves) and records which linear
parameters were removed in `marginalized_names: tuple[str, ...]` (static). Field
access delegates to `_values` via `__getattr__`, so `params.period` works as expected.

Creation:

```python
# From an existing full-parameter instance (drops named linear fields):
marg = full_params.marginalize("K", "v0")  # partial
marg = full_params.marginalize()             # all linear params

# Classmethod shortcut (sampler construction path):
marg = RVParameters.marginalized(period=..., eccentricity=..., phase_peri=..., arg_peri=...)
```

For combined astrometry + RV runs, the wrapper is created from
`GaiaAstrometryParameters` (6 nonlinear params). The distinction between run types is
carried by `Samples._data_type`, not by the parameter class.

### The `period` convention

Parameter structs store `period: Quantity["time"]`. The period prior is a
`dist.LogUniform(period_min, period_max)` in the native time unit of the data (days by
default). The prior samples period directly — no log-space transform is needed in the
sampler. The sampler constructs param structs as:

```python
period = Quantity(period_sample, data.time.unit)
```

This keeps the sampler unit-agnostic: if the data is in days the period will be in
days; if it is in years it will be in years. The likelihood is also unit-agnostic
because `_solve_kepler` computes the mean anomaly via the shared building block:

```
M = mean_anomaly(dt, params.period)
```

The ratio `dt / period` is dimensionless regardless of units.

### `phase_peri` vs `t_peri`

The nonlinear structs use `phase_peri = t_peri / period` (dimensionless, range 0–1)
rather than an absolute `t_peri`. This decouples the phase from the period scale,
which simplifies the prior (uniform on \[0, 1\]) and avoids the need to specify a
reference epoch in the prior.

______________________________________________________________________

## Likelihood layer (`harv.likelihood`)

### `AbstractLikelihood[ParamT]`

The generic base class. Uses PEP 695 syntax (`class AbstractLikelihood[ParamT: eqx.Module]`) to satisfy static type checkers. Declares two abstract members:

- `param_names: tuple[str, ...]` — names of the nonlinear parameters this likelihood
  consumes (class attribute, not an instance field, so not a pytree leaf).
- `log_prob(params: ParamT) -> jax.Array` — scalar log-likelihood for a single
  parameter sample.

The design guarantees that `jax.vmap(lik.log_prob)(params_batch)` works out-of-the-box
when `params_batch` is a pytree of stacked JAX arrays (i.e. a batch of param structs
with leading batch dimension).

### `RVLikelihood`

`RVLikelihood` is the unified radial velocity likelihood class. It closes over a
`RadialVelocityData` object and supports three evaluation modes:

1. **Marginalized** (`linear_prior` provided, `params` is `MarginalizedParameters`):
   analytically integrates over \[K, v₀\] via `MarginalizedLinear`. Supports partial
   marginalization via `params.marginalized_names`.
1. **Multi-survey marginalized** (`indicator_matrix` provided): appends
   instrument-offset columns and marginalizes \[K, v₀, δ₁, …, δₖ\] jointly.
1. **Explicit** (`linear_prior` is `None`, `params` is `RVParameters`): evaluates the
   Gaussian data log-likelihood directly.

For each nonlinear parameter sample it:

1. Solves Kepler's equation for (sin f, cos f) via `_solve_kepler`.
1. Builds the (n_obs, 2) design matrix `[rv_amplitude, 1]`.
1. If marginalized, constructs a `MarginalizedLinear` distribution (numpyro-ext) and
   calls `.log_prob()`. If explicit, evaluates `dm @ [K, v₀]` + Gaussian log-prob.

The `design_matrix(params)` method exposes the full design matrix (including indicator
columns for multi-survey) for reuse by MCMC builders and other consumers.

The `sample_conditional_linear(params, key)` method builds a `MarginalizedLinear`
from the design matrix, resolved linear prior, and data errors, then draws one sample
from the posterior conditioned on the observed data. This encapsulates the
conditional-sampling step used by the rejection sampler's linear parameter sampling
phase, keeping the strategy code simple.

#### Linear prior as a function of nonlinear parameters

The current implementation stores `linear_prior: dist.MultivariateNormal` as a fixed
distribution. This means the prior on K cannot depend on the nonlinear parameters.

However, there is a physically well-motivated prior on K that *does* depend on period
and eccentricity: the Joker-style prior that is uniform in companion mass. Because K
is related to mass by

```
K = (2πG/P)^(1/3) · m₂ sin i / (m₁ + m₂)^(2/3) / √(1 − e²)
```

a uniform prior on m₂ induces a period- and eccentricity-dependent prior on K. This
cannot be captured by a fixed `dist.MultivariateNormal`.

**Planned design:** The `linear_prior` field should accept either a fixed
`dist.MultivariateNormal` **or** a callable satisfying the `LinearPriorCallable`
protocol (defined in `harv.likelihood.helpers`) with signature
`__call__(params) -> dist.MultivariateNormal`. Inside `log_prob`, the implementation
calls `linear_prior(params)` if the field is callable, otherwise uses it directly.
Because equinox Modules are valid JAX pytrees and can hold parameters (e.g. the
reference mass m₁), this works cleanly under `jax.vmap` and `jax.jit`. Until this
is implemented, users who need a mass-based K prior should pre-transform their K
samples after the fact using the posterior period and eccentricity.

### `GaiaAstrometryLikelihood`

Same structure as `RVLikelihood`, but for astrometry. Supports marginalized (with
`linear_prior`) and explicit (without) evaluation modes. The (n_obs, 6) design matrix
columns are \[α₀, δ₀, μ_α, μ_δ, ϖ, a\], following Appendix A of
[Holl et al. 2022](https://arxiv.org/abs/2206.05726). The Thiele-Innes constants
are computed on-the-fly from the nonlinear orientation parameters.

The `design_matrix(params)` method exposes the full design matrix for reuse.

The `sample_conditional_linear(params, key)` method works identically to
`RVLikelihood.sample_conditional_linear` — builds a `MarginalizedLinear` from the
design matrix, resolved linear prior, and along-scan data, then draws one conditional
sample.

### `CompositeLikelihood`

Combines multiple `AbstractLikelihood` components by summing their log-likelihoods.
Shared nonlinear parameters (e.g. `period` appears in both the RV and astrometry
models) are automatically de-duplicated in `param_names` by order of first appearance.
Each component's `log_prob` reads only the fields it needs from the shared params
struct via duck typing — passing a `MarginalizedParameters` wrapping
`GaiaAstrometryParameters` to a component that expects RV fields works because
the wrapper carries a superset of the required attributes.

This is the canonical way to combine heterogeneous datasets, including astrometry + RV:

```python
composite = CompositeLikelihood(
    rv=RVLikelihood(data=rv_data, linear_prior=rv_prior),
    astro=GaiaAstrometryLikelihood(data=gaia_data, linear_prior=astro_prior),
)
log_liks = jax.jit(jax.vmap(composite.log_prob))(params_batch)
```

Each component holds its own `linear_prior`, which may be a fixed
`dist.MultivariateNormal` or a `LinearPriorCallable`. When callable, it receives
whatever params struct is passed by the caller — for combined data this is a
`MarginalizedParameters` wrapper, giving the callable access to all nonlinear parameters.

#### `LinearPriorCallable` protocol

`LinearPriorCallable` (defined in `harv.likelihood.helpers`) is a
`typing.Protocol` that formalises the callable linear-prior interface:

```python
class LinearPriorCallable(Protocol):
    def __call__(self, params: Any) -> dist.MultivariateNormal: ...
```

Any `eqx.Module` (or other object) whose `__call__` matches this signature
satisfies the protocol. It is `@runtime_checkable`, but the codebase uses
`isinstance(lp, dist.MultivariateNormal)` to distinguish fixed priors from
callables — the isinstance check is a static Python test, safe under JIT.

#### `_sub_mvn` — block-diagonal sub-distribution extraction

`_sub_mvn(mvn, indices)` (in `harv.likelihood.helpers`) extracts a
sub-block from a block-diagonal `MultivariateNormal` by fancy-indexing
both `loc` and `scale_tril`:

```python
def _sub_mvn(mvn, indices):
    idx = jnp.array(indices)
    return dist.MultivariateNormal(
        loc=mvn.loc[idx],
        scale_tril=mvn.scale_tril[jnp.ix_(idx, idx)],
    )
```

This is correct when the selected parameters are independent from the
unselected ones (block-diagonal covariance), which is the case for the
default combined astrometry + RV linear prior. Both `_IndexedCallable`
and `_CombinedStrategy.build_likelihood` delegate to `_sub_mvn`.

#### `_IndexedCallable` — splitting a joint callable prior

When the user provides a single joint `linear_prior` callable covering all linear
parameters (e.g. an 8-dim prior for combined astrometry + RV), the sampler wraps it
in `_IndexedCallable` to produce per-component sub-priors. Each instance holds a
static `indices: tuple[int, ...]` and delegates to `_sub_mvn`:

```python
def __call__(self, params):
    full = self.wrapped(params)  # full joint MultivariateNormal
    return _sub_mvn(full, self.indices)
```

`jnp.ix_` constructs the open mesh needed for 2-D fancy indexing on `scale_tril`.
Because `indices` is a static field, it is resolved at JAX trace time and compiles
away completely — no runtime overhead. Using index arrays (rather than `start:stop`
slices) means the design is robust to non-contiguous or reordered parameter layouts.

______________________________________________________________________

## Prior (`harv.priors.rejection.RejectionPrior`)

`RejectionPrior` holds numpyro distributions over all nonlinear parameters and a
linear prior for the linear/observational parameters. It is an `eqx.Module`.

### Constructing a prior

The preferred API is the `default_*` class methods, which cover the common
configurations with sensible defaults:

```python
RejectionPrior.default_rv()
RejectionPrior.default_astrometry()
RejectionPrior.default_combined()
```

Direct `__init__` construction is supported for custom configurations — all fields are
keyword arguments. The `default_*` methods exist purely as convenience wrappers around
`__init__`, not as alternative constructors that expose different internals. This
pattern keeps the class simple and avoids a proliferation of factory methods.

### Nonlinear parameter priors

| Field          | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `period`       | Prior on period in data time units; default `LogUniform(min, max)` |
| `eccentricity` | Typically `Beta(0.867, 3.03)` following Kipping (2013)             |
| `phase_peri`   | Typically `Uniform(0, 1)`                                          |
| `cos_i`        | Astrometry/combined only; absent for RV-only                       |
| `arg_peri`     | RV or combined; absent if not needed                               |
| `lon_asc_node` | Astrometry or combined; absent if not needed                       |

### Linear parameter prior

`linear_prior: dist.MultivariateNormal | LinearPriorCallable` is passed through to the
likelihood during the sampler's likelihood evaluation step. The callable form takes
the nonlinear param struct and returns a `dist.MultivariateNormal` — see §Linear
prior as a function of nonlinear parameters.

For single-dataset cases (RV-only or astrometry-only) `linear_prior` covers exactly
the linear parameters of that dataset. For combined astrometry + RV, `linear_prior`
covers all 8 linear parameters `[α₀, δ₀, μ_α, μ_δ, ϖ, a, K, v₀]` in that order.
The sampler splits this into per-component sub-priors using `_IndexedCallable` (for
callables) or `_sub_mvn` (for fixed MVNs) before constructing the
`CompositeLikelihood`.

The `default_*` constructors accept `linear_prior_scale` convenience arguments.
For combined data, `default_combined` takes separate `linear_prior_scale_astro` (for
the 6 astrometric parameters, in the data's position unit) and `linear_prior_scale_rv`
(for the 2 RV parameters, in the data's velocity unit), constructing a block-diagonal
8×8 covariance matrix. Direct `__init__` construction accepts any pre-built
`dist.MultivariateNormal` for full control.

### Multi-survey RV offsets

When multiple instruments observe the same star, their zero-points may differ by an
additive offset. The intended design: an `offsets` dict maps instrument names to
optional distributions over the per-instrument offset:

```python
prior = RejectionPrior.default_rv(
    offsets={"espresso": dist.Normal(0, 5.0)}
    # "keck" absent → reference instrument, offset = 0
)
```

The offsets are additional linear parameters appended to the design matrix, one column
per non-reference instrument. The linear parameters for a two-instrument (keck + espresso)
case are `[K, v₀, δ_espresso]`.

`default_rv` automatically extends the joint `linear_prior` MVN to include the offset
dimensions when `offsets` is provided. Each non-reference offset prior must be a
`dist.Normal`; its `loc` and `scale` are incorporated as additional diagonal blocks.

In `SourceData` with multiple RV datasets, the sampler stacks all observations in dict
order and builds an indicator matrix (constant across parameter samples) that selects
which rows belong to each non-reference instrument. The indicator matrix is stored on
`RVLikelihood` and appended to the base `[K, v₀]` design matrix at `log_prob` time.
Named access to offset samples works via `samples["espresso"]` (the instrument name
becomes a linear parameter key).

### SB2 and hierarchical systems

For SB2 systems the linear parameters expand to \[K₁, K₂, v₀\]: two semi-amplitudes
(with opposite sign in the design matrix) plus a shared systemic velocity. The
`default_sb2()` constructor covers this case with a 3-dimensional linear prior.

For hierarchical systems beyond SB2 (e.g. triple stars) a separate prior class may
be more maintainable than overloading `RejectionPrior` further. The design sketch in
`api.py` suggests `RejectionSB2Prior` as a dedicated class. This is a placeholder for
future work — the immediate priority is getting the single-star multi-survey and basic
SB2 cases working correctly.

______________________________________________________________________

## Rejection sampler (`harv.samplers.rejection.RejectionSampler`)

Implements the rejection sampling algorithm from
[Price-Whelan et al. 2017](https://arxiv.org/abs/1701.08160) (The Joker). The core
idea: because the likelihood is analytically marginalized over linear parameters, it
can be evaluated cheaply for millions of nonlinear prior samples, making rejection
sampling efficient.

**Algorithm:**

1. **Prior sampling.** Draw `n_prior_samples` from the nonlinear prior. Period is
   sampled directly in the data's time unit via `dist.LogUniform`; angles are
   dimensionless. All samples are raw JAX arrays (units attached later).

1. **Likelihood evaluation** (batched). For each batch of `batch_size` samples,
   construct param structs, build the appropriate likelihood objects, and evaluate
   `jax.vmap(lik.log_prob)(params_batch)` using `jax.lax.fori_loop` to control
   memory usage.

1. **Rejection.** Normalize weights to `max` and accept samples where
   `Uniform() < weight`.

1. **Linear parameter sampling.** For each accepted nonlinear sample, call
   `likelihood.sample_conditional_linear(params, key)` to sample the linear
   parameters from their conditional posterior given the data.

1. **Return** a `Samples` object.

### Data type inference

`_infer_and_validate_data_type` inspects the concrete type of `data`:

- `GaiaAstrometryData` or any `AbstractAstrometryData` → `"astrometry"`
- `RadialVelocityData` or any `AbstractRadialVelocityData` → `"rv"`
- `SourceData` with astrometry and RV → `"combined"`
- `SourceData` with multiple RV datasets → **currently treated as multi-survey RV**
  (not SB2 — see §Planned: `SystemData`)
- `SystemData` (planned) → `"sb2"`

### Data-type strategy pattern

All data-type-specific logic (data extraction, likelihood construction, parameter
building, linear sampling) is encapsulated in `_DataTypeStrategy` subclasses in
`_strategies.py`. The sampler and numpyro model builders are kept data-type-generic.

**Component-generic design:** each strategy produces a tuple of `_ComponentSlice`
objects — lightweight frozen dataclasses that carry each likelihood component's
sub-likelihood, its global column indices into the joint linear vector, and
unit-stripped observations/errors. The numpyro model builders loop over these
components instead of branching on hardcoded astro/rv fields. This means adding a new
data type (photometry, relative astrometry, SB2) requires only a new strategy
subclass and likelihood class — zero changes to `_ModelContext` or the builders.

`_ModelContext` (in `_numpyro.py`) holds only generic fields: `prior`, `strategy`,
`time_unit`, `nonlinear_cls`, `nonlinear_priors`, `lik`, `components`, and
`all_linear_names`. Multi-survey offset names are computed by
`strategy.all_linear_names()` rather than being bolted on in the context builder.

### `batch_size` and GPU support

The `batch_size` field (default 100,000) is a static equinox field that controls how
many samples are stacked and vmapped at once. The likelihood evaluation is pure JAX
and runs on any device (CPU, GPU, TPU) without code changes.

**CPU:** `fori_loop` serialises the batches, which keeps peak memory bounded at
`batch_size × n_obs × (parameter footprint)`. The default of 100,000 is suitable for
typical laptops and workstations.

**GPU:** `fori_loop` also serialises on GPU, preventing the device from saturating all
its cores across batches. On GPU, it is almost always better to use a single large
`vmap` over all samples (i.e. set `batch_size = n_prior_samples`) and let JAX/XLA
schedule the work across the device's streaming multiprocessors. A future enhancement
would auto-select `batch_size` based on `jax.devices()[0].device_kind` — using
`n_prior_samples` on GPU and 100,000 on CPU — but for now callers should set it
manually when running on GPU:

```python
sampler = RejectionSampler(prior, batch_size=n_prior_samples)  # GPU-friendly
```

______________________________________________________________________

## `Samples` container (`harv.samplers.samples.Samples`)

Stores the posterior samples returned by `RejectionSampler.run()`.

### Design

| Internal field        | Content                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `_nonlinear`          | `dict[str, jax.Array]` — nonlinear parameter samples                   |
| `_linear`             | `jax.Array` shape `(n_samples, n_linear)`                              |
| `_orbit_cls`          | Static reference to the nonlinear param class (e.g. `RVParameters`)    |
| `_full_cls`           | Static tuple of full param classes (e.g. `(RVParameters,)`)            |
| `_linear_param_units` | Static tuple of unit strings for `_linear` columns, set by the sampler |
| `_metadata`           | Static dict with `t_ref` and any extra info                            |

`_linear_param_names` is derived on demand from `_full_cls[i].linear_param_names`
(concatenated across all full classes). Unit restoration for each linear parameter uses
`_linear_param_units`, which the sampler populates from the actual data units (e.g.
`rv_data.rv.unit` for K and v₀, `astro_data.al_position.unit` for astrometric params).

Dict-style access (`samples["period"]`) dispatches to appropriate unit restoration:

- `"period"` → `Quantity` in data time units (stored directly in `_nonlinear`)
- `"log_period"` → dimensionless array (derived as `log10(period)`)
- `"t_peri"` → `Quantity` (derived from `phase_peri * period + t_ref`)
- `"inclination"` → `Quantity` in radians (derived from `arccos(cos_i)`)
- Linear params (`"K"`, `"v0"`, `"ra0"`, etc.) → `Quantity` with units from `_linear_param_units`

`Samples` supports `median()`, `percentile()`, `summary()`, HDF5 serialization
(`to_hdf5` / `from_hdf5`), and a corner plot via arviz (`plot_corner`).

______________________________________________________________________

## Simulation utilities (`harv.simulate`)

### `simulate_rv_sb1_data` (currently `simulate_rv_data`)

Generates a synthetic `RadialVelocityData` object for a single-lined spectroscopic
binary (SB1). Parameters default to random draws if not specified. Returns
`(data, true_params)`. Uses NumPy random number generation (not JAX) because this is a
one-off setup step, not on the hot path.

The `simulate_rv_sb1_data` name leaves a clear namespace for sibling functions:

- `simulate_rv_sb2_data` — generates primary + secondary `RadialVelocityData` for a
  double-lined binary (planned, requires `SystemData`)
- `simulate_rv_multisurv_data` — generates a `SourceData` with multiple instruments
  and per-instrument zero-point offsets (planned)

### `simulate_gaia_epoch_astrometry`

Generates a synthetic `GaiaAstrometryData` object with 5-parameter astrometry plus
Keplerian orbital motion. Includes a simplified (sinusoidal) parallax factor model via
`fake_parallax_factor`. For real Gaia data, the parallax factors come from the Gaia
epoch astrometry tables directly.

______________________________________________________________________

## Key design decisions and trade-offs

### Why `cos_i` instead of `i`?

Inclination `i` has a prior that is uniform in `cos(i)` for an isotropically
distributed orbit population. Sampling `cos_i ~ Uniform(-1, 1)` is therefore the
natural prior, and it avoids the singularity at `i = 0` or `i = π`. The raw `cos_i`
value is stored throughout; inclination in radians is only a derived quantity exposed
via `Samples["inclination"]`.

### Why Thiele-Innes rather than (ω, Ω, i, a) directly?

The Thiele-Innes constants (A, B, F, G) appear linearly in the astrometric model. This
means `a` (the semi-major axis in angular units) is a *linear* parameter and can be
marginalized analytically. Fitting for (ω, Ω, i, a) directly would make `a` a
nonlinear parameter. The price is that the Thiele-Innes constants mix orientation with
amplitude, but since we marginalize them out, that is acceptable.

### Why `MarginalizedLinear` from numpyro-ext?

Analytic marginalization over Gaussian linear parameters given a Gaussian prior is a
standard result, but implementing it carefully (handling the Woodbury identity,
numerics, gradients) is non-trivial. numpyro-ext provides a tested implementation that
also gives us `.conditional()` to draw from the posterior conditional — which is
exactly what the rejection sampler needs for the linear parameter sampling step.

### Why `eqx.field(static=True)` for `batch_size`, `_linear_param_names`, etc.?

Fields marked `static=True` in equinox are not treated as pytree leaves — they are
compared structurally (by value) when JAX traces a new JIT-compiled function. This is
appropriate for metadata that controls the computation graph (like `batch_size`) or
for strings that have no gradient. Concretely, if `batch_size` changes, JAX re-traces;
if it stays the same, the cached compilation is reused.

______________________________________________________________________

## Planned features and known gaps

### `SystemData` for SB2

A dedicated two-component data container (see §Planned: `SystemData` above). The SB2
model requires two design matrices with columns \[K₁, 0, 1\] and \[0, -K₂, 1\] for the
primary and secondary respectively, sharing a common systemic velocity v₀.

### Combined astrometry + multi-survey RV

**This is a known bug / unimplemented gap. `_CombinedStrategy` raises
`NotImplementedError` if `SourceData` contains both `GaiaAstrometryData` and more
than one `RadialVelocityData`.**

The RV-only multi-survey case (no astrometry) is fully implemented — see
`RVLikelihood` (with `indicator_matrix`) and `_RVStrategy`. The combined case requires
the same indicator-matrix treatment applied to the RV block of the joint linear prior:

- The joint linear prior becomes `(6 + 2 + n_non_ref)`-dimensional:
  `[ra0, dec0, pmra, pmdec, parallax, a, K, v₀, δ₁, …, δₖ]`.
- The `RVLikelihood` within the `CompositeLikelihood` must hold an `indicator_matrix`
  and append it to the RV design matrix columns inside `log_prob`.
- `_CombinedStrategy.extract_data` must stack the RV datasets (as `_RVStrategy`
  does) and build the indicator matrix.
- `default_combined` must extend the RV block of the covariance matrix when `offsets`
  is provided.
- `_extra_linear_names` in `Samples` must carry the offset instrument names through.

**Test:** add an `xfail` test in `tests/unit/samplers/` that constructs a
`SourceData` with one `GaiaAstrometryData` and two `RadialVelocityData` datasets,
runs `RejectionSampler.run`, and asserts it raises `NotImplementedError` (mark
`strict=True` so the test fails if the error is accidentally suppressed, and update
to a passing test once the feature is implemented).

### Iterative rejection sampling

The Joker's iterative scheme grows the sample batch exponentially until enough
posterior samples are accepted. Useful when the likelihood is very constraining (few
or highly precise observations):

```python
samples = sampler.run_iterative(
    data,
    n_prior_samples=1_000_000,
    n_requested_samples=1024,
    growth_factor=128,
)
```

### Adaptive / importance sampling

For sources with many accepted samples (multi-modal or broad posterior), a standard
rejection step wastes most prior draws. An adaptive or importance-weighted scheme
would recycle the rejected samples.

### MAP refinement

After rejection sampling, optimize from each posterior sample to find the nearest
posterior mode. Useful for getting precise parameter estimates before handing off to
MCMC.

### MCMC initialization

**Implemented** — `RejectionSampler.init_mcmc` in `samplers/rejection.py`.

The rejection-sampler posterior provides warm-start positions for MCMC chains.
`init_mcmc` takes the `Samples` object returned by `run`, the observed data, and an
optional numpyro kernel *class*.  It builds a numpyro model automatically from the
sampler's prior and data, draws one starting position per chain from the posterior,
and returns a `_WarmStartMCMC` wrapper whose `run()` injects those positions
automatically.

Two model variants are supported via the `marginalized` argument:

**Marginalized** (`marginalized=True`, default) — MCMC explores only the nonlinear
subspace; linear parameters are analytically integrated out, identical to rejection
sampling.  Sample sites: nonlinear parameter names only.

```python
import jax.random as jr

prior = RejectionPrior.default_rv(period_min=50, period_max=200)
sampler = RejectionSampler(prior)
samples = sampler.run(rv_data, n_prior_samples=500_000)

mcmc = sampler.init_mcmc(
    samples, rv_data,
    num_chains=4, num_warmup=2_000, num_samples=10_000,
)
mcmc.run(jr.PRNGKey(0))
posterior = mcmc.get_samples()  # keys: period, eccentricity, phase_peri, arg_peri
```

**Full** (`marginalized=False`) — MCMC samples all parameters jointly.  Linear
parameters are drawn from the prior's `MultivariateNormal` as a joint latent site
`"_linear"`; each component is also exposed as a named `deterministic` site (e.g.
`"K"`, `"v0"`, `"semi_major_axis"`) for easy access via `get_samples()`.  The
correlation structure of the linear prior is preserved.

```python
mcmc = sampler.init_mcmc(
    samples, rv_data, marginalized=False,
    num_chains=4, num_warmup=2_000, num_samples=10_000,
)
mcmc.run(jr.PRNGKey(0))
posterior = mcmc.get_samples()  # adds K, v0 (as deterministics) to the above
```

**Warm-start positions:**

- Marginalized model: `init_params` contains the nonlinear parameter arrays, shape
  `(num_chains,)` per key.
- Full model: same nonlinear arrays, plus `"_linear"` with shape
  `(num_chains, n_linear)` drawn from `samples._linear`.  Named deterministic sites
  (`"K"`, `"v0"`, …) are computed from `"_linear"` at runtime and do not appear in
  `init_params`.

`_WarmStartMCMC` (in `samplers/samples.py`) is a thin wrapper around
`numpyro.infer.MCMC`.  It delegates all attributes (`get_samples`, `print_summary`, …)
to the underlying MCMC object via `__getattr__`, and only overrides `run` to inject
`init_params` unless the caller provides their own.

### Absolute and relative astrometry

`AbstractAstrometryData` exists as a base for future data types:

- **Absolute astrometry** (RA/Dec timeseries from ground-based or HST observations) —
  stub exists in `data.py`, commented out.
- **Relative astrometry** (separation and position angle from direct imaging) — not
  yet started.

### Visualization

**`Samples.plot_corner()`** — implemented; uses arviz `plot_pair` for a corner plot of
the posterior.

**`Samples.plot(data=source_data)`** — implemented in `samplers/samples.py`.  Selects
panels automatically based on `data_type`:

| `data_type`    | Panel(s)                                              |
| -------------- | ----------------------------------------------------- |
| `"rv"`         | Phase-folded RV curve; data points + posterior curves |
| `"astrometry"` | On-sky orbital ellipses (posterior samples only)      |
| `"combined"`   | Both panels side by side                              |

```python
fig = samples.plot(data=rv_data)           # RV: data + model curves
fig = samples.plot(data=source_data)       # combined: two panels
fig = samples.plot(n_samples=100)          # astrometry: orbits only
```

**RV panel** — phase-folds observations at the median posterior period; overlays `n_samples`
model curves drawn from the posterior.  Multi-survey datasets are coloured by instrument.

**Astrometry panel** — plots the photocentric orbit as an ellipse in (ΔRA, ΔDec) using
the Thiele-Innes constants derived from each posterior sample.  Gaia along-scan
measurements are 1-D projections and cannot be shown as 2-D sky positions, so only
model curves appear.

The numpy Kepler solver `_kepler_newton` (module-level in `samplers/samples.py`) is
used for plotting; it is a simple Newton-Raphson iteration and is intentionally separate
from the JAX `_solve_kepler` used inside likelihoods.

______________________________________________________________________

## API sketch (from `api.py`)

The intended user-facing interface for common use cases:

```python
# Minimal RV-only case:
data = RadialVelocityData(time, rv, rv_err)
prior = RejectionPrior.default_rv()
sampler = RejectionSampler(prior)
samples = sampler.run(data, n_prior_samples=100_000)

# With max posterior samples:
samples = sampler.run(data, n_prior_samples=100_000, max_posterior_samples=128)

# Multi-instrument RV with zero-point offsets:
data = SourceData(
    keck=RadialVelocityData(time1, rv1, rv_err1),
    espresso=RadialVelocityData(time2, rv2, rv_err2),
)
prior = RejectionPrior.default_rv(
    offsets={"espresso": dist.Normal(0, 5.0)}
    # keck is the reference instrument; its offset is fixed to 0
)

# Combined astrometry + RV:
data = SourceData(
    keck=RadialVelocityData(time, rv, rv_err),
    gaia=GaiaAstrometryData(...),
)
prior = RejectionPrior.default_combined()
samples = sampler.run(data, n_prior_samples=1_000_000)

# SB2 (not yet implemented):
sb2_data = SystemData(
    RadialVelocityData(time, rv1, rv_err1),  # primary
    RadialVelocityData(time, rv2, rv_err2),  # secondary
)
prior = RejectionPrior.default_sb2()

# Post-sampling analysis:
samples["period"]          # Quantity in data time units
samples["eccentricity"]    # dimensionless array
samples.median("K")        # median semi-amplitude
samples.summary()          # dict of all statistics
samples.plot_corner()      # arviz corner plot
samples.plot(data=data)    # phase-folded RV/astrometry overlay (planned)
samples.to_hdf5("out.h5")  # persistence
```
