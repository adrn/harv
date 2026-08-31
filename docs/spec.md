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

1. **No global state.** Component models close over data; samplers combine models and
   priors; random state passes explicitly as JAX key values.

______________________________________________________________________

## Type annotations and runtime checking

### Annotation conventions

All module fields and function signatures use **jaxtyping** shape-and-dtype annotations
built on top of **unxt.Quantity**. The canonical aliases live in `harv.custom_types`:

| Alias                 | Definition                                                              | Use for                                       |
| --------------------- | ----------------------------------------------------------------------- | --------------------------------------------- |
| `ScalarQTime`         | `Real[Q["time"], ""]`                                                   | Scalar time quantities (period, t_peri, …)    |
| `ScalarQLength`       | `Real[Q["length"], ""]`                                                 | Scalar length quantities (semi-major axis, …) |
| `ScalarQMass`         | `Real[Q["mass"], ""]`                                                   | Scalar mass quantities                        |
| `ScalarQSpeed`        | `Real[Q["speed"], ""]`                                                  | Scalar velocity quantities                    |
| `ScalarQAngle`        | `Real[Q["angle"], ""]`                                                  | Scalar angle quantities                       |
| `ScalarQAngularSpeed` | `Real[Q["angular_speed"], ""]`                                          | Scalar angular speed quantities               |
| `ScalarQDimless`      | `Real[Q["dimensionless"], ""]`                                          | Scalar dimensionless quantities               |
| `Vec3QLength`         | `Real[Q["length"], "3"]`                                                | 3-vector position returns                     |
| `Vec3QSpeed`          | `Real[Q["speed"], "3"]`                                                 | 3-vector velocity returns                     |
| `BatchVec3QLength`    | `Real[Q["length"], "3 *batch"]`                                         | Batched 3-vector positions                    |
| `BatchVec3QSpeed`     | `Real[Q["speed"], "3 *batch"]`                                          | Batched 3-vector velocities                   |
| `BatchQTime`, etc.    | `Real[Q[dim], "*batch"]`                                                | Batched Quantities (scalar or array)          |
| `BatchFloat`          | `Float[jax.Array, "*batch"] \| np.floating \| float \| ...`             | Dimensionless batched inputs                  |
| `NTime`, `NAngle`, …  | `Real[Q[dim], "n"]`                                                     | 1-d arrays of observations                    |
| `NFloatArray`         | `Float[jax.Array, "n"]`                                                 | Plain JAX float arrays                        |
| `ScalarFloat`         | `Float[jax.Array, ""] \| np.floating \| float \| int \| ScalarQDimless` | Dimensionless scalar *inputs*                 |

Dimension literal aliases (`Time = Literal["time"]`, `Speed = Literal["speed"]`, etc.)
are also exported for use in `Q[Time]`-style annotations elsewhere.

### `ScalarFloat` and `float_converter`

Dimensionless scalar fields (e.g. eccentricity, sin/cos of angles) accept a wide union
of input types via `ScalarFloat` and normalize them to bare `Float[jax.Array, ""]` at
storage time using `float_converter`:

```python
class KeplerianBody(eqx.Module):
    eccentricity: ScalarFloat = eqx.field(converter=float_converter)
```

`float_converter` calls `ustrip(AllowValue, "", x)`, which strips units from a
dimensionless `Q` or passes through plain scalars, always producing a 0-d JAX
array.

### Annotation semantics

Field annotations describe the **accepted input type**, not necessarily the stored type.
When a field has a `converter`, the stored type is whatever the converter returns. For
example, `eccentricity: ScalarFloat` accepts `float`, `int`, `jax.Array`, or a
dimensionless `Q`, but after `float_converter` the stored value is always
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

Setting the environment variable `HARV_NO_TYPECHECK` skips the hooks entirely. It
exists for the benchmark harness (`benchmarks/`, see `docs/running-benchmarks.md`):
beartype decorates Python-level functions, and the hot paths are called inside
`jax.vmap` under `eqx.filter_jit`, so the checks run once at *trace* time. That does
not affect warm timings, but it does inflate the first-call compile cost the
benchmarks report. Leave it unset everywhere else -- the test suite wants the checks.

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
├── custom_types.py          # Unit-dimension Literal aliases + Batch* type aliases
├── distributions.py         # QuantityDistribution (QD) unit-aware wrapper
├── data/                    # Observation data classes + stack/indicator helpers
│   ├── datasets.py          # AbstractData, GaiaAstrometryData, RVData
│   ├── containers.py        # SystemData, SourceData
│   └── helpers.py           # stack_datasets, build_indicator_matrix
├── kepler/                  # Orbit mechanics (JAX)
│   ├── orbits.py            # Low-level building blocks and orbit functions
│   ├── body.py              # KeplerianBody
│   ├── orientation.py       # KeplerianOrientation + Thiele-Innes
│   ├── nbody_system.py      # AbstractNBodySystem, TwoBodySystem
│   └── constants.py         # G, c
├── models/                  # Component models (likelihood + parameterization)
│   ├── parameterizations/    # Parameter declarations and design matrices
│   │   ├── _base.py         # AbstractParameterization base class
│   │   ├── rv.py            # StandardRV, EcoswEsinwRV
│   │   ├── gaia.py          # StandardGaiaAstrometry, ThieleInnesGaiaAstrometry
│   │   └── fourier.py       # FourierRV, FourierGaiaAstrometry (Kepler-free)
│   ├── component.py         # AbstractComponentModel (marginalization, numpyro)
│   ├── rv.py                # RVModel (final)
│   ├── astrometry.py        # GaiaAstrometryModel (final)
│   ├── joint.py             # JointModel (composition of components)
│   └── _helpers.py          # PriorDist, LinearPriorCallable, _needs_explicit_sampling
├── extensions/              # Pluggable model modifiers
│   ├── base.py              # ParamInfo, AbstractExtension
│   ├── jitter.py            # Jitter (excess variance)
│   ├── trend.py             # MonomialTrend
│   ├── multi_survey.py      # MultiSurveyOffset
│   └── gp.py                # GP (Gaussian Process covariance)
├── samplers/
│   ├── base.py              # AbstractSampler (shared base)
│   ├── rejection_prior.py   # HarvPrior
│   ├── custom_priors.py     # PeriodDependentKPrior, _make_log_period_prior
│   ├── rejection.py         # RejectionSampler
│   ├── numpyro.py           # NumpyroSampler (MCMC with warm-start)
│   └── samples.py           # Samples container
├── periodogram/             # Periodogram-informed interim period priors
│   ├── grid.py              # frequency_grid
│   ├── core.py              # periodogram(), PeriodogramResult
│   ├── distribution.py      # LogGridDensity
│   ├── priors.py            # tempered_period_prior, peak_period_prior, attach_ln_pint
│   └── io.py                # save_period_prior / load_period_prior
├── plot.py                  # get_t_grid and plotting utilities
└── simulate/                # Synthetic data generators
    ├── rv.py                # simulate_rv_sb1_data, simulate_rv_multisurv_data
    ├── astrometry.py        # simulate_gaia_epoch_astrometry
    ├── scanlaw.py           # Gaia scanning law utilities
    └── source.py            # Source motion models (for simulation)
```

______________________________________________________________________

## Data layer (`harv.data`)

### `AbstractData`

The root base class for all observational datasets. Carries a `time: Q["time"]`
array (barycentric TCB) and an optional keyword-only `t_ref` reference epoch
(defaults to the mean observation time via `__check_init__`). Subclasses add the
observed quantities and their uncertainties. Declares abstract class variables
`_obs_name` and `_err_name` that point to the observation and error field names.

### `GaiaAstrometryData`

`GaiaAstrometryData` (via `AbstractAstrometryData`) stores the Gaia epoch astrometry
for a single source:

| Field             | Units         | Description                             |
| ----------------- | ------------- | --------------------------------------- |
| `time`            | time          | Barycentric observation times           |
| `al_position`     | angle (mas)   | Along-scan position residuals           |
| `al_position_err` | angle (mas)   | Per-observation 1σ uncertainties        |
| `scan_angle`      | angle (rad)   | Scan angle ψ of Gaia's field of view    |
| `parallax_factor` | dimensionless | AL parallax factor H_ϖ(t)               |
| `t_ref`           | time          | Reference epoch (defaults to mean time) |

The along-scan model is (see §Gaia astrometry likelihood), following the Gaia local plane
coordinate convention (Lindegren & Bastian, GAIA-C3-TN-LU-LL-061-08, Eqs. 4 & 6):

```
y_AL(t) = α₀ sin(θ) + δ₀ cos(θ)
         + (μ_α sin(θ) + μ_δ cos(θ)) · (t − t_ref)
         + ϖ · H_ϖ(t)
         + a · [(B X + G Y) sin(θ) + (A X + F Y) cos(θ)]
```

where θ is the position angle of the scan, A, B, F, G are Thiele-Innes constants that
encode the orbit orientation, f is the true anomaly, and X, Y are the Thiele-Innes
orbital coordinates:

```
X = (1 − e²) cos(f) / (1 + e cos(f))      [= cos(E) − e]
Y = (1 − e²) sin(f) / (1 + e cos(f))      [= √(1−e²) sin(E)]
```

`GaiaAstrometryData` has a `plot(ax, **kwargs)` method that renders the observations
as error-bars on the given matplotlib `Axes`. Default style: black markers with grey
error bars; all keyword arguments are forwarded to `ax.errorbar()` and override the
defaults.

### `RVData`

`RVData` stores RV observations from a single instrument:

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

`RVData` has a `plot(ax, *, rv_unit=None, add_labels=True, relative_to_t_ref=False, phase_fold=None, **kwargs)` method that renders the observations as error-bars on the
given matplotlib `Axes`. Default style: black markers with grey error bars; all keyword
arguments are forwarded to `ax.errorbar()` and override the defaults.

- `phase_fold`: a `Q["time"]` period. When provided, the x-axis shows
  `(time - t_ref) / phase_fold mod 1` (orbital phase in \[0, 1)) instead of absolute
  time. Mutually exclusive with `relative_to_t_ref`.

### Indexing data objects

All concrete `AbstractData` subclasses (`RVData`, `GaiaAstrometryData`) support
integer and slice indexing to extract a subset of observations:

```python
data[0]      # first observation — returns a length-1 RVData (1-d shape preserved)
data[:10]    # first 10 observations
data[mask]   # boolean mask
```

Fields whose shape matches the observation count are sliced; scalar fields (`t_ref`)
are passed through unchanged. Integer keys are promoted to length-1 slices so all
arrays remain 1-d.

### `SourceData`

`SourceData` is a named dictionary of datasets for a single source. It is the natural
container for multi-instrument observations and for combined astrometry + RV analyses:

```python
data = SourceData(
    gaia=GaiaAstrometryData(...),
    keck=RVData(...),
    espresso=RVData(...),
)
```

Each dataset is accessed by name (`data["keck"]`). `SourceData` provides
`get_datasets_by_type(dtype)`, `keys()`, `values()`, and `items()` for iteration,
plus a `plot(ax=None, *, add_legend=True, color_cycler=None, **kwargs)` method that
inherits the shared implementation on `AbstractDatasetContainer`. `SourceData.plot()`
raises `TypeError` if the contained datasets are not all the same concrete type, since
overlaying heterogeneous datasets (e.g. RV in km/s and astrometry in mas) on one axes
is meaningless. Use `get_datasets_by_type(...)` to filter first if needed.

**Important:** `SourceData` is for heterogeneous or multi-instrument data for a
*single stellar photocenter*. It is *not* the right container for SB2 systems (see
§`SystemData` for SB2).

### `SystemData`

`SystemData(eqx.Module)` is a generic named-component container for multi-body
systems. Each component holds a `DatasetType` (e.g. `RVData`,
`GaiaAstrometryData`) representing observations of a distinct physical body or
photocenter in a gravitationally bound system.

```python
data = SystemData(
    primary=RVData(time_1, rv_1, rv_err_1),
    secondary=RVData(time_2, rv_2, rv_err_2),
)
data["primary"]   # → RVData
len(data)         # → 2
```

Components are passed as keyword arguments; the number and names are
user-defined (not restricted to "primary"/"secondary").

`SystemData` provides the same dict-like interface as `SourceData`:
`__getitem__`, `keys()`, `values()`, `items()`, `get_datasets_by_type(dtype)`,
`plot(...)` (inherited; no homogeneity check is needed because the constructor
already enforces one), plus:

- `t_ref` — delegates to the first component's `t_ref`
- `_get_obs()` — concatenates observations across all components (key order)
- `_get_obs_err()` — concatenates uncertainties across all components (key order)

`SystemData` is explicitly *not* a `SourceData`. The components measure different
physical bodies (e.g. anti-phase RV motion in an SB2), not the same source through
different instruments. Eventually `SystemData` and `SourceData` will compose:
per-component spectroscopy in a `SystemData` alongside unresolved photocenter
astrometry in `SourceData` or a standalone `GaiaAstrometryData`.

### Helper functions

- `stack_datasets(datasets: dict[str, AbstractData]) -> AbstractData` — concatenates
  multiple datasets of the same type into a single stacked dataset. Scalar fields
  like `t_ref` are recomputed from the concatenated time array via `__check_init__`.

- `build_indicator_matrix(datasets: dict[str, AbstractData], reference: str) -> tuple[AbstractData, jax.Array | None, tuple[str, ...] | None]` — stacks datasets
  and builds an indicator matrix for multi-survey data. Returns
  `(stacked_data, indicator_matrix, non_reference_instrument_names)`. The indicator
  matrix has shape `(n_obs_total, n_non_reference)` with 1s marking observations
  from each non-reference instrument.

______________________________________________________________________

## Kepler mechanics (`harv.kepler`)

### Shared building blocks (`harv.kepler.orbits`)

Core orbit computation functions used by `harv.kepler`, `harv.models`, and
`harv.simulate`. All three consumers call these building blocks instead of duplicating
the math.

`mean_anomaly` and `true_anomaly_from_mean` accept and return `Q` objects
so callers never need to strip units themselves:

- `mean_anomaly(dt: BatchQTime, period: ScalarQTime) -> BatchQAngle` — `M = 2π · dt / period`
- `true_anomaly_from_mean(M: BatchQAngle, eccentricity: ScalarFloat) -> (sin f, cos f)` — solve Kepler's equation

`rv_shape` and `thiele_innes_ABFG` remain pure functions on raw JAX arrays
or dimensionless `Q` objects, because their inputs are always already dimensionless at every call site:

- `rv_shape(sin_f, cos_f, eccentricity, arg_peri)` — RV shape function: cos(ω+f) + e·cos(ω)
- `thiele_innes_ABFG(cos_ω, sin_ω, cos_Ω, sin_Ω, cos_i)` — unit Thiele-Innes constants (a=1)

Orbital-element conversions translate between equivalent element sets. They accept
and return `Q` objects, and back the parameterization-conversion machinery (see
"Parameterization conversion"):

- `campbell_from_thiele_innes(A, B, F, G)` — invert physical Thiele-Innes constants
  to Campbell elements `(semi_major_axis, arg_peri, lon_asc_node, cos_i)`; adopts the
  `cos_i ≥ 0` convention and wraps angles into `[0, 2π)`.
- `thiele_innes_from_campbell(semi_major_axis, arg_peri, lon_asc_node, cos_i)` — the
  forward direction, returning physical Thiele-Innes constants `(A, B, F, G)`.
- `ecc_omega_from_ecosw_esinw(ecosw, esinw)` — `(e, ω) = (√(ecosw²+esinw²), atan2(esinw, ecosw))`.
- `ecosw_esinw_from_ecc_omega(eccentricity, arg_peri)` — `(e·cos ω, e·sin ω)`.

Higher-level convenience functions compose these building blocks:

- `compute_true_anomaly_components(time, period, eccentricity, t_peri)` — returns (sin f, cos f) at given times
- `rv_at_times(times, period, eccentricity, t_peri, arg_peri, rv_semiamp, v_sys)` — evaluates the full RV model
- `astrometric_orbit_at_times(times, period, eccentricity, t_peri, arg_peri, cos_i, lon_asc_node, semi_major_axis)` — returns (Δra, Δdec) offsets

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
in 3D, accounting for the orbit orientation. Both accept `BatchQTime` and return
`BatchVec3QLength` / `BatchVec3QSpeed` respectively. Alternative constructors:

- `from_masses(period, e, m_total, m_body, t_peri)` — uses Kepler's 3rd law to
  derive the barycentric semi-major axis from the total system mass and this body's mass.
- `get_mass(m_total)` — returns the body mass using Kepler's 3rd law.

`KeplerianBody` is the *physical* orbit model. The models layer uses its own
lighter-weight parameterizations (see "Parameterizations") that are shaped to the
specific inference problem.

### `TwoBodySystem`

Combines a primary mass with a `KeplerianBody` companion. Derives total mass and
companion mass from Kepler's third law. Provides barycentric and relative
positions/velocities for both components via `position_barycentric(time, body_idx)`,
`position_relative(time)`, `velocity_barycentric(time, body_idx)`, and
`velocity_relative(time)`.

### Mass functions (`harv.kepler.masses`)

Pure, unit-aware functions that turn posterior orbital elements into physical
masses and physical orbit sizes. They are shape-agnostic (scalar or batched
inputs) and `jax.jit` / `jax.vmap` friendly. Used by the `Samples` derived
quantities (see "`Samples` container").

- `binary_mass_function(period, rv_semiamp, eccentricity) -> Q["mass"]` —
  `f(m) = P K^3 (1 - e^2)^{3/2} / (2 pi G) = m_2^3 sin^3 i / (m_1 + m_2)^2`,
  returned in `Msun`.
- `astrometric_mass_function(a_physical, period) -> Q["mass"]` —
  `f(m) = 4 pi^2 a^3 / (G P^2)`, returned in `Msun`. With `a` the primary's
  barycentric (photocentre) orbit size this equals `m_2^3 / (m_1 + m_2)^2`
  (dark/faint-companion assumption).
- `companion_mass_from_mass_function(mass_function, m1, sini=1.0) -> Q["mass"]`
  — solves `m_2^3 sin^3 i / (m_1 + m_2)^2 = f` for `m_2` by bisection;
  `sini=1` yields the minimum companion mass.
- `semi_major_axis_physical(a_angular, parallax) -> Q["length"]` — physical
  semi-major axis `a = (a_angular / parallax)` in `AU`.

______________________________________________________________________

## Layered architecture overview

The package follows a layered separation of concerns:

1. **Parameterizations** (`harv.models.parameterizations`) -- declare parameter
   names, units, roles (linear / nonlinear), and build design matrices. This is
   the single source of truth for what parameters a model has.

1. **Component models** (`harv.models.component`, `harv.models.extensions`,
   `harv.models.rv`, `harv.models.astrometry`) -- combine data + parameterization +
   extensions,evaluate log-likelihoods (marginalized or explicit), generate numpyro
   models. Extensions are pluggable modifiers that add parameters and/or alter the
   design matrix or covariance (jitter, trends, offsets, GP).

1. **Composition** (`harv.models.joint`) -- `JointModel` composes multiple
   component models with shared orbital parameters.

1. **Samplers** (`harv.samplers`) -- thin wrappers that draw prior samples,
   evaluate model log-probs, and perform rejection / MCMC. Samplers do not
   hardcode parameter names; they discover them from the model and prior.

______________________________________________________________________

## Parameter metadata (`harv.models.extensions.base.ParamInfo`)

`ParamInfo(eqx.Module)` is a frozen descriptor for a single model parameter:

| Field    | Type   | Default | Description                                    |
| -------- | ------ | ------- | ---------------------------------------------- |
| `name`   | `str`  | --      | Parameter name (must not contain `"."`)        |
| `unit`   | `str`  | --      | Physical unit string (e.g. `"day"`, `"km/s"`)  |
| `linear` | `bool` | `False` | Whether the parameter enters the design matrix |

Parameter names must not contain `"."` -- dots are reserved for
`JointModel` component-qualified keys (e.g. `"rv.jitter"`).

______________________________________________________________________

## Parameterizations (`harv.models.parameterizations`)

### `AbstractParameterization`

An `eqx.Module` that declares which parameters a model uses and how to
build the design matrix. Subclasses implement:

- `params() -> tuple[ParamInfo, ...]` -- all parameter descriptors
  (nonlinear first, then linear).
- `design_matrix(sin_f, cos_f, ..., nl_values)` -- build the design matrix
  from true-anomaly components and unit-stripped nonlinear values.
- `default_prior(**kwargs) -> HarvPrior` -- return a `HarvPrior` with
  sensible default distributions for the parameters this parameterization
  declares.  The signature (which scale arguments are accepted) is
  parameterization-specific.  The base-class definition raises
  `NotImplementedError`; every concrete `@final` parameterization overrides it.

Derived convenience methods:

- `nonlinear_params()` -- filter to nonlinear entries.
- `linear_params()` -- filter to linear entries.

### `StandardRV`

Standard RV parameterization: `(period, eccentricity, phase_peri, arg_peri, rv_semiamp, v_sys)`.

- Nonlinear: `period`, `eccentricity`, `phase_peri`, `arg_peri`.
- Linear: `rv_semiamp`, `v_sys`.
- Design matrix shape: `(n_obs, 2)` with columns `[rv_shape(t), 1]`.

Also provides `eccentricity(nl_values)` and `strip_nl_for_design(nl_values)`.

`default_prior(*, period_min, period_max, sigma_K0, sigma_v0, P0=Q(1, "yr"), **kwargs)`
returns a `HarvPrior` with:

- `period`: `LogUniform(period_min, period_max)` wrapped in `QD`
- `eccentricity`: `Beta(0.867, 3.03)` (Kipping 2013)
- `phase_peri`: `Uniform(0, 1)`
- `arg_peri`: `Uniform(0, 2π)`
- `rv_semiamp` linear prior: `PeriodDependentKPrior(sigma_K0, P0)` — a callable
  that scales the K prior with period and eccentricity
- `v_sys` linear prior: `QD(Normal(0, sigma_v0), unit)`

Any nonlinear or linear prior can be overridden by name via `**kwargs`.

### `EcoswEsinwRV`

Alternative RV parameterization using `e*cos(omega)` and `e*sin(omega)`:

- Nonlinear: `period`, `ecosw`, `esinw`, `phase_peri`.
- Linear: `rv_semiamp`, `v_sys`.
- Design matrix shape: `(n_obs, 2)` -- same columns, different internal derivation.

This parameterization has better sampling geometry for low eccentricities.

`default_prior(*, period_min, period_max, sigma_K0, sigma_v0, P0=Q(1, "yr"), **kwargs)`
returns a `HarvPrior` with the same period / `phase_peri` / linear (`rv_semiamp`,
`v_sys`) priors as `StandardRV.default_prior`, plus:

- `ecosw`: `Uniform(-1, 1)`
- `esinw`: `Uniform(-1, 1)`

Independent `Uniform(-1, 1)` priors on `ecosw` and `esinw` do **not** match the
implicit prior under `e ~ Kipping(0.867, 3.03)` × `omega ~ Uniform(0, 2π)`.
This is the simplest sensible default for this parameterization; users wanting a
matched prior should sample with `StandardRV` and convert (or override via
`**kwargs`).

**The default prior admits unbound orbits.** `default_prior` puts independent
`Uniform(-1, 1)` priors on `ecosw` and `esinw`, whose support is a *square*, while a
bound orbit requires the *unit disk* (`e = sqrt(ecosw² + esinw²) < 1`). About 21% of
draws (`1 - π/4`) land outside it with `e >= 1`, where the default `rv_semiamp` prior's
`(1 - e²)^(-1/2)` is `NaN`. Those draws must be rejected: pass
`ignore_non_finite=True`, or a single `NaN` propagates through the `max` reduction and
leaves `max_log_likelihood` and every evidence statistic `NaN`. See
`docs/sharp-bits.md`.

### `StandardGaiaAstrometry`

Standard Gaia epoch-astrometry parameterization:

- Nonlinear: `period`, `eccentricity`, `phase_peri`, `arg_peri`, `lon_asc_node`, `cos_i`.
- Linear: `ra0`, `dec0`, `pmra`, `pmdec`, `parallax`, `semi_major_axis`.
- Design matrix shape: `(n_obs, 6)` following Holl et al. (2022), Appendix A.

The design matrix columns are
`[sin(psi), cos(psi), sin(psi)*dt, cos(psi)*dt, H_parallax, TI_orbit]`
where the Thiele-Innes orbital element combines the (A, B, F, G) constants
with the X, Y orbital coordinates.

`default_prior(*, period_min, period_max, sigma_a0, sigma_parallax, sigma_pos,
sigma_vtan, P0=Q(1, "yr"), **kwargs)` returns a `HarvPrior` with:

- `period`, `eccentricity`, `phase_peri`, `arg_peri`: same defaults as
  `StandardRV.default_prior`
- `cos_i`: `Uniform(-1, 1)`
- `lon_asc_node`: `Uniform(0, 2π)`
- `semi_major_axis`: `PeriodDependentSemiMajorAxisPrior(sigma_a0, P0)` — a
  callable that scales the semi-major axis prior with period and parallax
- `parallax`: `QD(HalfNormal(sigma_parallax), "mas")` — explicitly sampled
  (not marginalized) by default, because the Gaia catalog parallax is derived
  from the same epoch data
- `ra0`, `dec0`: `QD(Normal(0, sigma_pos), "mas")`
- `pmra`, `pmdec`: `ParallaxDependentProperMotionPrior(sigma_v0=sigma_vtan)` —
  a callable that scales the proper-motion prior with parallax, keeping the
  prior fixed in velocity space

Any nonlinear or linear prior can be overridden by name via `**kwargs`. When a
linear prior is supplied directly (e.g. `parallax=QD(...)`), the corresponding
scale argument (`sigma_parallax`, etc.) must be omitted — passing both raises
`TypeError`.

### `ThieleInnesGaiaAstrometry`

Alternative Gaia parameterization that moves the four Thiele-Innes constants
`(A, B, F, G)` from the nonlinear to the linear parameter set, reducing the
nonlinear space from 6-D to 3-D.  This is the approach described in Hsieh et al.
("Astrometric Orbit Fitting with Marginalization over Linear Parameters").

- Nonlinear: `period`, `eccentricity`, `phase_peri`.
- Linear: `ra0`, `dec0`, `pmra`, `pmdec`, `parallax`, `ti_A`, `ti_B`, `ti_F`, `ti_G`.
- Design matrix shape: `(n_obs, 9)`.

By default a Jacobian correction is applied: a flat prior on the Thiele-Innes
constants is not equivalent to a flat prior on the physical Campbell elements
`(a_0, ω, Ω, cos i)`.  The zeroth-order correction (evaluated at the conditional-mean
TI constants following Hsieh et al.) multiplies the marginal likelihood by the factor
`(a_0 + δ_a)^{-m} (sin²i + δ_s)^{-1}`, where `m = 3` for a uniform prior on `a_0`
and `m = 4` for a log-uniform prior.

The correction can be disabled with `apply_jacobian_correction=False`, which makes
`linear_log_prior_correction` return `None` (no correction) — appropriate when the
priors are genuinely intended to be flat in the Thiele-Innes constants.

Constructor parameters:

| Parameter                   | Type           | Default | Description                                                                         |
| --------------------------- | -------------- | ------- | ----------------------------------------------------------------------------------- |
| `a_floor`                   | `float \| None` | `None`  | Floor on `a_0` (in obs units, e.g. mas).  **Required when the correction is on.**   |
| `sin2i_floor`               | `float \| None` | `None`  | Floor on `sin²i` for the Jacobian denominator.  Falls back to `0.01` when `None`.   |
| `log_uniform_in_a`          | `bool \| None`  | `None`  | Use log-uniform prior on `a_0` (`m=4`).  Falls back to `False` when `None`.          |
| `apply_jacobian_correction` | `bool`         | `True`  | Whether to apply the Jacobian correction.                                           |

Validation (enforced in `__check_init__`):

- When `apply_jacobian_correction=True`, `a_floor` must be supplied (non-`None`);
  `sin2i_floor` and `log_uniform_in_a` are optional and fall back to their defaults.
- When `apply_jacobian_correction=False`, none of `a_floor`, `sin2i_floor`, or
  `log_uniform_in_a` may be supplied — they must all be left as `None`.

The recommended constructor is `ThieleInnesGaiaAstrometry.from_data(data)`, which
sets `a_floor = Med(σ_AL) / sqrt(N)` automatically.  Pass
`from_data(data, apply_jacobian_correction=False)` to construct a correction-free
parameterization without deriving `a_floor`.

After sampling with this parameterization, convert the Thiele-Innes linear parameters
to Campbell elements via
`samples.convert_parameterization(source=ThieleInnesGaiaAstrometry(...), target=StandardGaiaAstrometry())`
or the convenience wrapper `samples.thiele_innes_to_campbell()`.

**Limitation**: the RV forward model is not linear in `(A, B, F, G)`, so joint
RV+astrometry fits must use `StandardGaiaAstrometry`.

`default_prior(*, period_min, period_max, sigma_a0, sigma_parallax, sigma_pos,
sigma_vtan, P0=Q(1, "yr"), **kwargs)` returns a `HarvPrior` with:

- Nonlinear: `period` (log-uniform), `eccentricity` (Kipping 2013),
  `phase_peri` (`Uniform(0, 1)`).
- Linear: `ra0`, `dec0`, `pmra`, `pmdec`, `parallax` -- same defaults as
  `StandardGaiaAstrometry.default_prior`.
- Linear (TI constants): `ti_A`, `ti_B`, `ti_F`, `ti_G` -- each gets a
  `PeriodDependentSemiMajorAxisPrior(sigma_a0, P0)` callable, mirroring the
  default on `StandardGaiaAstrometry.semi_major_axis`.  The four TI constants
  are linear projections of the angular semi-major axis onto the (RA, Dec)
  sky plane modulated by `sin`/`cos` of the orientation angles, so each is
  bounded by `a_0` and shares its scale.  Using a parallax-dependent prior is
  also necessary for unit consistency (TI constants are angular; `sigma_a0`
  is a length).

The Jacobian correction (`apply_jacobian_correction=True`) restores the
correct posterior under a flat-Campbell-elements prior.

### `FourierRV` and `FourierGaiaAstrometry` (Kepler-free)

Two **Kepler-free** parameterizations replace the Keplerian orbit with a
truncated Fourier series in the mean longitude `M = 2π(t − t_ref)/P` whose
coefficients are all *linear*. The only nonlinear parameter is `period`: the
periastron phase is absorbed into each `(cos, sin)` amplitude pair, and
eccentricity distortion of the orbit shape is absorbed by the higher
harmonics. **No Kepler solve occurs** — `RVModel` / `GaiaAstrometryModel`
dispatch (at trace time, on the parameterization type) to a mean-longitude
branch instead of `_solve_kepler`.

`n_terms: int` is a **static** field; the parameter list is computed from it
(the same pattern as `MonomialTrend.order` / `MultiSurveyOffset`). `n_terms =
0` is the valid **null (no-signal) model**, which the periodogram uses as its
base model.

- `FourierRV(n_terms=H)` — nonlinear: `period`. Linear: `cos_amp_k`,
  `sin_amp_k` for `k = 1..H` (unit kind `speed`), plus `v_sys`. Design matrix
  `(n_obs, 2H + 1)`: `[cos(kM), sin(kM)] for k = 1..H` plus a constant column.
- `FourierGaiaAstrometry(n_terms=H)` — nonlinear: `period`. Linear: the same
  five astrometric-solution parameters as `StandardGaiaAstrometry` (`ra0`,
  `dec0`, `pmra`, `pmdec`, `parallax`), plus per harmonic the Thiele-Innes-like
  amplitudes `ti_A_k`, `ti_B_k`, `ti_F_k`, `ti_G_k`. Design matrix
  `(n_obs, 5 + 4H)`; per harmonic the four columns are
  `[cos(kM)·cosψ, cos(kM)·sinψ, sin(kM)·cosψ, sin(kM)·sinψ]` — the
  circular-orbit Thiele-Innes structure (i.e. `ThieleInnesGaiaAstrometry` at
  `e = 0`).

`default_prior(...)` requires **explicit** scales — `sigma_amp` (applied to
every harmonic amplitude; individual amplitudes may be overridden by name) plus
`sigma_v0` (RV) or `sigma_pos` / `sigma_pm` / `sigma_parallax` (Gaia). There is
deliberately **no data-driven default**: nothing in these classes inspects the
data. In particular there is **no centering**, so the `v_sys` / `ra0`/`dec0`
priors must be appropriate for the data's actual offsets. Every Gaia linear
prior is a plain Gaussian (including `parallax`, a zero-mean nuisance by
default) so the model marginalizes analytically; override `parallax=` with a
catalog-informed `Normal` when known.

These are first-class parameterizations: extensions that add linear columns
(`MultiSurveyOffset`, `MonomialTrend`) and the `RejectionSampler` work as
usual. They exist primarily to drive the periodogram through the standard
model/likelihood machinery (see "Periodogram and interim period priors").
Being Kepler-free they carry no orbital elements, so orbital-element-specific
analysis raises cleanly: `Samples` from these parameterizations do not
advertise the derived `t_peri` key (it requires `phase_peri`), and
`binary_mass_function` / `companion_mass` / `convert_parameterization` /
Gaia sky-orbit plotting are not applicable.

### Parameter naming convention

All parameter names follow the rule: **use the standard descriptive name; abbreviate
only when the abbreviation is itself a recognized domain term.** Examples:

| Parameter name    | Rationale                                                                         |
| ----------------- | --------------------------------------------------------------------------------- |
| `period`          | Full word -- unambiguous                                                          |
| `eccentricity`    | Full word -- unambiguous                                                          |
| `phase_peri`      | Descriptive compound                                                              |
| `arg_peri`        | `arg` is the standard abbreviation for *argument* in orbital mechanics            |
| `rv_semiamp`      | `rv` is universally recognized; avoids ambiguity with astrometric semi-major axis |
| `v_sys`           | $v\_\\text{sys}$ is the standard notation for systemic velocity                   |
| `pmra`, `pmdec`   | `pm` is the standard abbreviation for *proper motion*                             |
| `ra0`, `dec0`     | `ra` and `dec` are standard coordinate abbreviations                              |
| `semi_major_axis` | Full descriptive name -- no universally short form                                |
| `cos_i`           | Stores cosine of inclination directly (prior is uniform in `cos_i`)               |
| `lon_asc_node`    | Descriptive; `lon` abbreviates *longitude*                                        |

**Physics symbols vs parameter names:** In mathematical descriptions (equations,
docstrings explaining the model), the physics symbols $K$ (semi-amplitude) and $v_0$
(systemic velocity) are standard and should be used. The API-level parameter names
(`rv_semiamp`, `v_sys`) appear in function signatures, dict keys, and struct fields.

### The `period` convention

The period prior is typically a `dist.LogUniform(period_min, period_max)` wrapped in a
`QD` to carry the unit. At sampling time, the sampler converts period draws from the
prior's unit to the data's time unit before constructing parameter values.

### `phase_peri` vs `t_peri`

Models use `phase_peri = t_peri / period` (dimensionless, range 0-1) rather than an
absolute `t_peri`. This decouples the phase from the period scale, simplifies the
prior (uniform on [0, 1]), and avoids the need to specify a reference epoch in the
prior. `Samples` exposes a derived `"t_peri"` key that reconstructs the absolute time
as `phase_peri * period + t_ref`.

### Parameterization conversion

Parameter values can be converted between supported parameterizations without
re-running the sampler.

The standalone helper
`harv.samplers.convert_parameterization(nonlinear, linear, *, source, target)`
converts two parameter dictionaries and returns new `(nonlinear, linear)`
dictionaries in the target representation.

`Samples.convert_parameterization(source=..., target=...)` wraps the same logic and
returns a new `Samples`, preserving `metadata`, `data_type`, and
`linear_extension_names`.

The first implementation supports **single-component** parameterizations only:

- RV: `StandardRV <-> EcoswEsinwRV`
- Gaia astrometry: `StandardGaiaAstrometry <-> ThieleInnesGaiaAstrometry`

Any extra parameters not declared by the source parameterization (for example,
extension parameters like jitter or polynomial-trend coefficients) are preserved
unchanged. Joint / namespaced sample dicts are out of scope for this first pass and
must raise a clear error.

### `Samples.thiele_innes_to_campbell()`

When sampling with `ThieleInnesGaiaAstrometry`, the posterior `Samples` object carries
the Thiele-Innes constants `ti_A, ti_B, ti_F, ti_G` as linear parameters.
`samples.thiele_innes_to_campbell()` is a convenience wrapper around
`Samples.convert_parameterization(...)` that converts them to the physical Campbell
elements `semi_major_axis, arg_peri, lon_asc_node, cos_i` using the standard inversion:

```
u = (A²+B²+F²+G²) / 2
v = A·G − B·F
a_0 = sqrt(u + sqrt(max(u² − v², 0)))
ω + Ω = atan2(B − F, A + G)
ω − Ω = atan2(−B − F, A − G)
cos i = |v / a_0²|    # cos_i ≥ 0 convention
```

The method returns a new `Samples` with the TI constants replaced by the Campbell
elements. If no TI constants are present, it is a no-op. The 2-fold degeneracy
inherent in pure astrometry (face-on reflections) means `cos_i` is not unique; the
convention `cos_i ≥ 0` is adopted.

______________________________________________________________________

## Extensions (`harv.models.extensions`)

### `AbstractExtension`

An `eqx.Module` base class providing three inference-time hooks:

1. `extra_params() -> tuple[ParamInfo, ...]` -- declare new parameters
   (nonlinear and/or linear). **Required** (abstract).
1. `modify_design_matrix(X, data, nl_values) -> jax.Array` -- append columns
   to the design matrix. Default: passthrough.
1. `modify_covariance(cov, data, nl_values) -> jax.Array` -- modify the data
   covariance (diagonal 1-d or full 2-d). Default: passthrough.

Extensions compose -- a model applies them in order, so earlier extensions'
columns appear before later ones in the design matrix.

Plot-specific behavior is handled privately by plotting helpers rather than by
the public extension base API.

### `Jitter`

Declares one nonlinear parameter (`jitter`). Adds `jitter**2` to the diagonal
of the covariance via `modify_covariance`. Works on both 1-d (diagonal) and
2-d (full) covariance representations.

```python
from harv.models.extensions import Jitter
ext = Jitter(param_unit="km/s")
```

### `MonomialTrend`

Appends monomial trend columns to the design matrix:

- **RV** (`astrometry=False`): columns `(t - t_ref)^k` for `k = 1..order`.
- **Astrometry** (`astrometry=True`): two columns per order
  `sin(psi) * dt^(k+1)` and `cos(psi) * dt^(k+1)`, with exponent `k+1` to
  avoid degeneracy with the base proper-motion columns.

Trend column names: `trend_1`, `trend_2`, ... (RV) or
`trend_ra_1`, `trend_dec_1`, ... (astrometry). All are linear parameters.

### `MultiSurveyOffset`

Stores a pre-computed indicator matrix and appends it as extra linear-parameter
columns. Each column corresponds to a non-reference instrument.

```python
from harv.models.extensions import MultiSurveyOffset
ext = MultiSurveyOffset(indicator_matrix, ("espresso", "keck"), "km/s")
```

### `GP`

Gaussian Process covariance extension. Adds the kernel matrix `K(t, t')` to
the observation covariance, enabling correlated-noise modeling while preserving
compatibility with the linear marginalization framework.

```python
from harv.models.extensions import GP, ParamInfo
gp = GP(
    kernel_builder=lambda hp: hp["gp_amp"] ** 2 * tinygp.kernels.ExpSquared(hp["gp_scale"]),
    hyperparams=(
        ParamInfo("gp_amp", "km/s"),
        ParamInfo("gp_scale", "day"),
    ),
    time_unit="day",
)
```

Requires `tinygp` (optional dependency).

______________________________________________________________________

## Component models (`harv.models`)

### `AbstractComponentModel`

The abstract base class for single-data-type models. Combines data,
parameterization, extensions, and linear prior into a single object that
evaluates the log-likelihood and generates numpyro models.

Subclasses must implement:

- `_param_infos() -> tuple[ParamInfo, ...]` -- all parameter descriptors.
- `_base_design_matrix(nl_values) -> jax.Array` -- base design matrix.
- `_strip_obs() -> (obs, obs_err)` -- unit-stripped observation arrays.
- `_obs_unit() -> str` -- observation unit string.

Subclasses declare these fields:

- `data` -- observation data.
- `parameterization` -- declares parameter names and design matrix.
- `linear_prior: dict | None` -- per-parameter priors for marginalization.
- `extensions: tuple` -- model extensions.

The base class methods `log_prob`, `sample_conditional_linear`, and
`numpyro_model` accept an optional `data` keyword that defaults to
`self.data` when `None`. Concrete subclasses do not need to override these
methods just to pass their data.

**Derived queries:**

- `_all_linear_names()` / `_all_nonlinear_names()` -- from `_param_infos()`.
- `_base_nonlinear_names()` -- base parameterization nonlinear names only
  (excludes extension params like jitter).
- `_auto_marginalized_names()` -- classifies linear priors: Gaussian ->
  marginalize, non-Gaussian -> explicit.
- `_linear_param_units()` -- map of linear parameter name to unit string.

### Linear prior classification (auto mode)

When `log_prob(values)` is called with only a flat dict of values (no explicit
`marginalized_names`), the model auto-classifies which linear params to
marginalize based on `linear_prior`:

| Prior type                    | Classification | Treatment                         |
| ----------------------------- | -------------- | --------------------------------- |
| `dist.Normal` or `QD(Normal)` | Gaussian       | Analytically marginalized         |
| `LinearPriorCallable`         | Callable       | Called, result marginalized       |
| `dist.Delta` or `QD(Delta)`   | Fixed          | Treated as explicit (value fixed) |
| `dist.HalfNormal`, etc.       | Non-Gaussian   | Sampled explicitly alongside NL   |

Non-Gaussian linear priors (e.g. `HalfNormal` for parallax) must have their
values present in the `values` dict alongside the nonlinear parameters. The
model extracts them automatically in auto mode.

### The `LinearPriorCallable` contract

A linear prior may be given as a callable, so that the prior on a linear parameter
can depend on the values of the nonlinear parameters. The contract is:

```python
LinearPriorCallable = Callable[[dict[str, Any]], QuantityDistribution | dist.Normal]
```

The callable receives a **plain dict** keyed by bare parameter name and must return a
`Normal` (bare, or wrapped in a `QuantityDistribution` to declare its unit). The dict
contains:

- every **nonlinear** parameter value sampled so far (`period`, `eccentricity`, …),
- every **explicit** (non-marginalized) **linear** parameter value sampled so far, and
- `eccentricity`, even under a parameterization that does not carry it as a
  parameter, when the parameterization can derive one.

The third bullet exists because callables like `PeriodDependentKPrior` are written
against the standard parameter names. `EcoswEsinwRV` carries `(ecosw, esinw)` instead,
so without this the default `rv_semiamp` prior could not be evaluated at all. The value
comes from `AbstractParameterization.derived_eccentricity(nl_values)`, which returns
`None` by default -- covering both parameterizations that already carry `eccentricity`
(nothing to derive) and the Kepler-free Fourier bases (no eccentricity exists) -- and is
overridden by `EcoswEsinwRV`. Shared priors in a `JointModel` are the one exception:
components need not agree on a parameterization, so no derivation is attempted there.

The second bullet is why `parallax` is readable by callable priors: with the default
`HalfNormal` prior it is classified non-Gaussian, so it is sampled explicitly rather
than analytically marginalized. If a user overrides it with a `Normal` prior it becomes
marginalized, disappears from the dict, and the parallax-dependent priors raise
`KeyError` with an explanatory message.

Values that carry units are `unxt.Q`-wrapped, so a callable can do
`ustrip("", params["period"] / self.P0)` without assuming a unit; dimensionless values
(e.g. `eccentricity`) are bare arrays. In a `JointModel`, component-specific values are
additionally reachable under their qualified `"component.param"` key.

Callables are resolved once per likelihood evaluation, inside the sampler trace, so they
must be JAX-traceable: no Python branching on parameter *values*. Branching on dict
*keys* (as the `parallax` guard above does) is fine — that is static structure.

### Three `log_prob` calling conventions

1. **Auto mode** (recommended): `model.log_prob(values)` where `values` is a
   flat dict containing nonlinear params and any explicit-linear params.
1. **Manual marginalization**: pass `marginalized_names` to control exactly
   which linear params to marginalize.
1. **Explicit evaluation**: pass `linear_values` without `marginalized_names`.

### `chi_squared`

`chi_squared(nl_values, linear_values, data) -> jax.Array` returns the
goodness-of-fit statistic `χ² = rᵀ C⁻¹ r` for one fully-specified parameter set,
where `r = y_obs - X y` is the residual and `C` is the extension-modified
observation covariance (so jitter inflation and GP covariances are included).
Unlike `log_prob`, it does not marginalize the linear parameters. It backs
`Samples.chi2` / `Samples.reduced_chi2`.

### `RVModel`

`@final` concrete model for radial velocity data. Models are **pure templates**:
no data or linear prior are stored as fields. Both are passed at call time.

| Field              | Type                         | Default        |
| ------------------ | ---------------------------- | -------------- |
| `parameterization` | `StandardRV \| EcoswEsinwRV` | `StandardRV()` |
| `extensions`       | `tuple`                      | `()`           |

### `GaiaAstrometryModel`

`@final` concrete model for Gaia epoch astrometry. Models are **pure templates**.

| Field              | Type                       | Default                    |
| ------------------ | -------------------------- | -------------------------- |
| `data`             | `GaiaAstrometryData`       | (required)                 |
| `parameterization` | `AbstractParameterization` | `StandardGaiaAstrometry()` |
| `linear_prior`     | `dict \| None`             | `None`                     |
| `extensions`       | `tuple`                    | `()`                       |

Overrides `_linear_param_units()` because astrometric linear params have mixed
units (mas vs mas/yr for proper motions).

### `JointModel`

Composes multiple `AbstractComponentModel` instances with shared orbital parameters.

| Field           | Type                                | Default    |
| --------------- | ----------------------------------- | ---------- |
| `components`    | `dict[str, AbstractComponentModel]` | (required) |
| `shared_params` | `tuple[str, ...]`                   | (required) |

**Factory classmethods:**

- `JointModel.for_sb2(prior, *, extensions=(), shared_params=None, shared_linear_params=None)` --
  build an SB2 `JointModel` from a `default_sb2_prior` prior. Automatically routes
  component-qualified linear-prior keys such as `"primary.rv_semiamp"` / `"secondary.rv_semiamp"`
  (or whatever names are in `component_names`) to the respective component models and declares all
  other linear-prior keys (e.g. `v_sys`) as shared. Defaults `shared_params` to
  `("period", "eccentricity", "phase_peri", "arg_peri")`.
- `JointModel.for_rv_and_gaia(components, *, shared_params=None)` -- build a
  JointModel for combined RV + Gaia astrometry. Same default shared_params.
  Defaults `shared_linear_params=()` — all linear priors must be fully qualified.

**Parameter namespacing:** Shared orbital params use bare names (`"period"`).
Component-specific nonlinear params use `"component_name.param_name"` convention
(e.g. `"rv.jitter"`). Linear params are per-component by default. Names listed
in `JointModel.shared_linear_params` are shared across all components: the
prior under that name must match across components, and the parameter appears
once at the top level of `sample_conditional_linear` and in flattened sample
output (no component namespacing). For SB2, the default
`shared_linear_params=("v_sys",)` ensures correct joint marginalization of the
systemic velocity.

**Key methods:**

- `log_prob(nl_values, data, *, linear_prior=None)` -- splits flat dict into per-component
  dicts, routes explicit linear values, sums component log-likelihoods.
- `sample_conditional_linear(nl_values, key, data, *, linear_prior=None)` -- returns
  `dict[str, dict[str, jax.Array]]` keyed by component name.
- `numpyro_model(nonlinear_priors, data, linear_prior, *, marginalized=True)` -- builds
  a joint numpyro model.
  **Explicit linear routing:** Non-Gaussian linear priors (e.g. HalfNormal parallax)
  are sampled alongside nonlinear params and appear in the flat `nl_values` dict.
  `JointModel._route_explicit_linear` copies them to the correct component's dict.

### Numpyro model generation

Each component model has a `numpyro_model(nonlinear_priors, *, marginalized=True)`
method that returns a no-argument callable suitable for `numpyro.infer.MCMC`:

- **Marginalized** (`marginalized=True`): samples nonlinear params via
  `numpyro.sample`, samples non-Gaussian linear params explicitly, then
  calls `model.log_prob(values)` in auto mode so that Gaussian linear params
  are analytically marginalized.
- **Full** (`marginalized=False`): samples all params (nonlinear + linear).
  Gaussian linear params are sampled jointly from their MVN; non-Gaussian
  ones are sampled individually.

`JointModel.numpyro_model` composes per-component log-probs with shared
nonlinear sampling.

### `QuantityDistribution` / `QD`

`QuantityDistribution` (alias `QD`, in `harv.distributions`) pairs a numpyro
distribution with the physical unit of its samples:

```python
from harv.distributions import QD  # or: from harv import QD

# Scalar (period in days):
qd = QD(dist.LogUniform(50., 2000.), "day")
sample = qd.sample(key)  # → Q(array, "day")

# Multivariate (mixed units):
qd = QD(
    dist.MultivariateNormal(loc=jnp.zeros(6), ...),
    ("mas", "mas", "mas/yr", "mas/yr", "mas", "mas"),
)
```

### `PeriodDependentKPrior`

`PeriodDependentKPrior` (in `harv.models.priors.custom_priors`) implements `LinearPriorCallable`.
It computes a period- and eccentricity-dependent scale for the RV semi-amplitude
prior, following the Joker's default:

```
σ_K(P, e) = σ_{K,0} · (P / P₀)^{-1/3} · (1 - e²)^{-1/2}
```

This keeps the prior approximately constant in companion mass at fixed primary mass.
`__call__` receives a `dict[str, Any]` containing `"period"` and `"eccentricity"` (see
[The `LinearPriorCallable` contract](#the-linearpriorcallable-contract)) and returns a
`QD(dist.Normal(0, σ_K_stripped), unit)`.

Fields:

- `sigma_K0: Q["speed"]` — scale at reference period
- `P0: Q["time"]` — reference period

### `PeriodDependentSemiMajorAxisPrior`

`PeriodDependentSemiMajorAxisPrior` (in `harv.models.priors.custom_priors`) implements
`LinearPriorCallable`. It computes a period- and parallax-dependent scale for the
astrometric semi-major axis prior:

```
σ_a(P, ϖ) = σ_{a,0} · (P / P₀)^{2/3} · ϖ
```

where `σ_{a,0}` is in physical length units (e.g. AU) and `ϖ` is the parallax in
mas. Since 1 AU at 1 mas parallax subtends 1 mas, the product gives the angular
semi-major axis scale in mas.

This keeps the prior approximately constant in companion mass at fixed primary mass
and accounts for the distance dependence of angular semi-major axis. Unlike the RV
amplitude, there is **no eccentricity dependence**.

`__call__` receives a `dict[str, Any]` containing `"period"`, `"eccentricity"`, and
`"parallax"` (parallax is available because it is explicitly sampled by default) and
returns `QD(dist.Normal(0, σ_a_stripped), "mas")`.  Raises `KeyError` if `parallax` has
been analytically marginalized away.

Fields:

- `sigma_a0: Q["length"]` — semi-major axis scale at reference period (e.g. AU)
- `P0: Q["time"]` — reference period

### `ParallaxDependentProperMotionPrior`

`ParallaxDependentProperMotionPrior` (in `harv.models.priors.custom_priors`) implements
`LinearPriorCallable`. It computes a parallax-dependent scale for the proper motion
prior, keeping the prior fixed in velocity space:

```
σ_μ(ϖ) = σ_{v,0} [AU/yr] · ϖ
```

where `σ_{v,0}` is converted from the user-supplied velocity units to AU/yr
internally, and `ϖ` is the parallax. Since 1 AU/yr at 1 mas parallax corresponds
to 1 mas/yr of proper motion, the product gives the angular proper motion scale in
the same angular unit as the parallax per year.

This keeps the velocity prior constant across distances — a source at larger
distance (smaller parallax) gets a proportionally smaller proper motion prior scale.

`__call__` receives a `dict[str, Any]` containing `"parallax"` (parallax is available
because it is explicitly sampled by default) and returns
`QD(dist.Normal(0, σ_μ), parallax_unit + "/yr")`.  Raises `KeyError` if `parallax` has
been analytically marginalized away.

Fields:

- `sigma_v0: Q["speed"]` — velocity dispersion scale (e.g. km/s)

______________________________________________________________________

## Prior (`harv.samplers.HarvPrior`)

`HarvPrior` holds numpyro distributions over all nonlinear parameters and a
per-parameter linear prior. It is an `eqx.Module`.

### Fields

| Field               | Type                                           | Description                                                |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| `nonlinear_priors`  | `dict[str, PriorDist]`                         | Nonlinear parameter priors                                 |
| `linear_prior`      | `LinearPriorDist`                              | Per-parameter linear priors                                |
| `offsets` parameter | `dict[str, QD \| None] \| None` (factory only) | Offset priors; non-ref entries merged into `linear_prior`  |
| `extension_priors`  | `dict[str, PriorDist]` (KW_ONLY, default `{}`) | Priors for extension params (jitter, GP hyperparams, etc.) |

### Constructing a prior

To build a default prior, call `default_prior(...)` on a concrete parameterization
instance.  Each parameterization owns its required scale-argument signature; see
the [Parameterizations](#parameterizations-harvmodelsparameterizations) section
for the per-parameterization defaults and signatures.

```python
import harv.models as hm

# RV
prior = hm.StandardRV().default_prior(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
)

# Alternative RV parameterization
prior = hm.EcoswEsinwRV().default_prior(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
)

# Gaia astrometry
prior = hm.StandardGaiaAstrometry().default_prior(
    period_min=Q(100, "day"),
    period_max=Q(3000, "day"),
    sigma_a0=Q(5.0, "AU"),
    sigma_parallax=Q(10.0, "mas"),
    sigma_pos=Q(100.0, "mas"),
    sigma_vtan=Q(50.0, "km/s"),
)
```

These factories do **not** provide parameter-less defaults — the user must supply
at minimum the period bounds and the relevant amplitude scale, since those depend
on the science case (binary stars, compact objects, and exoplanets have very
different characteristic scales and timescales).

Direct `__init__` construction is always supported for fully custom configurations.

Per-parameter overrides flow through `**kwargs` (e.g.
`hm.StandardRV().default_prior(..., eccentricity=dist.Uniform(0, 0.5))`).  Names not
declared by the parameterization land in `extension_priors` for resolution at
sampling time against the model's declared extension parameters.

Parallax is classified as explicit automatically (because `HalfNormal` cannot be
analytically marginalized).  For exoplanet searches where the catalog parallax
is trustworthy, override with a `Normal` prior and set
`marginalized_names=("parallax", ...)` on the sampler.

#### `default_sb2_prior` (module-level)

```python
from harv.samplers import default_sb2_prior

default_sb2_prior(
    *,
    period_min: Q["time"],     # required
    period_max: Q["time"],     # required
    sigma_K0: Q["speed"],      # required — RV amplitude scale
    sigma_v0: Q["speed"],      # required — systemic velocity scale
    P0: Q["time"] = Q(1.0, "yr"),
    component_names: tuple[str, str] = ("primary", "secondary"),
    **kwargs,          # per-parameter or extension prior overrides (e.g. jitter=QD(...))
) -> HarvPrior
```

A module-level factory (not a classmethod on `HarvPrior`) because SB2 is a
joint composition of two `StandardRV` components rather than a single
parameterization.  Pairs naturally with `JointModel.for_sb2(prior=...)`.

Same orbital defaults as `default_rv` but with linear parameters keyed by
component name:

- `{component_names[0]}.rv_semiamp`, `{component_names[1]}.rv_semiamp`: both use
  `PeriodDependentKPrior(sigma_K0, P0)`
- `v_sys`: `QD(Normal(0, sigma_v0), unit)` (shared across components)

### Multi-survey RV offsets

When multiple instruments observe the same star, their zero-points may differ by an
additive offset.  Pass non-reference offset priors as `**kwargs` keyed by instrument
name to `StandardRV().default_prior(...)`; the reference instrument's offset is
absorbed by `v_sys`.

```python
import harv.models as hm

prior = hm.StandardRV().default_prior(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    espresso=QD(dist.Normal(0, 5.0), "km/s"),  # offset for non-reference inst.
)
assert "espresso" in prior.extension_priors
```

The offsets are additional linear parameters appended to the design matrix by a
`MultiSurveyOffset` extension via `indicator_matrix`. Because they are in
`linear_prior`, they are passed directly to the model — no manual merging needed.

### Jitter (excess variance)

Jitter adds excess variance in quadrature to the observation errors:

$$\\sigma\_\\mathrm{eff} = \\sqrt{\\sigma\_\\mathrm{obs}^2 + s^2}$$

where $s$ is the jitter value sampled from its prior.

Jitter requires **two** things:

1. A prior — supplied as `jitter=QD(...)` in `**kwargs` to any `default_*` method, or
   directly in `extension_priors` when constructing `HarvPrior` manually.
1. A `Jitter` extension — passed as `extensions=(Jitter(param_unit=...), ...)` to the
   sampler. The sampler validates at run time that every declared extension parameter
   has a matching entry in `prior.extension_priors`.

```python
from harv.models.extensions import Jitter
from harv.samplers import RejectionSampler, HarvPrior
from harv.distributions import QD
import numpyro.distributions as dist

# Via default_prior **kwargs:
import harv.models as hm

prior = hm.StandardRV().default_prior(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    jitter=QD(dist.HalfNormal(1.0), "km/s"),  # stored in extension_priors
)
sampler = RejectionSampler(prior, extensions=(Jitter(param_unit="km/s"),))

# Or with explicit HarvPrior construction:
prior = HarvPrior(
    nonlinear_priors=...,
    linear_prior=...,
    extension_priors={"jitter": QD(dist.HalfNormal(1.0), "km/s")},
)
sampler = RejectionSampler(prior, extensions=(Jitter(param_unit="km/s"),))
```

For a `JointModel`, use the component-qualified key in `extension_priors`:

```python
prior = HarvPrior(
    nonlinear_priors=...,
    linear_prior=...,
    extension_priors={"rv.jitter": QD(dist.HalfNormal(1.0), "km/s")},
)
sampler = RejectionSampler(prior, joint)
```

Jitter is implemented via the `Jitter` extension, which adds `jitter**2` to the
observation covariance diagonal.

### `sample_nonlinear`

`sample_nonlinear(key, n_samples) -> dict[str, jax.Array]` draws from all nonlinear
priors. Returns bare JAX arrays regardless of whether the distribution is wrapped in
`QuantityDistribution`. This is a low-level primitive; user code should prefer
`HarvPrior.sample(...)` (below).

### `sample`

```python
prior.sample(
    key: jax.Array,
    n_samples: int,
    *,
    model: AbstractComponentModel | JointModel,
    return_logprobs: bool = False,
    marginalized_names: tuple[str, ...] | None = None,
) -> Samples
```

Draws a *complete* prior sample for ``model`` — base nonlinear params, any
nonlinear extension params declared by ``model.extensions`` (jitter, GP hypers,
…), and any linear params from ``linear_priors`` that are sampled explicitly
rather than analytically marginalized. ``model`` is required because the set of
extension and explicit-linear params is determined by the model template.

`marginalized_names` mirrors `RejectionSampler.marginalized_names` and controls
which linear params are marginalized (and so *not* drawn here). With the default
`None`, every linear param that can be marginalized is (the Gaussian ones),
leaving only the non-Gaussian linear priors explicit. When set, every linear
param not in the set is sampled explicitly — even Gaussian ones. To build a
prior library (in memory or via `make_prior_cache`) for a sampler with a custom
`marginalized_names`, pass the **same** value here so the explicit-linear keys
match what `run_with_samples` expects.

Returns a `Samples` container (the same one produced by the rejection sampler
for posteriors), with units restored from each `QuantityDistribution`. The
`linear` field is empty in the common Gaussian-linear case. `ln_likelihood`
is always `None`; `ln_prior` is populated when `return_logprobs=True`, summing
the nonlinear (base + extension) prior log-densities — matching the
convention used by `RejectionSampler.run(..., return_logprobs=True)`.

Use `prior.sample(...)` to (a) inspect a draw, (b) hand a pre-computed library
to `RejectionSampler.run_with_samples(...)`, or (c) seed a chunked HDF5 cache
via `make_prior_cache(...)`.

______________________________________________________________________

## Sampler base (`harv.samplers.base.AbstractSampler`)

Every sampler holds a prior and a model. `AbstractSampler` declares the shared
fields (`prior`, `model`, `marginalized_names`) and `get_extensions()`. Concrete
samplers (`RejectionSampler`, `NumpyroSampler`) add algorithm-specific fields (e.g.
`batch_size` on `RejectionSampler`) and implement their own `run()`. The
concrete samplers are marked `@final` per the project's abstract-final pattern.

Models are **pure templates** — they hold no data or linear prior. Both are passed
at call time (`run(data, ...)`) so the same model instance can be reused across
different datasets.

Constructor: `Sampler(prior, model)` where `model` is a fully-built
`AbstractComponentModel` or `JointModel`.

______________________________________________________________________

## Rejection sampler (`harv.samplers.rejection.RejectionSampler`)

Implements the rejection sampling algorithm from
[Price-Whelan et al. 2017](https://arxiv.org/abs/1701.08160) (The Joker). The core
idea: because the likelihood is analytically marginalized over linear parameters, it
can be evaluated cheaply for millions of nonlinear prior samples, making rejection
sampling efficient.

### Fields

| Field                | Type                                   | Description                                                  |
| -------------------- | -------------------------------------- | ------------------------------------------------------------ |
| `prior`              | `HarvPrior`                       | Prior distributions for sampling                             |
| `model`              | `AbstractComponentModel \| JointModel` | Model template (no data or linear prior stored)              |
| `marginalized_names` | `tuple[str, ...] \| None`              | Optional subset of linear params to analytically marginalize |
| `batch_size`         | `int` (static)                         | Samples vmapped at once (default: 100,000)                   |

`get_extensions()` walks the attached model: returns `model.extensions` for a
single component model, or `dict[component_name, tuple[Extension, ...]]` for a
`JointModel` (preserving per-component associations like `"primary.jitter"` vs
`"secondary.jitter"`). The same method is inherited by `NumpyroSampler`.

### Algorithm

1. **Prior sampling.** Draw `n_prior_samples` from the nonlinear priors in
   `HarvPrior`. Also samples any non-Gaussian explicit linear params and
   jitter parameters from their priors.

1. **Likelihood evaluation** (batched). For each batch of `batch_size` samples,
   wrap unit-bearing parameters as `Quantity` objects and evaluate
   `jax.vmap(model.log_prob)(values)`. If `marginalized_names` is not set, the
   model auto-classifies which linear params to marginalize from its own
   `linear_prior`. If `marginalized_names` is set on the sampler, that subset is
   passed through explicitly.
   Evaluated via `jax.lax.fori_loop` to bound memory.

1. **Selection.** One of two mutually exclusive policies:

   - **Rejection** (default). Normalize weights to `max` and accept samples
     where `Uniform() < weight`. The number of survivors depends on how
     constraining the data are.
   - **Top-K by weight** (`top_k=k`). Take the `k` prior samples with the
     largest importance weights, via `jax.lax.top_k` on the log-likelihood
     array, ordered by decreasing weight. Ranking by log-likelihood is
     identical to ranking by log-weight (the normalization is a per-run
     constant), so no `logsumexp` is needed and there is no intermediate
     normalized array to become `NaN` when every likelihood is non-finite.
     Non-finite log-likelihoods sort last and carry zero weight regardless of
     `ignore_non_finite`. The output length is exactly `k` for every dataset —
     see §Top-K selection.

1. **Cap** (rejection only). If `max_posterior_samples` is set and more than
   that many samples were accepted, randomly subsample to
   `max_posterior_samples` via `jax.random.choice(..., replace=False)`. This
   happens *before* linear-parameter sampling so the `jax.vmap` shape in step 5
   is stable across calls — important for population-scale loops where the same
   sampler is reused over many datasets. The `top_k` policy makes that shape
   static by construction and so needs no equivalent step.

1. **Linear parameter sampling.** For each (kept) accepted nonlinear sample,
   call `model.sample_conditional_linear(values, key)` to draw the marginalized
   linear parameters from their conditional posterior, honoring the sampler's
   `marginalized_names` override when present.

1. **Return** a `Samples` object.

### `run` method

```python
sampler.run(
    data: AbstractData | AbstractDatasetContainer,
    *,
    n_prior_samples: int,
    max_posterior_samples: int | None = None,
    top_k: int | None = None,
    seed: int = 0,
    ignore_non_finite: bool = False,
    return_logprobs: bool = False,
    return_evidence_stats: bool = False,
) -> Samples
```

`data` is the first positional argument and is passed through to `model.log_prob`
at each evaluation. It is validated at the entry point: a single-component model
requires an `AbstractData` subclass (`RVData`, `GaiaAstrometryData`); a `JointModel`
requires an `AbstractDatasetContainer` (`SystemData`, `SourceData`) keyed by
component name (e.g. `SourceData(rv=rv_data, astro=astro_data)`). Passing a bare
`dict` or any other object raises `TypeError`.

- `max_posterior_samples` -- cap on the number of accepted samples returned.
  Mutually exclusive with `top_k`; passing both raises `ValueError`.
- `top_k` -- when set, skip rejection and return exactly `top_k` samples by
  importance weight (see §Top-K selection). Default: `None`.
- `ignore_non_finite` -- when `True`, any `NaN` or infinite log-likelihoods
  are treated as rejected samples by replacing them with `-inf` before the
  rejection step. Default: `False`. On the `top_k` path this is a no-op:
  non-finite log-likelihoods always sort last and carry zero weight.
- `return_logprobs` -- when `True`, the returned `Samples` carries per-sample
  log-probabilities: `ln_likelihood` (the marginal log-likelihood) and
  `ln_prior` (the summed nonlinear-prior log-density). These enable
  `Samples.map_sample()` and the `Samples.ln_posterior` property. Default:
  `False`. Supported by both `RejectionSampler.run` and `NumpyroSampler.run`.
- `return_evidence_stats` -- when `True`, the returned `Samples.metadata`
  carries the prior-Monte-Carlo evidence statistics `logZ_int`,
  `logZ_int_mcse`, `logZ_int_ess`, `max_log_likelihood`, and `n_prior_samples`
  (used by the population-inference reweighting and by
  `Samples.acceptance_diagnostics()`; see "Interpreting acceptance"). The
  under-resolution warning is emitted regardless of this flag. Default:
  `False`.

### Interpreting acceptance

The rejection step accepts each prior draw with probability `exp(L − max L)`,
where `max L` is the maximum marginal log-likelihood **over the drawn prior
samples**. The accepted-sample *count* is therefore only a meaningful posterior
size once `max L` has converged to the true peak `L*`. When the likelihood is
sharply peaked (high SNR, dense sampling), a broad prior may never sample near
the peak: `max L` sits far below `L*`, and the sampler "accepts" a handful of
poor fits simply because it never saw a good one. Concentrating the prior (e.g.
a periodogram-informed period prior) then *finds* the peak, raising `max L` — so
it can report **fewer** accepted samples against the correct (higher) bar even
though it resolved the posterior far better. **Comparing raw accept counts
across priors is misleading until `max_log_likelihood` has converged.**

The reliable diagnostic is the evidence effective sample size
(`logZ_int_ess = (Σ L)² / Σ L²`): the number of prior draws that effectively
contribute to the marginal-likelihood integral. When it is O(1), the integral —
and the `max`-normalization — is dominated by a single draw, so the run is
under-resolved.

- `run(...)` and `run_with_samples(...)` emit a `UserWarning` when
  `logZ_int_ess < 3` (the evidence is dominated by ≲3 effective draws),
  regardless of `return_evidence_stats`. It is a filterable `UserWarning`;
  silence it in population loops via `warnings.catch_warnings`.
- `Samples.acceptance_diagnostics()` (requires `return_evidence_stats=True`)
  returns `{n_prior_samples, n_accepted, evidence_ess, max_log_likelihood,
  logZ_int, well_resolved, message}` for inspection.

**Recommended workflow for peaked likelihoods:** use the rejection sampler
(ideally with a periodogram-informed period prior) to *locate* the mode — check
that `max_log_likelihood` stops rising as `n_prior_samples` increases and across
seeds — then continue with `NumpyroSampler(prior, model).run(data,
init_samples=...)` to draw the posterior. In this regime the rejection stage is
a mode-finder, not a posterior sampler: even with the period pinned, the joint
(eccentricity, phase, `arg_peri`) volume at high SNR is a tiny acceptance target.
  Forced on by `top_k`.
- `return_evidence_stats` -- when `True`, add prior-Monte-Carlo evidence
  statistics to `Samples.metadata`, estimated from the **full** `(M,)`
  log-likelihood array before selection. Default: `False`. Forced on by
  `top_k`. Keys:

  | key | meaning |
  | --- | --- |
  | `logZ_int` | log-evidence, `logsumexp(ln L) - ln M` |
  | `logZ_int_mcse` | delta-method MC standard error on `logZ_int`, `sqrt(max(0, 1/ESS - 1/M))` |
  | `logZ_int_ess` | Kish effective sample size of the importance weights, `(Σ L)² / Σ L²` |
  | `max_log_likelihood` | `max(ln L)` over the library |
  | `n_prior_samples` | library size `M` |

  `logZ_int_ess` is the diagnostic for whether the prior library resolved this
  posterior at all: `ESS ≲ 10` means it did not, and the result is a
  localization rather than a posterior.

### Top-K selection (`top_k`)

Rejection returns a data-dependent number of rows — ~1000 for an unconstrained
system, ~1 for a well-constrained one. Population-scale workflows need a uniform
output: one `run_with_samples` call per system, every call yielding the same
number of rows so the results form a rectangular table. `top_k=k` provides that
by keeping the `k` highest-importance-weight prior draws *with their weights*
instead of accept/rejecting.

The output shape is static in `k`, which removes the recompile-per-acceptance-count
problem at its source: the boolean rejection mask forces a device→host sync and a
fresh trace of the per-sample conditional Gaussian solve for every distinct
acceptance count, which `max_posterior_samples` only works around. A `top_k`
gather is by index, so the `jax.vmap` in the linear-parameter step sees one shape
forever.

`top_k` forces `return_logprobs=True` and `return_evidence_stats=True`, because
`Samples["weight"]` is reconstructed from `ln_likelihood` plus the `logZ_int` and
`n_prior_samples` metadata. It adds one further metadata key:

| key | meaning |
| --- | --- |
| `weight_captured` | `Samples["weight"].sum()` — the fraction of total posterior mass the returned `k` samples capture |

Two diagnostics are reported because they answer different questions, and a
system can pass one while failing the other:

- `logZ_int_ess` — *did the library sample this posterior?*
- `weight_captured` — *was `k` big enough?* ~1.0 means ample; 0.1 means 90% of
  the posterior mass was truncated away.

**The returned samples are weighted, and truncation biases them.** Treating them
as equal-weight posterior draws is wrong, and `Σ w f / Σ w` over a truncated
top-K set is biased whenever `weight_captured` is not close to 1 — however large
`k` is in absolute terms. See `docs/sharp-bits.md`.

Errors: `ValueError` if `top_k` is combined with `max_posterior_samples`, if
`top_k < 1`, or if `top_k` exceeds the prior library size (returning fewer than
`top_k` rows would defeat the fixed-shape contract the caller depends on).

### `batch_size` and GPU support

The `batch_size` field controls how many samples are vmapped at once within a
`fori_loop`. On CPU, the default of 100,000 is appropriate. On GPU, set
`batch_size = n_prior_samples` to let XLA fully utilize the device.

This guidance is measured by the `batch_size` curve in `docs/benchmarks.md`; update
it here if the measurement disagrees. Note that `batch_size` is currently the *only*
device-related knob -- harv contains no device-placement or sharding code (see the
`shard_map` TODO in `harv/samplers/rejection.py`), so the benchmark numbers are the
single-device baseline any future multi-device work is measured against.

### `summary` method

`summary() -> str` returns a plain-ASCII, sectioned-table description of how the
configured sampler will treat each parameter — useful for inspecting a `(prior, model,
marginalized_names)` setup before running. It performs **no sampling** and is
side-effect-free (it suppresses the non-Gaussian-marginalization warning that `run`
emits, since the table itself surfaces that information).

The string contains:

- a header with the model class, the parameterization class (per component for a
  `JointModel`), the active extensions, and a `parameters` line giving the count of
  sampled vs. marginalized parameters;
- a **Nonlinear parameters** table — base orbital params plus any nonlinear extension
  params (marked `(ext)`), which are always sampled explicitly;
- a **Linear parameters** table classifying each linear param as:
  - `marginalized` — analytically integrated out (Gaussian prior, in the effective
    marginalized set);
  - `sampled` — a non-Gaussian linear prior (e.g. a `HalfNormal` parallax) that cannot be
    marginalized and is drawn explicitly;
  - `sampled (could marg.)` — a Gaussian/linear prior that *could* be marginalized but is
    excluded via `marginalized_names`.

Each row shows the prior-distribution type and unit. The classification reuses the same
`effective_linear_prior` / marginalized-name resolution that `run` uses, so the summary
matches the actual run behavior. Print it with `print(sampler.summary())`.

### Pre-computed prior samples (`run_with_samples`)

When the same prior library is reused across many datasets, draw it once with
`HarvPrior.sample(...)` (in memory) or `make_prior_cache(...)` (HDF5) and feed
it back via `run_with_samples`:

```python
sampler.run_with_samples(
    data: AbstractData | AbstractDatasetContainer,
    prior_samples: Samples | str | os.PathLike,
    *,
    max_posterior_samples: int | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    ignore_non_finite: bool = False,
    return_logprobs: bool = False,
    return_evidence_stats: bool = False,
    randomize_prior_order: bool = True,
) -> Samples
```

`prior_samples` dispatches on type:

- `Samples` — in-memory cache returned by `prior.sample(...)`. Requires every
  key the (prior, model) bundle expects to be present; any missing keys raise
  `ValueError` listing them. Extra keys are ignored, so a superset cache (e.g. a
  jitter cache reused by a non-jitter sampler) is reused safely.
- `str | os.PathLike` — path to an HDF5 cache (see `make_prior_cache`). The
  file is streamed `batch_size` rows at a time via contiguous h5py slices, so
  the file may be much larger than RAM. Same key handling as the in-memory
  branch: missing keys raise, extra keys are ignored.

`randomize_prior_order` (HDF5 path only): when `True` (default), batch *order*
is permuted via `np.random.default_rng(seed).permutation(n_batches)`. Each
batch is still a single contiguous h5py slice — no random seeks, no read
amplification. Set to `False` for strictly sequential reads (reproducibility /
debugging).

The other keyword arguments behave exactly as on `run`. This is the intended
entry point for `top_k`: one shared prior library, one call per system, exactly
`top_k` rows out of every call. Top-K selection depends only on the
log-likelihoods, so it selects the same library draws either way and is
unaffected by `randomize_prior_order`.

### Building a prior cache (`make_prior_cache`)

```python
from harv.samplers import make_prior_cache

make_prior_cache(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
    n_samples: int,
    filename: str | os.PathLike,
    *,
    key: jax.Array,
    batch_size: int = 100_000,
    return_logprobs: bool = False,
    marginalized_names: tuple[str, ...] | None = None,
) -> None
```

Writes `n_samples` prior draws to an HDF5 file without materializing more than
`batch_size` rows in memory. Each batch is drawn from an independent subkey
via `jax.random.fold_in(key, i)`, so on-disk row order is i.i.d.

`marginalized_names` is forwarded to `HarvPrior.sample` and must match the
consuming sampler's `marginalized_names` (see above).

The on-disk layout is identical to `Samples.to_hdf5` (see "`to_hdf5` /
`from_hdf5`" below): nonlinear and linear datasets under `nonlinear/` and
`linear/` groups (with `@unit` attrs, each dataset keeping its parameter's
dtype), metadata under `metadata/`, and `ln_prior` as a top-level dataset when
`return_logprobs=True`. A prior cache is therefore loadable directly via
`Samples.from_hdf5(path)` for inspection.

### MCMC sampling (`NumpyroSampler`)

MCMC functionality lives on `NumpyroSampler(prior, model)`. The `run()` method
takes the data and an optional `Samples` warm-start from `RejectionSampler.run()`,
builds a numpyro model automatically from the component model's `numpyro_model()`
method, draws one starting position per chain from the rejection posterior (if
provided), runs MCMC, and returns a new `Samples` object.

```python
mcmc_sampler = NumpyroSampler(prior, model)
mcmc_samples = mcmc_sampler.run(
    data,
    init_samples=rej_samples,
    seed=42,
    num_warmup=500,
    num_samples=1000,
    num_chains=4,
)
```

Both sampler classes own marginalization policy. Set
`marginalized_names=(...)` on `RejectionSampler` or `NumpyroSampler` to request
an explicit subset of linear parameters to marginalize. Any non-Gaussian linear
priors are still sampled explicitly even if they appear in that tuple.

`NumpyroSampler.run` also accepts `return_logprobs` (default `False`). When
`True` the returned `Samples` carries `ln_likelihood` (the marginal
log-likelihood, re-evaluated via `model.log_prob` over the posterior draws) and
`ln_prior` (the summed nonlinear-prior log-density). This requires
`marginalized=True` and no `extra_model`; other configurations raise
`NotImplementedError`.

Two model variants are supported via `marginalized`:

- `marginalized=True` (default): MCMC explores nonlinear subspace only; Gaussian
  linear params are analytically marginalized inside the likelihood, then
  conditionally sampled afterward to populate the returned `Samples`.
- `marginalized=False`: MCMC samples all parameters jointly (nonlinear + linear).

### `NumpyroSampler.optimize(samples, data, *, seed=None, max_passes=10, tol=1e-4) -> Samples`

Refines each input sample to the local posterior MAP using BFGS via
`numpyro.optim.Minimize` (which wraps `jax.scipy.optimize.minimize`) with an
`AutoDelta` guide. Each sample in *samples* is used as a warm start for an
independent BFGS run that maximises `log_prior + marginal_log_likelihood`.
Because `jax.scipy.optimize.minimize` BFGS often quits early when its line
search fails, `optimize` restarts BFGS up to `max_passes` times from the
previous result, breaking when the loss change falls below `tol` and emitting
`UserWarning` if it never does.

Linear parameters at the returned MAP are taken as the conditional posterior
**mean** (equal to the conditional MAP since the conditional is Gaussian), not
a random draw. This matters when the conditional posterior is highly correlated
(e.g. Thiele-Innes constants with sub-orbit data coverage), where a draw can be
many sigma from the mean along degenerate directions and the nonlinear
TI→Campbell conversion would amplify that noise.

The returned `Samples` carries `ln_likelihood` and `ln_prior` re-evaluated at
the optimised points. Particularly useful when `RejectionSampler.run` returns a
single sample inside a posterior mode -- this moves it from a random acceptance
point to the mode peak.

Only the marginalized path is supported (matches `run(marginalized=True)`).
`extra_model` is not supported.

______________________________________________________________________

## `Samples` container (`harv.samplers.samples.Samples`)

Stores the posterior samples returned by `RejectionSampler.run()` or
`NumpyroSampler.run()`.

### Fields

| Field                    | Type                       | Description                                                  |
| ------------------------ | -------------------------- | ------------------------------------------------------------ |
| `nonlinear`              | `dict[str, Q]`             | Nonlinear parameter samples with units                       |
| `linear`                 | `dict[str, Q]`             | Linear parameter samples with units                          |
| `metadata`               | `dict[str, Any]` (static)  | JSON-friendly scalars only — see invariant below             |
| `linear_extension_names` | `tuple[str, ...]` (static) | Linear extension param names (offsets, trends, etc.)         |
| `data_type`              | `str` (static)             | Model class name (e.g. `"RVModel"`, `"GaiaAstrometryModel"`) |
| `ln_likelihood`          | `jax.Array \| None`        | Per-sample marginal log-likelihood (see `return_logprobs`)   |
| `ln_prior`               | `jax.Array \| None`        | Per-sample nonlinear-prior log-density (see `return_logprobs`) |

`ln_likelihood` and `ln_prior` are optional pytree leaves: they are `None`
unless the sampler was run with `return_logprobs=True`. They are carried
through slicing, `wrap_angles`, and `convert_parameterization`, and persisted
by `to_hdf5` / `from_hdf5`.

#### `metadata` invariant

`metadata` is an `eqx.field(static=True)` dict, so its contents are *not*
pytree leaves -- they participate in JIT-cache identity. To keep this
well-defined, the dict must contain only JSON-friendly scalars
(`int` / `float` / `str` / `bool`); a JAX array (including a `unxt.Q`, which
wraps one) placed in a static field triggers equinox's
"A JAX array is being set as static!" warning and breaks the invariant.

Quantity-valued entries are stored in **split form**: the value goes under
`<name>` (as a `float` / `int`) and the unit string under `<name>_unit`.
One convention applies in-memory and on disk -- the samplers produce this
shape, `to_hdf5` writes the dict entries one-for-one as HDF5 attrs, and
`from_hdf5` loads them back the same way. Keys harv writes itself:

- `t_ref` (`float`) + `t_ref_unit` (`str`) -- the reference epoch in the
  source data's time unit.
- `num_chains` (`int`) -- written by `NumpyroSampler.run()`.
- `logZ_int`, `logZ_int_mcse`, `logZ_int_ess`, `max_log_likelihood` (`float`)
  and `n_prior_samples` (`int`) -- written by `RejectionSampler` when
  `return_evidence_stats=True` or `top_k` is set. See §`run` method.
- `weight_captured` (`float`) -- written by `RejectionSampler` when `top_k` is
  set. See §Top-K selection.

For Q-aware reads, use `samples.meta` -- a `collections.abc.Mapping` view
that reassembles `<name>` + `<name>_unit` pairs into `Q` instances on the
fly and hides the `_unit` companions from iteration:

```python
samples.meta["t_ref"]      # Q(0.0, "day")
samples.meta["num_chains"] # 1 (no _unit companion -> bare value)
list(samples.meta)         # ["t_ref", "num_chains"] (no "t_ref_unit")
```

Drop down to `samples.metadata` for raw dict access (e.g. when you need
to construct a new `Samples` with the same metadata).

### Dict-style and index access

`samples["key"]` dispatches to appropriate unit restoration:

- Nonlinear params (`"period"`, `"eccentricity"`, `"phase_peri"`, etc.) → `Q`
  with units
- Linear params (`"rv_semiamp"`, `"v_sys"`, `"ra0"`, etc.) → `Q` with units
- Derived keys:
  - `"log_period"` → dimensionless array (`log10(period in data time units)`)
  - `"t_peri"` → `Q` (derived from `phase_peri * period + t_ref`)
  - `"inclination"` → `Q` in radians (derived from `arccos(cos_i)`)
  - `"binary_mass_function"` → `Q` in `Msun` (present only for RV samples)
  - `"semi_major_axis_AU"` → `Q` in `AU` (present only for astrometry samples
    carrying `semi_major_axis` and `parallax`)
  - `"weight"` → dimensionless array; equivalent to the `weight` property
    below. Deliberately **not** listed by `keys()`, since `keys()` enumerates
    model parameters and drives the default axes of `plot_corner` / `to_arviz`
    and the all-key form of `median()`.

Integer, slice, or array keys return a new `Samples` with all parameter arrays
sliced along the *leading* axis:

```python
samples[0]       # first sample — returns Samples with shape (1,) arrays
samples[:100]    # first 100 samples
samples[mask]    # boolean mask
```

Integer keys are promoted to length-1 slices so all arrays remain at least 1-d.
Static fields (`data_type`, `metadata`, `linear_extension_names`) are passed
through unchanged.

### Extra parameter columns

`Samples.nonlinear` may carry **extra dimensionless columns** beyond the
parameters declared by the model — per-sample derived quantities attached by
tools or user code (e.g. Jacobian factors for population reweighting). Extra
columns behave exactly like parameters: they flow through indexing/slicing,
`pad_and_stack_samples`, and `to_hdf5` / `from_hdf5` unchanged. When stacking,
every input must carry the same key set, so attach extra columns to *every*
per-source `Samples` before stacking.

Reserved extra-column names:

- `ln_pint_period` — the per-sample interim period prior log-density (per unit
  natural-log period), written by `harv.periodogram.attach_ln_pint` (see
  "Periodogram and interim period priors"). The key is exported as
  `harv.periodogram.LN_PINT_PERIOD_KEY`.

Parameter arrays may carry one or more leading batch dimensions -- for example
`(N_stars, K_max)` after [`pad_and_stack_samples`](#stacking-per-entity-samples).
In that case, integer / slice / array indexing slices the leading axis (the
batch axis), and `n_samples` is the trailing-axis length (samples per entity);
the leading shape is exposed via `batch_shape`.

### Stacking per-entity Samples

`harv.samplers.pad_and_stack_samples(samples_list, *, pad_value=float('nan')) -> tuple[Samples, jax.Array]`
combines a sequence of per-entity `Samples` (each with 1-D parameter arrays of
possibly differing length) into one batched `Samples` of shape `(N, K_max)`
plus a `(N, K_max)` boolean mask that is `True` at non-padded positions. All
inputs must share `data_type`, `linear_extension_names`, and the set of
nonlinear / linear keys with matching units per key (mismatches raise
`ValueError`). `ln_likelihood` and `ln_prior` are stacked iff every input
carries them, with `-inf` as the log-space padding sentinel; otherwise the
stacked `Samples` has `None` for those fields. `metadata` is inherited from
the first entry.

### Methods

- `keys() -> list[str]` — nonlinear + linear + derived parameter names
- `n_samples -> int` — number of samples per batch entry (trailing-axis
  length; equals total samples for a flat 1-D `Samples`)
- `batch_shape -> tuple[int, ...]` — leading batch dimensions
  (empty tuple for a flat `Samples`; e.g. `(N_stars,)` after
  `pad_and_stack_samples`)
- `median(key=None)` — median of one key or all keys
- `percentile(key, percentiles=(16, 50, 84))` — compute percentiles
- `summary(params=None)` — dict of statistics (median, mean, std, p16, p84)
- `wrap_angles() -> Samples` — return a new `Samples` enforcing the convention
  `K >= 0`, `a >= 0`. Applied in two steps: (1) negative `rv_semiamp` is flipped
  by shifting `arg_peri` by `pi`, which flips *both* `rv_semiamp` and
  `semi_major_axis`; (2) any `semi_major_axis` still negative afterward is
  flipped by shifting `lon_asc_node` by `pi`, which flips `semi_major_axis`
  alone (`lon_asc_node` does not enter the RV model). Both shifted angles are
  wrapped to `[0, 2*pi)`. A single `arg_peri` shift cannot make both `K` and `a`
  positive when their signs disagree, so the `lon_asc_node` shift is required.
  The orbit predicted by the wrapped sample is identical to the original. No-op
  when `arg_peri` is missing or no entries are negative.
- `convert_parameterization(source=..., target=...) -> Samples` — convert the stored
  parameter values between supported single-component RV or Gaia parameterizations.
  Extra non-base parameters are preserved unchanged; unsupported families or
  namespaced sample dicts raise a clear error.
- `thiele_innes_to_campbell() -> Samples` — convenience wrapper for the Gaia
  `ThieleInnesGaiaAstrometry -> StandardGaiaAstrometry` conversion.
- `to_arviz(params=None)` -- export to `arviz.InferenceData`
- `to_hdf5(filename)` / `from_hdf5(filename)` -- HDF5 persistence
- `plot_corner(params=None, truths=None, **kwargs)` — corner plot via arviz
- `ln_posterior -> jax.Array` — per-sample log-posterior (`ln_prior +
  ln_likelihood`); raises `ValueError` if either was not stored
- `weight -> jax.Array` — per-sample importance weight,
  `exp(ln_likelihood - logsumexp(ln L))`, normalized over the **full** prior
  library. Reconstructed rather than stored: the normalization is
  `logZ_int + ln(n_prior_samples)`, both from the evidence metadata that `top_k`
  / `return_evidence_stats=True` writes. Because the normalization spans the
  whole library, `weight.sum()` is the posterior mass these samples capture and
  is **less than 1** whenever samples were truncated, so expectations need
  `w / w.sum()`. Raises `ValueError` if `ln_likelihood` or the evidence metadata
  is missing, or if the `Samples` is batched — `pad_and_stack_samples` inherits
  metadata from the first entry only, so a stacked normalization would be
  silently wrong for the others. Also reachable as `samples["weight"]`.

#### Sample analysis

Ported from `thejoker.samples_analysis`. Methods that need the observed time
sampling take the `data` object; all support single-component samples only
(namespaced joint-model samples raise a clear error).

- `map_sample(return_index=False) -> Samples` — the maximum a posteriori sample
  (highest `ln_posterior`), as a length-1 `Samples`. Requires `return_logprobs`.
- `acceptance_diagnostics() -> dict` — whether the rejection run resolved the
  posterior (see "Interpreting acceptance"). Requires `return_evidence_stats`.
- `period_unimodal(data) -> bool` — whether the period samples lie in one mode.
- `period_modes(data, n_clusters=2) -> (bool, Q, ndarray)` — K-means clustering
  of `log(period)` into modes (needs the optional `scikit-learn` dependency).
- `max_phase_gap(data) -> ndarray` — largest circular phase-coverage gap, per
  sample.
- `phase_coverage(data, n_bins=10) -> ndarray` — fraction of phase bins occupied.
- `periods_spanned(data) -> ndarray` — number of periods spanned by the data.
- `phase_coverage_per_period(data) -> ndarray` — max observations within one
  period.

#### Goodness of fit

- `chi2(data, model) -> jax.Array` — per-sample :math:`\chi^2` against the data,
  evaluated from the model prediction; see
  `AbstractComponentModel.chi_squared`. The `model` argument is the
  single-component model used for the fit.
- `reduced_chi2(data, model, *, dof=None) -> jax.Array` — per-sample reduced
  :math:`\chi^2` (`chi2 / dof`). `dof` defaults to `n_obs - n_params`, counting
  every fitted parameter (orbital + linear + extension); pass `dof=` to override.

`chi2` differs from the stored `ln_likelihood`, which is the *marginal*
log-likelihood (linear parameters integrated out) rather than a goodness-of-fit
statistic.

#### Derived physical quantities

These wrap the pure functions in `harv.kepler.masses` (see "Mass functions").

- `binary_mass_function() -> Q` — RV binary mass function, in `Msun`.
- `semi_major_axis_AU() -> Q` — physical semi-major axis (`AU`) from the angular
  size and parallax (astrometry samples).
- `companion_mass(m1, *, sini=None) -> Q` — companion mass given the primary
  mass `m1`. RV samples use the binary mass function (default `sini=1`, i.e. the
  minimum companion mass); astrometry samples use the dark-companion astrometric
  mass function and ignore `sini`.
- `minimum_companion_mass(m1) -> Q` — convenience for `companion_mass(m1, sini=1)`.

______________________________________________________________________

## Periodogram and interim period priors (`harv.periodogram`)

The rejection sampler's acceptance rate is dominated by how much prior mass
falls near the data's true period. `harv.periodogram` builds a **per-source
interim period prior** from a periodogram of that source's data: prior mass
concentrates near plausible periods (dramatically higher acceptance at fixed
`n_prior_samples`), while a log-uniform mixture "floor" preserves full period
support so the samplings remain valid for downstream hierarchical inference.

### The Δ log-marginal-likelihood statistic

The periodogram scans **period only**, and owns no likelihood machinery of its
own: it is a thin `jax.vmap` of `model.log_prob(...)` over the period grid,
using the Kepler-free Fourier parameterizations (see "`FourierRV` and
`FourierGaiaAstrometry`"), whose amplitudes are all linear and therefore
analytically marginalized by the standard model path.

```
delta_ln_likelihood(f) = ln L(f) − ln L_base
```

- `ln L(f)` = `model.log_prob({"period": 1/f, ...}, data, linear_priors=...)`
  with `FourierX(n_terms=H)`.
- `ln L_base` = the same call with `FourierX(n_terms=0)` — the null model (RV:
  the constant offset alone; Gaia: the 5-parameter astrometric solution
  alone). It is period-independent, so it is evaluated **once**.

Δ is therefore a per-frequency log Bayes factor of "base + orbit harmonics" vs.
the base model, **under exactly the priors supplied** by the required `prior`
argument. Because the base columns appear in *both* models, their power cancels
in Δ — for Gaia this is what suppresses scan-law / parallax / proper-motion
signal (no spurious peaks at one year or the scan-law periods).

`eccentricity = 0` is adopted inside the periodogram: the Fourier trial model
has no eccentricity, but `e = 0` is passed alongside `period` so that
eccentricity-dependent amplitude priors (e.g. `PeriodDependentKPrior`) resolve
through the standard prior machinery. It is ignored by the Fourier design
matrix.

`n_terms` (default 2) is **capped per dataset** to keep the trial model
overdetermined — at least two observations per linear column, floored at 1 —
emitting a `UserWarning` when reduced. Column counts are derived from the
parameterization and any linear extensions, so extension columns count against
the same budget. This is a correctness safeguard, not just an optimization: on
sparse data an overfit trial model fits almost any trial period, so spurious
alias peaks dominate the periodogram and a prior built from it can *hurt*
acceptance. The cap engages only when the harmonics could not be reliably
estimated anyway; where multi-term genuinely helps (eccentric orbits with
adequate sampling) it does not engage. `PeriodogramResult.n_terms` reports the
effective value used.

**Containers** (`SourceData`, `SystemData`): each dataset's Δ is evaluated on
the shared grid and summed — one periodogram per source; per-dataset Δ are kept
in `PeriodogramResult.per_dataset`. Multiple RV instruments are handled by
passing a `MultiSurveyOffset` extension (offset columns are marginalized in
both models, as with any model).

Other data types raise `NotImplementedError` (2-d absolute/relative astrometry
is future work; see "Planned features").

### `frequency_grid`

```python
frequency_grid(
    data=None, *,
    period_min,               # required; its unit sets the grid unit (1/unit)
    period_max=None,          # default: max_period_factor * t_span
    t_span=None,              # alternative to data (exactly one required)
    samples_per_peak=5,       # oversampling per peak width 1/t_span
    max_period_factor=1.0,
    n_grid=None,              # explicit grid size override
) -> Q["frequency"]           # uniform in frequency, ascending
```

**Cross-source shape stability:** pass the same `(period_min, period_max,
n_grid)` (or one precomputed grid) for every source in a population so the
resulting prior pytree structure is identical and the sampler JIT-compiles
once for all sources.

### `periodogram` and `PeriodogramResult`

```python
periodogram(
    data,                     # RVData | GaiaAstrometryData | container
    frequency_grid=None,      # explicit grid; exclusive with grid kwargs
    *,
    prior,                    # REQUIRED: HarvPrior (or {dataset_name: HarvPrior})
    period_min=None, period_max=None, samples_per_peak=5, n_grid=None,
    n_terms=2,
    extensions=(),            # linear-column extensions (or per-dataset mapping)
) -> PeriodogramResult
```

`PeriodogramResult` is an `eqx.Module` with fields `frequency`,
`delta_ln_likelihood`, `ln_likelihood_base`, `t_span`, `t_ref`, optional
`per_dataset` (per-dataset Δ for container inputs), and static `n_terms`;
plus `period` (property, `1/frequency`), `max_period()`, and `plot(ax=None,
x="period" | "frequency")`.

### Priors are explicit

`prior` is **required** and is an ordinary `HarvPrior` for the Fourier trial
model, normally built by `FourierRV(n_terms=H).default_prior(...)` /
`FourierGaiaAstrometry(n_terms=H).default_prior(...)`. The periodogram makes
**no data-driven prior choices** — it never inspects the data to set a scale,
does no centering, and keeps no table of column names: Δ is a log Bayes factor
under exactly the priors given. Consequences:

- The `v_sys` / `ra0` / `dec0` priors must suit the data's actual offsets
  (there is no centering). Δ becomes invariant to a constant offset only in the
  limit that the offset prior is wide enough to absorb it.
- Amplitude priors may be `LinearPriorCallable`s (e.g. `PeriodDependentKPrior`)
  — they resolve per trial period through the standard prior machinery, with
  `e = 0` adopted, and so intentionally tilt Δ. The Occam factors are constant
  across the grid only for period-independent amplitude priors.
- Extensions adding linear columns (`MultiSurveyOffset`, `MonomialTrend`) are
  passed via `extensions=` and apply to both the trial and base models; their
  priors come from `prior.extension_priors` as usual. Extensions with nonlinear
  parameters (`Jitter`, `GP`) raise `TypeError` — the periodogram can neither
  scan nor marginalize them.
- Prior/data mismatches raise `TypeError`: extra nonlinear priors (e.g. handing
  it a `StandardRV` prior), unknown linear names, or missing linear entries.
  For containers, `prior` may be a `{dataset_name: HarvPrior}` mapping (a
  single prior may be shared when all datasets are of one type).

**Open question (deliberately unsettled).** What amplitude scale to *recommend*
is not settled. A scale comparable to the data RMS under-estimates the true
amplitude in the partial-arc regime (baseline < period) and biases the peak
toward short periods; a broader or period-dependent prior mitigates it, but
choosing a default needs a study against a converged period posterior (dense
rejection / MCMC) across many seeds and regimes. Until then the API requires
the user to choose. See the `TODO` in
`harv.models.parameterizations.fourier`.

### `LogGridDensity`

A numpyro `Distribution` over `x > 0` whose pdf is **piecewise-linear in
`u = ln x`** on fixed knots `(ln_grid, log_density)` — the shared backbone of
both prior builders:

- `log_density` is the *unnormalized* log-density w.r.t. `d(ln x)`;
  normalization is trapezoid-exact. Zero density (`-inf` log-density) knots
  are allowed.
- `log_prob(x)` is the density **per unit x** (same convention as
  `dist.LogUniform`); `log_prob_ln(x) = log_prob(x) + ln x` is the density per
  unit `ln x` and is invariant under a change of x's unit.
- `cdf` / `icdf` are closed-form per segment (piecewise-quadratic CDF;
  "citardauq" quadratic inversion, stable as the slope → 0); `sample` is
  inverse-CDF. All operations are shape-static and jit/vmap-safe; instances
  with equal knot counts share a pytree structure.
- `support` is `constraints.interval(exp(ln_grid[0]), exp(ln_grid[-1]))`, so
  `biject_to` and hence `NumpyroSampler` MCMC continuation work. Caveat: the
  gradient of `log_prob` is discontinuous at the knots (acceptable for NUTS
  in practice).

Wrapped in a `QD` (e.g. `QD(LogGridDensity(...), "day")`) it is a **drop-in
period prior**: pass it via the `period=` override of any `default_prior(...)`
or set `nonlinear_priors["period"]` directly. **No sampler changes are
involved anywhere in this feature.**

### Prior builders

Both builders map a `PeriodogramResult` onto ln-period knots on the requested
domain `[period_min, period_max]` (defaults: the grid range; the domain may
extend beyond the grid, where Δ continues flat at 0 so the tempered prior is
floor-like there), and both **mix in a log-uniform floor of weight `floor`**
(λ, default 0.1). Since a log-uniform is constant in `ln P`, the mixture is
itself a grid density — one distribution class covers everything.

```python
tempered_period_prior(result, *, beta=1.0, floor=0.1,
                      period_min=None, period_max=None, unit=None) -> QD
```

Density per unit ln-period `∝ (1−λ)·exp(β·Δ)/Z + λ·log-uniform`. `beta=0`
reduces to an exact log-uniform; `beta=1` treats the periodogram as a
likelihood times log-uniform. Note that on high-SNR data `exp(Δ)` can be
narrower than the grid spacing; the piecewise-linear density then smears the
peak to roughly one knot spacing (increase `samples_per_peak` to resolve it).

```python
peak_period_prior(result, *, height_drop=10.0, max_peaks=8, peak_width=None,
                  floor=0.1, period_min=None, period_max=None, unit=None) -> QD
```

Amplitude-agnostic alternative: strict local maxima of Δ **within `height_drop`
nats of the global maximum** each get a top-hat in ln-period of full frequency
width `peak_width` (default `1/t_span`) with **equal mass** `1/n_peaks`
regardless of amplitude. The criterion is *relative to the best peak*, so it is
scale-invariant across data types — RV periodograms span hundreds of nats,
while astrometry periodograms (the orbit is a small perturbation on the
marginalized 5-parameter astrometric signal) span only a few; an absolute
threshold would silently reject every astrometry peak. Candidate maxima within
one peak width of a stronger peak are suppressed (real periodograms carry
hundreds of spurious local maxima), and at most `max_peaks` survivors are kept —
bounding the dilution so each peak carries at least `(1−floor)/max_peaks`. The
global maximum always qualifies, so the fallback (a `UserWarning` degrading to
a pure log-uniform on the same knots, preserving the pytree structure) fires
only for a perfectly flat or monotonic periodogram.

Usage:

```python
import harv.periodogram as hp

result = hp.periodogram(data, period_min=Q(2.0, "day"), period_max=Q(2000.0, "day"))
prior = hm.StandardRV().default_prior(
    period=hp.tempered_period_prior(result, beta=1.0, floor=0.1),
    sigma_K0=Q(30.0, "km/s"),
    sigma_v0=Q(10.0, "km/s"),
)
samples = RejectionSampler(prior, RVModel()).run(data, n_prior_samples=100_000)
```

### Hierarchical inference bookkeeping (interim priors)

Per-source interim priors remain valid for Hogg/Myers/Bovy-style population
reweighting: the per-source estimator
`Z_n(α)/Z_int,n ≈ (1/K_n) Σ_k p(θ_nk|α) / p_int,n(θ_nk)` is importance
sampling of each source's integral, and conditioned on the data each source's
interim prior is a fixed, exactly-normalized proposal that is divided out
exactly (its data-dependence does not bias the estimator). Requirements:

1. **Support** — `p_int,n(P) > 0` wherever the population prior can put mass.
   The λ floor guarantees this and bounds the importance weights by `1/λ`
   relative to a log-uniform interim prior. `floor=0` voids the guarantee
   (a `UserWarning` is emitted).
1. **Per-source evaluability** — the reweighting needs `ln p_int,n` at each
   retained sample. `attach_ln_pint(samples, period_prior)` evaluates and
   stores it as the reserved extra column `ln_pint_period` (see "Extra
   parameter columns"), which flows through `pad_and_stack_samples` into the
   population step. It works for any scalar-unit period prior, including
   `QD(LogUniform, ...)` for the classic shared-prior case.
1. **Interim evidence** — the per-source `Z_int,n` from
   `run(..., return_evidence_stats=True)` (`metadata["logZ_int"]`) enters the
   population likelihood as usual; nothing changes with per-source priors.

**Measure convention:** the stored `ln_pint_period` is the log-density **per
unit natural-log period** (`log_prob(P) + ln(P/unit)`), which is invariant
under the prior's time unit. Convert to a density in `log10 P` by adding
`ln(ln 10)`; to a density in P (unit u) by subtracting `ln(P/u)`. Population
densities must be expressed in the same measure before forming weight ratios.

`save_period_prior(file, prior, *, group="interim_period_prior",
metadata=None)` / `load_period_prior(file, *, group=...)` persist the prior
spec (knots + log-densities + unit + scalar provenance attrs) as a small HDF5
group. The group can live in the **same file** as `Samples.to_hdf5` output —
`Samples.from_hdf5` reads only its own groups — so a per-source posterior file
can carry its interim prior alongside the samples. The round trip is exact
(the stored arrays are the constructor arguments).

### Per-source priors vs. the shared prior cache

Per-source interim priors are incompatible with reusing one
`make_prior_cache` library across sources (the period column's distribution
differs per source). The intended path is **per-source on-the-fly prior
sampling** with a shared grid configuration: identical knot counts give an
identical prior pytree structure, so the sampler JIT-compiles once for the
whole population. A cache-resampling utility (replacing the period column of
a shared cache per source) is future work — see "Planned features".

______________________________________________________________________

## Plotting utilities (`harv.plot`)

### `get_t_grid`

`get_t_grid(times: BatchQTime, period: Q["time"])` returns a dense time grid
for plotting orbit curves. The grid spans from `min(times) - span_factor*range/2` to
`max(times) + span_factor*range/2`, with spacing determined by
`period / n_points_per_period`.

### `plot_rv`

```python
plot_rv(
    samples,
    data=None,
    extensions=(),
    *,
    n_samples=128,
    time_grid=None,
    show_signal_components=False,
    phase_fold_median=False,
    apply_median_offsets=True,
    plot_kwargs=None,
    data_plot_kwargs=None,
    extra_err_plot_kwargs=None,
    color_cycler=None,
    ax=None,
    **kwargs,
)
```

Draw posterior RV curves over observed data. Handles multi-instrument offsets,
phase folding, and optional extension contributions (GP conditional mean,
polynomial trend overlays, jitter error bars). When `ax=None` a new figure is
created and returned; otherwise draws into `ax` and returns `None`.

- `samples` -- `Samples` from rejection or MCMC sampling.
- `data` -- `RVData`, `SourceData`, `SystemData`, or `None`. When `None`, only
  posterior orbit curves are drawn (no data points).
- `extensions` -- tuple of `AbstractExtension` instances. `plot_rv()` has
  built-in plotting support for currently supported plot-aware extensions,
  specifically GP conditional-mean overlays, monomial trend overlays in the
  time-domain branch, and jitter-driven error-bar widening.
- `time_grid` -- optional explicit time grid used to evaluate and plot the
  posterior orbit curves. When omitted, `plot_rv()` builds a default dense grid
  from the data baseline (or a one-period phase grid when phase folding).
- `show_signal_components` -- when `True`, switch from plotting the total RV
  model to plotting the Keplerian curve and the combined extension-driven RV
  contribution as separate curves. This decomposition view is only supported
  for time-domain RV plots with observed data.
- `phase_fold_median` -- when `True`, fold to orbital phase using the median-period
  sample. Only the reference orbit is drawn (multiple samples on a phase axis
  defined by one period would be misleading). When plot-aware extensions are
  present, the reference sample's extension contribution is subtracted from the
  data before folding so the Keplerian orbit overlays the phase-folded points.
- `apply_median_offsets` -- when `True` (default), shift non-reference instrument
  data by the posterior median offset parameter.

### `plot_gaia_astrometry`

```python
plot_gaia_astrometry(
    samples,
    data,
    extensions=(),
    *,
    data_plot_kwargs=None,
    sky_orbit_kwargs=None,
    figsize=(10, 5),
    axes=None,
    **kwargs,
)
```

Draw a two-panel goodness-of-fit figure for a **single** Gaia epoch-astrometry
posterior sample. `samples` must contain exactly one sample — select one
beforehand with `samples[i]` (by index) or `samples.map_sample()` (the maximum
a posteriori sample); a `Samples` with any other number of samples raises
`ValueError`.

1. **Sky-projected orbital ellipse** for the sample (delegates to
   `plot_gaia_sky_orbit`), with each Gaia epoch shown as a scan-direction
   segment at the model-predicted photocenter offset.
1. **Along-scan position residual vs time** — the observed `al_position` minus
   the full predicted model for the sample (orbital wobble + parallax + proper
   motion + zero-point), drawn with measurement error bars and a dashed line at
   zero.

When `axes=None`, a new 1x2 figure is created and returned; otherwise draws into
the two given axes `(sky, residual)` and returns `None`.

### `plot_gaia_sky_orbit`

```python
plot_gaia_sky_orbit(
    orbit_params,
    data=None,
    *,
    n_grid=500,
    errorbar_scale=1.0,
    plot_kwargs=None,
    data_plot_kwargs=None,
    ax=None,
    **kwargs,
)
```

Draw a single astrometric photocenter orbit on the sky for one set of orbital
parameters. When `data` is provided, each Gaia epoch is rendered as a short line
segment in the scan direction at the model-predicted photocenter offset, with
half-length equal to `errorbar_scale * al_position_err`.

- `orbit_params` -- dict with keys `period`, `eccentricity`, `t_peri`,
  `arg_peri`, `cos_i`, `lon_asc_node`, `semi_major_axis`. `t_peri` is the
  *absolute* periastron time (i.e. `t_ref + phase_peri * period`).
- `data` -- `GaiaAstrometryData` or `None`.

______________________________________________________________________

## Serialization utilities (`harv.io`)

### `save_sampler` / `load_sampler`

```python
harv.save_sampler(path, sampler)  # -> None
harv.load_sampler(path)           # -> sampler
```

Persist a fully-constructed sampler (prior, parameterization, extensions) to
disk using Python `pickle`. The round-trip preserves all Python objects including
numpyro distribution objects in static pytree fields.

```python
import harv

sampler = harv.RejectionSampler(prior, extensions=(jitter,))
harv.save_sampler("sampler.pkl", sampler)

# Later:
sampler2 = harv.load_sampler("sampler.pkl")
samples = sampler2.run(data, seed=0)
```

______________________________________________________________________

## Simulation utilities (`harv.simulate`)

### `simulate_rv_sb1_data`

Generates a synthetic `RVData` for a single-lined spectroscopic binary.
All orbital parameters have random defaults if not specified. Returns
`(data, true_params)`. Uses NumPy RNG (not JAX) because this is a one-off setup step.

### `simulate_rv_multisurv_data`

Generates a `SourceData` with multiple `RVData` instruments and
per-instrument zero-point offsets. Takes an `instruments` dict mapping instrument
names to their offset (or `None` for the reference instrument). Returns
`(source_data, true_params)`.

### `simulate_gaia_epoch_astrometry`

Generates a synthetic `GaiaAstrometryData` with 5-parameter astrometry plus
Keplerian orbital motion. Includes a simplified (sinusoidal) parallax factor model via
`fake_parallax_factor`. For real Gaia data, the parallax factors come from the Gaia
epoch astrometry tables directly. Returns `(data, true_params)`.

### `GaiaReducedCommandedScanLaw`

Loads and queries the Gaia commanded scanning law (as processed HEALPix-indexed data)
to produce realistic observation times and scan angles for simulation.

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

### Why `eqx.field(static=True)` for metadata fields?

Fields marked `static=True` in equinox are not treated as pytree leaves — they are
compared structurally (by value) when JAX traces a new JIT-compiled function. This is
appropriate for metadata that controls the computation graph (like `batch_size`) or
for strings that have no gradient. Concretely, if `batch_size` changes, JAX re-traces;
if it stays the same, the cached compilation is reused.

______________________________________________________________________

## Planned features and known gaps

### JointModel non-marginalized MCMC

`JointModel.numpyro_model(marginalized=False)` currently raises
`NotImplementedError`. Implementing this requires composing per-component
full-model numpyro builders with shared nonlinear param sampling.

### Batch inference over many datasets

A common population-level workflow is: define a single prior, generate a large library
of prior samples once, then run rejection sampling against many datasets (e.g. thousands
of Gaia sources). The current API creates a separate model and `RejectionSampler` per
dataset, which means:

1. **Redundant prior sampling** -- the same prior draws are regenerated for every
   dataset even though they only depend on the prior, not the data.
1. **JIT retracing** -- if datasets have different numbers of observations (different
   array shapes), JAX recompiles the likelihood evaluation kernel for each new shape.

The planned design separates prior sampling from likelihood evaluation:

- **Prior samples are drawn once** and reused across all datasets.
- **Likelihood evaluation is batched** over datasets, with automatic padding/masking
  to a common observation count so that a single JIT-compiled kernel handles all
  datasets without retracing.
- A high-level entry point handles the padding, batching, rejection step, and
  linear-parameter sampling internally.
- The implementation should support **chunked/batched execution** over datasets to
  control memory usage, and be designed with **multi-device and GPU parallelism** in
  mind (e.g. `jax.pmap` or `jax.experimental.shard_map` over devices).

### Iterative rejection sampling

The Joker's iterative scheme grows the sample batch exponentially until enough
posterior samples are accepted. Useful when the likelihood is very constraining.

### Prior-cache resampling for per-source interim priors

`harv.periodogram` interim period priors are per-source, so they cannot reuse
a single shared `make_prior_cache` library. A planned utility would take a
shared prior cache and, per source, resample/reweight the period column to a
tailored interim prior — combining the cache's one-time prior-draw cost with
per-source period priors.

### Periodogram for 2-d astrometry

`harv.periodogram.periodogram` currently supports `RVData` and
`GaiaAstrometryData` (both 1-d observables). Absolute and relative astrometry
(2-d position time series) will need a 2-d periodogram variant once those data
and model types exist.

### Absolute and relative astrometry

Future data and model types:

- **Absolute astrometry** (RA/Dec timeseries from ground-based or HST observations)
- **Relative astrometry** (separation and position angle from direct imaging)

These will require new parameterization and component model implementations following
the same abstract-final pattern as `RVModel` and `GaiaAstrometryModel`.

### Source motion models (`harv.simulate.source`)

`simulate/source.py` contains an incomplete `AbstractSource` hierarchy for modeling
astrometric source motion (linear proper motion, small-angle approximation,
accelerated motion from a Keplerian companion). Not yet functional.

### Pluggable trend basis

The current `MonomialTrend` extension uses a monomial basis. To support alternative
bases (Chebyshev, B-splines), replace with a `TrendBasis` protocol:

```
class TrendBasis(Protocol):
    n_basis: int
    names: tuple[str, ...]          # one per output column
    def __call__(
        self, times: jax.Array, t_ref: float,
    ) -> jax.Array:                 # (n_obs, n_basis)
        ...
```

**Key contract**: The basis must NOT include a constant column (order 0), since that
role is already filled by `v_sys` / `ra0` / `dec0`.

### JIT compile time (TTFX)

First-call JIT compile time is measured per parameterization by the benchmark
harness (`docs/benchmarks.md`, "First-call compile cost") as `cold - warm` against a
cleared JIT cache. It has not yet been *optimized*. Potential approaches: smaller
pytrees, pre-compilation of hot paths, compile-time caching.

Compile cost is paid once per distinct input shape, so it amortizes to nothing in a
population loop over sources with identical data shapes, and can dominate a one-off
fit.

______________________________________________________________________

## API sketch

The intended user-facing interface for common use cases:

```python
import numpyro.distributions as dist
from unxt import Q
from harv.data import RVData
from harv.distributions import QD
from harv.models import RVModel, GaiaAstrometryModel, JointModel
from harv.models.extensions import Jitter, MultiSurveyOffset
from harv.samplers import NumpyroSampler, HarvPrior, RejectionSampler

# --- Minimal RV-only case ---
import harv.models as hm

data = RVData(time, rv, rv_err)
prior = hm.StandardRV().default_prior(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
)
sampler = RejectionSampler(prior, RVModel())
samples = sampler.run(data, n_prior_samples=500_000)

# With max posterior samples:
samples = sampler.run(data, n_prior_samples=500_000, max_posterior_samples=128)

# --- RV with custom extensions and parameterization ---
from harv.models.parameterizations.rv import EcoswEsinwRV
sampler = RejectionSampler(
    prior,
    RVModel(parameterization=EcoswEsinwRV(), extensions=(Jitter(param_unit="km/s"),)),
)
samples = sampler.run(data, n_prior_samples=500_000)

# --- Periodogram-informed interim period prior ---
import harv.periodogram as hp

# The periodogram runs a Kepler-free Fourier model; its priors are explicit:
fourier_prior = hm.FourierRV(n_terms=2).default_prior(
    period_min=Q(2, "day"),
    period_max=Q(2000, "day"),
    sigma_amp=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
)
result = hp.periodogram(
    data, prior=fourier_prior, period_min=Q(2, "day"), period_max=Q(2000, "day")
)
prior = hm.StandardRV().default_prior(
    period=hp.tempered_period_prior(result, beta=1.0, floor=0.1),  # or peak_period_prior
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
)
samples = RejectionSampler(prior, RVModel()).run(data, n_prior_samples=100_000)
samples = hp.attach_ln_pint(samples, prior.nonlinear_priors["period"])  # for reweighting

# --- Gaia astrometry only ---
prior = hm.StandardGaiaAstrometry().default_prior(
    period_min=Q(0.3, "yr"),
    period_max=Q(10, "yr"),
    sigma_a0=Q(1e3, "AU"),
    sigma_parallax=Q(100.0, "mas"),
    sigma_pos=Q(1e3, "mas"),
    sigma_vtan=Q(200, "km/s"),
)
sampler = RejectionSampler(prior, GaiaAstrometryModel())
samples = sampler.run(gaia_data, n_prior_samples=1_000_000)

# --- Joint astrometry + RV ---
# All linear-prior keys must be qualified ("rv.rv_semiamp", "astro.parallax", etc.)
joint = JointModel.for_rv_and_gaia(
    components={
        "rv": RVModel(extensions=(Jitter(param_unit="km/s"),)),
        "astro": GaiaAstrometryModel(),
    },
)
sampler = RejectionSampler(joint_prior, joint)
samples = sampler.run(
    SourceData(rv=rv_data, astro=gaia_data), n_prior_samples=1_000_000
)

# --- MCMC continuation ---
mcmc_sampler = NumpyroSampler(prior, RVModel())
mcmc_samples = mcmc_sampler.run(
    data, init_samples=samples, num_chains=4, num_warmup=500, num_samples=2000, seed=42,
)

# --- Post-sampling analysis ---
mcmc_samples["period"]          # Quantity in data time units
mcmc_samples["eccentricity"]    # dimensionless array
mcmc_samples.median("rv_semiamp")        # median semi-amplitude
mcmc_samples.summary()          # dict of all statistics
mcmc_samples.plot_corner()                  # arviz corner plot
harv.plot_rv(mcmc_samples, data)                       # RV curve with data overlay
harv.plot_gaia_astrometry(mcmc_samples.map_sample(), data=gaia_data)  # single-sample plot
harv.save_sampler("sampler.pkl", sampler)   # persist sampler
mcmc_samples.to_hdf5("out.h5")             # persistence
```
