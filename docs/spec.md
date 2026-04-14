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

1. **No global state.** Likelihoods close over data; a `Model` combines prior and data;
   random state passes explicitly as JAX key values.

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
| `ScalarQAngularSpeed` | `Real[Q["angular speed"], ""]`                                          | Scalar angular speed quantities               |
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
├── data/                    # Observation data classes + stack/indicator helpers
│   ├── datasets.py          # AbstractData, GaiaAstrometryData, RVData
│   ├── containers.py        # SystemData, SourceData, InputData
│   └── helpers.py           # stack_datasets, build_indicator_matrix
├── distributions.py         # QuantityDistribution (QD) unit-aware wrapper
├── kepler/                  # Orbit mechanics (JAX)
│   ├── orbits.py            # Low-level building blocks and orbit functions
│   ├── body.py              # KeplerianBody
│   ├── orientation.py       # KeplerianOrientation + Thiele-Innes
│   ├── nbody_system.py      # AbstractNBodySystem, TwoBodySystem
│   └── constants.py         # G, c
├── likelihood/              # Log-likelihood evaluators
│   ├── base.py              # AbstractLikelihood[DataT, ParamT]
│   ├── params.py            # Parameter structs (eqx.Module pytrees)
│   ├── helpers.py           # _solve_kepler, _resolve_linear_prior_mvn, LinearPriorCallable
│   ├── rv.py                # RVLikelihood, SB2RVLikelihood
│   ├── gaia_astrometry.py   # GaiaAstrometryLikelihood
│   ├── composite.py         # CompositeLikelihood
│   └── astrometry.py        # Stub: future absolute/relative astrometry
├── model.py              # Model class combining prior + data
├── samplers/
│   ├── rejection_prior.py   # RejectionPrior
│   ├── custom_priors.py     # PeriodDependentKPrior, _make_log_period_prior
│   ├── rejection.py         # RejectionSampler
│   ├── numpyro.py           # NumpyroSampler + numpyro model builders for MCMC
│   └── samples.py           # Samples container + WarmStartMCMC
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

`RVData` has a `plot(ax, **kwargs)` method that renders the observations as error-bars
on the given matplotlib `Axes`. Default style: black markers with grey error bars; all
keyword arguments are forwarded to `ax.errorbar()` and override the defaults.

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
`get_datasets_by_type(dtype)`, `keys()`, `values()`, and `items()` for iteration.

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
plus:

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

Core orbit computation functions used by `harv.kepler`, `harv.likelihood`, and
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

`KeplerianBody` is the *physical* orbit model. The likelihood layer uses its own
lighter-weight parameter structs (see §Parameter structs) that are shaped to the
specific inference problem.

### `TwoBodySystem`

Combines a primary mass with a `KeplerianBody` companion. Derives total mass and
companion mass from Kepler's third law. Provides barycentric and relative
positions/velocities for both components via `position_barycentric(time, body_idx)`,
`position_relative(time)`, `velocity_barycentric(time, body_idx)`, and
`velocity_relative(time)`.

______________________________________________________________________

## Parameter structs (`harv.likelihood.params`)

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
a **marginalized wrapper** (created on-the-fly via `.marginalized()`).

`AbstractParameters` declares 4 nonlinear orbital fields shared by every data type:

| Field          | Type         | Description                          |
| -------------- | ------------ | ------------------------------------ |
| `period`       | `BatchQTime` | Orbital period                       |
| `eccentricity` | `BatchFloat` | Orbital eccentricity                 |
| `phase_peri`   | `BatchFloat` | Fractional phase at perihelion (0–1) |
| `arg_peri`     | `BatchFloat` | Argument of pericenter               |

`nonlinear_param_names` and `linear_param_names` are `ClassVar[tuple[str, ...]]` on
each subclass. `linear_param_names` is declared explicitly on each concrete class.
`nonlinear_param_names` is computed automatically by `__init_subclass__` from the set
difference of all dataclass fields minus `linear_param_names`. The auto-computation
calls `dataclasses.dataclass(cls)` to force early field registration before equinox
processes the class.

**Full parameter structs:**

| Struct                     | Nonlinear (beyond base 4) | Linear fields                                                                  | Optional nonlinear            |
| -------------------------- | ------------------------- | ------------------------------------------------------------------------------ | ----------------------------- |
| `RVParameters`             | —                         | `rv_semiamp: BatchQSpeed`, `v_sys: BatchQSpeed`                                | `jitter: BatchQSpeed \| None` |
| `SB2RVParameters`          | —                         | `rv_semiamp_1: BatchQSpeed`, `rv_semiamp_2: BatchQSpeed`, `v_sys: BatchQSpeed` | `jitter: BatchQSpeed \| None` |
| `GaiaAstrometryParameters` | `cos_i`, `lon_asc_node`   | `ra0`, `dec0`, `pmra`, `pmdec`, `parallax`, `semi_major_axis`                  | `jitter: BatchQAngle \| None` |

Optional nonlinear parameters (declared in `_optional_nonlinear_param_names`) default
to `None` and are excluded from the auto-computed `nonlinear_param_names`. They do not
need to appear in the prior. When `None`, they are static pytree leaves and do not
interfere with `jax.vmap`.

**`MarginalizedParameters` wrapper** (nonlinear parameters only; linear parameters are
analytically marginalized out):

A single `MarginalizedParameters(eqx.Module)` wrapper is used for all data types. It
stores non-marginalized field values in a `values: dict[str, Any]` (pytree leaves) and
records which linear parameters were removed in `marginalized_names: tuple[str, ...]`
(static). Field access delegates to `values` via `__getattr__`, so `params.period`
works as expected. The `source_cls` static field records the full parameter class the
wrapper was derived from.

Creation:

```python
# Classmethod shortcut (sampler construction path):
marg = RVParameters.marginalized(period=..., eccentricity=..., phase_peri=..., arg_peri=...)

# Or with partial marginalization (only marginalize rv_semiamp, not v_sys):
marg = RVParameters.marginalized("rv_semiamp", period=..., eccentricity=..., phase_peri=..., arg_peri=..., v_sys=...)
```

### Parameter naming convention

All parameter names follow the rule: **use the standard descriptive name; abbreviate
only when the abbreviation is itself a recognized domain term.** Examples:

| Parameter name    | Rationale                                                                         |
| ----------------- | --------------------------------------------------------------------------------- |
| `period`          | Full word — unambiguous                                                           |
| `eccentricity`    | Full word — unambiguous                                                           |
| `phase_peri`      | Descriptive compound                                                              |
| `arg_peri`        | `arg` is the standard abbreviation for *argument* in orbital mechanics            |
| `rv_semiamp`      | `rv` is universally recognized; avoids ambiguity with astrometric semi-major axis |
| `v_sys`           | $v\_\\text{sys}$ is the standard notation for systemic velocity                   |
| `pmra`, `pmdec`   | `pm` is the standard abbreviation for *proper motion*                             |
| `ra0`, `dec0`     | `ra` and `dec` are standard coordinate abbreviations                              |
| `semi_major_axis` | Full descriptive name — no universally short form                                 |
| `cos_i`           | Stores cosine of inclination directly (prior is uniform in `cos_i`)               |
| `lon_asc_node`    | Descriptive; `lon` abbreviates *longitude*                                        |

**Physics symbols vs parameter names:** In mathematical descriptions (equations,
docstrings explaining the model), the physics symbols $K$ (semi-amplitude) and $v_0$
(systemic velocity) are standard and should be used. The API-level parameter names
(`rv_semiamp`, `v_sys`) appear in function signatures, dict keys, and struct fields.

### The `period` convention

Parameter structs store `period: Q["time"]`. The period prior is typically a
`dist.LogUniform(period_min, period_max)` wrapped in a `QD` to
carry the unit. At sampling time, the sampler converts period draws from the prior's
unit to the data's time unit before constructing parameter structs.

### `phase_peri` vs `t_peri`

The nonlinear structs use `phase_peri = t_peri / period` (dimensionless, range 0–1)
rather than an absolute `t_peri`. This decouples the phase from the period scale,
which simplifies the prior (uniform on \[0, 1\]) and avoids the need to specify a
reference epoch in the prior. `Samples` exposes a derived `"t_peri"` key that
reconstructs the absolute time as `phase_peri * period + t_ref`.

______________________________________________________________________

## Likelihood layer (`harv.likelihood`)

### `AbstractLikelihood[DataT, ParamT]`

The generic base class, parameterized by data type and parameter type. It is an
`eqx.Module` with the following fields:

| Field                        | Type                              | Description                                                  |
| ---------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `data`                       | `DataT`                           | Observation data                                             |
| `linear_marginalized_prior`  | `LinearPriorDist \| None`         | Per-parameter Gaussian priors for analytical marginalization |
| `offsets_marginalized_prior` | `Mapping[str, PriorDist] \| None` | Per-instrument offset priors (multi-survey RV)               |
| `trend_marginalized_prior`   | `Mapping[str, PriorDist] \| None` | Per-trend-column Gaussian priors                             |
| `indicator_matrix`           | `jax.Array \| None`               | Multi-survey indicator matrix                                |
| `instrument_names`           | `tuple[str, ...] \| None`         | Non-reference instrument names                               |

The `log_prob(params, offsets=None)` method dispatches between marginalized and explicit
evaluation based on whether `linear_marginalized_prior` is set and whether `params` is a
`MarginalizedParameters` instance.

Abstract methods that subclasses must implement:

- `design_matrix(params) -> jax.Array` — build the design matrix
- `linear_param_units -> dict[str, str]` — units of linear parameters (property)

Shared methods provided by the base:

- `log_prob(params, offsets=None) -> jax.Array` — dispatches to `_log_prob_marginalized`
  or `_log_prob_explicit` based on the params type and prior configuration.
- `sample_conditional_linear(params, key, offsets=None) -> dict[str, AbstractQuantity]`
  — sample linear parameters from their conditional posterior given data and nonlinear
  parameters. Used by the rejection sampler after acceptance.
- `linear_unmarginalized_param_values(params, offsets)` — extract values for
  non-marginalized linear parameters (used internally for partial marginalization).

### Per-parameter linear prior (`LinearPriorDist`)

The `linear_marginalized_prior` field accepts a `dict[str, PriorDist | LinearPriorCallable]`
where each entry specifies the prior for one linear parameter. Each entry is classified:

| Prior type                    | Classification  | Treatment                                                                          |
| ----------------------------- | --------------- | ---------------------------------------------------------------------------------- |
| `QD(Normal)` or `dist.Normal` | Gaussian        | Analytically marginalized via joint MVN                                            |
| `LinearPriorCallable`         | Param-dependent | Called with `params` to produce a `QD(Normal)` or `dist.Normal`, then marginalized |

`_resolve_linear_prior_mvn` (in `helpers.py`) resolves all entries into a joint diagonal
`dist.MultivariateNormal`, converting units to the data's native units using
`linear_param_units`.

A `LinearPriorCallable` is a `typing.Protocol` (runtime-checkable) with signature:

```python
class LinearPriorCallable(Protocol):
    def __call__(
        self, params: AbstractParameters | MarginalizedParameters
    ) -> QuantityDistribution | dist.Normal: ...
```

Callables that return Q-valued distributions **must** wrap them in
`QuantityDistribution` (or `QD`), not pass bare `dist.Normal` with Q loc/scale. numpyro
distributions do not natively support `Q` parameters.

### `RVLikelihood`

`RVLikelihood(AbstractLikelihood[RVData, RVParameters])` is the unified
radial velocity likelihood class. It supports:

1. **Marginalized** (`linear_marginalized_prior` provided, `params` is
   `MarginalizedParameters`): analytically integrates over \[K, v₀\] via a
   `MarginalizedLinear` distribution (from numpyro-ext).
1. **Multi-survey marginalized** (`indicator_matrix` and `offsets_marginalized_prior`
   provided): appends instrument-offset columns to the design matrix and marginalizes
   \[K, v₀, δ₁, …, δₖ\] jointly.
1. **Explicit** (`linear_marginalized_prior` is `None`, `params` is `RVParameters`):
   evaluates the Gaussian log-likelihood directly at specified K, v₀ values.

The design matrix has columns `[rv_amplitude, 1]` (base), plus one column per
non-reference instrument when an indicator matrix is present.

### Polynomial trend support

Both `RVLikelihood` and `GaiaAstrometryLikelihood` support polynomial velocity/position
trends via the `trend_order: int` field. The trend is a monomial basis:

- **RV**: columns `[(t - t_ref)^1, (t - t_ref)^2, ..., (t - t_ref)^k]` for
  `trend_order = k`. The constant term is NOT included (already captured by `v_sys`).
- **Astrometry**: each order *k* adds **two** columns (RA and Dec projected along the
  scan angle): `cos(ψ)·dt^(k+1)` and `sin(ψ)·dt^(k+1)`, where `dt = (t - t_ref)`.
  The `+1` offset is because the base astrometric model already includes `dt^1`
  proper motion columns.

Column ordering in the combined design matrix is:
`(*linear_param_names, *trend_column_names, *instrument_names)`.

Trend column names are auto-generated: `trend_1`, `trend_2`, ... for RV, and
`trend_ra_1`, `trend_dec_1`, `trend_ra_2`, ... for astrometry.

Trend priors are passed via `trend_marginalized_prior` on the likelihood and
`trend_priors` / `trend_order` on `RejectionPrior`.

**Pluggable basis (future):** The current implementation uses a power-law monomial
basis. To support alternative bases (Chebyshev, B-splines), replace the
`_build_trend_columns` helper with a `TrendBasis` protocol. See
"Pluggable trend basis" under Planned features for the full design.

### `SB2RVLikelihood`

`SB2RVLikelihood(AbstractLikelihood[SystemData, SB2RVParameters])` handles
double-lined spectroscopic binary observations. The design matrix stacks primary
and secondary observations:

- Primary rows: `[rv_shape, 0, 1]` (K₁ column active)
- Secondary rows: `[0, -rv_shape, 1]` (K₂ column active, negated for anti-phase)

The three linear parameters are `rv_semiamp_1`, `rv_semiamp_2`, and `v_sys`.
Polynomial trends are appended after the 3 base columns and span the full stacked
observation vector.

SB2 + multi-survey offsets are NOT currently supported.

### `GaiaAstrometryLikelihood`

`GaiaAstrometryLikelihood(AbstractLikelihood[GaiaAstrometryData, GaiaAstrometryParameters])`
follows the same structure. The (n_obs, 6) design matrix columns are
\[α₀, δ₀, μ_α, μ_δ, ϖ, a\], following Appendix A of
[Holl et al. 2022](https://arxiv.org/abs/2206.05726). The Thiele-Innes constants
are computed on-the-fly from the nonlinear orientation parameters (`cos_i`,
`lon_asc_node`, `arg_peri`).

### `CompositeLikelihood`

Combines multiple `AbstractLikelihood` components by summing their log-likelihoods.
Stored as a `dict[str, AbstractLikelihood]` of named components (passed as
`**kwargs` to `__init__`). Each component holds its own `linear_marginalized_prior`
and evaluates independently.

`log_prob` takes a `dict[str, MarginalizedParameters]` keyed by component name, and
an optional `offsets` dict. Each component reads only its corresponding params entry.

```python
composite = CompositeLikelihood(
    rv=RVLikelihood(data=rv_data, linear_marginalized_prior=rv_prior),
    astro=GaiaAstrometryLikelihood(data=gaia_data, linear_marginalized_prior=astro_prior),
)
log_liks = jax.jit(jax.vmap(composite.log_prob))(params_dict_batch)
```

`CompositeLikelihood` also exposes `linear_param_units` (property) and dict-style
access (`keys()`, `values()`, `items()`, `__getitem__`, `__len__`) over its components.

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

`PeriodDependentKPrior` (in `harv.samplers.custom_priors`) implements `LinearPriorCallable`.
It computes a period- and eccentricity-dependent scale for the RV semi-amplitude
prior, following the Joker's default:

```
σ_K(P, e) = σ_{K,0} · (P / P₀)^{-1/3} · (1 - e²)^{-1/2}
```

This keeps the prior approximately constant in companion mass at fixed primary mass.
`__call__` returns a `QD(dist.Normal(0, σ_K_stripped), unit)`.

Fields:

- `sigma_K0: Q["speed"]` — scale at reference period
- `P0: Q["time"]` — reference period

### `PeriodDependentSemiMajorAxisPrior`

`PeriodDependentSemiMajorAxisPrior` (in `harv.samplers.custom_priors`) implements
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

`__call__` receives a parameter struct with `.period`, `.eccentricity`, and `.parallax`
fields (parallax is available because it is explicitly sampled by default) and returns
`QD(dist.Normal(0, σ_a_stripped), "mas")`.

Fields:

- `sigma_a0: Q["length"]` — semi-major axis scale at reference period (e.g. AU)
- `P0: Q["time"]` — reference period

### `ParallaxDependentProperMotionPrior`

`ParallaxDependentProperMotionPrior` (in `harv.samplers.custom_priors`) implements
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

`__call__` receives a parameter struct with a `.parallax` field (parallax is available
because it is explicitly sampled by default) and returns
`QD(dist.Normal(0, σ_μ), parallax_unit + "/yr")`.

Fields:

- `sigma_v0: Q["speed"]` — velocity dispersion scale (e.g. km/s)

______________________________________________________________________

## Prior (`harv.samplers.RejectionPrior`)

`RejectionPrior` holds numpyro distributions over all nonlinear parameters and a
per-parameter linear prior. It is an `eqx.Module`.

### Fields

| Field               | Type                                                 | Description                                                          |
| ------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| `nonlinear_priors`  | `dict[str, PriorDist]`                               | Nonlinear parameter priors                                           |
| `linear_prior`      | `LinearPriorDist`                                    | Per-parameter linear priors                                          |
| `marginalize_names` | `tuple[str, ...] \| None` (KW_ONLY)                  | Which linear params to marginalize; `None` = all                     |
| `offsets`           | `dict[str, dict[str, QD \| None]] \| None` (KW_ONLY) | Per-instrument offset priors keyed by data type then instrument name |
| `trend_order`       | `int` (KW_ONLY, default 0)                           | Polynomial trend order (0 = no trend)                                |
| `trend_priors`      | `dict[str, LinearPriorDist] \| None` (KW_ONLY)       | Per-trend-column Gaussian priors                                     |
| `jitter_priors`     | `dict[str, PriorDist] \| None` (KW_ONLY)             | Per-data-type jitter (excess variance) priors                        |

### Constructing a prior

The `default_*` class methods simplify construction of priors for common cases. They
are convenience wrappers around `__init__` that set up standard nonlinear priors and
linear prior structures. They do **not** provide parameter-less defaults — the user
must supply at minimum the period bounds and (for RV) the amplitude scale, since
these depend on the science case (binary stars, compact objects, exoplanets all have
different characteristic scales and timescales).

Direct `__init__` construction is always supported for fully custom configurations.

#### `default_rv`

```python
RejectionPrior.default_rv(
    *,
    period_min: Q["time"],     # required
    period_max: Q["time"],     # required
    sigma_K0: Q["speed"],      # required — RV amplitude scale
    sigma_v0: Q["speed"],      # required — systemic velocity scale
    P0: Q["time"] = Q(1.0, "yr"),
    offsets: dict[str, QD | None] | None = None,
    marginalize_names: tuple[str, ...] | None = None,
    trend_order: int = 0,
    trend_priors: dict[str, LinearPriorDist] | None = None,
    jitter_scale: Q["speed"] | None = None,  # excess variance scale (HalfNormal)
    **kwargs,                          # per-parameter prior overrides
) -> RejectionPrior
```

Constructs a prior with:

- `period`: `LogUniform(period_min, period_max)` wrapped in `QD`
- `eccentricity`: `Beta(0.867, 3.03)` (Kipping 2013)
- `phase_peri`: `Uniform(0, 1)`
- `arg_peri`: `Uniform(0, 2π)`
- `rv_semiamp` linear prior: `PeriodDependentKPrior(sigma_K0, P0)` — a callable that scales
  the K prior with period and eccentricity
- `v_sys` linear prior: `QD(Normal(0, sigma_v0), unit)`

Any nonlinear or linear prior can be overridden by passing the corresponding
parameter name as a keyword argument.  Valid names are the nonlinear and linear
parameter names from `RVParameters`: `period`, `eccentricity`, `phase_peri`,
`arg_peri`, `rv_semiamp`, `v_sys`.

#### `default_gaia_astrometry`

```python
RejectionPrior.default_gaia_astrometry(
    *,
    period_min: Q["time"],             # required
    period_max: Q["time"],             # required
    sigma_a0: Q["length"],             # required — semi-major axis scale
    sigma_parallax: Q["angle"],        # required — parallax prior scale
    sigma_pos: Q["angle"],             # required — position prior scale
    sigma_vtan: Q["speed"],            # required — tangential velocity dispersion scale
    P0: Q["time"] = Q(1.0, "yr"),
    marginalize_names: tuple[str, ...] | None = None,
    trend_order: int = 0,
    trend_priors: dict[str, LinearPriorDist] | None = None,
    jitter_scale: Q["angle"] | None = None,  # excess variance scale (HalfNormal)
    **kwargs,                                  # per-parameter prior overrides
) -> RejectionPrior
```

Constructs a prior with:

- `period`, `eccentricity`, `phase_peri`, `arg_peri`: same defaults as RV
- `cos_i`: `Uniform(-1, 1)`
- `lon_asc_node`: `Uniform(0, 2π)`
- `semi_major_axis`: `PeriodDependentSemiMajorAxisPrior(sigma_a0, P0)` — a
  callable that scales the semi-major axis prior with period and parallax
- `parallax`: `QD(HalfNormal(sigma_parallax), "mas")` — explicitly
  sampled (not marginalized) by default, because the Gaia catalog parallax is
  derived from the same epoch data
- `ra0`, `dec0`: `QD(Normal(0, sigma_pos), "mas")`
- `pmra`, `pmdec`: `ParallaxDependentProperMotionPrior(sigma_v0=sigma_vtan)` — a
  callable that scales the proper motion prior with parallax, keeping the prior
  fixed in velocity space

Any nonlinear or linear prior can be overridden by passing the corresponding
parameter name as a keyword argument.  Valid names are the nonlinear and linear
parameter names from `GaiaAstrometryParameters`: `period`, `eccentricity`,
`phase_peri`, `arg_peri`, `cos_i`, `lon_asc_node`, `ra0`, `dec0`, `pmra`, `pmdec`,
`parallax`, `semi_major_axis`.

Parallax is classified as explicit automatically by `__check_init__` because
`HalfNormal` cannot be analytically marginalized.  For exoplanet searches where
the catalog parallax is trustworthy, users can override with a `Normal` prior
and include `"parallax"` in `marginalize_names`.

#### `default_sb2`

```python
RejectionPrior.default_sb2(
    *,
    period_min: Q["time"],     # required
    period_max: Q["time"],     # required
    sigma_K0: Q["speed"],      # required — RV amplitude scale
    sigma_v0: Q["speed"],      # required — systemic velocity scale
    P0: Q["time"] = Q(1.0, "yr"),
    marginalize_names: tuple[str, ...] | None = None,
    trend_order: int = 0,
    trend_priors: dict[str, LinearPriorDist] | None = None,
    jitter_scale: Q["speed"] | None = None,  # excess variance scale (HalfNormal)
    **kwargs,                          # per-parameter prior overrides
) -> RejectionPrior
```

Same defaults as `default_rv` but with three linear parameters:

- `rv_semiamp_1`, `rv_semiamp_2`: both use `PeriodDependentKPrior(sigma_K0, P0)`
- `v_sys`: `QD(Normal(0, sigma_v0), unit)`

### Multi-survey RV offsets

When multiple instruments observe the same star, their zero-points may differ by an
additive offset. The `offsets` dict on `RejectionPrior` maps data-type keys (e.g.
`"rv"`) to instrument-name → `QD | None` dicts:

```python
prior = RejectionPrior.default_rv(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    offsets={
        "espresso": QD(dist.Normal(0, 5.0), "km/s"),
        # "keck" absent → reference instrument, offset = 0
    },
)
```

The offsets are additional linear parameters appended to the design matrix by the
`RVLikelihood` via `indicator_matrix`. The sampler constructs the likelihood with the
appropriate indicator matrix automatically when `SourceData` has multiple RV datasets.

### Jitter (excess variance)

The `jitter_priors` field on `RejectionPrior` provides per-data-type jitter parameters
that are added in quadrature to the observation errors:

$$\\sigma\_\\mathrm{eff} = \\sqrt{\\sigma\_\\mathrm{obs}^2 + s^2}$$

where $s$ is the jitter value. Jitter is an **optional nonlinear parameter** — it is
sampled from its prior but is not required. When `jitter_priors` is `None` (the
default), no jitter is applied and the behavior is identical to previous versions.

The `jitter_priors` dict is keyed by data-type label (`"rv"`, `"astrometry"`):

```python
prior = RejectionPrior.default_rv(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    jitter_scale=Q(1.0, "km/s"),  # HalfNormal(1.0 km/s)
)
```

The `default_*` convenience methods accept a `jitter_scale` keyword that creates a
`HalfNormal` prior with the given scale. For full control, pass `jitter_priors`
directly to `__init__`:

```python
prior = RejectionPrior(
    nonlinear_priors=...,
    linear_prior=...,
    jitter_priors={"rv": QD(dist.HalfNormal(1.0), "km/s")},
)
```

In combined (multi-data-type) fits, each data type has its own independent jitter
parameter. Internally these are stored with namespaced keys (`_jitter_rv`,
`_jitter_astrometry`) to avoid collision; in the output `Samples`, they appear as
`jitter_rv` and `jitter_astrometry`.

Each parameter struct carries a `jitter` field with appropriate units:

| Class                      | `jitter` type         |
| -------------------------- | --------------------- |
| `RVParameters`             | `BatchQSpeed \| None` |
| `GaiaAstrometryParameters` | `BatchQAngle \| None` |
| `SB2RVParameters`          | `BatchQSpeed \| None` |

The default is `None` (no jitter). When `None`, the parameter is a static pytree leaf,
so it does not interfere with `jax.vmap` over batched parameters.

### `sample_nonlinear`

`sample_nonlinear(key, n_samples) -> dict[str, jax.Array]` draws from all nonlinear
priors. Returns bare JAX arrays regardless of whether the distribution is wrapped in
`QuantityDistribution`.

______________________________________________________________________

## Rejection sampler (`harv.samplers.rejection.RejectionSampler`)

Implements the rejection sampling algorithm from
[Price-Whelan et al. 2017](https://arxiv.org/abs/1701.08160) (The Joker). The core
idea: because the likelihood is analytically marginalized over linear parameters, it
can be evaluated cheaply for millions of nonlinear prior samples, making rejection
sampling efficient.

### Fields

| Field        | Type             | Description                                |
| ------------ | ---------------- | ------------------------------------------ |
| `model`      | `Model`          | Model combining prior and data             |
| `batch_size` | `int` (static)   | Samples vmapped at once (default: 100,000) |

### Algorithm

1. **Prior sampling.** Draw `n_prior_samples` from the nonlinear prior. Period is
   converted from the prior's unit to the data's time unit.

1. **Likelihood evaluation** (batched). For each batch of `batch_size` samples,
   construct `MarginalizedParameters` structs, and evaluate
   `jax.vmap(lik.log_prob)(params_batch)` using `jax.lax.fori_loop` to bound memory.

1. **Rejection.** Normalize weights to `max` and accept samples where
   `Uniform() < weight`.

1. **Linear parameter sampling.** For each accepted nonlinear sample, call
   `likelihood.sample_conditional_linear(params, key)` to draw the linear
   parameters from their conditional posterior.

1. **Return** a `Samples` object.

### `run` method

```python
sampler.run(
    n_prior_samples: int,
    *,
    max_posterior_samples: int | None = None,
    seed: int = 0,
) -> Samples
```

### Data type inference

Data type inference and likelihood construction happen in `Model.__init__`. The
`Model` inspects the input data to determine the data type:

- `RVData` → data_type `"rv"`
- `GaiaAstrometryData` → data_type `"astrometry"`
- `SourceData` with both data types → data_type `"combined"`
- `SourceData` with multiple RV datasets → data_type `"rv"` with indicator matrix

The `Model` constructs the appropriate likelihood object(s) at init time and provides
methods to build parameter structs and evaluate the log-probability.

### `batch_size` and GPU support

The `batch_size` field controls how many samples are vmapped at once within a
`fori_loop`. On CPU, the default of 100,000 is appropriate. On GPU, set
`batch_size = n_prior_samples` to let XLA fully utilize the device.

### MCMC initialization (`NumpyroSampler`)

MCMC functionality lives on `NumpyroSampler(model)`. `init_mcmc` takes the `Samples`
object returned by `RejectionSampler.run()` and an optional numpyro kernel class. It
builds a numpyro model automatically from the model's prior and data, draws one
starting position per chain from the posterior, and returns a `WarmStartMCMC` wrapper
whose `run()` injects those positions automatically.

Two model variants are supported via `marginalized`:

- `marginalized=True` (default): MCMC explores nonlinear subspace only
- `marginalized=False`: MCMC samples all parameters jointly

______________________________________________________________________

## `Samples` container (`harv.samplers.samples.Samples`)

Stores the posterior samples returned by `RejectionSampler.run()`.

### Fields

| Field                | Type                        | Description                                 |
| -------------------- | --------------------------- | ------------------------------------------- |
| `nonlinear`          | `dict[str, Q]`              | Nonlinear parameter samples with units      |
| `linear`             | `dict[str, Q]`              | Linear parameter samples with units         |
| `orbit_cls`          | `type` (static)             | Nonlinear param class (e.g. `RVParameters`) |
| `full_cls`           | `tuple[type, ...]` (static) | Ordered tuple of full parameter classes     |
| `metadata`           | `dict[str, Any]` (static)   | Contains `t_ref` and extra info             |
| `extra_linear_names` | `tuple[str, ...]` (static)  | Multi-survey offset param names             |
| `data_type`          | `str` (static)              | `"rv"`, `"astrometry"`, or `"combined"`     |

### Dict-style access

`samples["key"]` dispatches to appropriate unit restoration:

- Nonlinear params (`"period"`, `"eccentricity"`, `"phase_peri"`, etc.) → `Q`
  with units
- Linear params (`"rv_semiamp"`, `"v_sys"`, `"ra0"`, etc.) → `Q` with units
- Derived keys:
  - `"log_period"` → dimensionless array (`log10(period in data time units)`)
  - `"t_peri"` → `Q` (derived from `phase_peri * period + t_ref`)
  - `"inclination"` → `Q` in radians (derived from `arccos(cos_i)`)

### Methods

- `keys() -> list[str]` — nonlinear + linear + derived parameter names
- `n_samples -> int` — number of posterior samples
- `median(key=None)` — median of one key or all keys
- `percentile(key, percentiles=(16, 50, 84))` — compute percentiles
- `summary(params=None)` — dict of statistics (median, mean, std, p16, p84)
- `to_hdf5(filename)` / `from_hdf5(filename)` — HDF5 persistence
- `plot_corner(params=None, truths=None, **kwargs)` — corner plot via arviz
- `plot(data=None, n_samples=None, phase_fold=False, plot_kwargs=None, data_plot_kwargs=None, **kwargs)` — phase-folded or time-domain RV / sky-plane orbit plots. `plot_kwargs` customises orbit curves (defaults: thin grey lines, `linewidth=0.5`, `alpha=0.15`, `color="#555555"`). `data_plot_kwargs` customises data error-bars (defaults: black markers, `marker="o"`, `markersize=4`).

______________________________________________________________________

## Plotting utilities (`harv.plot`)

### `get_t_grid`

`get_t_grid(times: BatchQTime, period: Q["time"])` returns a dense time grid
for plotting orbit curves. The grid spans from `min(times) - span_factor*range/2` to
`max(times) + span_factor*range/2`, with spacing determined by
`period / n_points_per_period`.

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

### SB2 + multi-survey offsets

`SB2RVLikelihood` does not yet support multi-survey offsets (`indicator_matrix`). If
needed, the secondary component's offset structure would have to be defined (e.g., does
each instrument have the same offset for both components?).

### Combined astrometry + multi-survey RV

The combined case (astrometry + multiple RV instruments) via `CompositeStrategy` is
partially implemented. Currently raises `NotImplementedError` if `SourceData`
contains both `GaiaAstrometryData` and more than one `RVData`.

### Batch inference over many datasets

A common population-level workflow is: define a single prior, generate a large library
of prior samples once, then run rejection sampling against many datasets (e.g. thousands
of Gaia sources). The current API creates a separate `Model(prior, data)` and
`RejectionSampler` per dataset, which means:

1. **Redundant prior sampling** — the same prior draws are regenerated for every dataset
   even though they only depend on the prior, not the data.
2. **JIT retracing** — if datasets have different numbers of observations (different
   array shapes), JAX recompiles the likelihood evaluation kernel for each new shape.

The planned design separates prior sampling from likelihood evaluation:

- **Prior samples are drawn once** and reused across all datasets.
- **Likelihood evaluation is batched** over datasets, with automatic padding/masking
  to a common observation count so that a single JIT-compiled kernel handles all
  datasets without retracing.
- A high-level entry point (e.g. a single function call) handles the
  padding, batching, rejection step, and linear-parameter sampling internally,
  so users do not need to manage these details.
- The implementation should support **chunked/batched execution** over datasets to
  control memory usage, and be designed with **multi-device and GPU parallelism** in
  mind (e.g. `jax.pmap` or `jax.experimental.shard_map` over devices).

### Iterative rejection sampling

The Joker's iterative scheme grows the sample batch exponentially until enough
posterior samples are accepted. Useful when the likelihood is very constraining.

### Absolute and relative astrometry

`AbstractAstrometryData` exists as a base for future data types:

- **Absolute astrometry** (RA/Dec timeseries from ground-based or HST observations)
- **Relative astrometry** (separation and position angle from direct imaging)

**TODO:** `AbsoluteAstrometryData` is currently commented out in `harv.data`. Before
implementing, define the required fields (time, RA, Dec, errors, covariance structure)
and add the corresponding likelihood class.

### Source motion models (`harv.simulate.source`)

`simulate/source.py` contains an incomplete `AbstractSource` hierarchy for modeling
astrometric source motion (linear proper motion, small-angle approximation,
accelerated motion from a Keplerian companion). The subclasses
(`LinearMotion3DSource`, `LinearMotionSmallAngleSource`, `Accelerating3DSource`)
are partially implemented and not yet functional.

**TODO:** Finish the `AbstractSource` hierarchy:

- Fix `LinearMotionSmallAngleSource.offset_sky` (references undefined `xyz_t`).
- Fix `Accelerating3DSource` (references nonexistent `SingleStarSource`).
- Define the `offset_sky` contract on `AbstractSource` (currently raises
  `NotImplementedError`).
- Integrate with `simulate_gaia_epoch_astrometry` once complete.

### Pluggable trend basis

The current polynomial trend implementation uses a monomial basis
`[(t-t_ref)^1, ..., (t-t_ref)^k]` via the `_build_trend_columns` helper. To support
alternative bases (Chebyshev, B-splines), replace this with a `TrendBasis` protocol::

```
class TrendBasis(Protocol):
    n_basis: int
    names: tuple[str, ...]          # one per output column
    def __call__(
        self, times: jax.Array, t_ref: float,
    ) -> jax.Array:                 # (n_obs, n_basis)
        ...
```

The monomial implementation becomes a concrete `MonomialBasis` class. A Chebyshev
basis would return columns evaluated on a normalized \[-1, 1\] domain mapped from the
observation time span. A B-spline basis would use a fixed knot vector.

**Key contract**: The basis must NOT include a constant column (order 0), since that
role is already filled by `v_sys` / `ra0` / `dec0`.

Changes required:

1. Define the `TrendBasis` protocol (in `trends.py` or `likelihood/rv.py`).
1. Replace `trend_order: int` fields with `trend_basis: TrendBasis | None`
   on `RVLikelihood` and `GaiaAstrometryLikelihood`.
1. Derive `trend_column_names` from `trend_basis.names`.
1. Update `RejectionPrior` factory methods to accept a basis.

______________________________________________________________________

## API sketch

The intended user-facing interface for common use cases:

```python
import numpyro.distributions as dist
from unxt import Q
from harv import Model
from harv.data import RVData, SourceData
from harv.distributions import QD
from harv.samplers import NumpyroSampler, RejectionPrior, RejectionSampler

# Minimal RV-only case:
data = RVData(time, rv, rv_err)
prior = RejectionPrior.default_rv(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
)
model = Model(prior, data)
sampler = RejectionSampler(model)
samples = sampler.run(n_prior_samples=500_000)

# With max posterior samples:
samples = sampler.run(n_prior_samples=500_000, max_posterior_samples=128)

# Multi-instrument RV with zero-point offsets:
data = SourceData(
    keck=RVData(time1, rv1, rv_err1),
    espresso=RVData(time2, rv2, rv_err2),
)
prior = RejectionPrior.default_rv(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    offsets={
        "espresso": QD(dist.Normal(0, 5.0), "km/s"),
        # keck is the reference instrument; its offset is fixed to 0
    },
)
model = Model(prior, data)
sampler = RejectionSampler(model)
samples = sampler.run(n_prior_samples=500_000)

# Gaia astrometry only:
prior = RejectionPrior.default_gaia_astrometry(
    period_min=Q(0.3, "yr"),
    period_max=Q(10, "yr"),
    sigma_a0=Q(1e3, "AU"),
    sigma_parallax=Q(100.0, "mas"),
    sigma_pos=Q(1e3, "mas"),
    sigma_vtan=Q(200, "km/s"),
)
model = Model(prior, gaia_data)
sampler = RejectionSampler(model)
samples = sampler.run(n_prior_samples=1_000_000)

# MCMC continuation:
mcmc = NumpyroSampler(model).init_mcmc(samples, num_chains=4, num_warmup=500, num_samples=2000)
mcmc.run(jr.key(0))

# Post-sampling analysis:
samples["period"]          # Quantity in data time units
samples["eccentricity"]    # dimensionless array
samples.median("rv_semiamp")        # median semi-amplitude
samples.summary()          # dict of all statistics
samples.plot_corner()      # arviz corner plot
samples.plot(data=data)    # phase-folded RV/astrometry overlay
samples.to_hdf5("out.h5")  # persistence
```
