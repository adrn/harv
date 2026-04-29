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
│   │   └── gaia.py          # StandardGaiaAstrometry, ThieleInnesGaiaAstrometry
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
│   ├── rejection_prior.py   # RejectionPrior
│   ├── custom_priors.py     # PeriodDependentKPrior, _make_log_period_prior
│   ├── rejection.py         # RejectionSampler
│   ├── numpyro.py           # NumpyroSampler (MCMC with warm-start)
│   └── samples.py           # Samples container
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

Derived convenience methods:

- `nonlinear_params()` -- filter to nonlinear entries.
- `linear_params()` -- filter to linear entries.

### `StandardRV`

Standard RV parameterization: `(period, eccentricity, phase_peri, arg_peri, rv_semiamp, v_sys)`.

- Nonlinear: `period`, `eccentricity`, `phase_peri`, `arg_peri`.
- Linear: `rv_semiamp`, `v_sys`.
- Design matrix shape: `(n_obs, 2)` with columns `[rv_shape(t), 1]`.

Also provides `eccentricity(nl_values)` and `strip_nl_for_design(nl_values)`.

### `EcoswEsinwRV`

Alternative RV parameterization using `e*cos(omega)` and `e*sin(omega)`:

- Nonlinear: `period`, `ecosw`, `esinw`, `phase_peri`.
- Linear: `rv_semiamp`, `v_sys`.
- Design matrix shape: `(n_obs, 2)` -- same columns, different internal derivation.

This parameterization has better sampling geometry for low eccentricities.

### `StandardGaiaAstrometry`

Standard Gaia epoch-astrometry parameterization:

- Nonlinear: `period`, `eccentricity`, `phase_peri`, `arg_peri`, `lon_asc_node`, `cos_i`.
- Linear: `ra0`, `dec0`, `pmra`, `pmdec`, `parallax`, `semi_major_axis`.
- Design matrix shape: `(n_obs, 6)` following Holl et al. (2022), Appendix A.

The design matrix columns are
`[sin(psi), cos(psi), sin(psi)*dt, cos(psi)*dt, H_parallax, TI_orbit]`
where the Thiele-Innes orbital element combines the (A, B, F, G) constants
with the X, Y orbital coordinates.

### `ThieleInnesGaiaAstrometry`

Alternative Gaia parameterization that moves the four Thiele-Innes constants
`(A, B, F, G)` from the nonlinear to the linear parameter set, reducing the
nonlinear space from 6-D to 3-D.  This is the approach described in Hsieh et al.
("Astrometric Orbit Fitting with Marginalization over Linear Parameters").

- Nonlinear: `period`, `eccentricity`, `phase_peri`.
- Linear: `ra0`, `dec0`, `pmra`, `pmdec`, `parallax`, `ti_A`, `ti_B`, `ti_F`, `ti_G`.
- Design matrix shape: `(n_obs, 9)`.

The Jacobian correction is **always applied**: a flat prior on the Thiele-Innes
constants is not equivalent to a flat prior on the physical Campbell elements
`(a_0, ω, Ω, cos i)`.  The zeroth-order correction (evaluated at the conditional-mean
TI constants following Hsieh et al.) multiplies the marginal likelihood by the factor
`(a_0 + δ_a)^{-m} (sin²i + δ_s)^{-1}`, where `m = 3` for a uniform prior on `a_0`
and `m = 4` for a log-uniform prior.

Constructor parameters:

| Parameter          | Type    | Default | Description                                             |
| ------------------ | ------- | ------- | ------------------------------------------------------- |
| `a_floor`          | `float` | —       | Floor on `a_0` (in obs units, e.g. mas).  **Required.** |
| `sin2i_floor`      | `float` | `0.01`  | Floor on `sin²i` for the Jacobian denominator.          |
| `log_uniform_in_a` | `bool`  | `False` | Use log-uniform prior on `a_0` (`m=4`).                 |

The recommended constructor is `ThieleInnesGaiaAstrometry.from_data(data)`, which
sets `a_floor = Med(σ_AL) / sqrt(N)` automatically.

After sampling with this parameterization, convert the Thiele-Innes linear parameters
to Campbell elements via `samples.thiele_innes_to_campbell()`.

**Limitation**: the RV forward model is not linear in `(A, B, F, G)`, so joint
RV+astrometry fits must use `StandardGaiaAstrometry`.

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

### `Samples.thiele_innes_to_campbell()`

When sampling with `ThieleInnesGaiaAstrometry`, the posterior `Samples` object carries
the Thiele-Innes constants `ti_A, ti_B, ti_F, ti_G` as linear parameters.
`samples.thiele_innes_to_campbell()` converts them to the physical Campbell elements
`semi_major_axis, arg_peri, lon_asc_node, cos_i` using the standard inversion:

```
u = (A²+B²+F²+G²) / 2
v = A·G − B·F
a_0 = sqrt(u + sqrt(max(u² − v², 0)))
ω + Ω = atan2(B − F, A + G)
ω − Ω = atan2(−B − F, A − G)
cos i = |v / a_0²|    # cos_i ≥ 0 convention
```

The method returns a new `Samples` with the TI constants replaced by the Campbell
elements.  The 2-fold degeneracy inherent in pure astrometry (face-on reflections)
means `cos_i` is not unique; the convention `cos_i ≥ 0` is adopted.

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

### Three `log_prob` calling conventions

1. **Auto mode** (recommended): `model.log_prob(values)` where `values` is a
   flat dict containing nonlinear params and any explicit-linear params.
1. **Manual marginalization**: pass `marginalized_names` to control exactly
   which linear params to marginalize.
1. **Explicit evaluation**: pass `linear_values` without `marginalized_names`.

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
  build an SB2 `JointModel` from a `RejectionPrior.default_sb2` prior. Automatically routes
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

| Field               | Type                                           | Description                                                |
| ------------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| `nonlinear_priors`  | `dict[str, PriorDist]`                         | Nonlinear parameter priors                                 |
| `linear_prior`      | `LinearPriorDist`                              | Per-parameter linear priors                                |
| `offsets` parameter | `dict[str, QD \| None] \| None` (factory only) | Offset priors; non-ref entries merged into `linear_prior`  |
| `extension_priors`  | `dict[str, PriorDist]` (KW_ONLY, default `{}`) | Priors for extension params (jitter, GP hyperparams, etc.) |

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
    **kwargs,          # per-parameter or extension prior overrides (e.g. jitter=QD(...))
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
parameter name as a keyword argument. Valid names are the nonlinear and linear
parameter names from `StandardRV`: `period`, `eccentricity`, `phase_peri`,
`arg_peri`, `rv_semiamp`, `v_sys`.

#### `default_gaia_astrometry`

```python
RejectionPrior.default_gaia_astrometry(
    *,
    period_min: Q["time"] | None = None,
    period_max: Q["time"] | None = None,
    sigma_a0: Q["length"] | None = None,        # required unless semi_major_axis= given
    sigma_parallax: Q["angle"] | None = None,   # required unless parallax= given
    sigma_pos: Q["angle"] | None = None,        # required unless ra0= and dec0= given
    sigma_vtan: Q["speed"] | None = None,       # required unless pmra= and pmdec= given
    P0: Q["time"] = Q(1.0, "yr"),
    **kwargs,          # per-parameter or extension prior overrides (e.g. jitter=QD(...))
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
parameter name as a keyword argument. Valid names are the nonlinear and linear
parameter names from `StandardGaiaAstrometry`: `period`, `eccentricity`,
`phase_peri`, `arg_peri`, `cos_i`, `lon_asc_node`, `ra0`, `dec0`, `pmra`, `pmdec`,
`parallax`, `semi_major_axis`.

When a linear prior is supplied directly via `**kwargs`, the corresponding scale
argument (`sigma_parallax`, `sigma_pos`, `sigma_a0`, or `sigma_vtan`) must be
omitted — passing both raises `TypeError`.

Parallax is classified as explicit automatically because `HalfNormal` cannot be
analytically marginalized. For exoplanet searches where the catalog parallax is
trustworthy, users can override with a `Normal` prior and set
`marginalized_names=("parallax", ...)` on the sampler.

#### `default_sb2`

```python
RejectionPrior.default_sb2(
    *,
    period_min: Q["time"],     # required
    period_max: Q["time"],     # required
    sigma_K0: Q["speed"],      # required — RV amplitude scale
    sigma_v0: Q["speed"],      # required — systemic velocity scale
    P0: Q["time"] = Q(1.0, "yr"),
    **kwargs,          # per-parameter or extension prior overrides (e.g. jitter=QD(...))
) -> RejectionPrior
```

Same defaults as `default_rv` but with three linear parameters:

- `rv_semiamp_1`, `rv_semiamp_2`: both use `PeriodDependentKPrior(sigma_K0, P0)`
- `v_sys`: `QD(Normal(0, sigma_v0), unit)`

### Multi-survey RV offsets

When multiple instruments observe the same star, their zero-points may differ by an
additive offset. Pass `offsets` to `default_rv()`: keys are instrument names,
`None` marks the reference, non-`None` entries are `QD` priors that get merged
into `linear_prior` automatically. The `linear_extension_names` field records which
`linear_prior` entries are linear extension parameters (offsets, trends, etc.)
used to populate `Samples.linear_extension_names`.

```python
prior = RejectionPrior.default_rv(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    offsets={
        "keck": None,                              # reference instrument
        "espresso": QD(dist.Normal(0, 5.0), "km/s"),
    },
)
assert "espresso" in prior.linear_prior
assert prior.linear_extension_names == ("espresso",)
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
   directly in `extension_priors` when constructing `RejectionPrior` manually.
1. A `Jitter` extension — passed as `extensions=(Jitter(param_unit=...), ...)` to the
   sampler. The sampler validates at run time that every declared extension parameter
   has a matching entry in `prior.extension_priors`.

```python
from harv.models.extensions import Jitter
from harv.samplers import RejectionSampler, RejectionPrior
from harv.distributions import QD
import numpyro.distributions as dist

# Via default_rv **kwargs:
prior = RejectionPrior.default_rv(
    period_min=Q(50, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),
    sigma_v0=Q(10, "km/s"),
    jitter=QD(dist.HalfNormal(1.0), "km/s"),  # stored in extension_priors
)
sampler = RejectionSampler(prior, extensions=(Jitter(param_unit="km/s"),))

# Or with explicit RejectionPrior construction:
prior = RejectionPrior(
    nonlinear_priors=...,
    linear_prior=...,
    extension_priors={"jitter": QD(dist.HalfNormal(1.0), "km/s")},
)
sampler = RejectionSampler(prior, extensions=(Jitter(param_unit="km/s"),))
```

For a `JointModel`, use the component-qualified key in `extension_priors`:

```python
prior = RejectionPrior(
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
`QuantityDistribution`.

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
| `prior`              | `RejectionPrior`                       | Prior distributions for sampling                             |
| `model`              | `AbstractComponentModel \| JointModel` | Model template (no data or linear prior stored)              |
| `marginalized_names` | `tuple[str, ...] \| None`              | Optional subset of linear params to analytically marginalize |
| `batch_size`         | `int` (static)                         | Samples vmapped at once (default: 100,000)                   |

`get_extensions()` walks the attached model: returns `model.extensions` for a
single component model, or `dict[component_name, tuple[Extension, ...]]` for a
`JointModel` (preserving per-component associations like `"primary.jitter"` vs
`"secondary.jitter"`). The same method is inherited by `NumpyroSampler`.

### Algorithm

1. **Prior sampling.** Draw `n_prior_samples` from the nonlinear priors in
   `RejectionPrior`. Also samples any non-Gaussian explicit linear params and
   jitter parameters from their priors.

1. **Likelihood evaluation** (batched). For each batch of `batch_size` samples,
   wrap unit-bearing parameters as `Quantity` objects and evaluate
   `jax.vmap(model.log_prob)(values)`. If `marginalized_names` is not set, the
   model auto-classifies which linear params to marginalize from its own
   `linear_prior`. If `marginalized_names` is set on the sampler, that subset is
   passed through explicitly.
   Evaluated via `jax.lax.fori_loop` to bound memory.

1. **Rejection.** Normalize weights to `max` and accept samples where
   `Uniform() < weight`.

1. **Linear parameter sampling.** For each accepted nonlinear sample, call
   `model.sample_conditional_linear(values, key)` to draw the marginalized
   linear parameters from their conditional posterior, honoring the sampler's
   `marginalized_names` override when present.

1. **Return** a `Samples` object.

### `run` method

```python
sampler.run(
    data,
    *,
    n_prior_samples: int,
    max_posterior_samples: int | None = None,
    seed: int = 0,
  ignore_non_finite: bool = False,
) -> Samples
```

`data` is the first positional argument and is passed through to `model.log_prob`
at each evaluation. For a `JointModel`, pass `data` as a dict keyed by component
name (e.g. `{"rv": rv_data, "astro": astro_data}`).

- `ignore_non_finite` -- when `True`, any `NaN` or infinite log-likelihoods
  are treated as rejected samples by replacing them with `-inf` before the
  rejection step. Default: `False`.

### `batch_size` and GPU support

The `batch_size` field controls how many samples are vmapped at once within a
`fori_loop`. On CPU, the default of 100,000 is appropriate. On GPU, set
`batch_size = n_prior_samples` to let XLA fully utilize the device.

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

Two model variants are supported via `marginalized`:

- `marginalized=True` (default): MCMC explores nonlinear subspace only; Gaussian
  linear params are analytically marginalized inside the likelihood, then
  conditionally sampled afterward to populate the returned `Samples`.
- `marginalized=False`: MCMC samples all parameters jointly (nonlinear + linear).

______________________________________________________________________

## `Samples` container (`harv.samplers.samples.Samples`)

Stores the posterior samples returned by `RejectionSampler.run()` or
`NumpyroSampler.run()`.

### Fields

| Field                    | Type                       | Description                                                  |
| ------------------------ | -------------------------- | ------------------------------------------------------------ |
| `nonlinear`              | `dict[str, Q]`             | Nonlinear parameter samples with units                       |
| `linear`                 | `dict[str, Q]`             | Linear parameter samples with units                          |
| `metadata`               | `dict[str, Any]` (static)  | Contains `t_ref` and extra info                              |
| `linear_extension_names` | `tuple[str, ...]` (static) | Linear extension param names (offsets, trends, etc.)         |
| `data_type`              | `str` (static)             | Model class name (e.g. `"RVModel"`, `"GaiaAstrometryModel"`) |

### Dict-style and index access

`samples["key"]` dispatches to appropriate unit restoration:

- Nonlinear params (`"period"`, `"eccentricity"`, `"phase_peri"`, etc.) → `Q`
  with units
- Linear params (`"rv_semiamp"`, `"v_sys"`, `"ra0"`, etc.) → `Q` with units
- Derived keys:
  - `"log_period"` → dimensionless array (`log10(period in data time units)`)
  - `"t_peri"` → `Q` (derived from `phase_peri * period + t_ref`)
  - `"inclination"` → `Q` in radians (derived from `arccos(cos_i)`)

Integer, slice, or array keys return a new `Samples` with all parameter arrays
sliced along the sample axis:

```python
samples[0]       # first sample — returns Samples with shape (1,) arrays
samples[:100]    # first 100 samples
samples[mask]    # boolean mask
```

Integer keys are promoted to length-1 slices so all arrays remain 1-d. Static
fields (`data_type`, `metadata`, `linear_extension_names`) are passed through unchanged.

### Methods

- `keys() -> list[str]` — nonlinear + linear + derived parameter names
- `n_samples -> int` — number of posterior samples
- `median(key=None)` — median of one key or all keys
- `percentile(key, percentiles=(16, 50, 84))` — compute percentiles
- `summary(params=None)` — dict of statistics (median, mean, std, p16, p84)
- `wrap_angles() -> Samples` — return a new `Samples` where any negative
  `rv_semiamp` and/or `semi_major_axis` entries are flipped to positive and the
  corresponding `arg_peri` values are shifted by `pi` (mod `2*pi`). The orbit
  predicted by the wrapped sample is identical to the original; the convention
  it enforces is `K >= 0`, `a >= 0`, `arg_peri in [0, 2*pi)`. No-op when
  `arg_peri` is missing or no entries are negative.
- `to_arviz(params=None)` -- export to `arviz.InferenceData`
- `to_hdf5(filename)` / `from_hdf5(filename)` -- HDF5 persistence
- `plot_corner(params=None, truths=None, **kwargs)` — corner plot via arviz

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
    n_samples=128,
    phase_fold_median=False,
    plot_kwargs=None,
    data_plot_kwargs=None,
    sky_orbit_kwargs=None,
    figsize=(10, 5),
    axes=None,
    **kwargs,
)
```

Draw a two-panel figure for Gaia epoch astrometry posteriors:

1. **Along-scan position vs time (or phase)** with multi-sample posterior orbit
   overlays. The median proper-motion and zero-point offsets are subtracted from
   the data so the parallax + orbital signal is visible. When `phase_fold_median`
   is true, the median parallax contribution is also subtracted (parallax has
   annual period and would smear when folded at the orbital period), and only
   the reference orbit is drawn.
1. **Sky-projected orbital ellipse** for the median-period sample (delegates to
   `plot_gaia_sky_orbit`).

When `axes=None`, a new 1x2 figure is created and returned; otherwise draws into
the two given axes and returns `None`.

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

First-call JIT compile time has not been systematically benchmarked or optimized.
Potential approaches: smaller pytrees, pre-compilation of hot paths, compile-time
caching.

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
from harv.samplers import NumpyroSampler, RejectionPrior, RejectionSampler

# --- Minimal RV-only case ---
data = RVData(time, rv, rv_err)
prior = RejectionPrior.default_rv(
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

# --- Gaia astrometry only ---
prior = RejectionPrior.default_gaia_astrometry(
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
samples = sampler.run({"rv": rv_data, "astro": gaia_data}, n_prior_samples=1_000_000)

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
harv.plot_gaia_astrometry(mcmc_samples, data=gaia_data)  # two-panel astrometry plot
harv.save_sampler("sampler.pkl", sampler)   # persist sampler
mcmc_samples.to_hdf5("out.h5")             # persistence
```
