"""Run configuration: ``hq.toml`` loaded into frozen, validated dataclasses.

See ``packages/hq/docs/spec.md``, "Configuration (``hq.toml``)". Every key in
the spec's tables is a field here, and nothing else is accepted: unknown keys,
missing required keys, and wrong types raise :class:`ConfigError` naming the
table and key.
"""

__all__ = (
    "CatalogConfig",
    "Config",
    "ConfigError",
    "DataConfig",
    "MCMCConfig",
    "PrepareConfig",
    "PriorCacheConfig",
    "RejectionConfig",
    "ResultsConfig",
    "RunConfig",
    "ServeConfig",
)

import dataclasses
import os
import tomllib
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, final

from astropy.time import Time

Kind = Literal["rv", "gaia_astrometry"]

KINDS: tuple[str, ...] = typing.get_args(Kind)
_MCMC_SELECT_RULES = ("under_resolved", "all")
_CHAIN_METHODS = ("sequential", "parallel", "vectorized")

# Keys of [data] that belong to one run kind only.
_KIND_DATA_KEYS: dict[str, tuple[str, ...]] = {
    "rv": ("rv", "rv_err", "rv_unit"),
    "gaia_astrometry": (
        "al_position",
        "al_position_err",
        "scan_angle",
        "parallax_factor",
        "al_position_unit",
        "scan_angle_unit",
    ),
}
_KIND_DATA_DEFAULTS: dict[str, dict[str, str]] = {
    "rv": {"rv_unit": "km/s"},
    "gaia_astrometry": {"al_position_unit": "mas", "scan_angle_unit": "deg"},
}


class ConfigError(ValueError):
    """An invalid ``hq.toml`` or model file; the message names the offending key."""


def _require_positive(table: str, **values: int | float) -> None:
    for key, value in values.items():
        if value <= 0:
            msg = f"[{table}] {key} must be positive, got {value!r}"
            raise ConfigError(msg)


def _require_choice(table: str, key: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        msg = f"[{table}] {key} must be one of {list(choices)}, got {value!r}"
        raise ConfigError(msg)


@final
@dataclass(frozen=True)
class RunConfig:
    """The ``[run]`` table."""

    name: str
    kind: Kind
    seed: int
    model_file: Path = Path("prior.py")

    def __post_init__(self) -> None:
        _require_choice("run", "kind", self.kind, KINDS)


@final
@dataclass(frozen=True)
class DataConfig:
    """The ``[data]`` table.

    Kind-specific keys are ``None`` for the other kind; which ones are required
    is checked against ``[run] kind`` when the config is loaded.
    """

    file: Path
    source_id: str
    time: str
    format: str | None = None
    hdu: int | str | None = None
    time_format: str = "jd"
    time_scale: str = "tdb"
    # kind = "rv"
    rv: str | None = None
    rv_err: str | None = None
    rv_unit: str | None = None
    # kind = "gaia_astrometry"
    al_position: str | None = None
    al_position_err: str | None = None
    scan_angle: str | None = None
    parallax_factor: str | None = None
    al_position_unit: str | None = None
    scan_angle_unit: str | None = None

    def __post_init__(self) -> None:
        _require_choice("data", "time_format", self.time_format, tuple(Time.FORMATS))
        _require_choice("data", "time_scale", self.time_scale, Time.SCALES)


@final
@dataclass(frozen=True)
class PrepareConfig:
    """The ``[prepare]`` table."""

    min_n_obs: int = 3

    def __post_init__(self) -> None:
        _require_positive("prepare", min_n_obs=self.min_n_obs)


@final
@dataclass(frozen=True)
class CatalogConfig:
    """The ``[catalog]`` table."""

    file: Path
    source_id: str
    format: str | None = None
    hdu: int | str | None = None
    columns: list[str] | None = None


@final
@dataclass(frozen=True)
class PriorCacheConfig:
    """The ``[prior_cache]`` table."""

    n_samples: int
    batch_size: int = 100_000

    def __post_init__(self) -> None:
        _require_positive(
            "prior_cache", n_samples=self.n_samples, batch_size=self.batch_size
        )


@final
@dataclass(frozen=True)
class RejectionConfig:
    """The ``[rejection]`` table."""

    top_k: int
    batch_size: int = 100_000
    min_evidence_ess: float = 3.0
    ignore_non_finite: bool = False
    randomize_prior_order: bool = True

    def __post_init__(self) -> None:
        _require_positive("rejection", top_k=self.top_k, batch_size=self.batch_size)


@final
@dataclass(frozen=True)
class MCMCConfig:
    """The ``[mcmc]`` table."""

    select: str = "under_resolved"
    require_unimodal: bool = True
    num_chains: int = 4
    num_warmup: int = 1000
    num_samples: int = 1000
    chain_method: str = "sequential"
    max_r_hat: float = 1.05
    min_ess_bulk: float = 400.0

    def __post_init__(self) -> None:
        _require_choice("mcmc", "select", self.select, _MCMC_SELECT_RULES)
        _require_choice("mcmc", "chain_method", self.chain_method, _CHAIN_METHODS)
        _require_positive(
            "mcmc",
            num_chains=self.num_chains,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
        )


@final
@dataclass(frozen=True)
class ResultsConfig:
    """The ``[results]`` table."""

    flush_n_sources: int = 1000
    flush_seconds: float = 600.0
    compact_n_sources: int = 100_000

    def __post_init__(self) -> None:
        _require_positive(
            "results",
            flush_n_sources=self.flush_n_sources,
            flush_seconds=self.flush_seconds,
            compact_n_sources=self.compact_n_sources,
        )


@final
@dataclass(frozen=True)
class ServeConfig:
    """The ``[serve]`` table."""

    host: str = "127.0.0.1"
    port: int = 8000


# table name -> (dataclass, required table?)
_TABLES: dict[str, tuple[type, bool]] = {
    "run": (RunConfig, True),
    "data": (DataConfig, True),
    "prepare": (PrepareConfig, False),
    "catalog": (CatalogConfig, False),
    "prior_cache": (PriorCacheConfig, True),
    "rejection": (RejectionConfig, True),
    "mcmc": (MCMCConfig, False),
    "results": (ResultsConfig, False),
    "serve": (ServeConfig, False),
}


@final
@dataclass(frozen=True)
class Config:
    """A validated run configuration, loaded from ``hq.toml``.

    Each table is an attribute holding a frozen dataclass of its keys. The
    optional ``[catalog]`` and ``[mcmc]`` tables are ``None`` when absent; the
    other optional tables take their defaults. Paths (``run.model_file``,
    ``data.file``, ``catalog.file``) are resolved against the run directory,
    the directory holding ``hq.toml``; absolute paths are kept as given.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> run_dir = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (run_dir / "hq.toml").write_text('''
    ... [run]
    ... name = "demo"
    ... kind = "rv"
    ... seed = 1
    ... [data]
    ... file = "obs.fits"
    ... source_id = "id"
    ... time = "jd"
    ... rv = "rv"
    ... rv_err = "rv_err"
    ... [prior_cache]
    ... n_samples = 1_000_000
    ... [rejection]
    ... top_k = 256
    ... ''')
    >>> config = Config.from_file(run_dir / "hq.toml")
    >>> config.run.kind, config.data.rv_unit, config.rejection.batch_size
    ('rv', 'km/s', 100000)
    >>> config.data.file.name, config.data.file.is_absolute()
    ('obs.fits', True)
    >>> config.mcmc is None
    True
    """

    config_path: Path
    run: RunConfig
    data: DataConfig
    prior_cache: PriorCacheConfig
    rejection: RejectionConfig
    prepare: PrepareConfig = PrepareConfig()
    catalog: CatalogConfig | None = None
    mcmc: MCMCConfig | None = None
    results: ResultsConfig = ResultsConfig()
    serve: ServeConfig = ServeConfig()

    @property
    def run_dir(self) -> Path:
        """The directory holding ``hq.toml``; relative paths resolve against it."""
        return self.config_path.parent

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Config":
        """Load and validate an ``hq.toml`` file.

        Parameters
        ----------
        path
            Path to the ``hq.toml`` file.

        Returns
        -------
            The validated configuration.

        Raises
        ------
        ConfigError
            If the file is not valid TOML, or any table or key is unknown,
            missing, of the wrong type, or has an invalid value.
        """
        config_path = Path(path).resolve()
        try:
            raw = tomllib.loads(config_path.read_text())
        except tomllib.TOMLDecodeError as err:
            msg = f"{config_path} is not valid TOML: {err}"
            raise ConfigError(msg) from err

        unknown = sorted(set(raw) - set(_TABLES))
        if unknown:
            msg = f"unknown table(s) or top-level key(s) {unknown} in {config_path}"
            raise ConfigError(msg)

        run_dir = config_path.parent
        tables: dict[str, Any] = {}
        for name, (table_cls, required) in _TABLES.items():
            if name not in raw:
                if required:
                    msg = f"missing required table [{name}] in {config_path}"
                    raise ConfigError(msg)
                continue
            values = raw[name]
            if not isinstance(values, dict):
                msg = f"[{name}] must be a table"
                raise ConfigError(msg)
            if name == "data":
                values = _check_data_kind(values, raw["run"].get("kind"))
            tables[name] = _build_table(name, table_cls, values, run_dir)

        return cls(config_path=config_path, **tables)


def _check_data_kind(values: dict[str, Any], kind: Any) -> dict[str, Any]:
    """Reject the other kind's [data] keys, require this kind's, apply defaults."""
    # Report a typo ("rv_er") as unknown before it can look like a missing key.
    unknown = sorted(set(values) - {f.name for f in dataclasses.fields(DataConfig)})
    if unknown:
        msg = f"[data] unknown key(s) {unknown}"
        raise ConfigError(msg)
    if kind not in _KIND_DATA_KEYS:
        return values  # RunConfig reports the bad kind
    for other, keys in _KIND_DATA_KEYS.items():
        if other == kind:
            continue
        stray = sorted(set(values) & set(keys))
        if stray:
            msg = f"[data] key(s) {stray} are for kind = {other!r}, not {kind!r}"
            raise ConfigError(msg)
    defaults = _KIND_DATA_DEFAULTS[kind]
    missing = [
        k for k in _KIND_DATA_KEYS[kind] if k not in values and k not in defaults
    ]
    if missing:
        msg = f"[data] missing required key(s) {missing} for kind = {kind!r}"
        raise ConfigError(msg)
    return {**defaults, **values}


def _build_table(
    name: str, table_cls: type, values: dict[str, Any], run_dir: Path
) -> Any:
    fields = {f.name: f for f in dataclasses.fields(table_cls)}
    hints = typing.get_type_hints(table_cls)

    unknown = sorted(set(values) - set(fields))
    if unknown:
        msg = f"[{name}] unknown key(s) {unknown}"
        raise ConfigError(msg)
    required = [
        k
        for k, f in fields.items()
        if f.default is dataclasses.MISSING and k not in values
    ]
    if required:
        msg = f"[{name}] missing required key(s) {required}"
        raise ConfigError(msg)

    kwargs = {
        key: _convert(name, key, value, hints[key], run_dir)
        for key, value in values.items()
    }
    # Defaulted paths (e.g. run.model_file) resolve against run_dir too.
    for key, f in fields.items():
        if key not in kwargs and isinstance(f.default, Path):
            kwargs[key] = run_dir / f.default
    return table_cls(**kwargs)


def _convert(table: str, key: str, value: Any, hint: Any, run_dir: Path) -> Any:
    """Check a TOML value against a field's annotation and convert it."""
    origin = typing.get_origin(hint)
    if hint is Path:
        if not isinstance(value, str):
            raise _type_error(table, key, value, "a path string")
        return run_dir / value  # an absolute value replaces run_dir
    if origin in (typing.Union, types.UnionType):
        return _convert_union(table, key, value, hint, run_dir)
    if origin is list:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise _type_error(table, key, value, _describe(hint))
        return list(value)
    if origin is Literal:
        hint = str  # the choice itself is checked by the dataclass
    # bool is an int subclass in Python, but not in TOML or in this schema.
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, hint) or (hint is int and isinstance(value, bool)):
        raise _type_error(table, key, value, _describe(hint))
    return value


def _convert_union(table: str, key: str, value: Any, hint: Any, run_dir: Path) -> Any:
    """Convert to the first option of ``X | Y | None`` that accepts the value."""
    for option in typing.get_args(hint):
        if option is type(None):
            continue
        try:
            return _convert(table, key, value, option, run_dir)
        except ConfigError:
            continue
    raise _type_error(table, key, value, _describe(hint))


def _describe(hint: Any) -> str:
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return " or ".join(
            _describe(a) for a in typing.get_args(hint) if a is not type(None)
        )
    if typing.get_origin(hint) is list:
        return "a list of strings"  # list[str] is the only list field
    return {str: "a string", int: "an integer", float: "a number", bool: "a boolean"}[
        hint
    ]


def _type_error(table: str, key: str, value: Any, expected: str) -> ConfigError:
    return ConfigError(f"[{table}] {key} must be {expected}, got {value!r}")
