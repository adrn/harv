"""Shared fixtures: small survey-style runs built from harv's simulators.

``rv_run``, ``gaia_run`` and ``sb2_run`` write an input table (one row per
observation, as a survey would deliver it), a catalog, and an ``hq.toml`` +
``prior.py`` from the ``hq init`` templates, and return the run directory with
the expected contents. Later phases build their end-to-end tests on these.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from unxt import Q, ustrip

from harv.kepler.orbits import rv_at_times
from harv.simulate import simulate_gaia_epoch_astrometry, simulate_rv_sb1_data
from harv_hq import init_run

JD_OFFSET = 2_460_000.0


@dataclass
class RunFixture:
    run_dir: Path
    table: Table  # the input table as written, before any cuts
    catalog: Table
    n_obs: dict  # source_id -> observations that survive the built-in cuts
    kept: dict  # the n_obs entries that also pass min_n_obs (the prepared sources)


def _rv_ids(n):
    # APOGEE-style IDs with "+", plus two with "/" (no characters are reserved).
    ids = [f"2M{i:08d}+{7 * i:07d}" for i in range(n - 2)]
    return [*ids, "J1234/5678", "J8765/4321"]


def _write_config(run_dir, kind, data_file, extra=""):
    path = run_dir / "hq.toml"
    text = path.read_text()
    text = text.replace(
        'file = "epoch_astrometry.fits"'
        if kind == "gaia_astrometry"
        else 'file = "observations.fits"',
        f'file = "{data_file}"',
    )
    text = text.replace("n_samples = 10_000_000", "n_samples = 20_000")
    text = text.replace("top_k = 512", "top_k = 64")
    path.write_text(text + extra)


@pytest.fixture
def rv_run(tmp_path):
    """An RV run: 20 sources with string IDs, 2-10 epochs, a few bad rows."""
    return make_rv_run(tmp_path)


@pytest.fixture(scope="session")
def rv_run_factory():
    """``make_rv_run``, for fixtures with a wider scope than ``rv_run``."""
    return make_rv_run


def make_rv_run(base):
    """Write the ``rv_run`` fixture's run under ``base`` (for wider scopes)."""
    run_dir = init_run(base / "run", kind="rv")
    ids = _rv_ids(20)
    rows = {"source_id": [], "time": [], "rv": [], "rv_err": [], "snr": []}
    n_obs = {}
    for i, sid in enumerate(ids):
        n = 2 if i in (0, 9) else 3 + (i % 8)  # two sources fall below min_n_obs
        data, _ = simulate_rv_sb1_data(seed=i, n_obs=n)
        rows["source_id"] += [sid] * n
        rows["time"] += list(np.asarray(ustrip("day", data.time)) + JD_OFFSET)
        rows["rv"] += list(np.asarray(ustrip("km/s", data.rv)))
        rows["rv_err"] += list(np.asarray(ustrip("km/s", data.rv_err)))
        rows["snr"] += [50.0] * n
        n_obs[sid] = n

    table = Table(rows)
    # Rows the built-in cuts drop (non-finite RV, zero uncertainty), and one a
    # select_rows hook on "snr" would drop.
    bad = ids[3]
    table.add_row(
        {
            "source_id": bad,
            "time": JD_OFFSET + 1,
            "rv": np.nan,
            "rv_err": 1.0,
            "snr": 50.0,
        }
    )
    table.add_row(
        {"source_id": bad, "time": JD_OFFSET + 2, "rv": 1.0, "rv_err": 0.0, "snr": 50.0}
    )
    table.add_row(
        {"source_id": bad, "time": JD_OFFSET + 3, "rv": 1.0, "rv_err": 1.0, "snr": 1.0}
    )
    n_obs[bad] += 1  # the low-snr row survives the built-in cuts
    table["rv"].unit = "km/s"
    table["rv_err"].unit = "km/s"
    table.write(run_dir / "observations.ecsv")

    # Catalog: two sources missing, one extra source not in the data.
    catalog = Table(
        {
            "source_id": [*ids[2:], "2M99999999+9999999"],
            "bp_rp": np.linspace(0.5, 2.0, len(ids) - 1),
            "abs_g": np.linspace(-1.0, 8.0, len(ids) - 1),
        }
    )
    catalog["abs_g"].unit = "mag"
    catalog.write(run_dir / "catalog.ecsv")

    _write_config(
        run_dir,
        "rv",
        "observations.ecsv",
        extra='\n[catalog]\nfile = "catalog.ecsv"\nsource_id = "source_id"\n',
    )
    kept = {sid: n for sid, n in n_obs.items() if n >= 3}  # template min_n_obs = 3
    return RunFixture(
        run_dir=run_dir, table=table, catalog=catalog, n_obs=n_obs, kept=kept
    )


@pytest.fixture
def gaia_run(tmp_path):
    """A Gaia astrometry run: 6 sources with 19-digit integer IDs."""
    return make_gaia_run(tmp_path)


@pytest.fixture(scope="session")
def gaia_run_factory():
    """``make_gaia_run``, for fixtures with a wider scope than ``gaia_run``."""
    return make_gaia_run


def make_gaia_run(base):
    """Write the ``gaia_run`` fixture's run under ``base`` (for wider scopes)."""
    run_dir = init_run(base / "run", kind="gaia_astrometry")
    rows = {k: [] for k in ("source_id", "time", "al", "al_err", "psi", "plx_factor")}
    n_obs = {}
    for i in range(6):
        sid = 4_295_806_720_000_000_000 + i
        n = 12 + i
        data, _ = simulate_gaia_epoch_astrometry(
            seed=i, n_obs=n, parallax=Q(20.0, "mas"), al_error=Q(0.1, "mas")
        )
        rows["source_id"] += [sid] * n
        rows["time"] += list(np.asarray(ustrip("day", data.time)) + JD_OFFSET)
        rows["al"] += list(np.asarray(ustrip("mas", data.al_position)))
        rows["al_err"] += list(np.asarray(ustrip("mas", data.al_position_err)))
        rows["psi"] += list(np.asarray(ustrip("rad", data.scan_angle)))
        rows["plx_factor"] += list(np.asarray(data.parallax_factor))
        n_obs[sid] = n
    table = Table(rows)
    table["psi"].unit = "rad"  # overrides the config's scan_angle_unit = "deg"
    table.write(run_dir / "epoch_astrometry.ecsv")

    path = run_dir / "hq.toml"
    text = path.read_text()
    for old, new in [
        ('al_position = "al_position"', 'al_position = "al"'),
        ('al_position_err = "al_position_err"', 'al_position_err = "al_err"'),
        ('scan_angle = "scan_angle"', 'scan_angle = "psi"'),
        ('parallax_factor = "parallax_factor"', 'parallax_factor = "plx_factor"'),
    ]:
        text = text.replace(old, new)
    path.write_text(text)
    _write_config(run_dir, "gaia_astrometry", "epoch_astrometry.ecsv")
    return RunFixture(  # every source has >= 12 epochs; template min_n_obs = 10
        run_dir=run_dir, table=table, catalog=Table(), n_obs=n_obs, kept=n_obs
    )


# The orbit every sb2_run source shares. hq's mass_ratio is
# |K_primary| / |K_secondary| = M_secondary / M_primary.
SB2_ORBIT = {
    "period": Q(137.0, "day"),
    "eccentricity": 0.25,
    "arg_peri": Q(1.1, "rad"),
    "time_peri": Q(11.0, "day"),
    "v_sys": Q(-8.0, "km/s"),
}
SB2_SEMIAMPS = (Q(28.0, "km/s"), Q(41.0, "km/s"))
SB2_MASS_RATIO = 28.0 / 41.0
# source_id -> (primary epochs, secondary epochs)
SB2_EPOCHS = {
    101: (24, 24),  # well sampled
    102: (10, 6),
    103: (8, 2),  # sparse secondary: kept, the primary has min_n_obs
    104: (6, 0),  # no secondary: dropped
    105: (2, 2),  # neither component has min_n_obs: dropped
}


@pytest.fixture
def sb2_run(tmp_path):
    """An SB2 run: 5 sources with integer IDs, long format, labels 1 and 2."""
    return make_sb2_run(tmp_path)


@pytest.fixture(scope="session")
def sb2_run_factory():
    """``make_sb2_run``, for fixtures with a wider scope than ``sb2_run``."""
    return make_sb2_run


def make_sb2_run(base):
    """Write the ``sb2_run`` fixture's run under ``base`` (for wider scopes).

    Both stars follow one orbit, the secondary at ``arg_peri + pi``, as in
    harv's SB2 integration test. Each source's secondary is measured at a
    subset of the primary's epochs, so some epochs have one row, some two.
    """
    run_dir = init_run(base / "run", kind="sb2")
    rows = {k: [] for k in ("source_id", "time", "rv", "rv_err", "component")}
    n_obs = {}
    for sid, (n_1, n_2) in SB2_EPOCHS.items():
        rng = np.random.default_rng(sid)
        times = Q(np.sort(rng.uniform(0.0, 600.0, n_1)), "day")
        for label, n, semiamp, shift, err in (
            (1, n_1, SB2_SEMIAMPS[0], 0.0, 0.6),
            (2, n_2, SB2_SEMIAMPS[1], np.pi, 0.8),
        ):
            t = times[np.sort(rng.choice(n_1, n, replace=False))]
            rv = rv_at_times(
                t,
                SB2_ORBIT["period"],
                SB2_ORBIT["eccentricity"],
                SB2_ORBIT["time_peri"],
                SB2_ORBIT["arg_peri"] + Q(shift, "rad"),
                semiamp,
                SB2_ORBIT["v_sys"],
            )
            rows["source_id"] += [sid] * n
            rows["time"] += list(np.asarray(ustrip("day", t)) + JD_OFFSET)
            rows["rv"] += list(
                np.asarray(ustrip("km/s", rv)) + err * rng.normal(size=n)
            )
            rows["rv_err"] += [err] * n
            rows["component"] += [label] * n
        n_obs[sid] = n_1 + n_2
    table = Table(rows)
    table.write(run_dir / "observations.ecsv")

    # A period range around the injected one, so a small cache resolves it.
    prior = run_dir / "prior.py"
    prior.write_text(
        prior.read_text()
        .replace('period_min=Q(2, "day")', 'period_min=Q(64, "day")')
        .replace('period_max=Q(4096, "day")', 'period_max=Q(256, "day")')
    )
    _write_config(run_dir, "sb2", "observations.ecsv")
    kept = {sid: n_obs[sid] for sid in (101, 102, 103)}
    return RunFixture(
        run_dir=run_dir, table=table, catalog=Table(), n_obs=n_obs, kept=kept
    )
