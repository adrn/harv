"""Shared fixtures: small survey-style runs built from harv's simulators.

``rv_run`` and ``gaia_run`` write an input table (one row per observation, as a
survey would deliver it), a catalog, and an ``hq.toml`` + ``prior.py`` from the
``hq init`` templates, and return the run directory with the expected
contents. Later phases build their end-to-end tests on these.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from unxt import Q, ustrip

from harv.simulate import simulate_gaia_epoch_astrometry, simulate_rv_sb1_data
from harv_hq import init_run

JD_OFFSET = 2_460_000.0


@dataclass
class RunFixture:
    run_dir: Path
    table: Table  # the input table as written, before any cuts
    catalog: Table
    n_obs: dict  # source_id -> observations that survive the built-in cuts


def _rv_ids(n):
    # APOGEE-style IDs with "+", plus two with "/" (no characters are reserved).
    ids = [f"2M{i:08d}+{7 * i:07d}" for i in range(n - 2)]
    return [*ids, "J1234/5678", "J8765/4321"]


def _write_config(run_dir, kind, data_file, extra=""):
    path = run_dir / "hq.toml"
    text = path.read_text()
    text = text.replace(
        'file = "observations.fits"'
        if kind == "rv"
        else 'file = "epoch_astrometry.fits"',
        f'file = "{data_file}"',
    )
    text = text.replace("n_samples = 10_000_000", "n_samples = 20_000")
    text = text.replace("top_k = 512", "top_k = 64")
    path.write_text(text + extra)


@pytest.fixture
def rv_run(tmp_path):
    """An RV run: 20 sources with string IDs, 2-10 epochs, a few bad rows."""
    run_dir = init_run(tmp_path / "run", kind="rv")
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
    return RunFixture(run_dir=run_dir, table=table, catalog=catalog, n_obs=n_obs)


@pytest.fixture
def gaia_run(tmp_path):
    """A Gaia astrometry run: 6 sources with 19-digit integer IDs."""
    run_dir = init_run(tmp_path / "run", kind="gaia_astrometry")
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
    return RunFixture(run_dir=run_dir, table=table, catalog=Table(), n_obs=n_obs)
