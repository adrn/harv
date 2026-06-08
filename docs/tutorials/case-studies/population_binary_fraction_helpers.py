"""Helper function for the binary population inference case study tutorial."""

import pathlib

import h5py
import numpy as np
from unxt import Q

from harv.data import RVData


def load_population(path: pathlib.Path | str) -> tuple[list[RVData], dict]:
    """Load the simulated population produced by ``simulate-population-data.ipynb``.

    Returns
    -------
    datasets
        List of ``RVData``, one per star.  All entries share the same
        number of epochs so the rejection sampler's JIT cache is reused
        across stars.
    truths
        Per-star and population-level truth values:

        - per-star arrays ``M1`` (Msun), ``M2`` (Msun, NaN for singles),
          ``period`` (day, NaN), ``eccentricity`` (NaN), ``sini`` (NaN),
          ``arg_peri`` (rad, NaN), ``t_peri`` (day, NaN), ``K`` (km/s,
          0 for singles), ``v_sys`` (km/s), and boolean ``is_binary`` /
          ``is_close_binary``;
        - scalars ``simulation_binary_fraction`` (raw 0.40 input),
          ``binary_fraction_true`` (recovered close-binary truth, the
          target of Section 4), ``ecc_mean_true`` / ``ecc_std_true``
          (mean and stddev of the truth TruncatedNormal[0, 1] eccentricity
          distribution — target of Section 3), ``P_min_cut_day``,
          ``P_max_cut_day``, ``M2_cut_msun``, ``seed``;
        - convenience ``ecc_dist_true = (mean, std)`` tuple.
    """
    datasets: list[RVData] = []
    truths: dict = {}
    with h5py.File(path, "r") as f:
        t_arr = np.asarray(f["time"][:])
        rv_arr = np.asarray(f["rv"][:])
        e_arr = np.asarray(f["rv_err"][:])
        t_unit = f["time"].attrs["unit"]
        rv_unit = f["rv"].attrs["unit"]
        e_unit = f["rv_err"].attrs["unit"]
        for n in range(t_arr.shape[0]):
            datasets.append(  # noqa: PERF401
                RVData(
                    time=Q(t_arr[n], t_unit),
                    rv=Q(rv_arr[n], rv_unit),
                    rv_err=Q(e_arr[n], e_unit),
                )
            )
        g = f["truths"]
        for name in g:
            truths[name] = np.asarray(g[name][:])
        for k in g.attrs:
            v = g.attrs[k]
            truths[k] = float(v) if isinstance(v, (np.floating, float)) else v
    truths["ecc_dist_true"] = (truths["ecc_mean_true"], truths["ecc_std_true"])
    return datasets, truths
