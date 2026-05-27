"""Chunked HDF5 prior-sample cache.

Writes a large library of prior samples to disk without materializing
``n_samples`` rows in memory.  The on-disk layout matches
:meth:`harv.samplers.Samples.to_hdf5` exactly, so a prior cache is just a
:class:`~harv.samplers.Samples` file with ``ln_likelihood`` absent and the
``linear`` group typically empty.  See the module-level docstring of
:mod:`harv.samplers.samples` for the full layout.

Use :meth:`harv.samplers.RejectionSampler.run_with_samples` to consume the
cache from disk, batch by batch.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import jax
import jax.random as jr
import numpy as np

if TYPE_CHECKING:
    from harv.models.component import AbstractComponentModel
    from harv.models.joint import JointModel
    from harv.models.priors import HarvPrior
    from harv.samplers.samples import Samples

__all__ = ("make_prior_cache",)


def make_prior_cache(
    prior: "HarvPrior",
    model: "AbstractComponentModel | JointModel",
    n_samples: int,
    filename: str | os.PathLike[str],
    *,
    key: jax.Array,
    batch_size: int = 100_000,
    return_logprobs: bool = False,
) -> None:
    """Write ``n_samples`` prior draws to an HDF5 cache, in batches.

    Each batch is drawn from an independent subkey via
    :func:`jax.random.fold_in`, so the on-disk order is i.i.d. and sequential
    reads of the file are statistically equivalent to random reads.

    Parameters
    ----------
    prior
        Prior to draw from.
    model
        Component or joint model template defining which extension parameters
        need priors.  See :meth:`HarvPrior.sample`.
    n_samples
        Total number of samples to write.
    filename
        Output HDF5 path.
    key
        JAX random key.  Independent subkeys per batch are derived via
        :func:`jax.random.fold_in`.
    batch_size
        Number of samples generated and written per chunk.  Memory usage scales
        with ``batch_size``, not ``n_samples``.  Default: 100,000.
    return_logprobs
        If ``True``, write ``ln_prior`` to the cache so it can be reused by
        downstream tools without recomputation.

    Examples
    --------
    >>> import jax
    >>> from unxt import Q
    >>> from harv import HarvPrior, RVModel, StandardRV
    >>> from harv.samplers import make_prior_cache
    >>> prior = StandardRV().default_prior(  # doctest: +SKIP
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> make_prior_cache(  # doctest: +SKIP
    ...     prior, RVModel(), 1_000_000, "cache.h5",
    ...     key=jax.random.key(0), batch_size=100_000,
    ... )
    """
    if n_samples <= 0:
        msg = f"n_samples must be positive, got {n_samples}."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}."
        raise ValueError(msg)

    path = Path(os.fspath(filename))

    n_batches = (n_samples + batch_size - 1) // batch_size

    # Probe the prior for keys / units / linear_extension_names / ln_prior
    # presence by drawing a single 1-sample batch.  This avoids hand-tracking
    # the same prior-resolution logic that lives in HarvPrior.sample.
    probe = prior.sample(key, 1, model=model, return_logprobs=return_logprobs)
    nonlinear_keys = list(probe.nonlinear)
    linear_keys = list(probe.linear)
    nonlinear_units = {k: str(probe.nonlinear[k].unit) for k in nonlinear_keys}
    linear_units = {k: str(probe.linear[k].unit) for k in linear_keys}

    with h5py.File(path, "w") as f:
        nl_group = f.create_group("nonlinear")
        lin_group = f.create_group("linear")
        meta_group = f.create_group("metadata")
        meta_group.attrs["data_type"] = type(model).__name__
        meta_group.attrs["linear_extension_names"] = ",".join(
            probe.linear_extension_names
        )
        meta_group.attrs["n_samples"] = n_samples

        nl_datasets: dict[str, h5py.Dataset] = {}
        for name in nonlinear_keys:
            ds = nl_group.create_dataset(name, shape=(n_samples,), dtype="float32")
            ds.attrs["unit"] = nonlinear_units[name]
            nl_datasets[name] = ds

        lin_datasets: dict[str, h5py.Dataset] = {}
        for name in linear_keys:
            ds = lin_group.create_dataset(name, shape=(n_samples,), dtype="float32")
            ds.attrs["unit"] = linear_units[name]
            lin_datasets[name] = ds

        lp_dataset: h5py.Dataset | None = None
        if return_logprobs:
            lp_dataset = f.create_dataset(
                "ln_prior", shape=(n_samples,), dtype="float32"
            )

        for i in range(n_batches):
            start = i * batch_size
            stop = min(start + batch_size, n_samples)
            n_this = stop - start

            sub_key = jr.fold_in(key, i)
            batch: Samples = prior.sample(
                sub_key, n_this, model=model, return_logprobs=return_logprobs
            )

            for name in nonlinear_keys:
                nl_datasets[name][start:stop] = np.asarray(batch.nonlinear[name].value)
            for name in linear_keys:
                lin_datasets[name][start:stop] = np.asarray(batch.linear[name].value)
            if lp_dataset is not None and batch.ln_prior is not None:
                lp_dataset[start:stop] = np.asarray(batch.ln_prior)
