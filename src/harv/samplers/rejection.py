"""Rejection sampler for orbital parameter inference."""

import os
import uuid
from pathlib import Path
from typing import Any, NamedTuple, cast, final

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from unxt import AbstractQuantity, Q
from unxt.quantity import ustrip

from harv.data.containers import InputData
from harv.distributions import QuantityDistribution
from harv.models._helpers import (
    _evaluate_nonlinear_log_prior,
    _needs_explicit_sampling,
    _unwrap_dist,
)
from harv.models.component import AbstractComponentModel
from harv.models.joint import JointModel
from harv.models.priors import HarvPrior
from harv.samplers._prior_resolution import (
    effective_linear_prior_from_prior as _effective_linear_prior_from_prior,
)
from harv.samplers._prior_resolution import (
    nonlinear_extension_priors_from_model as _nonlinear_extension_priors_from_model,
)
from harv.samplers._prior_resolution import (
    resolve_effective_marginalized_names as _resolve_effective_marginalized_names,
)
from harv.samplers._prior_resolution import (
    validate_extension_priors as _validate_extension_priors,
)
from harv.samplers.base import AbstractSampler, _validate_data
from harv.samplers.samples import Samples

__all__ = ("RejectionSampler",)


class _PreparedSamplerModel(NamedTuple):
    """Normalized model-preparation bundle shared across sampler entry paths."""

    model: AbstractComponentModel | JointModel
    nonlinear_extension_priors: dict[str, Any]
    effective_linear_prior: dict[str, Any] | None
    effective_marginalized_names: tuple[str, ...] | None
    linear_extension_names: tuple[str, ...]


def _prepare_sampler_model(
    prior: HarvPrior,
    model: AbstractComponentModel | JointModel,
    marginalized_names: tuple[str, ...] | None,
) -> _PreparedSamplerModel:
    """Prepare a normalized model/prior bundle for rejection or MCMC sampling.

    Walks the attached ``model`` to extract nonlinear extension priors and
    computes the effective linear prior at run-time from ``prior.linear_prior``
    plus any linear-extension parameters declared on the model's extensions.
    """
    nonlinear_extension_priors, linear_extension_names = (
        _nonlinear_extension_priors_from_model(prior, model)
    )
    effective_linear_prior = _effective_linear_prior_from_prior(prior, model)
    _validate_extension_priors(prior, model, effective_linear_prior)
    effective_marginalized_names = _resolve_effective_marginalized_names(
        effective_linear_prior,
        marginalized_names,
    )
    return _PreparedSamplerModel(
        model=model,
        nonlinear_extension_priors=nonlinear_extension_priors,
        effective_linear_prior=effective_linear_prior,
        effective_marginalized_names=effective_marginalized_names,
        linear_extension_names=linear_extension_names,
    )


def _wrap_unit_values(
    values: dict[str, Any],
    nonlinear_priors: dict[str, Any],
    base_names: frozenset[str],
) -> dict[str, Any]:
    """Wrap QuantityDistribution-sampled base params in Q objects.

    Extension params (jitter, etc.) are left as raw scalars.
    TODO: why are the extension params left as raw scalars?
    """
    result = dict(values)
    for name, d in nonlinear_priors.items():
        if isinstance(d, QuantityDistribution) and name in base_names:
            result[name] = Q(result[name], str(d.unit))
    return result


@final
class RejectionSampler(AbstractSampler):
    """Rejection sampler for Keplerian orbital parameters.

    Implements rejection sampling with analytic marginalization over linear
    parameters. Configure once, then call :meth:`run` with each dataset.

    Parameters
    ----------
    prior
        Prior distributions for nonlinear (and optionally linear) parameters.
    model
        Fully constructed model template (no data, no linear_priors).
    marginalized_names
        Linear parameter names to analytically marginalize. If None, all
        Gaussian linear parameters are auto-classified for marginalization.
    batch_size
        Number of samples to process per batch. Smaller values use less memory
        but may be slower. Default: 100_000.

    Examples
    --------
    >>> from unxt import Q
    >>> import jax.numpy as jnp
    >>> from harv import HarvPrior, RejectionSampler, RVData
    >>> from harv.models.rv import RVModel
    >>> data = RVData(  # doctest: +SKIP
    ...     time=Q(jnp.linspace(0, 100, 5), "day"),
    ...     rv=Q(jnp.zeros(5), "km/s"),
    ...     rv_err=Q(jnp.full(5, 1.0), "km/s"),
    ... )
    >>> prior = HarvPrior.default_rv(  # doctest: +SKIP
    ...     period_min=Q(2.0, "day"),
    ...     period_max=Q(1000.0, "day"),
    ...     sigma_K0=Q(30.0, "km/s"),
    ...     sigma_v0=Q(50.0, "km/s"),
    ... )
    >>> sampler = RejectionSampler(prior, RVModel())  # doctest: +SKIP
    >>> samples = sampler.run(data, n_prior_samples=100_000)  # doctest: +SKIP
    """

    batch_size: int = eqx.field(static=True, default=100_000)

    def run(
        self,
        data: InputData,
        *,
        n_prior_samples: int,
        max_posterior_samples: int | None = None,
        seed: int | None = None,
        ignore_non_finite: bool = False,
        return_logprobs: bool = False,
    ) -> Samples:
        """Run rejection sampling.

        Parameters
        ----------
        data
            Observed data: an :class:`~harv.data.AbstractData` subclass
            (e.g. :class:`~harv.data.RVData`,
            :class:`~harv.data.GaiaAstrometryData`) for single-component
            models, or an :class:`~harv.data.AbstractDatasetContainer`
            (e.g. :class:`~harv.data.SystemData`,
            :class:`~harv.data.SourceData`) for :class:`~harv.JointModel`.
        n_prior_samples
            Number of samples to draw from the prior.
        max_posterior_samples
            Maximum number of posterior samples to return. If None, returns all
            accepted samples.
        seed
            Random number seed. If not specified, picks a seed based on the
            current time.
        ignore_non_finite
            If ``True``, any ``NaN`` or infinite log-likelihood values are
            treated as rejected samples by replacing them with ``-inf`` before
            the rejection step. If ``False`` (default), non-finite values are
            left unchanged.
        return_logprobs
            If ``True``, store per-sample log-probabilities on the returned
            :class:`~harv.samplers.samples.Samples`: ``ln_likelihood`` (the
            marginal log-likelihood) and ``ln_prior`` (the summed nonlinear
            prior log-density).  These enable :meth:`Samples.map_sample` and
            the :attr:`Samples.ln_posterior` property.  Default ``False``.

        Returns
        -------
            Posterior samples container.
        """
        _validate_data(data, self.model)

        prepared = _prepare_sampler_model(
            self.prior,
            self.model,
            self.marginalized_names,
        )

        # if not specified, pick a different random seed each run:
        _seed: int = uuid.uuid4().int >> 96 if seed is None else seed

        key = jr.key(_seed)
        sample_key, rej_key = jr.split(key)

        # generate prior samples and evaluate (marginalized) log likelihoods in batches
        # TODO: only return accepted samples and acceptance rate to conserve memory?

        prior_samples, log_likelihoods = self._sample_prior_and_evaluate_batched(
            prepared.model,
            sample_key,
            n_prior_samples,
            prepared.nonlinear_extension_priors,
            prepared.effective_linear_prior or {},
            prepared.effective_marginalized_names,
            data,
        )

        return self._finalize_posterior(
            data=data,
            prepared=prepared,
            prior_samples=prior_samples,
            log_likelihoods=log_likelihoods,
            rej_key=rej_key,
            key=key,
            ignore_non_finite=ignore_non_finite,
            max_posterior_samples=max_posterior_samples,
            return_logprobs=return_logprobs,
        )

    def _finalize_posterior(
        self,
        *,
        data: Any,
        prepared: _PreparedSamplerModel,
        prior_samples: dict[str, jax.Array],
        log_likelihoods: jax.Array,
        rej_key: jax.Array,
        key: jax.Array,
        ignore_non_finite: bool,
        max_posterior_samples: int | None,
        return_logprobs: bool,
    ) -> Samples:
        """Shared downstream: rejection -> linear -> trim -> Samples assembly.

        Both :meth:`run` and :meth:`run_with_samples` call this once the
        flat ``prior_samples`` dict and matching ``log_likelihoods`` array
        have been produced.  ``key`` seeds the linear-sampling and
        max-posterior-samples subsamples (via :func:`jax.random.fold_in`);
        ``rej_key`` seeds the per-sample uniform draws.
        """
        model = prepared.model
        nonlinear_extension_priors = prepared.nonlinear_extension_priors
        effective_linear_prior = prepared.effective_linear_prior or {}
        effective_marginalized_names = prepared.effective_marginalized_names
        linear_extension_names = prepared.linear_extension_names

        if ignore_non_finite:
            log_likelihoods = jnp.where(
                jnp.isfinite(log_likelihoods), log_likelihoods, -jnp.inf
            )

        accepted_mask = self._rejection_step(rej_key, log_likelihoods)
        accepted_nonlinear = {k: v[accepted_mask] for k, v in prior_samples.items()}
        accepted_log_likelihood = log_likelihoods[accepted_mask]

        linear_key = jr.fold_in(key, 2)
        # TODO: support oversampling of linear parameters?
        linear_samples = self._sample_linear_parameters(
            model,
            linear_key,
            accepted_nonlinear,
            effective_marginalized_names,
            data,
            effective_linear_prior,
        )

        if max_posterior_samples is not None:
            n_accepted = len(next(iter(accepted_nonlinear.values())))
            if n_accepted > max_posterior_samples:
                idx_key = jr.fold_in(key, 3)
                idx = jr.choice(
                    idx_key,
                    n_accepted,
                    shape=(max_posterior_samples,),
                    replace=False,
                )
                accepted_nonlinear = {k: v[idx] for k, v in accepted_nonlinear.items()}
                linear_samples = {k: v[idx] for k, v in linear_samples.items()}
                accepted_log_likelihood = accepted_log_likelihood[idx]

        # Build nonlinear dict as Quantities with units from the prior.
        # Base orbital params come from prior.nonlinear_priors.
        # Extension nonlinear params (e.g. jitter) come from the prepared model-key map.
        _all_nl_priors: dict[str, Any] = dict(self.prior.nonlinear_priors)
        _all_nl_priors.update(nonlinear_extension_priors)

        nonlinear_q: dict[str, AbstractQuantity] = {}
        for k, v in accepted_nonlinear.items():
            if k not in _all_nl_priors:
                continue
            d = _all_nl_priors[k]
            unit = str(d.unit) if isinstance(d, QuantityDistribution) else ""
            nonlinear_q[k] = Q(v, unit)

        # t_ref is uniformly exposed by both AbstractData and
        # AbstractDatasetContainer; no branching needed.
        t_ref = data.t_ref

        metadata: dict[str, Any] = {}
        if t_ref is not None:
            # Strip to a plain Python float so a JAX-traced array never lands in a
            # static metadata dict (which would trigger an equinox UserWarning).
            _t_unit = str(t_ref.unit)
            metadata["t_ref"] = float(ustrip(_t_unit, t_ref))
            metadata["t_ref_unit"] = _t_unit

        ln_likelihood_arr: jax.Array | None = None
        ln_prior_arr: jax.Array | None = None
        if return_logprobs:
            ln_likelihood_arr = accepted_log_likelihood
            ln_prior_arr = _evaluate_nonlinear_log_prior(
                _all_nl_priors, accepted_nonlinear
            )

        return Samples(
            nonlinear=cast("dict[str, Q]", nonlinear_q),
            linear=cast("dict[str, Q]", linear_samples),
            data_type=type(model).__name__,
            metadata=metadata,
            linear_extension_names=linear_extension_names,
            ln_likelihood=ln_likelihood_arr,
            ln_prior=ln_prior_arr,
        )

    def run_with_samples(
        self,
        data: InputData,
        prior_samples: "Samples | str | os.PathLike[str]",
        *,
        max_posterior_samples: int | None = None,
        seed: int | None = None,
        ignore_non_finite: bool = False,
        return_logprobs: bool = False,
        randomize_prior_order: bool = True,
    ) -> Samples:
        """Run rejection against pre-computed prior samples.

        Parallel to :meth:`run` but skips the prior-sampling step: the caller
        supplies the prior draws either in memory (a :class:`Samples` returned
        by :meth:`HarvPrior.sample`) or as a path to an HDF5 file produced by
        :func:`~harv.samplers.make_prior_cache`. Useful when the same prior
        library is reused across many datasets — generate once, evaluate many
        times.

        Parameters
        ----------
        data
            Observed data; same conventions as :meth:`run`.
        prior_samples
            Either a :class:`Samples` instance (in-memory cache, e.g. from
            ``prior.sample(...)``) or an ``str`` / ``os.PathLike`` path to an
            HDF5 file. The HDF5 path is streamed batch by batch, so the file
            may be much larger than RAM.
        max_posterior_samples
            See :meth:`run`.
        seed
            See :meth:`run`.
        ignore_non_finite
            See :meth:`run`.
        return_logprobs
            See :meth:`run`.
        randomize_prior_order
            Disk-streaming branch only: when ``True`` (default), batches are
            read from the HDF5 file in a random order (drawn from ``seed``).
            Each batch is still a single contiguous h5py slice, so disk I/O
            is unchanged. Set to ``False`` for strictly sequential reads
            (reproducibility / debugging). Ignored for the in-memory branch.

        Returns
        -------
        Samples
            Posterior samples, equivalent in shape and contents to
            :meth:`run`'s return.
        """
        _validate_data(data, self.model)

        prepared = _prepare_sampler_model(
            self.prior,
            self.model,
            self.marginalized_names,
        )

        _seed: int = uuid.uuid4().int >> 96 if seed is None else seed
        key = jr.key(_seed)
        rej_key = jr.fold_in(key, 1)

        if isinstance(prior_samples, Samples):
            flat_samples, log_likelihoods = self._evaluate_in_memory(
                prepared, prior_samples, data
            )
        else:
            flat_samples, log_likelihoods = self._evaluate_from_hdf5(
                prepared,
                Path(os.fspath(prior_samples)),
                data,
                seed=_seed,
                randomize_prior_order=randomize_prior_order,
            )

        return self._finalize_posterior(
            data=data,
            prepared=prepared,
            prior_samples=flat_samples,
            log_likelihoods=log_likelihoods,
            rej_key=rej_key,
            key=key,
            ignore_non_finite=ignore_non_finite,
            max_posterior_samples=max_posterior_samples,
            return_logprobs=return_logprobs,
        )

    def _expected_prior_keys(
        self, prepared: _PreparedSamplerModel
    ) -> tuple[set[str], set[str]]:
        """Return ``(expected_nonlinear_keys, expected_explicit_linear_keys)``.

        The flat sample dict the sampler consumes contains base nonlinear +
        extension nonlinear under the first set, plus any non-Gaussian
        explicit-linear params under the second set.
        """
        nonlinear_keys = set(self.prior.nonlinear_priors) | set(
            prepared.nonlinear_extension_priors
        )
        explicit_linear_keys: set[str] = set()
        eff_linear = prepared.effective_linear_prior or {}
        marg = prepared.effective_marginalized_names
        if marg is not None:
            marg_set = set(marg)
            for name in eff_linear:
                if name not in marg_set:
                    explicit_linear_keys.add(name)
        else:
            for name, d in eff_linear.items():
                if _needs_explicit_sampling(d):
                    explicit_linear_keys.add(name)
        return nonlinear_keys, explicit_linear_keys

    def _flatten_prior_samples(
        self,
        prior_samples: Samples,
        prepared: _PreparedSamplerModel,
    ) -> dict[str, jax.Array]:
        """Unit-strip a :class:`Samples` container into the flat sampler dict.

        Validates that the prior-samples key set matches what ``prepared``
        expects (base nonlinear + extension nonlinear in ``nonlinear``, any
        explicit-linear params in ``linear``).  Raises ``ValueError`` with the
        diff on mismatch.
        """
        expected_nl, expected_lin = self._expected_prior_keys(prepared)
        got_nl = set(prior_samples.nonlinear)
        got_lin = set(prior_samples.linear)

        missing = (expected_nl - got_nl) | (expected_lin - got_lin)
        extra = (got_nl - expected_nl) | (got_lin - expected_lin)
        if missing or extra:
            msg = (
                "Prior samples key mismatch for this (prior, model) setup. "
                f"Missing: {sorted(missing)}; unexpected: {sorted(extra)}."
            )
            raise ValueError(msg)

        flat: dict[str, jax.Array] = {}
        for name, qty in prior_samples.nonlinear.items():
            unit = str(qty.unit) or ""
            flat[name] = jnp.asarray(ustrip(unit, qty) if unit else qty.value)
        for name, qty in prior_samples.linear.items():
            unit = str(qty.unit) or ""
            flat[name] = jnp.asarray(ustrip(unit, qty) if unit else qty.value)
        return flat

    def _pad_to_batch_multiple(
        self, flat: dict[str, jax.Array], n_prior_samples: int
    ) -> tuple[dict[str, jax.Array], int]:
        """Pad each array in ``flat`` so its length is a multiple of ``batch_size``.

        ``_evaluate_log_likelihoods_batched`` assumes ``n_total = n_batches *
        batch_size``.  Pad with repeats of the last element; the trailing
        evaluations are discarded by the caller.
        """
        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        n_total = n_batches * self.batch_size
        if n_total == n_prior_samples:
            return flat, n_total
        padded: dict[str, jax.Array] = {}
        pad = n_total - n_prior_samples
        for k, v in flat.items():
            padded[k] = jnp.concatenate([v, jnp.broadcast_to(v[-1:], (pad,))])
        return padded, n_total

    def _evaluate_in_memory(
        self,
        prepared: _PreparedSamplerModel,
        prior_samples: Samples,
        data: Any,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """In-memory branch of :meth:`run_with_samples`."""
        flat = self._flatten_prior_samples(prior_samples, prepared)
        n_prior_samples = prior_samples.n_samples
        padded, _ = self._pad_to_batch_multiple(flat, n_prior_samples)

        log_likelihoods = self._evaluate_log_likelihoods_batched(
            prepared.model,
            padded,
            prepared.effective_linear_prior or {},
            prepared.effective_marginalized_names,
            data,
        )
        trimmed = {k: v[:n_prior_samples] for k, v in padded.items()}
        return trimmed, log_likelihoods[:n_prior_samples]

    def _evaluate_from_hdf5(
        self,
        prepared: _PreparedSamplerModel,
        path: Path,
        data: Any,
        *,
        seed: int,
        randomize_prior_order: bool,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Disk-streaming branch of :meth:`run_with_samples`.

        Reads the prior cache in ``batch_size``-row contiguous slices and
        evaluates ``model.log_prob`` per batch via
        :meth:`_evaluate_log_likelihoods_one_batch`.  When
        ``randomize_prior_order`` is true, the batch order is permuted so the
        accumulated samples are not biased toward the start of the file.
        """
        expected_nl, expected_lin = self._expected_prior_keys(prepared)
        expected_keys = expected_nl | expected_lin

        with h5py.File(path, "r") as f:
            nl_group = f["nonlinear"]
            # ``Samples.to_hdf5`` always writes both groups (linear may be empty);
            # ``make_prior_cache`` matches that layout.
            lin_group = f["linear"]

            available_nl = set(nl_group)
            available_lin = set(lin_group)
            available = available_nl | available_lin
            missing = expected_keys - available
            if missing:
                msg = (
                    f"Prior cache at {path} is missing required keys: "
                    f"{sorted(missing)}. Expected: {sorted(expected_keys)}."
                )
                raise ValueError(msg)

            # All datasets share a common length.
            first_key = next(iter(expected_keys))
            src_group = nl_group if first_key in available_nl else lin_group
            n_prior_samples = int(src_group[first_key].shape[0])

            n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size

            if randomize_prior_order:
                batch_order = np.random.default_rng(seed).permutation(n_batches)
            else:
                batch_order = np.arange(n_batches)

            log_lik_chunks: list[jax.Array] = []
            sample_chunks: dict[str, list[jax.Array]] = {k: [] for k in expected_keys}

            for i in batch_order:
                start = int(i) * self.batch_size
                stop = min(start + self.batch_size, n_prior_samples)
                actual = stop - start

                batch: dict[str, jax.Array] = {}
                for k in expected_keys:
                    grp = nl_group if k in available_nl else lin_group
                    arr = np.asarray(grp[k][start:stop])
                    # Pad the final batch to ``batch_size`` so the JIT cache
                    # is reused (single static shape).
                    if actual < self.batch_size:
                        pad = self.batch_size - actual
                        arr = np.concatenate([arr, np.repeat(arr[-1:], pad)])
                    batch[k] = jnp.asarray(arr)

                log_lik = self._evaluate_log_likelihoods_one_batch(
                    prepared.model,
                    batch,
                    prepared.effective_linear_prior or {},
                    prepared.effective_marginalized_names,
                    data,
                )
                # Drop padding before accumulating.
                log_lik_chunks.append(log_lik[:actual])
                for k in expected_keys:
                    sample_chunks[k].append(batch[k][:actual])

        log_likelihoods = jnp.concatenate(log_lik_chunks)
        flat = {k: jnp.concatenate(v) for k, v in sample_chunks.items()}
        return flat, log_likelihoods

    @eqx.filter_jit
    def _sample_prior_and_evaluate_batched(
        self,
        model: AbstractComponentModel | JointModel,
        key: jax.Array,
        n_prior_samples: int,
        ext_nl_priors: dict[str, Any],
        eff_linear: dict[str, Any],
        marginalize_names: "tuple[str, ...] | None",
        # data is correlated with the (polymorphic) model and is dispatched
        # through model.log_prob; the static type cannot be narrowed here.
        data: Any,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        """Sample prior and evaluate likelihoods in batches.

        The model's ``log_prob`` is called either in auto mode or with the
        sampler-resolved ``marginalized_names`` override.
        """
        prior = self.prior

        n_batches = (n_prior_samples + self.batch_size - 1) // self.batch_size
        n_total = n_batches * self.batch_size

        key, nl_key = jr.split(key)
        prior_samples = prior.sample_nonlinear(nl_key, n_total)

        # Sample explicit linear params (non-Gaussian, not analytically marginalized).
        # Use the effective marginalize_names computed by _resolve_extension_priors
        # (which auto-classified non-Gaussian entries at run-time).
        if isinstance(eff_linear, dict):
            if marginalize_names is not None:
                marg_set = set(marginalize_names)
                for name, d in eff_linear.items():
                    if name not in marg_set:
                        key, k = jr.split(key)
                        prior_samples[name] = _unwrap_dist(d).sample(k, (n_total,))
            else:
                # marginalize_names is None => auto-marginalize Gaussians, sample rest
                for name, d in eff_linear.items():
                    if _needs_explicit_sampling(d):
                        key, k = jr.split(key)
                        prior_samples[name] = _unwrap_dist(d).sample(k, (n_total,))

        # Sample extension nonlinear parameters (jitter, GP hypers, etc.).
        if ext_nl_priors:
            key, ext_key = jr.split(key)
            ext_keys = jr.split(ext_key, len(ext_nl_priors))
            for (model_key, d), k in zip(ext_nl_priors.items(), ext_keys, strict=True):
                prior_samples[model_key] = _unwrap_dist(d).sample(k, (n_total,))

        log_likelihoods = self._evaluate_log_likelihoods_batched(
            model, prior_samples, eff_linear, marginalize_names, data
        )

        trimmed = {k: prior_samples[k][:n_prior_samples] for k in prior_samples}
        return trimmed, log_likelihoods[:n_prior_samples]

    @eqx.filter_jit
    def _evaluate_log_likelihoods_batched(
        self,
        model: AbstractComponentModel | JointModel,
        prior_samples: dict[str, jax.Array],
        eff_linear: dict[str, Any],
        marginalize_names: "tuple[str, ...] | None",
        # data is correlated with the (polymorphic) model and is dispatched
        # through model.log_prob; the static type cannot be narrowed here.
        data: Any,
    ) -> jax.Array:
        """Evaluate the marginal log-likelihood for a pre-sampled prior dict.

        Reshapes the flat ``prior_samples`` dict (each value of length
        ``n_total = n_batches * batch_size``) into batches of size ``batch_size``,
        evaluates ``model.log_prob`` for each batch under ``jax.lax.fori_loop``,
        and returns the flattened length-``n_total`` log-likelihood array.

        Used by both :meth:`_sample_prior_and_evaluate_batched` (fresh-draw path)
        and :meth:`run_with_samples` (in-memory cached path).
        """
        prior = self.prior
        base_names = model._base_nonlinear_names()
        model_keys = tuple(prior_samples.keys())

        n_total = prior_samples[model_keys[0]].shape[0]
        n_batches = n_total // self.batch_size

        batched: dict[str, jax.Array] = {
            k: prior_samples[k].reshape(n_batches, self.batch_size) for k in model_keys
        }

        def body_fn(i: int, acc: jax.Array) -> jax.Array:
            raw = {k: batched[k][i] for k in model_keys}
            wrapped = _wrap_unit_values(raw, prior.nonlinear_priors, base_names)
            if marginalize_names is None:
                return acc.at[i].set(
                    jax.vmap(
                        lambda s: model.log_prob(
                            s, data, linear_priors=eff_linear or None
                        )
                    )(wrapped)
                )
            return acc.at[i].set(
                jax.vmap(
                    lambda sample: model.log_prob(
                        sample,
                        data,
                        linear_priors=eff_linear or None,
                        marginalized_names=marginalize_names,
                    )
                )(wrapped)
            )

        log_liks_batched = jax.lax.fori_loop(
            0, n_batches, body_fn, jnp.zeros((n_batches, self.batch_size))
        )
        return log_liks_batched.flatten()

    @eqx.filter_jit
    def _evaluate_log_likelihoods_one_batch(
        self,
        model: AbstractComponentModel | JointModel,
        batch: dict[str, jax.Array],
        eff_linear: dict[str, Any],
        marginalize_names: "tuple[str, ...] | None",
        data: Any,
    ) -> jax.Array:
        """Evaluate ``model.log_prob`` over a single batch of prior samples.

        Used by the HDF5-streaming branch of :meth:`run_with_samples`, where
        a Python-level loop reads one chunk at a time from disk and feeds it
        through this jit-compiled vmap.
        """
        prior = self.prior
        base_names = model._base_nonlinear_names()
        wrapped = _wrap_unit_values(batch, prior.nonlinear_priors, base_names)
        if marginalize_names is None:
            return jax.vmap(
                lambda s: model.log_prob(s, data, linear_priors=eff_linear or None)
            )(wrapped)
        return jax.vmap(
            lambda s: model.log_prob(
                s,
                data,
                linear_priors=eff_linear or None,
                marginalized_names=marginalize_names,
            )
        )(wrapped)

    @staticmethod
    @jax.jit
    def _rejection_step(key: jax.Array, log_likelihoods: jax.Array) -> jax.Array:
        """Compute rejection mask."""
        max_log_likelihood = jnp.max(log_likelihoods)
        weights = jnp.where(
            jnp.isfinite(max_log_likelihood),
            jnp.exp(log_likelihoods - max_log_likelihood),
            jnp.zeros_like(log_likelihoods),
        )
        uniform_draws = jr.uniform(key, shape=log_likelihoods.shape)
        return uniform_draws < weights

    def _sample_linear_parameters(  # noqa: C901
        self,
        model: AbstractComponentModel | JointModel,
        key: jax.Array,
        nonlinear_samples: dict[str, jax.Array],
        marginalized_names: tuple[str, ...] | None,
        # data is correlated with the (polymorphic) model and is dispatched
        # through model.sample_conditional_linear; cannot be narrowed here.
        data: Any,
        linear_priors: dict[str, Any] | None,
    ) -> dict[str, AbstractQuantity]:
        """Sample linear parameters from conditional posterior using vmap.

        The model's ``sample_conditional_linear`` uses the sampler-resolved
        ``marginalized_names`` override when one is provided.
        """
        prior = self.prior

        n_samples = len(next(iter(nonlinear_samples.values())))
        if n_samples == 0:
            if isinstance(model, JointModel):
                names: list[str] = []
                for comp in model.components.values():
                    names.extend(comp._all_linear_names())
            else:
                names = list(model._all_linear_names())
            return {name: Q(jnp.zeros(0), "") for name in names}

        keys = jr.split(key, n_samples)
        base_names = model._base_nonlinear_names()
        model_keys = tuple(nonlinear_samples.keys())

        def _sample_one(key: jax.Array, sample: dict[str, jax.Array]) -> dict[str, Any]:
            raw = {k: sample[k] for k in model_keys}
            wrapped = _wrap_unit_values(raw, prior.nonlinear_priors, base_names)
            return model.sample_conditional_linear(
                wrapped,
                key,
                data,
                linear_priors=linear_priors,
                marginalized_names=marginalized_names,
            )

        filtered = {k: nonlinear_samples[k] for k in model_keys}
        result = jax.vmap(_sample_one)(keys, filtered)

        # Attach units from the model's linear_param_units
        if isinstance(model, JointModel):
            # Detect which per-component param names appear in more than one
            # component.  Colliding names (e.g. both "rv_semiamp" in an SB2)
            # are namespaced as "comp_name.param_name" to avoid silent overwrites.
            name_counts: dict[str, int] = {}
            for comp in model.components.values():
                for name in comp._all_linear_names():
                    name_counts[name] = name_counts.get(name, 0) + 1

            # Shared linear params that appear at the top level (not in
            # per-component sub-dicts) should use bare names.
            final: dict[str, AbstractQuantity] = {}
            first_comp_name = next(iter(model.components))
            first_comp = model.components[first_comp_name]
            shared_units = first_comp._linear_param_units(data[first_comp_name])

            for k, value in result.items():
                if isinstance(value, dict):
                    # Per-component sub-dict.
                    comp_name = k
                    comp = model.components[comp_name]
                    units = comp._linear_param_units(data[comp_name])
                    for nm, arr in value.items():
                        final_name = (
                            f"{comp_name}.{nm}" if name_counts.get(nm, 1) > 1 else nm
                        )
                        final[final_name] = Q(arr, units.get(nm, ""))
                else:
                    # Shared top-level param (joint path).
                    final[k] = Q(value, shared_units.get(k, ""))
            return final
        units = model._linear_param_units(data)
        return {name: Q(arr, units.get(name, "")) for name, arr in result.items()}
