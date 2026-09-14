# Running harv at scale

This document contains some guidance for population-scale rejection sampling with
`harv`.
The standard setup is: one shared prior samples library / cache file, one
`run_with_samples` call per source, thousands to millions of sources.

The first section below is about hardware and settings. The second is about the
failure modes that only show up once you are running many sources at once,
which are mostly ways of getting a wrong answer that looks ok.


## Making it fast

Per-source wall time is, to a good approximation,

$$
t \approx M \times N_\text{obs} \times \tau_{mn}
$$

where $M$ is the prior library size (number of prior samples), $N_\text{obs}$ is the
(padded) number of epochs, and $\tau_{mn}$ is the time to evaluate the model at one
prior draw.
$\tau_{mn}$ depends on the model, the data type, and the parameterization.
Astrometric models carry more linear columns than radial velocity models and are
correspondingly slower, and `ThieleInnesGaiaAstrometry` evaluates a Jacobian
correction per sample on top of that.
We recommend benchmarking your own hardware and model on sources in your dataset that
span the range of epoch counts.

### Choosing a hardware path

`harv` does not have built-in support for distributed computing, beyond what is
available at a low level in JAX.
The `harv` internals do not handle device placement, sharding, `pmap`, or multi-device
execution.
So, currently, scaling out to multiple devices means running one process per
(user-defined) shard of sources.
This is embarrassingly parallel, so you can use whatever you have available.
We have tested using MPI/slurm and found that it works well.

This is not imperative, but we have found the following from our own experience:

- If you have hundreds of thousands to millions of sources, you will want to use a multi-node CPU cluster. You can scale to many more cores than you can even with a few GPUs.
- If you have tens of thousands to hundreds of thousands of sources, you can use a single GPU or multiple GPUs. Or just let it run on a multi-core CPU workstation.
- If you want to run a few to a few thousand sources with a larger prior library, you should use a GPU or multiple GPUs. The GPU advantage grows with prior library size.


### Settings that matter

`batch_size`
: How many prior draws are evaluated in one vmapped step, which sets the working-set
size of the `(batch_size, n_obs, n_linear)` intermediate. The default of 100,000 is a
reasonable place to start. In our measurements (see {doc}`benchmarks`) it barely matters
on GPU, since the device is already saturated at much smaller batches, and is worth a
factor of roughly 1.5 to 2 on a single CPU core.

Prior library backend
: Use an in-memory `Samples` library when it fits in RAM. `make_prior_cache` writes a
library larger than RAM to an HDF5 file and `run_with_samples` streams it back in
contiguous slices. The streaming penalty is small on CPU and substantial on GPU, where
the device finishes a batch faster than the disk can feed it.

Prior library size $M$
: Two things scale differently with $M$, so decide which one you need. Point estimates
(e.g., the best-fitting period, or whether a source is recovered at all) improve with
$M$ until the prior and the data set the answer. Past that point, ten times the samples
buys very little. The effective sample size keeps growing roughly linearly with $M$, so
if you need posterior *widths* rather than point estimates, a larger library will help.
In general, we recommend doing an initial run with a smaller prior library and then
refining the sampling for sources that are recovered poorly.

Period prior width
: Every decade of period the prior covers costs sampling density in the decades that
matter. Narrowing the period prior is usually the cheapest speedup available: in our own
tests with *Gaia* data, cutting the range from about eight decades to four bought about
as much period accuracy as a tenfold larger library, at the cost of the small fraction
of systems that fell outside the narrower range.

Selection policy
: Instead of doing rejection sampling, we recommend using the `top_k=True` option in
`RejectionSampler` for population work. It returns a fixed number of rows per source, so
results form a rectangular table and the `jax.vmap` over the conditional-linear solve is
traced once instead of once per distinct acceptance count. And one can always
post-process the `top_k` output to get the same results as rejection sampling.


## Making it right

Running with `top_k=True` does not accept or reject samples, it evaluates the marginal
likelihood for every prior draw in the library and keeps the `k` draws with the largest
importance weights. The rows you get back are therefore not equal-weight posterior
samples: a row's weight says how much of the posterior that row represents, and rows in
one source's output routinely differ by many orders of magnitude in weight (especially
for well-constrained sources).

However, be aware that the weights are normalized over the whole library, not over the
`k` rows you got back. They sum to `weight_captured`, the fraction of the posterior mass
your `k` rows captured, which is below 1 and can be far below it. Every average over the
returned samples has to renormalize. For example, the posterior mean of a function $f$
is

$$ \Sigma_k w_k \, f / \Sigma_k w_k $$

and not $\Sigma w f$.

When you plot weighted samples, color them by $\ln(w/w_\text{best})$ and cut on
cumulative weight rather than drawing all `k` rows.
`top_k=True` keeps the top `k` draws by rank, so a source with an effective sample size
of a few has a handful of rows carrying essentially all of the mass while the rest are
prior draws that happened to rank highest.
Drawn with equal visual weight, they more give you a sense of the prior rather than the
posterior samples.

### ESS is a resolution diagnostic, not a quality metric

`logZ_int_ess` is the effective sample size of the importance weights.
It counts how many library draws actually contribute to the evidence integral, which
answers one question: did the library resolve this posterior?
It does not say whether the orbit is well determined itself.

For a strongly detected system, ESS near 1 is expected.
The $\Delta\chi^2$ between the best library draw and the second best can be hundreds, so
tightening a prior or reparameterizing does not change the ESS much.
For these cases, we recommend following up the rejection sampling with a local MCMC
(i.e. use `NumpyroSampler`).

### Calibrate your uncertainties rather than fitting a jitter

Reported per-epoch uncertainties are often smaller than the true scatter.
*Gaia* is a good example: the published formal error omits calibration terms that are
genuinely there, so the ratio $r = \sigma_\text{true}/\sigma_\text{reported}$ has a
median above 1 and a long tail.
If you are running on hundreds of thousands to millions of sources, it is worth the time
to calibrate $r$ up front instead of letting a fitted jitter absorb it.

Fitting a jitter parameter is expensive for three reasons.
First, it is a
nonlinear parameter, so it comes at a cost of the library's prior resolution.
Second, it competes with the orbit model,
because an orbit the library cannot match exactly is cheaper to explain as excess
noise, which drives solutions to the edges of the prior.
Lastly, the uncertainty inflation often varies from source to source, so the jitter
prior has to be absolute and wide enough for the whole catalog, which wastes resolution
on individual stars.
