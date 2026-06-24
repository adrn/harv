# Getting Started with harv

### Radial velocity modeling

```python
from unxt import Q
import harv
import harv.models as hm

# Load or create RV data with explicit units - some sample RV data:
data = harv.RVData(
    time=Q([56000.0, 56100.0, 56250.0, 56400.0, 56600.0], "day"),
    rv=Q([12.3, -8.7, 5.1, -14.2, 10.8], "km/s"),
    rv_err=Q([1.2, 0.9, 1.1, 0.8, 1.0], "km/s"),
)

# Set up a prior with default structure (log-uniform in period, etc.)
prior = hm.StandardRV().default_prior(
    period_min=Q(10, "day"),
    period_max=Q(1000, "day"),
    sigma_K0=Q(30, "km/s"),   # RV semi-amplitude scale
    sigma_v0=Q(10, "km/s"),   # systemic velocity prior width
)

# Run the rejection sampler
sampler = harv.RejectionSampler(prior, harv.RVModel())

# Inspect how each parameter will be treated before running:
print(sampler.summary())   # which params are marginalized vs. sampled

samples = sampler.run(data, n_prior_samples=1_000_000, seed=42)

# Inspect results — quantities carry units:
print(f"Accepted {samples.n_samples} posterior samples")
print(f"Period: {samples['period']}")
print(f"Eccentricity: {samples['eccentricity']}")
```

### Gaia epoch astrometry

<!-- TODO: custom parallax prior parallax=harv.QD(dist.TruncatedNormal(0.5, 0.5, low=0.0), "mas"),  # parallax prior -->

```python
import numpyro.distributions as dist
import quaxed.numpy as jnp

astro_data = harv.GaiaAstrometryData(
    time=Q([958.110978, 994.910525, 995.086642, 1010.091395, 1076.918577], "day"),
    al_position=Q([147.066, 379.996, 378.656, 74.666, -293.923], "mas"),
    al_position_err=Q([0.370, 0.446, 0.428, 0.270, 0.247], "mas"),
    scan_angle=Q([-59.047, -5.114, -5.783, -68.579, -134.155], "deg"),
    parallax_factor=jnp.array([0.70828, -0.46657, -0.45946, 0.19659, -0.21379])
)

prior = hm.StandardGaiaAstrometry().default_prior(
    period_min=Q(50, "day"),
    period_max=Q(3000, "day"),
    sigma_a0=Q(1, "au"),   # astrometric semi-major axis prior scale
    sigma_parallax=Q(1.0, "mas"),  # parallax prior width
    sigma_pos=Q(1, "mas"),  # position offset prior width
    sigma_vtan=Q(30, "km/s"),  # tangential velocity prior width
)

sampler = harv.RejectionSampler(prior, harv.GaiaAstrometryModel())
samples = sampler.run(astro_data, n_prior_samples=1_000_000, seed=42)
```

### MCMC continuation

When the rejection sampler returns a small number of samples, you can refine with
NumPyro MCMC, started from the posterior samples:

```py
from harv.samplers import NumpyroSampler

samples = sampler.run(n_prior_samples=1_000_000, max_posterior_samples=128)
mcmc_sampler = NumpyroSampler(model=model, prior=prior)
mcmc_samples = mcmc_sampler.run(samples, seed=0)
```
