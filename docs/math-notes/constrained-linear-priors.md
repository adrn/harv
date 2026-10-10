# Marginalizing linear parameters with constrained priors

`harv` analytically marginalizes over linear parameters when their priors are Normal
(see {doc}`../concepts`).
But we often have linear parameters with non-Gaussian priors, or with physically
constrained domains (e.g., a true parallax is positive).
We support these through truncated Normal priors and Gaussian mixture priors, with the
restriction that at most two of the linear parameters may carry a constrained domain
(see below).
A second motivation is specific to modeling SB2 systems with radial velocity data:
For an SB2 system, we require the phase and argument of pericenter to be the same for
both components, which means that the sign of their semi-amplitudes must be opposite
($\mathrm{sign}(K_1) = -\mathrm{sign}(K_2)$).
Both of these are support constraints on linear parameters, so both are handled by the
same machinery, defined below.

## The truncated prior

When we constrain a multivariate Normal to a rectangular domain, we need the
normalization constant over the allowed region to keep the pdf properly normalized.
For a prior over parameter vector $\alpha$ restricted to a box $S$ (i.e. a rectangular
domain, and $S(\alpha)$ is a function that returns 1 if $\alpha \in S$ and 0 otherwise),
the prior is

$$
\begin{align}
p(\alpha) &= \mathcal{N}(\alpha \mid \mu, \Sigma_p) \\
p_S(\alpha) &= \frac{1}{Z_\mathrm{prior}} \, p(\alpha) \, S(\alpha) \\
Z_\mathrm{prior} &= \int_S \mathrm{d}\alpha \, p(\alpha) \\
&= \int_S \mathrm{d}\alpha \, \mathcal{N}(\alpha\mid\mu,\Sigma_p) \quad .
\end{align}
$$

where $\mu$ is the prior mean and $\Sigma_p$ is the prior covariance matrix, and I'm using the subscript $\cdot_S$ to indicate that a pdf is normalized over the domain $S$ and not $\mathbb{R}$ (default).
If we are in a regime where $\Sigma_p$ is diagonal (as is often the case for me), then $Z_\mathrm{prior}$ factorizes into a product of 1D integrals that are differences of CDF values at the truncation bounds of the domain in each parameter.

The likelihood is $p(y \mid \alpha)$ for some data $y$ (and, assume there are other parameters on the right of the conditional that are not being marginalized -- I've just dropped them for brevity), where $X$ is the design matrix and $C$ the observation covariance, so

$$
p(y\mid\alpha) = \mathcal N(y \mid X\alpha, C) \quad .
$$

The marginal likelihood we need within `harv` is:

$$
\begin{align}
p_S(y) &= \int_S \mathrm{d}\alpha \, p(y\mid \alpha) \, p_S(\alpha) \\
&= \frac{1}{Z_\mathrm{prior}} \, \int_S \mathrm{d}\alpha \, p(y\mid \alpha) \,
    p(\alpha) \quad .
\end{align}
$$

We can use the factorization rule $p(y \mid \alpha) \, p(\alpha) = p(\alpha \mid y) \, p(y)$ to rewrite the integral above as:

$$
\begin{align}
\int_S \mathrm{d}\alpha \, p(y\mid \alpha) \, p(\alpha) &=
    \int_S \mathrm{d}\alpha \, p(\alpha \mid y) \, p(y) \\
    &= p(y) \int_S \mathrm{d}\alpha \, p(\alpha \mid y)
\end{align}
$$

where

$$
p(y) = \int \mathrm{d}\alpha \, p(y \mid \alpha) \, p(\alpha)
$$

is the untruncated marginal likelihood equivalent of the truncated version above.
Also, $p(\alpha \mid y)$ here is the untruncated posterior.
Why is it useful to rewrite that? If $p(y \mid \alpha)$ and $p(\alpha)$ are Normal, then the posterior $p(\alpha \mid y)$ is also Normal.
If we define

$$
Z_\mathrm{post} = \int_S \mathrm{d}\alpha \, p(\alpha \mid y)
$$

then the marginal likelihood computed over the truncated domain $S$ is

$$
\ln p_S(y) = \ln p(y) + \ln Z_\mathrm{post} - \ln Z_\mathrm{prior}
$$

where again $p(y)$ is the value of the untruncated marginal likelihood.

As mentioned above, if $\Sigma_p$ is diagonal and the domain $S$ is rectangular, then $Z_\mathrm{prior}$ is straightforward to compute as products of differences of CDF values at the boundaries of the region.
However, even with a diagonal $\Sigma_p$, the posterior pdf $p(\alpha \mid y)$ will not in general have a diagonal covariance matrix.
For a Normal likelihood and a Normal prior over $\alpha$, the integral of the posterior pdf above results in

$$
\begin{align}
    Z_\mathrm{post} &= \int_S \mathrm{d}\alpha \, p(\alpha \mid y) \\
    &= \int_S \mathrm{d}\alpha \, \mathcal{N}(\alpha \mid \hat\alpha, \Sigma_\mathrm{post})
\end{align}
$$


The posterior mean $\hat{\alpha}$ and covariance $\Sigma_{\mathrm{post}}$ are given by the Normal conjugate expressions.
The posterior mean is

$$
\hat{\alpha} = \Sigma_{\mathrm{post}} \, (\Sigma_p^{-1} \, \mu + X^\top \, C^{-1} \, y)
$$

The precision matrix (inverse covariance), $\Lambda$, of the posterior pdf is

$$
\Sigma_{\mathrm{post}} = \Lambda^{-1} \\
\Lambda = \Sigma_p^{-1} + X^\top \, C^{-1} \, X
$$

which is generally a dense matrix because the $X^\top \, C^{-1} \, X$ term couples the linear parameters in the posterior constraints.
So the integral in the $Z_{\mathrm{post}}$ equation above is not separable in general, and it generally has to be computed numerically.
However, `harv` has special cases for 1D and 2D integrals (i.e. 1 or 2 linear parameters with constrained domains).

In 1D the integral is a difference of CDF values of that parameter's posterior marginal.

In 2D, the rectangle probability can be computed from the corners of the selection rectangle $S$.
If the two constrained parameters are $\alpha_1$ and $\alpha_2$, then the rectangle is defined by the lower and upper bounds $(l_1, h_1)$ and $(l_2, h_2)$ such that

$$
S = \left\{ \alpha : l_1 \le \alpha_1 \le h_1 \,\&\, l_2 \le \alpha_2 \le h_2 \right\}
    \quad .
$$

To compute this, we need the marginal posterior over the two constrained parameters $(\alpha_1, \alpha_2)$.
The remaining linear parameters are unconstrained, so they are integrated over all of $\mathbb{R}$ and drop out of $Z_\mathrm{post}$: marginalizing a multivariate Normal over a subset of its coordinates just keeps the un-marginalized entries of its mean and covariance.
What is left is a 2D Normal with marginal posterior mean $\hat\alpha = (\hat\alpha_1, \hat\alpha_2)$ and covariance $\Sigma_\mathrm{post}$.
Here, it is convenient to work in standardized Normal coordinates:
For $k \in {1, 2}$, we define

$$
s_k = \sqrt{(\Sigma_\mathrm{post})_{kk}} \quad , \quad
\rho = \frac{(\Sigma_\mathrm{post})_{12}}{s_1 \, s_2}
$$

where $s_k$ is the posterior standard deviation of $\alpha_k$ and the posterior correlation coefficient of the two constrained parameters is $\rho$.
We then define the shifted and scaled variables and bounds as

$$
z_k = \frac{\alpha_k - \hat\alpha_k}{s_k} \quad , \quad
\tilde{l}_k = \frac{l_k - \hat\alpha_k}{s_k} \quad , \quad
\tilde{h}_k = \frac{h_k - \hat\alpha_k}{s_k} \quad .
$$

In these coordinates the marginal posterior is a standard bivariate Normal (unit variance and zero mean) with correlation $\rho$, and the box becomes $\tilde{l}_k \le z_k \le \tilde{h}_k$, so each standardized bound is a distance from the posterior mean in units of the posterior standard deviation.

If $\Phi_2(a, b; \rho)$ is the CDF of a standard bivariate Normal with correlation $\rho$ (i.e. zero means and unit variances), then the probability within the rectangle using its four corners is:

$$
Z_\mathrm{post} = \Phi_2(\tilde{h}_1, \tilde{h}_2; \rho) - \Phi_2(\tilde{l}_1, \tilde{h}_2; \rho)
    - \Phi_2(\tilde{h}_1, \tilde{l}_2; \rho) + \Phi_2(\tilde{l}_1, \tilde{l}_2; \rho) \quad .
$$

For one-sided constraints (e.g., a `HalfNormal` prior), some of the bounds are infinite, and the corresponding corner terms simplify.
For example, if either argument is $-\infty$, then $\Phi_2 = 0$.
If one argument is $+\infty$, then $\Phi_2$ reduces to the univariate standard Normal CDF $\Phi$ of the other argument (e.g., $\Phi_2(a, +\infty; \rho) = \Phi(a)$).
In `harv` these reductions are structural and are applied at trace time, before any numerical evaluation.

For the remaining corners, we need to evaluate $\Phi_2$.
We do this using Plackett's identity, which says that the derivative of $\Phi_2$ with respect to the correlation is the bivariate Normal density:

$$
\frac{\partial \Phi_2(a, b; \rho)}{\partial \rho} = \phi_2(a, b; \rho)
    = \frac{1}{2\pi\sqrt{1-\rho^2}} \,
    \exp\left(-\frac{a^2 - 2\rho ab + b^2}{2(1-\rho^2)}\right) \quad .
$$

At $\rho = 0$ the two coordinates are independent, so $\Phi_2(a, b; 0) = \Phi(a) \, \Phi(b)$. Integrating the identity up from $\rho = 0$ then gives

$$
\Phi_2(a, b; \rho) = \Phi(a) \, \Phi(b) + \int_0^\rho \mathrm{d}t \, \frac{1}{2\pi\sqrt{1-t^2}} \,
    \exp\left(-\frac{a^2 - 2tab + b^2}{2(1-t^2)}\right) \quad .
$$

If we substitute $t = \sin\theta$, then $\mathrm{d}t = \cos\theta \, \mathrm{d}\theta$. This exactly cancels the factor $\sqrt{1-t^2} = \cos\theta$ in the denominator:

$$
\Phi_2(a, b; \rho) = \Phi(a) \, \Phi(b) + \frac{1}{2\pi} \int_0^{\arcsin\rho} \mathrm{d}\theta \,
    \exp\left(-\frac{a^2 - 2ab\sin\theta + b^2}{2\cos^2\theta}\right) \quad .
$$

This cancellation is why a fixed Gauss–Legendre quadrature rule is enough for any $|\rho| < 1$ (i.e. any posterior that is not exactly degenerate in the constrained pair).
In the $\theta$ form, the integrand is bounded and smooth over the whole integration range.
This keeps this whole computation compatible with `jit`, `vmap`, and `grad`.

For three or more constrained parameters the code raises `NotImplementedError`.
A $k$-dimensional Gaussian rectangle probability with dense covariance has no closed form CDF.


## Mixture priors

A Gaussian mixture prior does not need any new math or machinery because it is just a sum over Normal distributions, which we already support.
If the prior is

$$
p(\alpha) = \sum_c w_c \, \mathcal{N}(\alpha \mid \mu_c, \Sigma_{p,c}) \quad , \quad
\sum_c w_c = 1 \quad ,
$$

then each integral over $\alpha$ splits into a weighted sum of the per-component
integrals we already know how to do.
Writing $p_c(y)$, $Z_{\mathrm{post},c}$, and $Z_{\mathrm{prior},c}$ for the
single-component quantities from the sections above,

$$
p_S(y) = \frac{\sum_c w_c \, p_c(y) \, Z_{\mathrm{post},c}}
    {\sum_c w_c \, Z_{\mathrm{prior},c}} \quad .
$$

So a mixture is a weighted `logsumexp` over component log-likelihoods.
In `harv` the components are along the batch axis of the marginalized likelihood, which
already supports batching, so a single evaluation returns one $p_c(y)$ per component.

One thing to be careful about: every component shares the same domain $S$, so the
indicator factors out of the sum and the denominator is a single mixture normalization
constant.
This is the marginal likelihood for a *truncated mixture*.
We don't support a general mixture of truncated Gaussians, where each component has its
own domain.
