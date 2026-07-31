"""Unit tests for the Gaia astrometry and joint paths of harv.periodogram."""

import jax.numpy as jnp
from unxt import Q, ustrip

import harv.models as hm
import harv.periodogram as hp
from harv.data import SourceData
from harv.simulate import simulate_gaia_epoch_astrometry, simulate_rv_sb1_data

P_TRUE = Q(100.0, "day")


def _gaia_prior(n_terms: int = 2):
    """Explicit Fourier prior for the Gaia periodogram."""
    return hm.FourierGaiaAstrometry(n_terms=n_terms).default_prior(
        period_min=Q(1.0, "day"),
        period_max=Q(5000.0, "day"),
        sigma_amp=Q(20.0, "mas"),
        sigma_pos=Q(500.0, "mas"),
        sigma_pm=Q(500.0, "mas/yr"),
        sigma_parallax=Q(500.0, "mas"),
    )


def _rv_prior(n_terms: int = 2):
    return hm.FourierRV(n_terms=n_terms).default_prior(
        period_min=Q(1.0, "day"),
        period_max=Q(5000.0, "day"),
        sigma_amp=Q(30.0, "km/s"),
        sigma_v0=Q(50.0, "km/s"),
    )


def _sim_gaia(seed: int = 3, **kwargs: object):
    defaults: dict[str, object] = {
        "n_obs": 80,
        "period": P_TRUE,
        "eccentricity": 0.2,
        "semi_major_axis": Q(2.0, "mas"),
        "parallax": Q(20.0, "mas"),
        "mu_alpha": Q(15.0, "mas/yr"),
        "mu_delta": Q(-8.0, "mas/yr"),
        "al_error": Q(0.05, "mas"),
    }
    defaults.update(kwargs)
    return simulate_gaia_epoch_astrometry(seed=seed, **defaults)


def _peak_within_grid_steps(result, p_true, n_steps: int = 2) -> bool:
    f = ustrip("1/day", result.frequency)
    df = float(f[1] - f[0])
    f_true = 1.0 / float(ustrip("day", p_true))
    f_peak = float(f[jnp.argmax(result.delta_ln_likelihood)])
    return abs(f_peak - f_true) < n_steps * df


class TestGaiaRecovery:
    def test_orbit_recovery(self):
        data, _ = _sim_gaia()
        result = hp.periodogram(data, prior=_gaia_prior(), period_min=Q(20.0, "day"))
        assert _peak_within_grid_steps(result, P_TRUE)

    def test_scan_law_and_parallax_suppression(self):
        """With no orbit, parallax + proper motion produce no periodogram peak.

        The five base astrometric columns appear in both the base and
        trial-period models, so their power must cancel — in particular there
        must be no spurious peak near one year from the parallax signal.
        """
        data, _ = _sim_gaia(semi_major_axis=Q(0.0, "mas"))
        result = hp.periodogram(data, prior=_gaia_prior(), period_min=Q(20.0, "day"))
        assert float(jnp.max(result.delta_ln_likelihood)) < 5.0
        # Specifically check the region around 1 year:
        p = ustrip("day", result.period)
        near_year = (p > 300.0) & (p < 430.0)
        assert float(jnp.max(result.delta_ln_likelihood[near_year])) < 5.0


class TestJoint:
    def test_joint_source_data(self):
        gaia_data, _ = _sim_gaia()
        rv_data, _ = simulate_rv_sb1_data(
            seed=11,
            n_obs=30,
            period=P_TRUE,
            eccentricity=0.2,
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(0.3, "km/s"),
        )
        source = SourceData(gaia=gaia_data, rv=rv_data)
        priors = {"gaia": _gaia_prior(), "rv": _rv_prior()}
        result = hp.periodogram(source, prior=priors, period_min=Q(20.0, "day"))

        assert result.per_dataset is not None
        assert set(result.per_dataset) == {"gaia", "rv"}
        total = result.per_dataset["gaia"] + result.per_dataset["rv"]
        assert jnp.allclose(result.delta_ln_likelihood, total, atol=1e-3)
        assert _peak_within_grid_steps(result, P_TRUE)

    def test_joint_peak_at_least_single_dataset(self):
        """The summed Δ at the true period exceeds each dataset's alone."""
        gaia_data, _ = _sim_gaia()
        rv_data, _ = simulate_rv_sb1_data(
            seed=11,
            n_obs=30,
            period=P_TRUE,
            eccentricity=0.2,
            rv_semiamp=Q(5.0, "km/s"),
            rv_err=Q(0.3, "km/s"),
        )
        source = SourceData(gaia=gaia_data, rv=rv_data)
        f = hp.frequency_grid(source, period_min=Q(20.0, "day"))
        priors = {"gaia": _gaia_prior(), "rv": _rv_prior()}
        result = hp.periodogram(source, f, prior=priors)
        i_true = jnp.argmin(
            jnp.abs(ustrip("1/day", f) - 1.0 / float(ustrip("day", P_TRUE)))
        )
        joint_val = float(result.delta_ln_likelihood[i_true])
        for delta in result.per_dataset.values():
            assert joint_val >= float(delta[i_true]) - 1e-3
