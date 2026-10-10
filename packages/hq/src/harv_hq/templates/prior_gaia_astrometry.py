"""The model file for an hq Gaia astrometry run: the prior and model for every source.

See harv-hq's spec, "The model file (prior.py)". Choose the period range and
scales for your science case; harv provides no defaults for them.
"""

from unxt import Q

import harv.models as hm


def make_setup():
    prior = hm.StandardGaiaAstrometry().default_prior(
        period_min=Q(30, "day"),
        period_max=Q(4000, "day"),
        sigma_a0=Q(5.0, "AU"),
        sigma_parallax=Q(10.0, "mas"),
        sigma_pos=Q(100.0, "mas"),
        sigma_vtan=Q(50.0, "km/s"),
    )
    return prior, hm.GaiaAstrometryModel()


# Optional: quality cuts applied to the input table at `hq prepare`.
# def select_rows(table):
#     return table["al_position_err"] < 1.0
