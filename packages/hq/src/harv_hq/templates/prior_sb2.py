"""The model file for an hq SB2 run: the prior and model used for every source.

See harv-hq's spec, "The model file (prior.py)". Choose the period range and
amplitude scales for your science case; harv provides no defaults for them.
The two stars share the orbit and the systemic velocity; the secondary's
semi-amplitude is negative (default_sb2_prior's signed_semiamp=True).
"""

from unxt import Q

import harv.models as hm


def make_setup():
    prior = hm.default_sb2_prior(
        period_min=Q(2, "day"),
        period_max=Q(4096, "day"),
        sigma_K0=Q(30, "km/s"),
        sigma_v0=Q(100, "km/s"),
    )
    return prior, hm.JointModel.for_sb2(prior=prior)


# Optional: quality cuts applied to the input table at `hq prepare`.
# def select_rows(table):
#     return table["rv_err"] < 10.0
