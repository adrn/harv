"""Constants used in Keplerian calculations."""

from astropy.constants import G as G_astropy  # noqa: N811
from astropy.constants import c as c_astropy
from unxt import AbstractQuantity, Q

G: AbstractQuantity = Q.from_(G_astropy)
c: AbstractQuantity = Q.from_(c_astropy)
