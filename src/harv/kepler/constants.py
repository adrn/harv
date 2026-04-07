"""Constants used in Keplerian calculations."""

from astropy.constants import G as G_astropy  # noqa: N811
from astropy.constants import c as c_astropy
from unxt import AbstractQuantity, Quantity

G: AbstractQuantity = Quantity.from_(G_astropy)
c: AbstractQuantity = Quantity.from_(c_astropy)
