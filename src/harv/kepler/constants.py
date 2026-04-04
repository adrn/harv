"""Constants used in Keplerian calculations."""

from astropy.constants import G as G_astropy  # noqa: N811
from unxt import AbstractQuantity, Quantity

G: AbstractQuantity = Quantity.from_(G_astropy)
