"""Constants used in Keplerian calculations."""

from typing import Any

from astropy.constants import G as G_astropy  # noqa: N811
from unxt import Quantity

G: Quantity[Any] = Quantity.from_(G_astropy)
