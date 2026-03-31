"""Constants used in Keplerian calculations."""

from __future__ import annotations

from typing import Any

from astropy.constants import G as G_astropy  # noqa: N811
from unxt import Quantity

G: Quantity[Any] = Quantity.from_(G_astropy)
