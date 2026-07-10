"""Compute wet-bulb temperature values using the Stull (2011) approximation."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

np: Any = cast("Any", import_module("numpy"))
type Dyn = Any

_STULL_C1 = 0.151977
_STULL_C2 = 8.313659
_STULL_C3 = 1.676331
_STULL_C4 = 0.00391838
_STULL_C5 = 1.5
_STULL_C6 = 0.023101
_STULL_OFFSET = 4.686035


def wetbulb_stull(tair: object = 20.0, rh: object = 50.0) -> object:
    """Compute wet-bulb temperature (deg C) from air temperature (deg C) and RH (%)."""
    inputs = [np.atleast_1d(x).astype(float) for x in (tair, rh)]
    is_scalar = bool(len(inputs[0]) == 1)

    max_len = max(len(a) for a in inputs)
    for a in inputs:
        if len(a) != 1 and len(a) != max_len:
            msg = (
                "Length of inputs differ! Single value or vectors of same "
                "length required."
            )
            raise ValueError(msg)

    t = np.broadcast_to(inputs[0], max_len).copy()
    rh_pct = np.broadcast_to(inputs[1], max_len).copy()

    tw = (
        t * np.arctan(_STULL_C1 * np.sqrt(rh_pct + _STULL_C2))
        + np.arctan(t + rh_pct)
        - np.arctan(rh_pct - _STULL_C3)
        + _STULL_C4 * rh_pct**_STULL_C5 * np.arctan(_STULL_C6 * rh_pct)
        - _STULL_OFFSET
    )

    if is_scalar:
        return float(np.asarray(tw, dtype="float64").reshape(-1)[0])
    return tw
