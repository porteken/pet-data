"""Tests for the Stull wet-bulb temperature approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from wetbulb import wetbulb_stull

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _as_float(value: object) -> float:
    return float(np.asarray(value, dtype=float).item())


def _as_float_array(value: object) -> NDArray[np.float64]:
    return np.asarray(value, dtype=float)


class TestWetbulbStull:
    def test_scalar_output(self) -> None:
        result = wetbulb_stull(20.0, 50.0)
        assert isinstance(result, float)

    def test_known_value_20_50(self) -> None:
        assert _as_float(wetbulb_stull(20.0, 50.0)) == pytest.approx(13.699, abs=0.01)

    def test_known_value_25_50(self) -> None:
        assert _as_float(wetbulb_stull(25.0, 50.0)) == pytest.approx(17.998, abs=0.01)

    def test_known_value_0_50(self) -> None:
        assert _as_float(wetbulb_stull(0.0, 50.0)) == pytest.approx(-3.498, abs=0.01)

    def test_wetbulb_below_air_temp_when_not_saturated(self) -> None:
        assert _as_float(wetbulb_stull(30.0, 70.0)) < 30.0

    def test_wetbulb_near_air_temp_at_saturation(self) -> None:
        result = _as_float(wetbulb_stull(25.0, 100.0))
        assert result == pytest.approx(25.0, abs=0.5)

    def test_monotonic_with_relative_humidity(self) -> None:
        rh = np.array([20.0, 50.0, 80.0])
        tair = np.full_like(rh, 25.0)
        arr = _as_float_array(wetbulb_stull(tair, rh))
        assert arr[0] < arr[1] < arr[2]

    def test_vector_input(self) -> None:
        tair = np.array([20.0, 25.0, 30.0])
        rh = np.array([50.0, 50.0, 50.0])
        result = _as_float_array(wetbulb_stull(tair, rh))
        assert len(result) == 3

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="Length"):
            wetbulb_stull(np.array([20.0, 25.0]), np.array([50.0, 60.0, 70.0]))

    def test_scalar_broadcast_with_vector(self) -> None:
        tair = np.array([20.0, 25.0, 30.0])
        result = _as_float_array(wetbulb_stull(tair, 50.0))
        assert len(result) == 3

    def test_nan_propagates(self) -> None:
        result = _as_float(wetbulb_stull(float("nan"), 50.0))
        assert np.isnan(result)
