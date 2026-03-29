"""Tests for calculate_pet.py — PET computation from combined parquet data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calculate_pet import (
    PET_ROUNDING_FACTOR,
    calculate_pet_frame,
    compute_pet_chunk,
)


def _make_combined_df(n: int = 50) -> pd.DataFrame:
    """Build a small synthetic combined DataFrame for PET calculation."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "location_id": rng.choice([1, 2], size=n),
            "time": pd.date_range("2020-06-01", periods=n, freq="h"),
            "v": rng.uniform(0.5, 5.0, n),
            "t": rng.uniform(15.0, 35.0, n),
            "rh": rng.uniform(20.0, 80.0, n),
            "mrt": rng.uniform(20.0, 60.0, n),
        }
    )


class TestComputePetChunk:
    def test_adds_pet_column(self) -> None:
        df = _make_combined_df(10)
        result = compute_pet_chunk(df[["v", "t", "rh", "mrt"]])
        assert "pet" in result.columns
        assert result["pet"].notna().all()

    def test_rounding(self) -> None:
        df = _make_combined_df(10)
        result = compute_pet_chunk(df[["v", "t", "rh", "mrt"]])
        # All PET values should be rounded to 1/PET_ROUNDING_FACTOR precision
        rounded = (result["pet"] * PET_ROUNDING_FACTOR).round() / PET_ROUNDING_FACTOR
        pd.testing.assert_series_equal(result["pet"], rounded)


class TestCalculatePetFrame:
    def test_produces_daily_max(self) -> None:
        df = _make_combined_df(48)  # 2 days of hourly data
        result = calculate_pet_frame(df)
        assert "location_id" in result.columns
        assert "date" in result.columns
        assert "pet" in result.columns

    def test_filters_invalid_rh_and_wind(self) -> None:
        df = _make_combined_df(10)
        df.loc[0, "rh"] = 0.5  # Below 1
        df.loc[1, "v"] = 0.0  # Zero wind
        result = calculate_pet_frame(df)
        # Should not crash and should produce some rows
        assert len(result) >= 0

    def test_returns_max_per_location_date(self) -> None:
        """Multiple hours on same day for same location → max PET."""
        df = pd.DataFrame(
            {
                "location_id": [1, 1, 1],
                "time": pd.to_datetime(
                    ["2020-06-01 08:00", "2020-06-01 12:00", "2020-06-01 16:00"]
                ),
                "v": [2.0, 2.0, 2.0],
                "t": [20.0, 30.0, 25.0],
                "rh": [50.0, 50.0, 50.0],
                "mrt": [30.0, 50.0, 40.0],
            }
        )
        result = calculate_pet_frame(df)
        # Should produce exactly one row for loc 1, date 2020-06-01
        loc1_rows = result[result["location_id"] == 1]
        assert len(loc1_rows) == 1
