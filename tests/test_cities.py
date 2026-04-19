"""Tests for cities.py — city loading, filtering, and processing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cities import filter_bounding_box, process_cities


class TestFilterBoundingBox:
    def test_keeps_continental_us(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [40.0, 50.0, 20.0, 35.0],
                "lng": [-90.0, -80.0, -90.0, -100.0],
            }
        )
        result = filter_bounding_box(df)
        assert len(result) == 2  # 40/-90 and 35/-100 are inside

    def test_excludes_outside_bounds(self) -> None:
        df = pd.DataFrame({"lat": [10.0], "lng": [-130.0]})
        result = filter_bounding_box(df)
        assert len(result) == 0


class TestProcessCities:
    def test_returns_at_most_500_cities(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "city": [f"City{i}" for i in range(600)],
                "state": ["ST"] * 600,
                "lat": rng.uniform(25, 49, 600),
                "lng": rng.uniform(-124, -67, 600),
                "population": rng.integers(1000, 1_000_000, 600),
            }
        )
        result = process_cities(df)
        assert len(result) <= 500

    def test_columns_present(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["A", "B"],
                "state": ["X", "Y"],
                "lat": [30.0, 31.0],
                "lng": [-90.0, -91.0],
                "population": [100000, 200000],
            }
        )
        result = process_cities(df)
        assert set(result.columns) == {"location_id", "city", "state", "lat", "lng"}

    def test_keeps_highest_population_per_grid_cell(self) -> None:
        # Two cities that snap to the same grid cell
        df = pd.DataFrame(
            {
                "city": ["Small", "Big"],
                "state": ["ST", "ST"],
                "lat": [30.01, 30.02],  # Both snap to 30.0
                "lng": [-90.01, -90.02],  # Both snap to -90.0
                "population": [1000, 5000],
            }
        )
        result = process_cities(df)
        assert len(result) == 1
        assert result.iloc[0]["city"] == "Big"

    def test_location_id_starts_at_zero(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["A"],
                "state": ["X"],
                "lat": [30.0],
                "lng": [-90.0],
                "population": [100000],
            }
        )
        result = process_cities(df)
        assert result.iloc[0]["location_id"] == 0

    def test_population_ties_are_broken_stably_by_city_and_state(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["Wichita Falls", "Palm Bay"],
                "state": ["TX", "FL"],
                "lat": [33.91, 28.03],
                "lng": [-98.49, -80.58],
                "population": [104_898, 104_898],
            }
        )

        result = process_cities(df)

        assert result.iloc[0]["city"] == "Palm Bay"
        assert result.iloc[1]["city"] == "Wichita Falls"
