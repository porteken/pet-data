"""Tests for combine.py — weather + MRT merge logic."""

from __future__ import annotations

import pandas as pd

from combine import (
    _merge_weather_chunk,
)


class TestMergeWeatherChunk:
    def test_inner_join_on_timestamp_lat_lng(self) -> None:
        weather = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "lat": [30.0, 30.0],
                "lng": [-90.0, -90.0],
                "temperature_c": [20.0, 21.0],
                "wind_speed": [1.0, 1.5],
                "relative_humidity": [50.0, 55.0],
            }
        )
        mrt = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "lat": [30.0, 30.0],
                "lng": [-90.0, -90.0],
                "mean_radiant_temperature_c": [25.0, 26.0],
            }
        )
        cities = pd.DataFrame({"location_id": [1], "lat": [30.0], "lng": [-90.0]})
        result = _merge_weather_chunk(weather, mrt_df=mrt, cities_df=cities)
        assert len(result) == 2
        assert "mean_radiant_temperature_c" in result.columns
        assert "location_id" in result.columns

    def test_no_matching_rows_returns_empty(self) -> None:
        weather = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00"]),
                "lat": [30.0],
                "lng": [-90.0],
                "temperature_c": [20.0],
                "wind_speed": [1.0],
                "relative_humidity": [50.0],
            }
        )
        mrt = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-02 00:00"]),  # Different day
                "lat": [30.0],
                "lng": [-90.0],
                "mean_radiant_temperature_c": [25.0],
            }
        )
        cities = pd.DataFrame({"location_id": [1], "lat": [30.0], "lng": [-90.0]})
        result = _merge_weather_chunk(weather, mrt_df=mrt, cities_df=cities)
        assert len(result) == 0

    def test_location_id_already_in_weather(self) -> None:
        """When weather already has location_id, should not re-merge cities."""
        weather = pd.DataFrame(
            {
                "location_id": [1, 1],
                "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "lat": [30.0, 30.0],
                "lng": [-90.0, -90.0],
                "temperature_c": [20.0, 21.0],
                "wind_speed": [1.0, 1.5],
                "relative_humidity": [50.0, 55.0],
            }
        )
        mrt = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "lat": [30.0, 30.0],
                "lng": [-90.0, -90.0],
                "mean_radiant_temperature_c": [25.0, 26.0],
            }
        )
        cities = pd.DataFrame({"location_id": [1], "lat": [30.0], "lng": [-90.0]})
        result = _merge_weather_chunk(weather, mrt_df=mrt, cities_df=cities)
        assert len(result) == 2
        # location_id should come from weather, not from a second merge
        assert result["location_id"].iloc[0] == 1
