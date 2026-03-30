"""Tests for pull_weather.py — weather download helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pull_weather import (
    WEATHER_CDS_REQUEST_CONCURRENCY,
    _build_weather_date_range,
    _detect_weather_csv_header_row,
    _finalize_weather_frame,
    _normalize_weather_columns,
    _weather_partition_path,
)


class TestWeatherPartitionPath:
    def test_basic(self) -> None:
        assert _weather_partition_path(0) == "city_shard_index=0"
        assert _weather_partition_path(3) == "city_shard_index=3"


class TestBuildWeatherDateRange:
    def test_with_year_and_month(self) -> None:
        result = _build_weather_date_range(year=2020, month=6, start_year=None)
        assert "2020-06-01" in result

    def test_with_year_only(self) -> None:
        result = _build_weather_date_range(year=2020, month=None, start_year=None)
        assert "2020-01-01" in result

    def test_month_without_year_raises(self) -> None:
        with pytest.raises(ValueError, match="--month requires --year"):
            _build_weather_date_range(year=None, month=6, start_year=None)


class TestNormalizeWeatherColumns:
    def test_renames_standard_columns(self) -> None:
        df = pd.DataFrame(
            {
                "valid_time": ["2020-01-01"],
                "latitude": [30.0],
                "longitude": [-90.0],
                "u10": [1.0],
                "v10": [2.0],
                "t2m": [293.0],
                "d2m": [290.0],
            }
        )
        result = _normalize_weather_columns(df)
        assert "time" in result.columns
        assert "lat" in result.columns
        assert "lng" in result.columns

    def test_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"time": ["2020-01-01"], "u10": [1.0]})
        with pytest.raises(ValueError, match="missing expected columns"):
            _normalize_weather_columns(df)


class TestDetectWeatherCsvHeaderRow:
    def test_standard_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "weather.csv.gz"
        csv_path.write_text("valid_time,u10,v10,t2m,d2m\n2020-01-01,1,2,293,290\n")
        assert _detect_weather_csv_header_row(csv_path, encoding="utf-8") == 0

    def test_header_with_metadata_prefix(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "weather.csv.gz"
        csv_path.write_text(
            "# metadata line 1\n# metadata line 2\nvalid_time,u10,v10,t2m,d2m\n"
        )
        assert _detect_weather_csv_header_row(csv_path, encoding="utf-8") == 2


class TestFinalizeWeatherFrame:
    def test_computes_derived_columns(self) -> None:
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2020-01-01 00:00"]),
                "lat": [30.0],
                "lng": [-90.0],
                "u10": [3.0],
                "v10": [4.0],
                "t2m": [293.15],  # 20°C in K
                "d2m": [288.15],  # 15°C in K
            }
        )
        result = _finalize_weather_frame(df)
        assert "wind_speed" in result.columns
        assert "temperature_c" in result.columns
        assert "relative_humidity" in result.columns
        assert "timestamp" in result.columns
        assert "year" in result.columns
        # wind_speed should be sqrt(3^2 + 4^2) = 5
        assert result["wind_speed"].iloc[0] == pytest.approx(5.0)
        # temperature should be ~20°C
        assert result["temperature_c"].iloc[0] == pytest.approx(20.0)
        # u10, v10, t2m, d2m should be dropped
        assert "u10" not in result.columns
        assert "v10" not in result.columns

    def test_concurrency_limit(self) -> None:
        assert WEATHER_CDS_REQUEST_CONCURRENCY == 2
