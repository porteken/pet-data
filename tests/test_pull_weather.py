"""Tests for pull_weather.py batching helpers."""

from __future__ import annotations

import pytest

from pull_weather import (
    _clamp_batches_to_window,
    _resolve_explicit_window,
    _weather_partition_path,
)


class TestResolveExplicitWindow:
    def test_none_without_overrides(self) -> None:
        assert _resolve_explicit_window(2024, start_date=None, end_date=None) is None

    def test_valid_window_returns_hour_bounds(self) -> None:
        result = _resolve_explicit_window(
            2024,
            start_date="2024-05-01",
            end_date="2024-05-07",
        )
        assert result is not None
        start_hour, end_hour = result
        assert end_hour > start_hour

    def test_mismatched_year_raises(self) -> None:
        with pytest.raises(ValueError, match="requested --year"):
            _resolve_explicit_window(
                2024,
                start_date="2025-05-01",
                end_date="2025-05-07",
            )


class TestClampBatchesToWindow:
    def test_clamps_and_filters_batches(self) -> None:
        batches = [(0, 0, 23), (1, 24, 47), (2, 48, 71)]
        result = _clamp_batches_to_window(
            batches,
            start_hour=12,
            end_hour=50,
        )
        assert result == [(0, 12, 23), (1, 24, 47), (2, 48, 50)]


class TestWeatherPartitionPath:
    def test_builds_partition_path(self) -> None:
        assert (
            _weather_partition_path(2024, 2, 7)
            == "city_shard_index=2/year=2024/batch_index=0007"
        )
