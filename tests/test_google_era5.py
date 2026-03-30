"""Tests for google_era5.py — ERA5 batch/shard helpers and constants."""

from __future__ import annotations

import pytest

from google_era5 import (
    DEFAULT_BATCH_HOURS,
    _era5_partition_path,
    _iter_time_batches,
    _resolve_era5_max_workers,
    _select_time_shard_batches,
    _year_time_slice,
)


class TestDefaultBatchHours:
    def test_is_monthly(self) -> None:
        assert DEFAULT_BATCH_HOURS == 24 * 30


class TestYearTimeSlice:
    def test_returns_tuple(self) -> None:
        start_h, end_h = _year_time_slice(2024)
        assert isinstance(start_h, int)
        assert isinstance(end_h, int)
        assert end_h > start_h

    def test_year_span_is_correct(self) -> None:
        start_h, end_h = _year_time_slice(2024)
        # 2024 is a leap year: 8784 hours
        assert end_h - start_h + 1 == 8784

    def test_non_leap_year(self) -> None:
        start_h, end_h = _year_time_slice(2023)
        assert end_h - start_h + 1 == 8760


class TestIterTimeBatches:
    def test_covers_full_year(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        start_h, end_h = _year_time_slice(2024)
        assert batches[0][1] == start_h
        assert batches[-1][2] == end_h

    def test_batch_count_monthly(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=720)
        # 8784 / 720 = 12.2 → 13 batches
        assert len(batches) == 13

    def test_no_gaps_between_batches(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        for i in range(1, len(batches)):
            prev_end = batches[i - 1][2]
            curr_start = batches[i][1]
            assert curr_start == prev_end + 1

    def test_batch_indices_sequential(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        assert [b[0] for b in batches] == list(range(len(batches)))

    def test_raises_on_zero_batch_hours(self) -> None:
        with pytest.raises(ValueError, match="batch_hours"):
            _iter_time_batches(2024, batch_hours=0)


class TestSelectTimeShardBatches:
    def test_all_batches_assigned(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        all_assigned = []
        for shard_index in range(4):
            selected = _select_time_shard_batches(
                batches, time_shard_index=shard_index, time_shard_count=4
            )
            all_assigned.extend(selected)
        # Every batch should be assigned to exactly one shard
        assert sorted(b[0] for b in all_assigned) == sorted(b[0] for b in batches)

    def test_single_shard_returns_all(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        selected = _select_time_shard_batches(
            batches, time_shard_index=0, time_shard_count=1
        )
        assert len(selected) == len(batches)

    def test_four_shards_balanced(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        shard_sizes = []
        for shard_index in range(4):
            selected = _select_time_shard_batches(
                batches, time_shard_index=shard_index, time_shard_count=4
            )
            shard_sizes.append(len(selected))
        # No shard should differ by more than 1 from another
        assert max(shard_sizes) - min(shard_sizes) <= 1

    def test_raises_on_invalid_index(self) -> None:
        batches = _iter_time_batches(2024, batch_hours=DEFAULT_BATCH_HOURS)
        with pytest.raises(ValueError, match="time_shard_index"):
            _select_time_shard_batches(batches, time_shard_index=4, time_shard_count=4)


class TestEra5PartitionPath:
    def test_without_batch(self) -> None:
        result = _era5_partition_path(2024, 5)
        assert result == "year=2024/tile_id=5"

    def test_with_batch_index(self) -> None:
        result = _era5_partition_path(2024, 5, batch_index=3)
        assert result == "year=2024/tile_id=5/batch_index=0003"


class TestResolveEra5MaxWorkers:
    def test_minus_one_returns_cpu_count(self) -> None:
        result = _resolve_era5_max_workers(-1)
        assert result >= 1

    def test_explicit_value_passes_through(self) -> None:
        assert _resolve_era5_max_workers(8) == 8

    def test_raises_on_zero(self) -> None:
        with pytest.raises(ValueError, match="max_workers"):
            _resolve_era5_max_workers(0)
