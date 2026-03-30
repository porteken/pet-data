"""Tests for pull_mrt.py — MRT download orchestration and helpers."""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pull_mrt import (
    MRT_CDS_REQUEST_CONCURRENCY,
    MRT_CDS_REQUEST_SEMAPHORE,
    MRT_NETCDF_READ_LOCK,
    MRT_THREAD_LOCAL,
    _compute_tight_area,
    _mrt_partition_is_current,
    _mrt_partition_path,
    _normalize_mrt_columns,
    _snap_grid_series,
    _write_mrt_partition,
)


class TestMrtConcurrency:
    def test_concurrency_is_ten(self) -> None:
        assert MRT_CDS_REQUEST_CONCURRENCY == 10

    def test_semaphore_exists(self) -> None:
        assert isinstance(MRT_CDS_REQUEST_SEMAPHORE, threading.BoundedSemaphore)

    def test_netcdf_read_lock_exists(self) -> None:
        assert isinstance(MRT_NETCDF_READ_LOCK, type(threading.Lock()))

    def test_thread_local_exists(self) -> None:
        assert isinstance(MRT_THREAD_LOCAL, threading.local)


class TestMrtPartitionPath:
    def test_without_month(self) -> None:
        result = _mrt_partition_path("2020", 5)
        assert result == "year=2020/tile_id=5"

    def test_with_month(self) -> None:
        result = _mrt_partition_path("2020", 5, month=3)
        assert result == "year=2020/month=03/tile_id=5"


class TestNormalizeMrtColumns:
    def test_renames_and_converts(self) -> None:
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "latitude": [30.0, 30.5],
                "lon": [-90.0, -90.5],
                "mrt": [300.0, 305.0],  # Kelvin
            }
        )
        result = _normalize_mrt_columns(df)
        assert "lat" in result.columns
        assert "lng" in result.columns
        assert "timestamp" in result.columns
        assert "mean_radiant_temperature_c" in result.columns
        # Should convert K → C
        assert result["mean_radiant_temperature_c"].iloc[0] == pytest.approx(
            300.0 - 273.15, abs=0.5
        )

    def test_drops_height_column(self) -> None:
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(["2020-01-01"]),
                "latitude": [30.0],
                "lon": [-90.0],
                "mrt": [300.0],
                "height": [2.0],
            }
        )
        result = _normalize_mrt_columns(df)
        assert "height" not in result.columns

    def test_raises_on_missing_columns(self) -> None:
        df = pd.DataFrame({"time": [1], "latitude": [30.0]})
        with pytest.raises(ValueError, match="missing expected columns"):
            _normalize_mrt_columns(df)


class TestSnapGridSeries:
    def test_snaps_to_quarter_degree(self) -> None:
        series = pd.Series([30.12, 30.37, 30.62])
        result = _snap_grid_series(series)
        expected = pd.Series([30.0, 30.25, 30.5])
        pd.testing.assert_series_equal(result, expected, atol=0.01)


class TestWriteMrtPartition:
    def test_creates_parquet(self, tmp_path: Path) -> None:
        mrt_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-01"]),
                "lat": [30.0],
                "lng": [-90.0],
                "mean_radiant_temperature_c": [25.0],
            }
        )
        _write_mrt_partition(
            mrt_root=str(tmp_path),
            partition_path="year=2020/tile_id=1",
            mrt_df=mrt_df,
        )
        output_path = tmp_path / "year=2020" / "tile_id=1" / "mrt.parquet"
        assert output_path.exists()
        table = pq.read_table(output_path)
        assert table.num_rows == 1


class TestMrtPartitionIsCurrent:
    def test_returns_false_for_nonexistent(self, tmp_path: Path) -> None:
        result = _mrt_partition_is_current(
            str(tmp_path), "year=2020/tile_id=1", year=2020, month=1
        )
        assert result is False


class TestComputeTightArea:
    def test_single_cell(self) -> None:
        cells = pd.DataFrame({"grid_lat": [30.0], "grid_lon": [-90.0]})
        result = _compute_tight_area(cells)
        assert result == [30.0, -90.0, 30.0, -90.0]

    def test_multiple_cells(self) -> None:
        cells = pd.DataFrame(
            {"grid_lat": [30.0, 31.0, 30.5], "grid_lon": [-90.0, -89.0, -89.5]}
        )
        result = _compute_tight_area(cells)
        assert result == [31.0, -90.0, 30.0, -89.0]

    def test_tight_area_smaller_than_tile_box(self) -> None:
        cells = pd.DataFrame({"grid_lat": [25.5, 25.75], "grid_lon": [-80.5, -80.25]})
        result = _compute_tight_area(cells)
        # Tight area should cover just the cells, not a full 3-degree tile
        assert result[0] - result[2] == pytest.approx(0.25)  # north - south
        assert result[3] - result[1] == pytest.approx(0.25)  # east - west
