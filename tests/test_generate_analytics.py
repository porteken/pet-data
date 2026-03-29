"""Tests for generate_analytics.py — percentiles, forecast, change per decade."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generate_analytics import (
    PET_CSV_NAME,
    _discover_pet_shards,
    _load_pet_frame,
    _load_pet_frame_from_csv,
    _load_pet_frame_from_shards,
    _output_dir,
    _parse_partition_value,
    _select_tile_ids,
    generate_change_per_decade,
    generate_forecast,
    generate_percentiles,
)


@pytest.fixture()
def sample_pet_df() -> pd.DataFrame:
    """A minimal PET DataFrame spanning two years and two locations."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    rows = []
    for loc_id in [1, 2]:
        for d in dates:
            rows.append(
                {
                    "location_id": loc_id,
                    "date": d,
                    "pet": rng.uniform(10, 30),
                    "year": d.year,
                }
            )
    dates2 = pd.date_range("2021-01-01", periods=365, freq="D")
    for loc_id in [1, 2]:
        for d in dates2:
            rows.append(
                {
                    "location_id": loc_id,
                    "date": d,
                    "pet": rng.uniform(10, 30),
                    "year": d.year,
                }
            )
    return pd.DataFrame(rows)


class TestGeneratePercentiles:
    def test_produces_p10_p90_columns(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "p10" in df.columns
        assert "p90" in df.columns
        assert "year" in df.columns
        assert "location_id" in df.columns

    def test_p10_less_than_p90(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        df = pd.read_csv(path)
        assert (df["p10"] <= df["p90"]).all()

    def test_all_locations_present(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        df = pd.read_csv(path)
        assert set(df["location_id"]) == {1, 2}


class TestGenerateForecast:
    def test_produces_forecast_decades(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_forecast(sample_pet_df, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert set(df["year"]) == {2030, 2040, 2050}

    def test_forecast_has_pet_column(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_forecast(sample_pet_df, tmp_path)
        df = pd.read_csv(path)
        assert "pet" in df.columns
        assert bool(df["pet"].notna().all())

    def test_single_year_produces_empty_forecast(self, tmp_path: Path) -> None:
        """A location with only one year of data cannot fit a trend."""
        df = pd.DataFrame(
            {
                "location_id": [1] * 10,
                "date": pd.date_range("2020-01-01", periods=10),
                "pet": [20.0] * 10,
                "year": [2020] * 10,
            }
        )
        path = generate_forecast(df, tmp_path)
        result = pd.read_csv(path)
        assert len(result) == 0


class TestGenerateChangePerDecade:
    def test_produces_change_column(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_change_per_decade(sample_pet_df, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "change" in df.columns
        assert "decade" in df.columns

    def test_single_decade_produces_no_change(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "location_id": [1] * 365,
                "date": pd.date_range("2020-01-01", periods=365),
                "pet": [20.0] * 365,
                "year": [2020] * 365,
            }
        )
        path = generate_change_per_decade(df, tmp_path)
        result = pd.read_csv(path)
        assert len(result) == 0


class TestDiscoverPetShards:
    def test_discovers_shards(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=5"
        shard_dir.mkdir(parents=True)
        (shard_dir / PET_CSV_NAME).write_text("location_id,date,pet\n1,2020-01-01,10\n")

        result = _discover_pet_shards(tmp_path)
        assert 5 in result
        assert len(result[5]) == 1

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        result = _discover_pet_shards(tmp_path / "nonexistent")
        assert result == {}

    def test_ignores_files_without_tile_id(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020"
        shard_dir.mkdir(parents=True)
        (shard_dir / PET_CSV_NAME).write_text("location_id,date,pet\n1,2020-01-01,10\n")

        result = _discover_pet_shards(tmp_path)
        assert result == {}


class TestParsePartitionValue:
    def test_extracts_tile_id(self, tmp_path: Path) -> None:
        shard_path = tmp_path / "year=2020" / "tile_id=7" / "pet.csv"
        result = _parse_partition_value(shard_path, tmp_path, "tile_id")
        assert result == 7

    def test_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        shard_path = tmp_path / "year=2020" / "pet.csv"
        result = _parse_partition_value(shard_path, tmp_path, "tile_id")
        assert result is None


class TestSelectTileIds:
    def test_basic_selection(self) -> None:
        result = _select_tile_ids(
            [1, 2, 3, 4],
            requested_tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        assert result == [1, 2, 3, 4]

    def test_sharding(self) -> None:
        shard_0 = _select_tile_ids(
            [1, 2, 3, 4],
            requested_tile_ids=None,
            shard_index=0,
            shard_count=2,
        )
        shard_1 = _select_tile_ids(
            [1, 2, 3, 4],
            requested_tile_ids=None,
            shard_index=1,
            shard_count=2,
        )
        assert len(shard_0) == 2
        assert len(shard_1) == 2
        assert set(shard_0).isdisjoint(set(shard_1))

    def test_filter_by_requested(self) -> None:
        result = _select_tile_ids(
            [1, 2, 3, 4],
            requested_tile_ids=[2, 4],
            shard_index=0,
            shard_count=1,
        )
        assert result == [2, 4]

    def test_invalid_shard_count(self) -> None:
        with pytest.raises(ValueError, match="shard_count"):
            _select_tile_ids([1], requested_tile_ids=None, shard_index=0, shard_count=0)

    def test_invalid_shard_index(self) -> None:
        with pytest.raises(ValueError, match="shard_index"):
            _select_tile_ids([1], requested_tile_ids=None, shard_index=2, shard_count=2)


class TestLoadPetFrameFromShards:
    def test_loads_shard_csvs(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        (shard_dir / PET_CSV_NAME).write_text(
            "location_id,date,pet\n1,2020-01-01,10.0\n1,2020-01-02,12.0\n"
        )

        df = _load_pet_frame_from_shards(tmp_path, tile_ids=[1])
        assert len(df) == 2
        assert "year" in df.columns

    def test_missing_tiles_returns_empty(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        (shard_dir / PET_CSV_NAME).write_text(
            "location_id,date,pet\n1,2020-01-01,10.0\n"
        )

        df = _load_pet_frame_from_shards(tmp_path, tile_ids=[99])
        assert df.empty


class TestLoadPetFrameFromCsv:
    def test_loads_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pet.csv"
        csv_path.write_text(
            "location_id,date,pet\n0,2020-01-01,15.0\n1,2020-01-01,16.0\n"
        )
        df = _load_pet_frame_from_csv(csv_path, shard_index=0, shard_count=1)
        assert len(df) == 2

    def test_csv_sharding(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "pet.csv"
        csv_path.write_text(
            "location_id,date,pet\n0,2020-01-01,15.0\n1,2020-01-01,16.0\n"
        )
        df0 = _load_pet_frame_from_csv(csv_path, shard_index=0, shard_count=2)
        df1 = _load_pet_frame_from_csv(csv_path, shard_index=1, shard_count=2)
        assert len(df0) + len(df1) == 2


class TestLoadPetFrame:
    def test_prefers_shards_over_csv(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "pet_data"
        shard_dir = pet_root / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        (shard_dir / PET_CSV_NAME).write_text(
            "location_id,date,pet\n1,2020-01-01,10.0\n"
        )

        csv_path = tmp_path / "pet.csv"
        csv_path.write_text("location_id,date,pet\n99,2020-01-01,99.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(csv_path),
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert 1 in df["location_id"].values
        assert 99 not in df["location_id"].values

    def test_falls_back_to_csv(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "empty_shards"

        csv_path = tmp_path / "pet.csv"
        csv_path.write_text("location_id,date,pet\n5,2020-01-01,20.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(csv_path),
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert len(df) == 1
        assert df["location_id"].iloc[0] == 5

    def test_raises_when_no_data_found(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            pet_root=str(tmp_path / "empty"),
            pet_csv=str(tmp_path / "no_such.csv"),
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        with pytest.raises(FileNotFoundError, match="No PET shard CSV files"):
            _load_pet_frame(args)


class TestOutputDir:
    def test_partitioned_path(self) -> None:
        result = _output_dir(Path("analytics"), shard_index=3, shard_count=20)
        assert result == Path("analytics/shard_count=00020/shard_index=00003")
