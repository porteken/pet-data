"""Tests for generate_analytics.py — percentiles, forecast, change per decade."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generate_analytics import (
    _discover_pet_files,
    _load_pet_frame,
    _load_pet_frame_from_csv,
    _output_dir,
    _select_pet_files,
    generate_change_per_decade,
    generate_forecast,
    generate_percentiles,
)


def _write_pet_parquet(path: Path, content_csv: str) -> None:
    import io

    df = pd.read_csv(io.StringIO(content_csv))
    df.to_parquet(path, index=False)


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
        df = pd.read_parquet(path)
        assert "p10" in df.columns
        assert "p90" in df.columns
        assert "year" in df.columns
        assert "location_id" in df.columns

    def test_p10_less_than_p90(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        df = pd.read_parquet(path)
        assert (df["p10"] <= df["p90"]).all()

    def test_all_locations_present(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_percentiles(sample_pet_df, tmp_path)
        df = pd.read_parquet(path)
        assert set(df["location_id"]) == {1, 2}


class TestGenerateForecast:
    def test_produces_forecast_decades(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_forecast(sample_pet_df, tmp_path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert int(df["year"].min()) == 2022  # pyright: ignore[reportArgumentType]
        assert int(df["year"].max()) == 2100  # pyright: ignore[reportArgumentType]
        assert {"lower", "upper"}.issubset(df.columns)

    def test_forecast_has_pet_column(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_forecast(sample_pet_df, tmp_path)
        df = pd.read_parquet(path)
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
        result = pd.read_parquet(path)
        assert len(result) == 0


class TestGenerateChangePerDecade:
    def test_produces_change_column(
        self, sample_pet_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = generate_change_per_decade(sample_pet_df, tmp_path)
        assert path.exists()
        df = pd.read_parquet(path)
        assert "change" in df.columns
        assert "year" in df.columns

    def test_smoke_style_input_produces_future_decade_changes(
        self, tmp_path: Path
    ) -> None:
        df = pd.DataFrame(
            {
                "location_id": [1, 1],
                "date": pd.to_datetime(["2024-01-01", "2025-01-01"]),
                "pet": [20.0, 21.0],
                "year": [2024, 2025],
            }
        )
        path = generate_change_per_decade(df, tmp_path)
        result = pd.read_parquet(path)
        assert len(result) > 0
        assert int(result["year"].min()) == 2030  # pyright: ignore[reportArgumentType]

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
        result = pd.read_parquet(path)
        assert len(result) == 0


class TestDiscoverPetFiles:
    def test_discovers_batch_parquets(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020"
        shard_dir.mkdir(parents=True)
        _write_pet_parquet(
            shard_dir / "pet_batch_0000_00.parquet",
            "location_id,date,pet\n1,2020-01-01,10\n",
        )

        result = _discover_pet_files(tmp_path)
        assert len(result) == 1

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        result = _discover_pet_files(tmp_path / "nonexistent")
        assert result == []

    def test_ignores_non_batch_files(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020"
        shard_dir.mkdir(parents=True)
        (shard_dir / "pet.csv").write_text("location_id,date,pet\n1,2020-01-01,10\n")
        result = _discover_pet_files(tmp_path)
        assert result == []


class TestSelectPetFiles:
    def test_single_shard_returns_all(self) -> None:
        files = [Path(f"f{i}.parquet") for i in range(4)]
        result = _select_pet_files(files, shard_index=0, shard_count=1)
        assert result == files

    def test_two_shards_split_evenly(self) -> None:
        files = [Path(f"f{i}.parquet") for i in range(4)]
        shard_0 = _select_pet_files(files, shard_index=0, shard_count=2)
        shard_1 = _select_pet_files(files, shard_index=1, shard_count=2)
        assert len(shard_0) == 2
        assert len(shard_1) == 2
        assert set(shard_0).isdisjoint(set(shard_1))

    def test_shard_beyond_count_returns_empty(self) -> None:
        files = [Path("f0.parquet"), Path("f1.parquet")]
        result = _select_pet_files(files, shard_index=5, shard_count=10)
        assert result == []


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
        shard_dir = pet_root / "year=2020"
        shard_dir.mkdir(parents=True)
        _write_pet_parquet(
            shard_dir / "pet_batch_0000_00.parquet",
            "location_id,date,pet\n1,2020-01-01,10.0\n",
        )

        csv_path = tmp_path / "pet.csv"
        csv_path.write_text("location_id,date,pet\n99,2020-01-01,99.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(csv_path),
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
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert len(df) == 1
        assert df["location_id"].iloc[0] == 5

    def test_returns_empty_when_no_data_found(self, tmp_path: Path) -> None:
        args = argparse.Namespace(
            pet_root=str(tmp_path / "empty"),
            pet_csv=str(tmp_path / "no_such.csv"),
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert df.empty


class TestOutputDir:
    def test_partitioned_path(self) -> None:
        result = _output_dir(Path("analytics"), shard_index=3, shard_count=20)
        assert result == Path("analytics/shard_count=00020/shard_index=00003")
