"""Integration tests exercising real pipeline components end-to-end.

These tests use actual data on disk (or synthetic data written to temp dirs)
and exercise the real codepaths — no mocking of core logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_pet_shard(root: Path, year: int, tile_id: int, rows: list[dict]) -> Path:
    """Write a PET shard CSV to the standard partition layout."""
    shard_dir = root / f"year={year}" / f"tile_id={tile_id}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / "pet.csv.gz"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _write_combined_parquet(
    root: Path,
    year: int,
    tile_id: int,
    n_locations: int = 2,
    hours: int = 24 * 3,
) -> Path:
    """Write a synthetic combined parquet file for one shard."""
    rng = np.random.default_rng(year * 100 + tile_id)
    timestamps = pd.date_range(f"{year}-01-01", periods=hours, freq="h")
    records = []
    for loc_id in range(1, n_locations + 1):
        for ts in timestamps:
            records.append(
                {
                    "location_id": loc_id,
                    "lat": 30.0 + loc_id * 0.25,
                    "lng": -100.0 + loc_id * 0.25,
                    "time": ts,
                    "t": float(rng.uniform(5, 35)),
                    "v": float(rng.uniform(0.5, 10)),
                    "rh": float(rng.uniform(20, 90)),
                    "mrt": float(rng.uniform(10, 60)),
                }
            )
    df = pd.DataFrame(records)
    out_dir = root / f"year={year}" / f"tile_id={tile_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "combined.parquet"
    df.to_parquet(path, index=False)
    return path


def _write_analytics_csv(
    root: Path,
    shard_count: int,
    shard_index: int,
    filename: str,
    rows: list[dict],
) -> Path:
    out_dir = root / f"shard_count={shard_count:05d}" / f"shard_index={shard_index:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. generate_analytics.py — real pipeline with synthetic PET shards
# ---------------------------------------------------------------------------


class TestGenerateAnalyticsIntegration:
    """Run generate_analytics functions against realistic multi-year data."""

    @pytest.fixture()
    def pet_dirs(self, tmp_path: Path) -> dict[str, Path]:
        pet_root = tmp_path / "pet_data_csv"
        out_dir = tmp_path / "analytics_out"
        rng = np.random.default_rng(42)

        # Generate 5 years of data across 3 tiles with 2 locations each
        for year in [2000, 2005, 2010, 2015, 2020]:
            for tile_id in [1, 2, 3]:
                dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                rows = [
                    {
                        "location_id": tile_id * 100 + loc,
                        "date": d.strftime("%Y-%m-%d"),
                        "pet": round(float(rng.uniform(5, 35)), 1),
                    }
                    for loc in [1, 2]
                    for d in dates
                ]
                _write_pet_shard(pet_root, year, tile_id, rows)

        return {"pet_root": pet_root, "out_dir": out_dir}

    def test_full_analytics_pipeline_multi_tile(
        self, pet_dirs: dict[str, Path]
    ) -> None:
        from generate_analytics import (
            _load_pet_frame,
            generate_change_per_decade,
            generate_forecast,
            generate_percentiles,
        )

        args = argparse.Namespace(
            pet_root=str(pet_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert not df.empty
        assert set(df["year"].unique()) == {2000, 2005, 2010, 2015, 2020}
        assert len(df["location_id"].unique()) == 6  # 3 tiles × 2 locs

        out_dir = pet_dirs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Percentiles
        pct_path = generate_percentiles(df, out_dir)
        pct_df = pd.read_csv(pct_path)
        assert len(pct_df) == 30  # 5 years × 6 locations
        assert set(pct_df.columns) == {"year", "location_id", "p10", "p90"}
        assert (pct_df["p10"] <= pct_df["p90"]).all()

        # Forecast
        fc_path = generate_forecast(df, out_dir)
        fc_df = pd.read_csv(fc_path)
        assert set(fc_df["year"]) == {2030, 2040, 2050}
        assert len(fc_df) == 18  # 6 locations × 3 decades
        assert bool(fc_df["pet"].notna().all())

        # Change per decade
        chg_path = generate_change_per_decade(df, out_dir)
        chg_df = pd.read_csv(chg_path)
        assert "decade" in chg_df.columns
        assert "change" in chg_df.columns
        # With years spanning 2000s/2010s/2020s, we get 2 change rows per loc
        assert len(chg_df) > 0

    def test_shard_partitioning_covers_all_tiles(
        self, pet_dirs: dict[str, Path]
    ) -> None:
        from generate_analytics import _load_pet_frame

        all_loc_ids = set()
        for shard_idx in range(3):
            args = argparse.Namespace(
                pet_root=str(pet_dirs["pet_root"]),
                pet_csv="nonexistent.csv.gz",
                tile_ids=None,
                shard_index=shard_idx,
                shard_count=3,
            )
            df = _load_pet_frame(args)
            all_loc_ids.update(df["location_id"].unique())
        assert len(all_loc_ids) == 6

    def test_tile_id_filter(self, pet_dirs: dict[str, Path]) -> None:
        from generate_analytics import _load_pet_frame

        args = argparse.Namespace(
            pet_root=str(pet_dirs["pet_root"]),
            pet_csv="nonexistent.csv.gz",
            tile_ids=[2],
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert set(df["location_id"].unique()) == {201, 202}


# ---------------------------------------------------------------------------
# 2. calculate_pet.py — PET computation from combined parquet
# ---------------------------------------------------------------------------


class TestCalculatePetIntegration:
    """Run real PET calculations against synthetic combined data."""

    @pytest.fixture()
    def combined_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "combined_data_parquet"
        _write_combined_parquet(root, 2000, 1, n_locations=2, hours=48)
        return root

    def test_pet_calculation_produces_valid_output(self, combined_root: Path) -> None:
        from calculate_pet import calculate_pet_frame

        df = pd.read_parquet(
            combined_root / "year=2000" / "tile_id=1" / "combined.parquet"
        )
        result = calculate_pet_frame(df)

        assert not result.empty
        assert set(result.columns) == {"location_id", "date", "pet"}
        assert result["pet"].notna().all()
        # PET values should be reasonable (roughly -20 to 50 for most conditions)
        assert (result["pet"] >= -30).all()
        assert (result["pet"] <= 60).all()

    def test_pet_daily_max_aggregation(self, combined_root: Path) -> None:
        from calculate_pet import calculate_pet_frame

        df = pd.read_parquet(
            combined_root / "year=2000" / "tile_id=1" / "combined.parquet"
        )
        result = calculate_pet_frame(df)

        # With 48 hours of data for 2 locations, we expect at most 2 days × 2 locations = 4 rows
        assert len(result) <= 4
        # Each location-date combo should appear exactly once (daily max)
        assert result.groupby(["location_id", "date"]).size().max() == 1

    def test_pet_end_to_end_shard_discovery(
        self, combined_root: Path, tmp_path: Path
    ) -> None:
        from calculate_pet import calculate_pet_frame
        from shards import discover_parquet_shards, read_parquet_files, select_shards

        shard_mapping = discover_parquet_shards(str(combined_root))
        assert len(shard_mapping) == 1

        selected = select_shards(shard_mapping, year=2000)
        assert len(selected) == 1

        shard_key = selected[0]
        combined_df = read_parquet_files(
            str(combined_root),
            shard_mapping[shard_key],
            columns=["location_id", "time", "v", "t", "rh", "mrt"],
        )
        pet_df = calculate_pet_frame(combined_df)
        assert not pet_df.empty

        # Write output and verify CSV format
        out_dir = tmp_path / "pet_out" / shard_key.partition_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pet.csv.gz"
        pet_df.to_csv(out_path, index=False)

        loaded = pd.read_csv(out_path)
        assert set(loaded.columns) == {"location_id", "date", "pet"}
        assert len(loaded) == len(pet_df)


# ---------------------------------------------------------------------------
# 3. combine.py — weather + MRT merge logic with synthetic parquet
# ---------------------------------------------------------------------------


class TestCombineIntegration:
    """Test the merge logic that joins weather and MRT data."""

    @pytest.fixture()
    def weather_mrt_dirs(self, tmp_path: Path) -> dict[str, Path]:
        rng = np.random.default_rng(99)
        weather_root = tmp_path / "weather_data_parquet"
        mrt_root = tmp_path / "utci_data_parquet"

        timestamps = pd.date_range("2000-01-01", periods=24, freq="h")
        lats = [30.0, 30.25]
        lngs = [-100.0, -99.75]

        # Weather data
        weather_records = []
        for ts in timestamps:
            for lat, lng in zip(lats, lngs, strict=True):
                weather_records.append(
                    {
                        "timestamp": ts,
                        "lat": lat,
                        "lng": lng,
                        "temperature_c": float(rng.uniform(5, 35)),
                        "wind_speed": float(rng.uniform(0.5, 10)),
                        "relative_humidity": float(rng.uniform(20, 90)),
                    }
                )

        weather_df = pd.DataFrame(weather_records)
        weather_dir = weather_root / "year=2000" / "tile_id=1"
        weather_dir.mkdir(parents=True, exist_ok=True)
        weather_df.to_parquet(weather_dir / "weather.parquet", index=False)

        # MRT data — same timestamps/coords
        mrt_records = []
        for ts in timestamps:
            for lat, lng in zip(lats, lngs, strict=True):
                mrt_records.append(
                    {
                        "timestamp": ts,
                        "lat": lat,
                        "lng": lng,
                        "mean_radiant_temperature_c": float(rng.uniform(10, 60)),
                    }
                )

        mrt_df = pd.DataFrame(mrt_records)
        mrt_dir = mrt_root / "year=2000" / "tile_id=1"
        mrt_dir.mkdir(parents=True, exist_ok=True)
        mrt_df.to_parquet(mrt_dir / "mrt.parquet", index=False)

        return {"weather_root": weather_root, "mrt_root": mrt_root}

    def test_merge_weather_and_mrt(self, weather_mrt_dirs: dict[str, Path]) -> None:
        from combine import _merge_weather_chunk

        weather_df = pd.read_parquet(
            weather_mrt_dirs["weather_root"]
            / "year=2000"
            / "tile_id=1"
            / "weather.parquet",
        )
        mrt_df = pd.read_parquet(
            weather_mrt_dirs["mrt_root"] / "year=2000" / "tile_id=1" / "mrt.parquet",
        )
        cities_df = pd.DataFrame(
            {
                "location_id": [1, 2],
                "lat": [30.0, 30.25],
                "lng": [-100.0, -99.75],
            }
        )

        result = _merge_weather_chunk(weather_df, mrt_df=mrt_df, cities_df=cities_df)
        assert not result.empty
        assert "mean_radiant_temperature_c" in result.columns
        assert "temperature_c" in result.columns
        assert "location_id" in result.columns
        # 24 hours × 2 locations = 48 rows
        assert len(result) == 48

    def test_shard_discovery_across_weather_and_mrt(
        self,
        weather_mrt_dirs: dict[str, Path],
    ) -> None:
        from shards import discover_common_shards, discover_parquet_shards

        weather_shards = discover_parquet_shards(str(weather_mrt_dirs["weather_root"]))
        mrt_shards = discover_parquet_shards(str(weather_mrt_dirs["mrt_root"]))
        assert len(weather_shards) == 1
        assert len(mrt_shards) == 1

        common = discover_common_shards(
            str(weather_mrt_dirs["weather_root"]),
            str(weather_mrt_dirs["mrt_root"]),
        )
        assert len(common) == 1
        assert common[0].year == 2000
        assert common[0].tile_id == 1


# ---------------------------------------------------------------------------
# 4. cities.py + boxes.py — tile generation pipeline
# ---------------------------------------------------------------------------


class TestCitiesBoxesIntegration:
    """Test the cities → boxes tile-generation pipeline."""

    def test_cities_processing_pipeline(self) -> None:
        from cities import filter_bounding_box, process_cities

        # Create synthetic US city data
        cities_df = pd.DataFrame(
            {
                "city": ["CityA", "CityB", "CityC", "CityD", "CityE"],
                "state": ["CA", "TX", "NY", "FL", "WA"],
                "lat": [34.05, 29.76, 40.71, 25.76, 47.61],
                "lng": [-118.24, -95.37, -74.01, -80.19, -122.33],
                "population": [4000000, 2300000, 8300000, 470000, 750000],
            }
        )

        filtered = filter_bounding_box(cities_df)
        assert len(filtered) == 5  # all within continental US

        processed = process_cities(filtered)
        assert "location_id" in processed.columns
        assert len(processed) == 5  # all kept (fewer than 500)

    def test_boxes_tile_generation(self, tmp_path: Path) -> None:
        from boxes import generate_tile_outputs

        # Write a small cities.csv
        cities_path = tmp_path / "cities.csv"
        cities_df = pd.DataFrame(
            {
                "location_id": range(5),
                "city": ["CityA", "CityB", "CityC", "CityD", "CityE"],
                "state": ["CA", "TX", "NY", "FL", "WA"],
                "lat": [34.05, 29.76, 40.71, 25.76, 47.61],
                "lng": [-118.24, -95.37, -74.01, -80.19, -122.33],
            }
        )
        cities_df.to_csv(cities_path, index=False)

        output_dir = tmp_path / "output_tiles"
        generate_tile_outputs(cities_path, output_dir)

        # Verify all output files were created
        assert (output_dir / "snapped_cities.csv").exists()
        assert (output_dir / "tile_boxes.csv").exists()
        assert (output_dir / "unique_grid_cells.csv").exists()
        assert (output_dir / "city_to_tile.csv").exists()

        # Verify tile boxes have expected fields
        tile_boxes = pd.read_csv(output_dir / "tile_boxes.csv")
        assert "tile_id" in tile_boxes.columns
        assert "cds_area" in tile_boxes.columns
        assert "n_cities" in tile_boxes.columns
        assert len(tile_boxes) > 0
        # Each tile should have <= MAX_CITIES_PER_TILE
        assert (tile_boxes["n_cities"] <= 10).all()

        # Verify city_to_tile has all 5 cities
        city_to_tile = pd.read_csv(output_dir / "city_to_tile.csv")
        assert len(city_to_tile) == 5
        assert "tile_id" in city_to_tile.columns
        assert "location_id" in city_to_tile.columns

    def test_snapped_grid_consistency(self, tmp_path: Path) -> None:
        from boxes import generate_tile_outputs

        # Write 10 cities in a tight cluster (should share grid cells)
        cities_path = tmp_path / "cities.csv"
        rng = np.random.default_rng(7)
        cities_df = pd.DataFrame(
            {
                "location_id": range(10),
                "city": [f"City{i}" for i in range(10)],
                "state": ["TX"] * 10,
                "lat": [30.0 + rng.uniform(-0.1, 0.1) for _ in range(10)],
                "lng": [-95.0 + rng.uniform(-0.1, 0.1) for _ in range(10)],
            }
        )
        cities_df.to_csv(cities_path, index=False)

        output_dir = tmp_path / "output_tiles"
        generate_tile_outputs(cities_path, output_dir)

        unique_cells = pd.read_csv(output_dir / "unique_grid_cells.csv")
        # Tight cluster → few unique grid cells
        assert len(unique_cells) <= 4


# ---------------------------------------------------------------------------
# 5. load.py — CSV discovery and validation (no DB required)
# ---------------------------------------------------------------------------


class TestLoadIntegration:
    """Test load.py CSV discovery and validation without a database."""

    @pytest.fixture()
    def load_dirs(self, tmp_path: Path) -> dict[str, Path]:
        # PET shards
        pet_root = tmp_path / "pet_data_csv"
        for tile_id in [1, 2, 3]:
            shard_dir = pet_root / "year=2000" / f"tile_id={tile_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "location_id": [tile_id],
                    "date": ["2000-01-01"],
                    "pet": [20.0],
                }
            ).to_csv(shard_dir / "pet.csv.gz", index=False)

        # Analytics shards
        analytics_root = tmp_path / "analytics_data_csv"
        for shard_idx in range(2):
            shard_dir = (
                analytics_root / "shard_count=00002" / f"shard_index={shard_idx:05d}"
            )
            shard_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "year": [2000],
                    "location_id": [1],
                    "p10": [15.0],
                    "p90": [25.0],
                }
            ).to_csv(shard_dir / "percentiles.csv.gz", index=False)
            pd.DataFrame(
                {
                    "location_id": [1],
                    "year": [2030],
                    "pet": [22.0],
                }
            ).to_csv(shard_dir / "forecast.csv.gz", index=False)
            pd.DataFrame(
                {
                    "location_id": [1],
                    "decade": ["2010s"],
                    "change": [0.5],
                }
            ).to_csv(shard_dir / "change_per_decade.csv.gz", index=False)

        # Cities
        cities_path = tmp_path / "cities.csv"
        pd.DataFrame(
            {
                "location_id": [1, 2, 3],
                "city": ["A", "B", "C"],
                "state": ["CA", "TX", "NY"],
                "lat": [34.0, 30.0, 40.0],
                "lng": [-118.0, -95.0, -74.0],
            }
        ).to_csv(cities_path, index=False)

        return {
            "pet_root": pet_root,
            "analytics_root": analytics_root,
            "cities_csv": cities_path,
        }

    def test_pet_csv_discovery(self, load_dirs: dict[str, Path]) -> None:
        from load import _discover_csv_inputs

        paths = _discover_csv_inputs(
            "nonexistent.csv.gz",
            shard_root=str(load_dirs["pet_root"]),
            shard_file_name="pet.csv.gz",
            shard_partition_key=None,
        )
        assert len(paths) == 3

    def test_analytics_csv_discovery_with_shard_count(
        self,
        load_dirs: dict[str, Path],
    ) -> None:
        from load import _discover_csv_inputs

        paths = _discover_csv_inputs(
            "percentiles.csv.gz",
            shard_root=str(load_dirs["analytics_root"]),
            shard_file_name="percentiles.csv.gz",
            shard_count=2,
            shard_partition_key="shard_count",
        )
        assert len(paths) == 2

    def test_validate_shard_args(self) -> None:
        from load import _validate_load_shard_args

        _validate_load_shard_args(0, 1)
        _validate_load_shard_args(0, 5)
        _validate_load_shard_args(4, 5)

        with pytest.raises(ValueError, match="shard_count"):
            _validate_load_shard_args(0, 0)
        with pytest.raises(ValueError, match="shard_index"):
            _validate_load_shard_args(5, 5)
        with pytest.raises(ValueError, match="shard_index"):
            _validate_load_shard_args(-1, 5)

    def test_load_skips_without_db_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load.py main() should skip gracefully when no DB URI is set."""
        import load

        monkeypatch.setattr(load, "DB_URI", None)
        monkeypatch.setattr(
            "sys.argv",
            [
                "load.py",
                "--cities-csv",
                str(tmp_path / "cities.csv"),
                "--pet-root",
                str(tmp_path / "pet_data_csv"),
                "--analytics-root",
                str(tmp_path / "analytics_data_csv"),
            ],
        )
        # Should return without error when DB_URI is not set
        load.main()


# ---------------------------------------------------------------------------
# 6. shards.py — parquet shard discovery with real parquet files
# ---------------------------------------------------------------------------


class TestShardsIntegration:
    """Test shard discovery against actual parquet files on disk."""

    @pytest.fixture()
    def parquet_tree(self, tmp_path: Path) -> Path:
        root = tmp_path / "data"
        rng = np.random.default_rng(0)

        for year in [2000, 2001]:
            for tile_id in [1, 2]:
                shard_dir = root / f"year={year}" / f"tile_id={tile_id}"
                shard_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"x": rng.random(10).tolist()}).to_parquet(
                    shard_dir / "data.parquet",
                    index=False,
                )

        return root

    def test_discover_and_select(self, parquet_tree: Path) -> None:
        from shards import discover_parquet_shards, select_shards

        mapping = discover_parquet_shards(str(parquet_tree))
        assert len(mapping) == 4  # 2 years × 2 tiles

        selected_2000 = select_shards(mapping, year=2000)
        assert len(selected_2000) == 2
        assert all(sk.year == 2000 for sk in selected_2000)

        selected_tile1 = select_shards(mapping, tile_ids=[1])
        assert len(selected_tile1) == 2
        assert all(sk.tile_id == 1 for sk in selected_tile1)

    def test_read_parquet_columns(self, parquet_tree: Path) -> None:
        from shards import discover_parquet_shards, read_parquet_files

        mapping = discover_parquet_shards(str(parquet_tree))
        first_key = sorted(mapping)[0]
        df = read_parquet_files(str(parquet_tree), mapping[first_key], columns=["x"])
        assert "x" in df.columns
        assert len(df) == 10


# ---------------------------------------------------------------------------
# 7. Full pipeline: combined → PET → analytics (end-to-end)
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Exercise the full pipeline from combined data through analytics output."""

    @pytest.fixture()
    def pipeline_workspace(self, tmp_path: Path) -> dict[str, Path]:
        combined_root = tmp_path / "combined"
        pet_root = tmp_path / "pet_data_csv"
        analytics_root = tmp_path / "analytics"

        # Create 3 years of combined data across 2 tiles
        for year in [2000, 2001, 2010]:
            for tile_id in [1, 2]:
                _write_combined_parquet(
                    combined_root,
                    year,
                    tile_id,
                    n_locations=2,
                    hours=24 * 5,
                )

        return {
            "combined_root": combined_root,
            "pet_root": pet_root,
            "analytics_root": analytics_root,
        }

    def test_combined_to_pet_to_analytics(
        self,
        pipeline_workspace: dict[str, Path],
    ) -> None:
        from calculate_pet import calculate_pet_frame
        from generate_analytics import (
            generate_change_per_decade,
            generate_forecast,
            generate_percentiles,
        )
        from shards import discover_parquet_shards, read_parquet_files, select_shards

        combined_root = pipeline_workspace["combined_root"]
        pet_root = pipeline_workspace["pet_root"]
        analytics_root = pipeline_workspace["analytics_root"]

        # Step 1: Discover combined shards
        shard_mapping = discover_parquet_shards(str(combined_root))
        all_shards = select_shards(shard_mapping)
        assert len(all_shards) == 6  # 3 years × 2 tiles

        # Step 2: Calculate PET for each shard and write CSVs
        for shard_key in all_shards:
            combined_df = read_parquet_files(
                str(combined_root),
                shard_mapping[shard_key],
                columns=["location_id", "time", "v", "t", "rh", "mrt"],
            )
            pet_df = calculate_pet_frame(combined_df)
            assert not pet_df.empty

            out_dir = pet_root / shard_key.partition_path
            out_dir.mkdir(parents=True, exist_ok=True)
            pet_df.to_csv(out_dir / "pet.csv.gz", index=False)

        # Verify all PET shards were written
        pet_csvs = list(pet_root.rglob("pet.csv.gz"))
        assert len(pet_csvs) == 6

        # Step 3: Load PET shards and generate analytics
        from generate_analytics import _load_pet_frame

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        pet_frame = _load_pet_frame(args)
        assert not pet_frame.empty
        assert set(pet_frame["year"].unique()) == {2000, 2001, 2010}

        analytics_root.mkdir(parents=True, exist_ok=True)

        pct_path = generate_percentiles(pet_frame, analytics_root)
        pct_df = pd.read_csv(pct_path)
        assert len(pct_df) > 0

        fc_path = generate_forecast(pet_frame, analytics_root)
        fc_df = pd.read_csv(fc_path)
        assert set(fc_df["year"]) == {2030, 2040, 2050}

        chg_path = generate_change_per_decade(pet_frame, analytics_root)
        chg_df = pd.read_csv(chg_path)
        assert len(chg_df) > 0

    def test_sharded_pipeline_produces_consistent_results(
        self,
        pipeline_workspace: dict[str, Path],
    ) -> None:
        """Verify that splitting across shards produces same total output."""
        # First, write PET so we have something to load
        from calculate_pet import calculate_pet_frame
        from generate_analytics import _load_pet_frame
        from shards import discover_parquet_shards, read_parquet_files, select_shards

        combined_root = pipeline_workspace["combined_root"]
        pet_root = pipeline_workspace["pet_root"]

        shard_mapping = discover_parquet_shards(str(combined_root))
        for shard_key in select_shards(shard_mapping):
            combined_df = read_parquet_files(
                str(combined_root),
                shard_mapping[shard_key],
                columns=["location_id", "time", "v", "t", "rh", "mrt"],
            )
            pet_df = calculate_pet_frame(combined_df)
            out_dir = pet_root / shard_key.partition_path
            out_dir.mkdir(parents=True, exist_ok=True)
            pet_df.to_csv(out_dir / "pet.csv.gz", index=False)

        # Load all at once
        args_full = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv="nonexistent.csv.gz",
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df_full = _load_pet_frame(args_full)

        # Load in 2 shards
        all_sharded_rows = 0
        for idx in range(2):
            args_shard = argparse.Namespace(
                pet_root=str(pet_root),
                pet_csv="nonexistent.csv.gz",
                tile_ids=None,
                shard_index=idx,
                shard_count=2,
            )
            df_shard = _load_pet_frame(args_shard)
            all_sharded_rows += len(df_shard)

        # All rows should be covered without duplication
        assert all_sharded_rows == len(df_full)


# ---------------------------------------------------------------------------
# 8. pet_corrected.py — thermodynamic model sanity
# ---------------------------------------------------------------------------


class TestPetCorrectedIntegration:
    """Verify the PET thermodynamic model produces physically plausible results."""

    @staticmethod
    def _as_float(value: object) -> float:
        return float(np.asarray(value, dtype=float).item())

    @staticmethod
    def _as_float_array(value: object) -> NDArray[np.float64]:
        return np.asarray(value, dtype=float)

    def test_pet_varies_with_temperature(self) -> None:
        from pet_corrected import pet_corrected

        cold = pet_corrected(
            np.array([5.0]),
            np.array([10.0]),
            np.array([2.0]),
            np.array([50.0]),
            icl=0.5,
        )
        hot = pet_corrected(
            np.array([35.0]),
            np.array([50.0]),
            np.array([2.0]),
            np.array([50.0]),
            icl=0.5,
        )
        assert self._as_float(hot) > self._as_float(cold)

    def test_pet_batch_consistency(self) -> None:
        from pet_corrected import pet_corrected

        rng = np.random.default_rng(42)
        n = 100
        t = rng.uniform(5, 35, n)
        mrt = rng.uniform(10, 60, n)
        v = rng.uniform(0.5, 5, n)
        rh = rng.uniform(20, 80, n)

        result = self._as_float_array(pet_corrected(t, mrt, v, rh, icl=0.5))
        assert len(result) == n
        assert bool(np.all(np.isfinite(result)))


# ---------------------------------------------------------------------------
# 9. Real data on disk (if available)
# ---------------------------------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRealDataOnDisk:
    """Test using actual data files if they exist on disk."""

    @pytest.mark.skipif(
        not (
            PROJECT_ROOT / "pet_data_csv" / "year=2000" / "tile_id=11" / "pet.csv.gz"
        ).exists(),
        reason="Real PET data not available",
    )
    def test_generate_analytics_from_real_data(self, tmp_path: Path) -> None:
        from generate_analytics import (
            _load_pet_frame,
            generate_change_per_decade,
            generate_forecast,
            generate_percentiles,
        )

        args = argparse.Namespace(
            pet_root=str(PROJECT_ROOT / "pet_data_csv"),
            pet_csv=str(PROJECT_ROOT / "pet.csv.gz"),
            tile_ids=None,
            shard_index=0,
            shard_count=1,
        )
        df = _load_pet_frame(args)
        assert not df.empty

        out_dir = tmp_path / "analytics"
        out_dir.mkdir()
        generate_percentiles(df, out_dir)
        generate_forecast(df, out_dir)
        generate_change_per_decade(df, out_dir)

        assert (out_dir / "percentiles.csv.gz").exists()
        assert (out_dir / "forecast.csv.gz").exists()
        assert (out_dir / "change_per_decade.csv.gz").exists()

    @pytest.mark.skipif(
        not (
            PROJECT_ROOT
            / "combined_data_parquet"
            / "year=2000"
            / "tile_id=11"
            / "combined.parquet"
        ).exists(),
        reason="Real combined data not available",
    )
    def test_calculate_pet_from_real_combined(self) -> None:
        from calculate_pet import calculate_pet_frame

        df = pd.read_parquet(
            PROJECT_ROOT
            / "combined_data_parquet"
            / "year=2000"
            / "tile_id=11"
            / "combined.parquet",
        )
        result = calculate_pet_frame(df)
        assert not result.empty
        assert set(result.columns) == {"location_id", "date", "pet"}
        assert result["pet"].notna().all()

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "cities.csv").exists(),
        reason="cities.csv not available",
    )
    def test_boxes_from_real_cities(self, tmp_path: Path) -> None:
        from boxes import generate_tile_outputs

        artifacts = generate_tile_outputs(
            PROJECT_ROOT / "cities.csv",
            tmp_path / "tiles",
        )
        assert len(artifacts.city_records) == 500
        assert len(artifacts.tile_boxes) > 0
        # Verify all tiles have valid CDS areas
        for tile_box in artifacts.tile_boxes:
            assert tile_box.cds_area
