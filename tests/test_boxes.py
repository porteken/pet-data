"""Tests for boxes.py — grid snapping, tile building, and metadata generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from boxes import (
    BASE_TILE_DEG,
    CityRecord,
    MissingRequiredColumnsError,
    build_tile_artifacts,
    build_tile_box,
    floor_to_tile,
    generate_tile_outputs,
    read_city_records,
    snap_to_grid,
)


class TestSnapToGrid:
    def test_exact_grid_point(self) -> None:
        assert snap_to_grid(30.0) == 30.0

    def test_rounds_to_nearest_quarter(self) -> None:
        assert snap_to_grid(30.12) == 30.0
        assert snap_to_grid(30.13) == 30.25
        assert snap_to_grid(30.37) == 30.25
        assert snap_to_grid(30.38) == 30.5

    def test_negative_values(self) -> None:
        assert snap_to_grid(-90.12) == -90.0
        assert snap_to_grid(-90.13) == -90.25


class TestFloorToTile:
    def test_exact_tile_boundary(self) -> None:
        assert floor_to_tile(30.0) == 30.0

    def test_within_tile(self) -> None:
        assert floor_to_tile(31.5) == 30.0
        assert floor_to_tile(32.99) == 30.0

    def test_negative(self) -> None:
        assert floor_to_tile(-91.0) == -93.0


class TestBuildTileBox:
    def test_basic_tile_box(self) -> None:
        box = build_tile_box(
            tile_lat_min=30.0,
            tile_lon_min=-90.0,
            tile_id=0,
            n_cities=5,
            n_unique_cells=10,
        )
        assert box.tile_id == 0
        assert box.south == 30.0
        assert box.north == 30.0 + BASE_TILE_DEG
        assert box.west == -90.0
        assert box.east == -90.0 + BASE_TILE_DEG
        assert box.n_cities == 5
        assert box.n_unique_cells == 10


class TestReadCityRecords:
    def test_reads_valid_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "cities.csv"
        csv_path.write_text("city,lat,lng\nNew York,40.71,-74.01\nLA,34.05,-118.24\n")
        records = read_city_records(csv_path)
        assert len(records) == 2
        assert records[0].city == "New York"

    def test_snaps_coordinates(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "cities.csv"
        csv_path.write_text("city,lat,lng\nTest,40.71,-74.01\n")
        records = read_city_records(csv_path)
        assert records[0].grid_lat == snap_to_grid(40.71)
        assert records[0].grid_lon == snap_to_grid(-74.01)

    def test_missing_required_columns_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "cities.csv"
        csv_path.write_text("city,population\nNew York,8000000\n")
        with pytest.raises(MissingRequiredColumnsError):
            read_city_records(csv_path)

    def test_skips_rows_with_no_coordinates(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "cities.csv"
        csv_path.write_text("city,lat,lng\nOK,40.0,-74.0\nBad,,\n")
        records = read_city_records(csv_path)
        assert len(records) == 1

    def test_reads_location_id(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "cities.csv"
        csv_path.write_text("location_id,city,lat,lng\n42,Test,40.0,-74.0\n")
        records = read_city_records(csv_path)
        assert records[0].location_id == 42


class TestBuildTileArtifacts:
    def test_groups_cities_into_tiles(self) -> None:
        records = [
            CityRecord(city="A", lat=30.1, lng=-90.1, grid_lat=30.0, grid_lon=-90.0),
            CityRecord(city="B", lat=30.2, lng=-90.2, grid_lat=30.25, grid_lon=-90.25),
        ]
        artifacts = build_tile_artifacts(records)
        assert len(artifacts.tile_boxes) >= 1
        assert len(artifacts.unique_cells) >= 1
        assert len(artifacts.city_to_tile_rows) == 2

    def test_single_city(self) -> None:
        records = [
            CityRecord(city="A", lat=30.0, lng=-90.0, grid_lat=30.0, grid_lon=-90.0),
        ]
        artifacts = build_tile_artifacts(records)
        assert len(artifacts.tile_boxes) == 1
        assert artifacts.tile_boxes[0].n_cities == 1


class TestGenerateTileOutputs:
    def test_creates_all_output_files(self, tmp_path: Path) -> None:
        cities_csv = tmp_path / "cities.csv"
        cities_csv.write_text(
            "location_id,city,lat,lng\n0,Denver,39.74,-104.99\n1,NYC,40.71,-74.01\n"
        )
        out_dir = tmp_path / "tiles"
        artifacts = generate_tile_outputs(cities_csv, out_dir)

        assert (out_dir / "snapped_cities.csv").exists()
        assert (out_dir / "unique_grid_cells.csv").exists()
        assert (out_dir / "tile_boxes.csv").exists()
        assert (out_dir / "city_to_tile.csv").exists()
        assert len(artifacts.tile_boxes) >= 1
