"""Tests for boxes.py tile generation helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from boxes import (
    BASE_TILE_DEG,
    GRID_DEG,
    build_tile_artifacts,
    floor_to_tile,
    generate_tile_outputs,
    read_city_records,
    snap_to_grid,
)


class TestSnapHelpers:
    def test_snap_to_grid_uses_era5_resolution(self) -> None:
        assert snap_to_grid(30.12) == 30.0
        assert snap_to_grid(30.13) == 30.25

    def test_floor_to_tile_uses_macro_tile(self) -> None:
        assert floor_to_tile(32.75) == 30.0
        assert floor_to_tile(-117.25) == -120.0


class TestBuildTileArtifacts:
    def test_groups_cities_into_unique_grid_cells(self, tmp_path: Path) -> None:
        cities_path = tmp_path / "cities.csv"
        pd.DataFrame(
            [
                {
                    "location_id": 1,
                    "city": "A",
                    "lat": 32.76,
                    "lng": -117.24,
                },
                {
                    "location_id": 2,
                    "city": "B",
                    "lat": 32.74,
                    "lng": -117.26,
                },
                {
                    "location_id": 3,
                    "city": "C",
                    "lat": 40.75,
                    "lng": -73.99,
                },
            ],
        ).to_csv(cities_path, index=False)

        artifacts = build_tile_artifacts(read_city_records(cities_path))
        assert len(artifacts.city_records) == 3
        assert len(artifacts.unique_cells) == 2
        assert len(artifacts.tile_boxes) >= 1
        assert all(cell.tile_deg == BASE_TILE_DEG for cell in artifacts.unique_cells)

    def test_generate_tile_outputs_writes_expected_files(self, tmp_path: Path) -> None:
        cities_path = tmp_path / "cities.csv"
        out_dir = tmp_path / "output_tiles"
        pd.DataFrame(
            [
                {
                    "location_id": 7,
                    "city": "San Diego",
                    "lat": 32.75,
                    "lng": -117.25,
                },
                {
                    "location_id": 116,
                    "city": "Brownsville",
                    "lat": 26.0,
                    "lng": -97.5,
                },
            ],
        ).to_csv(cities_path, index=False)

        artifacts = generate_tile_outputs(cities_path, out_dir)
        assert len(artifacts.tile_boxes) >= 1
        assert (out_dir / "tile_boxes.csv").exists()
        assert (out_dir / "unique_grid_cells.csv").exists()
        assert (out_dir / "city_to_tile.csv").exists()

        city_to_tile = pd.read_csv(out_dir / "city_to_tile.csv")
        assert {"location_id", "tile_id", "grid_lat", "grid_lon"}.issubset(
            city_to_tile.columns,
        )


class TestConstants:
    def test_grid_resolution_matches_era5(self) -> None:
        assert GRID_DEG == 0.25
