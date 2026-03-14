"""Create 3x3 degree CDS tile metadata for snapped ERA5 city grid cells."""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

INPUT_CSV: Final[str] = "cities.csv"
OUTPUT_DIR: Final[str] = "output_tiles"
GRID_DEG: Final[float] = 0.25
TILE_DEG: Final[float] = 3.0
TILE_GRID_CELLS_PER_SIDE: Final[int] = int(TILE_DEG / GRID_DEG)
APPROX_GRID_CELLS_DOWNLOADED: Final[int] = TILE_GRID_CELLS_PER_SIDE**2
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({"lat", "lng"})

LOGGER = logging.getLogger(__name__)
CSVValue: TypeAlias = str | float | int
CSVRow: TypeAlias = dict[str, CSVValue]


@dataclass(frozen=True)
class CityRecord:
    """One city row with snapped ERA5 grid coordinates."""

    city: str
    lat: float
    lng: float
    grid_lat: float
    grid_lon: float
    location_id: int | None = None


@dataclass(frozen=True, order=True)
class GridCell:
    """One unique snapped ERA5 grid cell plus its tile bounds."""

    grid_lat: float
    grid_lon: float
    tile_lat_min: float
    tile_lon_min: float


@dataclass(frozen=True, order=True)
class TileKey:
    """Logical key for one 3x3 degree tile."""

    tile_lat_min: float
    tile_lon_min: float


@dataclass(frozen=True)
class TileBox:
    """CDS-ready metadata for one occupied tile."""

    tile_id: int
    tile_lat_min: float
    tile_lon_min: float
    north: float
    west: float
    south: float
    east: float
    cds_area: str
    n_unique_cells: int
    approx_grid_cells_downloaded: int


@dataclass(frozen=True)
class TileArtifacts:
    """All generated tile metadata collections."""

    city_records: list[CityRecord]
    unique_cells: list[GridCell]
    tile_boxes: list[TileBox]
    city_to_tile_rows: list[CSVRow]


class MissingRequiredColumnsError(ValueError):
    """Raised when the source city CSV is missing required columns."""

    def __init__(self, missing: list[str]) -> None:
        """Build an error that lists the missing CSV column names."""
        super().__init__(f"Missing required columns: {', '.join(missing)}")


def snap_to_grid(value: float, step: float = GRID_DEG) -> float:
    """Snap a coordinate to the nearest ERA5 grid center."""
    return round(value / step) * step


def floor_to_tile(value: float, tile_deg: float = TILE_DEG) -> float:
    """Return the lower tile edge that contains the coordinate."""
    return math.floor(value / tile_deg) * tile_deg


def build_tile_box(
    tile_lat_min: float,
    tile_lon_min: float,
    tile_id: int,
    n_unique_cells: int,
    tile_deg: float = TILE_DEG,
) -> TileBox:
    """Build a CDS area box in [North, West, South, East] order."""
    south = tile_lat_min
    north = tile_lat_min + tile_deg
    west = tile_lon_min
    east = tile_lon_min + tile_deg
    cds_area = f"[{north}, {west}, {south}, {east}]"
    return TileBox(
        tile_id=tile_id,
        tile_lat_min=tile_lat_min,
        tile_lon_min=tile_lon_min,
        north=north,
        west=west,
        south=south,
        east=east,
        cds_area=cds_area,
        n_unique_cells=n_unique_cells,
        approx_grid_cells_downloaded=APPROX_GRID_CELLS_DOWNLOADED,
    )


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _parse_optional_int(value: str | None) -> int | None:
    numeric_value = _parse_float(value)
    if numeric_value is None:
        return None

    try:
        return int(numeric_value)
    except (OverflowError, ValueError):
        return None


def read_city_records(input_path: Path | str = INPUT_CSV) -> list[CityRecord]:
    """Read cities and snap them to ERA5 grid cells."""
    path = Path(input_path)
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise MissingRequiredColumnsError(sorted(REQUIRED_COLUMNS))

        missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
        if missing:
            raise MissingRequiredColumnsError(missing)

        city_records: list[CityRecord] = []
        for row in reader:
            lat = _parse_float(row.get("lat"))
            lng = _parse_float(row.get("lng"))
            if lat is None or lng is None:
                continue

            city_records.append(
                CityRecord(
                    city=row.get("city", "") or "",
                    lat=lat,
                    lng=lng,
                    grid_lat=snap_to_grid(lat),
                    grid_lon=snap_to_grid(lng),
                    location_id=_parse_optional_int(row.get("location_id")),
                ),
            )

    return city_records


def build_tile_artifacts(city_records: list[CityRecord]) -> TileArtifacts:
    """Generate unique cells, occupied tile boxes, and city-to-tile rows."""
    unique_pairs = {(record.grid_lat, record.grid_lon) for record in city_records}
    unique_cells = [
        GridCell(
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            tile_lat_min=floor_to_tile(grid_lat),
            tile_lon_min=floor_to_tile(grid_lon),
        )
        for grid_lat, grid_lon in sorted(unique_pairs)
    ]

    tile_counts = Counter(
        TileKey(tile_lat_min=cell.tile_lat_min, tile_lon_min=cell.tile_lon_min)
        for cell in unique_cells
    )
    tile_boxes = [
        build_tile_box(
            tile_lat_min=tile_key.tile_lat_min,
            tile_lon_min=tile_key.tile_lon_min,
            tile_id=index,
            n_unique_cells=tile_counts[tile_key],
        )
        for index, tile_key in enumerate(sorted(tile_counts), start=1)
    ]
    tile_lookup = {
        TileKey(
            tile_lat_min=tile_box.tile_lat_min,
            tile_lon_min=tile_box.tile_lon_min,
        ): tile_box
        for tile_box in tile_boxes
    }

    city_to_tile_rows: list[CSVRow] = []
    for record in city_records:
        tile_key = TileKey(
            tile_lat_min=floor_to_tile(record.grid_lat),
            tile_lon_min=floor_to_tile(record.grid_lon),
        )
        tile_box = tile_lookup[tile_key]
        row: CSVRow = {
            "city": record.city,
            "lat": record.lat,
            "lng": record.lng,
            "grid_lat": record.grid_lat,
            "grid_lon": record.grid_lon,
            "tile_lat_min": tile_box.tile_lat_min,
            "tile_lon_min": tile_box.tile_lon_min,
            "tile_id": tile_box.tile_id,
            "cds_area": tile_box.cds_area,
        }
        if record.location_id is not None:
            row["location_id"] = record.location_id
        city_to_tile_rows.append(row)

    return TileArtifacts(
        city_records=city_records,
        unique_cells=unique_cells,
        tile_boxes=tile_boxes,
        city_to_tile_rows=city_to_tile_rows,
    )


def generate_tile_outputs(
    input_path: Path | str = INPUT_CSV,
    output_dir: Path | str = OUTPUT_DIR,
) -> TileArtifacts:
    """Generate and write all tile metadata CSVs."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifacts = build_tile_artifacts(read_city_records(input_path))
    include_location_id = any(
        record.location_id is not None for record in artifacts.city_records
    )

    snapped_city_rows: list[CSVRow] = []
    for record in artifacts.city_records:
        row: CSVRow = {
            "city": record.city,
            "lat": record.lat,
            "lng": record.lng,
            "grid_lat": record.grid_lat,
            "grid_lon": record.grid_lon,
        }
        if include_location_id and record.location_id is not None:
            row["location_id"] = record.location_id
        snapped_city_rows.append(row)

    unique_cell_rows: list[CSVRow] = [
        {
            "grid_lat": cell.grid_lat,
            "grid_lon": cell.grid_lon,
            "tile_lat_min": cell.tile_lat_min,
            "tile_lon_min": cell.tile_lon_min,
        }
        for cell in artifacts.unique_cells
    ]
    tile_box_rows: list[CSVRow] = [
        {
            "tile_id": tile_box.tile_id,
            "tile_lat_min": tile_box.tile_lat_min,
            "tile_lon_min": tile_box.tile_lon_min,
            "north": tile_box.north,
            "west": tile_box.west,
            "south": tile_box.south,
            "east": tile_box.east,
            "cds_area": tile_box.cds_area,
            "n_unique_cells": tile_box.n_unique_cells,
            "approx_grid_cells_downloaded": tile_box.approx_grid_cells_downloaded,
        }
        for tile_box in artifacts.tile_boxes
    ]

    _write_csv(
        outdir / "snapped_cities.csv",
        [
            *(("location_id",) if include_location_id else ()),
            "city",
            "lat",
            "lng",
            "grid_lat",
            "grid_lon",
        ],
        snapped_city_rows,
    )
    _write_csv(
        outdir / "unique_grid_cells.csv",
        ["grid_lat", "grid_lon", "tile_lat_min", "tile_lon_min"],
        unique_cell_rows,
    )
    _write_csv(
        outdir / "tile_boxes.csv",
        [
            "tile_id",
            "tile_lat_min",
            "tile_lon_min",
            "north",
            "west",
            "south",
            "east",
            "cds_area",
            "n_unique_cells",
            "approx_grid_cells_downloaded",
        ],
        tile_box_rows,
    )
    city_to_tile_fieldnames = [
        *(("location_id",) if include_location_id else ()),
        "city",
        "lat",
        "lng",
        "grid_lat",
        "grid_lon",
        "tile_lat_min",
        "tile_lon_min",
        "tile_id",
        "cds_area",
    ]
    _write_csv(
        outdir / "city_to_tile.csv",
        city_to_tile_fieldnames,
        artifacts.city_to_tile_rows,
    )

    LOGGER.info("Input cities: %s", f"{len(artifacts.city_records):,}")
    LOGGER.info("Unique snapped ERA5 cells: %s", f"{len(artifacts.unique_cells):,}")
    LOGGER.info("Occupied 3x3 tiles: %s", f"{len(artifacts.tile_boxes):,}")
    return artifacts


def _write_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: Sequence[Mapping[str, CSVValue]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate snapped cells and tile metadata on disk."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    generate_tile_outputs()


if __name__ == "__main__":
    main()
