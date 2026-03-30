"""Script to test tile box generation and print summary statistics."""

from boxes import build_tile_artifacts, read_city_records


def test_boxes() -> None:
    """Read city records and build tile artifacts, printing summary for each tile."""
    records = read_city_records("cities.csv")
    artifacts = build_tile_artifacts(records)
    for box in artifacts.tile_boxes:
        print(  # noqa: T201
            f"Tile {box.tile_id}: {box.cds_area}, approx_cells: {box.approx_grid_cells_downloaded}",
        )


if __name__ == "__main__":
    test_boxes()
