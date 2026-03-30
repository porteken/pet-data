"""Local pipeline to run PET calculation and analytics."""

import logging
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def run_command(cmd: list[str]) -> None:
    """Run a shell command and exit on failure."""
    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


def main() -> None:
    """Run the integrated project pipeline locally."""
    LOGGER.info("Starting local PET pipeline...")

    # Define directories
    dirs = [
        "output_tiles",
        "era5_data_parquet",
        "weather_data_parquet",
        "utci_data_parquet",
        "combined_data_parquet",
        "pet_data_csv",
        "analytics_data_csv",
    ]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)

    # Step 1: Geometry & locations
    run_command(["python", "cities.py"])
    run_command(
        [
            "python",
            "boxes.py",
            "--cities-csv",
            "cities.csv",
            "--cells-out",
            "output_tiles/unique_grid_cells.csv",
            "--boxes-out",
            "output_tiles/tile_boxes.csv",
            "--snapped-out",
            "output_tiles/snapped_cities.csv",
            "--city-tile-out",
            "output_tiles/city_to_tile.csv",
        ],
    )

    # Step 2: Data pulls - skipping these if external services are skipped,
    # but prompt says "except for pulling the data", so we will pull data.
    # Adjust year/month as appropriate. Here we just run defaults for 1 chunk.
    # Note: era5, weather, mrt
    run_command(
        [
            "python",
            "google_era5.py",
            "--year",
            "2025",
            "--out-dir",
            "./era5_data_parquet",
        ],
    )
    run_command(
        [
            "python",
            "pull_weather.py",
            "--year",
            "2025",
            "--out-dir",
            "./weather_data_parquet",
        ],
    )
    run_command(
        [
            "python",
            "pull_mrt.py",
            "--year",
            "2025",
            "--month",
            "7",
            "--out-dir",
            "./utci_data_parquet",
        ],
    )

    # Step 3: Combine
    run_command(
        [
            "python",
            "combine.py",
            "--year",
            "2025",
            "--era5-root",
            "./era5_data_parquet",
            "--weather-root",
            "./weather_data_parquet",
            "--mrt-root",
            "./utci_data_parquet",
            "--out-dir",
            "./combined_data_parquet",
            "--boxes-csv",
            "output_tiles/tile_boxes.csv",
        ],
    )

    # Step 4: PET Calculation
    run_command(
        [
            "python",
            "calculate_pet.py",
            "--year",
            "2025",
            "--combined-root",
            "./combined_data_parquet",
            "--out-dir",
            "./pet_data_csv",
        ],
    )

    # Step 5: Analytics
    run_command(
        [
            "python",
            "generate_analytics.py",
            "--pet-root",
            "./pet_data_csv",
            "--out-dir",
            "./analytics_data_csv",
            "--reference-years",
            "2000",
            "2001",
            "2002",
        ],
    )

    # Step 6: Load to DB
    run_command(
        [
            "python",
            "load.py",
            "--cities-csv",
            "cities.csv",
            "--pet-root",
            "./pet_data_csv",
            "--analytics-root",
            "./analytics_data_csv",
        ],
    )

    # Step 7: Export necessary CSVs to root
    LOGGER.info("Exporting summary files...")
    # Find generated CSVs in their directories and copy them to root
    for p in Path().rglob("pet_data_csv/**/*.csv"):
        shutil.copy(p, "pet.csv")
        break
    for p in Path().rglob("analytics_data_csv/**/*forecast.csv"):
        shutil.copy(p, "forecast.csv")
        break
    for p in Path().rglob("analytics_data_csv/**/*change_per_decade.csv"):
        shutil.copy(p, "change.csv")
        break
    for p in Path().rglob("analytics_data_csv/**/*percentiles.csv"):
        shutil.copy(p, "percentiles.csv")
        break

    LOGGER.info("Pipeline complete!")


if __name__ == "__main__":
    main()
