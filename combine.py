"""Combine weather or MRT data for specified cities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DATA_CONFIG = {
    "weather": {
        "parquet_dir": Path("weather_data_parquet"),
        "output_file": Path("weather.parquet"),
        "columns": [
            "location_id",
            "timestamp",
            "temperature_c",
            "wind_speed",
            "relative_humidity",
        ],
        "renames": {
            "wind_speed": "v",
            "temperature_c": "t",
            "relative_humidity": "rh",
            "timestamp": "time",
        },
        "desc": "weather partitions",
    },
    "mrt": {
        "parquet_dir": Path("utci_data_parquet"),
        "output_file": Path("mrt.parquet"),
        "columns": ["location_id", "timestamp", "mean_radiant_temperature_c"],
        "renames": {"mean_radiant_temperature_c": "mrt", "timestamp": "time"},
        "desc": "MRT partitions",
    },
}


def load_and_filter_data(
    file_path: Path,
    merge_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Load a Parquet file, merge with a DataFrame, and filter columns."""
    data_df = pd.read_parquet(file_path)
    merged_df = data_df.merge(
        merge_df,
        how="inner",
        on=["lat", "lng"],
        validate="many_to_one",
    )
    return cast("pd.DataFrame", merged_df[columns])


def main(data_type: str) -> None:
    """Load, merge, and combine data for all cities."""
    if data_type not in DATA_CONFIG:
        raise ValueError(
            "Invalid data type: " + data_type + ". Use 'weather' or 'mrt'.",
        )

    config = DATA_CONFIG[data_type]

    cities_csv = Path("cities.csv")
    if not cities_csv.exists():
        raise FileNotFoundError("Cities file not found: " + str(cities_csv))

    cities_df = pd.read_csv(cities_csv)
    logger.info("Loaded %d city records from %s.", len(cities_df), cities_csv)

    data_files = sorted(config["parquet_dir"].glob("*"))
    if not data_files:
        raise FileNotFoundError(
            "No Parquet files found in " + str(config["parquet_dir"]),
        )
    logger.info("Found %d Parquet partition(s) to process.", len(data_files))

    data_list = [
        load_and_filter_data(file_path, cities_df, config["columns"])
        for file_path in tqdm(data_files, desc="Loading " + config["desc"])
    ]

    combined_df = pd.concat(data_list, ignore_index=True)
    logger.info("Combined %d total rows.", len(combined_df))

    combined_df = combined_df.rename(columns=config["renames"])

    combined_df.to_parquet(config["output_file"], index=False)
    logger.info("Wrote output to %s.", config["output_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine weather or MRT data.")
    parser.add_argument(
        "type",
        choices=["weather", "mrt"],
        help="Data type to combine: 'weather' or 'mrt'",
    )
    args = parser.parse_args()
    main(args.type)
