"""Combine weather and MRT data for specified cities."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataConfig(TypedDict):
    """Shape of config entries for each source dataset."""

    parquet_dir: Path
    output_file: Path
    columns: list[str]
    renames: dict[str, str]
    desc: str


DATA_CONFIG: dict[Literal["weather", "mrt"], DataConfig] = {
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
COMBINED_OUTPUT_FILE = Path("combined_data.parquet")
MERGE_KEYS = ["location_id", "time"]
SOURCE_MERGE_KEYS = ["location_id", "timestamp"]
PARTITION_RE = re.compile(r"^(year|month)=\d+$")


def load_and_filter_data(
    file_path: Path,
    merge_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Load a Parquet file, merge with a DataFrame, and filter columns."""
    parquet_columns = sorted(
        {"lat", "lng", *[col for col in columns if col != "location_id"]},
    )
    data_df = pd.read_parquet(file_path, columns=parquet_columns)
    merged_df = data_df.merge(
        merge_df,
        how="inner",
        on=["lat", "lng"],
        validate="many_to_one",
    )
    return merged_df[columns]


def _partition_group_for_file(file_path: Path, root_dir: Path) -> str:
    """Return a stable group key (e.g., year/month) for each parquet file."""
    try:
        rel_parts = file_path.relative_to(root_dir).parts[:-1]
    except ValueError:
        rel_parts = file_path.parts[:-1]

    partition_parts = [part for part in rel_parts if PARTITION_RE.match(part)]
    return "/".join(partition_parts) if partition_parts else "unpartitioned"


def combine_data(data_type: str) -> pd.DataFrame:
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
    logger.info(
        "[%s] Loaded %d city records from %s.",
        data_type,
        len(cities_df),
        cities_csv,
    )

    data_files = sorted(config["parquet_dir"].rglob("*.parquet"))

    if not data_files:
        raise FileNotFoundError(
            "No Parquet files found in " + str(config["parquet_dir"]),
        )
    logger.info(
        "[%s] Found %d Parquet partition(s) to process.",
        data_type,
        len(data_files),
    )

    grouped_files: dict[str, list[Path]] = {}
    for file_path in data_files:
        group_key = _partition_group_for_file(file_path, config["parquet_dir"])
        grouped_files.setdefault(group_key, []).append(file_path)

    grouped_data: list[pd.DataFrame] = []
    for group_key in tqdm(
        sorted(grouped_files),
        desc="Loading " + config["desc"],
    ):
        group_frames = [
            load_and_filter_data(file_path, cities_df, config["columns"])
            for file_path in grouped_files[group_key]
        ]
        group_df = pd.concat(group_frames, ignore_index=True)
        before_dedup = len(group_df)
        group_df = group_df.drop_duplicates(subset=SOURCE_MERGE_KEYS, keep="last")
        dropped = before_dedup - len(group_df)
        if dropped > 0:
            logger.warning(
                "[%s] Removed %d duplicate rows in partition %s.",
                data_type,
                dropped,
                group_key,
            )
        grouped_data.append(group_df)

    combined_df = pd.concat(grouped_data, ignore_index=True)
    logger.info(
        "[%s] Combined %d total rows after dedupe.",
        data_type,
        len(combined_df),
    )

    combined_df = combined_df.rename(columns=config["renames"])

    combined_df.to_parquet(config["output_file"], index=False)
    logger.info("[%s] Wrote output to %s.", data_type, config["output_file"])
    return combined_df


def merge_weather_and_mrt(
    weather_df: pd.DataFrame,
    mrt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weather and MRT city datasets into one DataFrame."""
    combined_df = weather_df.merge(
        mrt_df,
        how="inner",
        on=MERGE_KEYS,
        validate="one_to_one",
    )
    logger.info("Merged weather and MRT into %d rows.", len(combined_df))
    return combined_df.sort_values(MERGE_KEYS).reset_index(drop=True)


def main() -> None:
    """Build weather + MRT city datasets and export merged output."""
    weather_df = combine_data("weather")
    mrt_df = combine_data("mrt")

    combined_df = merge_weather_and_mrt(weather_df, mrt_df)
    combined_df.to_parquet(COMBINED_OUTPUT_FILE, index=False)
    logger.info("Wrote merged output to %s.", COMBINED_OUTPUT_FILE)


if __name__ == "__main__":
    main()
