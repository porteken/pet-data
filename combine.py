"""Combine weather and MRT data for a specific year/month shard."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict, cast

DataFrame: TypeAlias = Any
DataType: TypeAlias = Literal["weather", "mrt"]

pd: Any = cast("Any", import_module("pandas"))
tqdm_module: Any = cast("Any", import_module("tqdm"))
tqdm_progress = tqdm_module.tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


class DataConfig(TypedDict):
    """Shape of config entries for each source dataset."""

    parquet_dir: Path
    columns: list[str]
    renames: dict[str, str]
    desc: str


DATA_CONFIG: dict[DataType, DataConfig] = {
    "weather": {
        "parquet_dir": Path("weather_data_parquet"),
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
        "columns": [
            "location_id",
            "timestamp",
            "mean_radiant_temperature_c",
        ],
        "renames": {
            "mean_radiant_temperature_c": "mrt",
            "timestamp": "time",
        },
        "desc": "MRT partitions",
    },
}

MERGE_KEYS = ["location_id", "time"]
SOURCE_MERGE_KEYS = ["location_id", "timestamp"]


def _empty_source_frame(config: DataConfig) -> DataFrame:
    """Return an empty DataFrame matching the renamed source schema."""
    output_columns = [
        "location_id",
        *[
            config["renames"].get(column_name, column_name)
            for column_name in config["columns"]
            if column_name != "location_id"
        ],
    ]
    return pd.DataFrame(columns=output_columns)


def load_and_filter_data(
    file_path: Path,
    merge_df: DataFrame,
    columns: list[str],
) -> DataFrame:
    """Load a parquet file, merge city IDs, and select requested columns."""
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
    return merged_df.loc[:, columns]


def _partition_dir(parquet_dir: Path, year: str, month: str) -> Path:
    """Return the PyArrow partition directory for a year/month shard."""
    return parquet_dir / f"year={year}" / f"month={int(month)}"


def combine_data(data_type: DataType, year: str, month: str) -> DataFrame:
    """Load and combine one source dataset for a specific year/month shard."""
    config = DATA_CONFIG[data_type]

    cities_path = Path("cities.csv")
    if not cities_path.exists():
        msg = f"Cities file not found: {cities_path}"
        raise FileNotFoundError(msg)

    cities_df = pd.read_csv(cities_path, usecols=["location_id", "lat", "lng"])

    target_dir = _partition_dir(config["parquet_dir"], year, month)
    if not target_dir.exists():
        LOGGER.warning(
            "[%s] No data found for %s-%s in %s. Returning empty frame.",
            data_type,
            year,
            month,
            target_dir,
        )
        return _empty_source_frame(config)

    data_files = sorted(target_dir.rglob("*.parquet"))
    if not data_files:
        LOGGER.warning(
            "[%s] No parquet files found for %s-%s in %s. Returning empty frame.",
            data_type,
            year,
            month,
            target_dir,
        )
        return _empty_source_frame(config)

    frames = [
        load_and_filter_data(file_path, cities_df, config["columns"])
        for file_path in tqdm_progress(data_files, desc=f"Loading {config['desc']}")
    ]

    combined_df = pd.concat(frames, ignore_index=True)
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=SOURCE_MERGE_KEYS,
        keep="last",
    )
    dropped = before_dedup - len(combined_df)
    if dropped > 0:
        LOGGER.warning(
            "[%s] Removed %d duplicate rows for %s-%s.",
            data_type,
            dropped,
            year,
            month,
        )

    combined_df = combined_df.rename(columns=config["renames"])
    LOGGER.info(
        "[%s] Prepared %d rows for %s-%s.",
        data_type,
        len(combined_df),
        year,
        month,
    )
    return combined_df


def merge_weather_and_mrt(weather_df: DataFrame, mrt_df: DataFrame) -> DataFrame:
    """Merge weather and MRT city datasets into one DataFrame."""
    combined_df = weather_df.merge(
        mrt_df,
        how="inner",
        on=MERGE_KEYS,
        validate="one_to_one",
    )
    LOGGER.info("Merged weather and MRT into %d rows.", len(combined_df))
    return combined_df.sort_values(MERGE_KEYS).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the combine step."""
    parser = argparse.ArgumentParser(
        description="Combine weather and MRT parquet shards for a year/month.",
    )
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    """Build the combined parquet shard for the requested year and month."""
    args = _parse_args()

    weather_df = combine_data("weather", args.year, args.month)
    mrt_df = combine_data("mrt", args.year, args.month)

    combined_df = merge_weather_and_mrt(weather_df, mrt_df)
    output_path = Path(f"combined_data_{args.year}_{args.month}.parquet")
    combined_df.to_parquet(output_path, index=False)
    LOGGER.info("Wrote merged output to %s.", output_path)


if __name__ == "__main__":
    main()
