"""Combine weather and MRT parquet shards discovered from partitioned datasets."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, cast

from shards import (
    ShardKey,
    discover_common_shards,
    discover_parquet_shards,
    read_parquet_files,
    select_shards,
)

DataFrame: TypeAlias = Any

pd: Any = cast("Any", import_module("pandas"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

WEATHER_COLUMNS = [
    "timestamp",
    "lat",
    "lng",
    "temperature_c",
    "wind_speed",
    "relative_humidity",
]
MRT_COLUMNS = [
    "timestamp",
    "lat",
    "lng",
    "mean_radiant_temperature_c",
]


def combine_shard(
    shard_key: ShardKey,
    weather_root: str,
    weather_files: list[str],
    mrt_root: str,
    mrt_files: list[str],
    cities_df: DataFrame,
    out_dir: str,
) -> Path | None:
    """Read one weather shard and one MRT shard, then write the merged parquet."""
    weather_df = read_parquet_files(
        weather_root,
        weather_files,
        columns=WEATHER_COLUMNS,
    )
    mrt_df = read_parquet_files(mrt_root, mrt_files, columns=MRT_COLUMNS)

    if weather_df.empty or mrt_df.empty:
        LOGGER.warning("Skipping empty shard %s.", shard_key.label)
        return None

    combined_df = weather_df.merge(
        mrt_df,
        how="inner",
        on=["timestamp", "lat", "lng"],
        validate="one_to_one",
    )
    combined_df = combined_df.merge(
        cities_df,
        how="inner",
        on=["lat", "lng"],
        validate="many_to_one",
    )

    combined_df = combined_df.rename(
        columns={
            "timestamp": "time",
            "temperature_c": "t",
            "wind_speed": "v",
            "relative_humidity": "rh",
            "mean_radiant_temperature_c": "mrt",
        },
    )
    combined_df = combined_df[
        ["location_id", "lat", "lng", "time", "t", "v", "rh", "mrt"]
    ].sort_values(["location_id", "time"])

    if combined_df.empty:
        LOGGER.warning(
            "Combined shard %s produced no matching city rows.",
            shard_key.label,
        )
        return None

    output_dir = Path(out_dir) / shard_key.partition_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.parquet"
    combined_df.to_parquet(output_path, index=False)
    LOGGER.info("Wrote %s rows to %s.", len(combined_df), output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine matching weather and MRT parquet shards into combined "
            "parquet shards."
        ),
    )
    parser.add_argument("--weather-root", default="weather_data_parquet")
    parser.add_argument("--mrt-root", default="utci_data_parquet")
    parser.add_argument("--out-dir", default="combined_data_parquet")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--tile-id", dest="tile_ids", action="append", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Combine the selected weather and MRT parquet shards."""
    args = _parse_args()
    cities_df = pd.read_csv(
        Path("cities.csv"),
        usecols=["location_id", "lat", "lng"],
    )
    weather_shards = discover_parquet_shards(args.weather_root)
    mrt_shards = discover_parquet_shards(args.mrt_root)
    common_shards = discover_common_shards(args.weather_root, args.mrt_root)
    selected_shards = select_shards(
        common_shards,
        year=args.year,
        month=args.month,
        tile_ids=args.tile_ids,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    if not selected_shards:
        LOGGER.warning("No matching shards found for the requested filters.")
        return

    for shard_key in selected_shards:
        combine_shard(
            shard_key=shard_key,
            weather_root=args.weather_root,
            weather_files=weather_shards[shard_key],
            mrt_root=args.mrt_root,
            mrt_files=mrt_shards[shard_key],
            cities_df=cities_df,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
