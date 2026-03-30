"""Combine weather and MRT parquet shards into year/tile combined outputs."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from boxes import OUTPUT_DIR
from shards import (
    ShardKey,
    discover_common_shards,
    discover_parquet_shards,
    read_parquet_files,
    resolve_filesystem,
    select_shards,
)
from shared_config import build_year_date_bounds

if TYPE_CHECKING:
    from collections.abc import Iterable

DataFrame: TypeAlias = Any

pd: Any = cast("Any", import_module("pandas"))
dataset_module: Any = cast("Any", import_module("pyarrow.dataset"))
fs_module: Any = cast("Any", import_module("pyarrow.fs"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

READ_FILE_CHUNK_SIZE = 16
READ_MAX_WORKERS = 8
MERGE_ROW_CHUNK_SIZE = 100_000
MERGE_MAX_WORKERS = 8

WEATHER_COLUMNS = [
    "timestamp",
    "lat",
    "lng",
    "temperature_c",
    "wind_speed",
    "relative_humidity",
]
WEATHER_COLUMNS_WITH_LOCATION_ID = [
    "location_id",
    *WEATHER_COLUMNS,
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
    *,
    weather_columns: list[str] = WEATHER_COLUMNS,
    weather_filters: object | None = None,
) -> Path | None:
    """Read one weather shard and one MRT shard, then write the merged parquet."""
    weather_df = _read_parquet_files_in_parallel(
        weather_root,
        weather_files,
        columns=weather_columns,
        filters=weather_filters,
    )
    mrt_df = _read_parquet_files_in_parallel(
        mrt_root,
        mrt_files,
        columns=MRT_COLUMNS,
    )

    if weather_df.empty:
        msg = f"Shard {shard_key.label}: weather data is empty."
        raise RuntimeError(msg)
    if mrt_df.empty:
        msg = f"Shard {shard_key.label}: MRT data is empty."
        raise RuntimeError(msg)

    combined_df = _merge_frames_in_parallel(
        weather_df=weather_df,
        mrt_df=mrt_df,
        cities_df=cities_df,
        shard_key=shard_key,
    )

    if combined_df.empty:
        msg = f"Shard {shard_key.label}: no rows after merging weather and MRT."
        raise RuntimeError(msg)

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

    output_dir = Path(out_dir) / shard_key.partition_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.parquet"
    combined_df.to_parquet(output_path, index=False, compression="zstd")
    LOGGER.info("Wrote %s rows to %s.", len(combined_df), output_path)
    return output_path


def _read_parquet_files_in_parallel(
    base_uri: str,
    file_paths: list[str],
    *,
    columns: list[str] | None = None,
    filters: object | None = None,
) -> DataFrame:
    if not file_paths:
        return pd.DataFrame(columns=columns or [])

    file_chunks = [
        file_paths[index : index + READ_FILE_CHUNK_SIZE]
        for index in range(0, len(file_paths), READ_FILE_CHUNK_SIZE)
    ]
    worker_count = min(
        len(file_chunks),
        READ_MAX_WORKERS,
        os.cpu_count() or 1,
    )
    if worker_count <= 1:
        return read_parquet_files(
            base_uri,
            file_paths,
            columns=columns,
            filters=filters,
        )

    LOGGER.info(
        "Reading %s parquet files from %s across %s chunks with %s workers.",
        len(file_paths),
        base_uri,
        len(file_chunks),
        worker_count,
    )
    frames: list[DataFrame] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                read_parquet_files,
                base_uri,
                file_chunk,
                columns=columns,
                filters=filters,
            )
            for file_chunk in file_chunks
        ]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def _merge_frames_in_parallel(
    *,
    weather_df: DataFrame,
    mrt_df: DataFrame,
    cities_df: DataFrame,
    shard_key: ShardKey,
) -> DataFrame:
    if "location_id" in weather_df.columns:
        weather_df = weather_df[
            weather_df["location_id"].isin(cities_df["location_id"])
        ].copy()
        if weather_df.empty:
            LOGGER.warning(
                "Weather shard %s produced no rows for tile %s.",
                shard_key.label,
                shard_key.tile_id,
            )
            return pd.DataFrame()

    weather_chunks = [
        weather_df.iloc[index : index + MERGE_ROW_CHUNK_SIZE].copy()
        for index in range(0, len(weather_df), MERGE_ROW_CHUNK_SIZE)
    ]
    if not weather_chunks:
        return pd.DataFrame()

    worker_count = min(
        len(weather_chunks),
        MERGE_MAX_WORKERS,
        os.cpu_count() or 1,
    )
    combined_frames: list[DataFrame]
    if worker_count <= 1:
        combined_frames = [
            _merge_weather_chunk(chunk, mrt_df=mrt_df, cities_df=cities_df)
            for chunk in weather_chunks
        ]
    else:
        LOGGER.info(
            "Merging shard %s across %s weather chunks with %s workers.",
            shard_key.label,
            len(weather_chunks),
            worker_count,
        )
        combined_frames = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _merge_weather_chunk,
                    chunk,
                    mrt_df=mrt_df,
                    cities_df=cities_df,
                )
                for chunk in weather_chunks
            ]
            for future in as_completed(futures):
                merged_chunk = future.result()
                if not merged_chunk.empty:
                    combined_frames.append(merged_chunk)

    if not combined_frames:
        LOGGER.warning(
            "Combined shard %s produced no matching city rows.",
            shard_key.label,
        )
        return pd.DataFrame()

    return pd.concat(combined_frames, ignore_index=True)


def _merge_weather_chunk(
    weather_chunk: DataFrame,
    *,
    mrt_df: DataFrame,
    cities_df: DataFrame,
) -> DataFrame:
    combined_chunk = weather_chunk.merge(
        mrt_df,
        how="inner",
        on=["timestamp", "lat", "lng"],
        validate="one_to_one",
    )
    if "location_id" not in combined_chunk.columns:
        combined_chunk = combined_chunk.merge(
            cities_df,
            how="inner",
            on=["lat", "lng"],
            validate="many_to_one",
        )
    return combined_chunk


def _discover_weather_city_shards(weather_root: str) -> dict[int, list[str]]:
    filesystem, root_path = resolve_filesystem(weather_root)
    file_infos = filesystem.get_file_info(
        fs_module.FileSelector(root_path, recursive=True),
    )

    shards: dict[int, list[str]] = {}
    for file_info in file_infos:
        if file_info.type != fs_module.FileType.File or not file_info.path.endswith(
            ".parquet",
        ):
            continue
        shard_index = _parse_partition_value(
            root_path,
            file_info.path,
            "city_shard_index",
        )
        if shard_index is None:
            continue
        shards.setdefault(shard_index, []).append(file_info.path)

    for file_paths in shards.values():
        file_paths.sort()
    return shards


def _parse_partition_value(root_path: str, file_path: str, key: str) -> int | None:
    relative_path = file_path.removeprefix(root_path).lstrip("/")
    for part in PurePosixPath(relative_path).parts:
        if not part.startswith(f"{key}="):
            continue
        try:
            return int(part.split("=", maxsplit=1)[1])
        except ValueError:
            return None
    return None


def _load_city_lookup(weather_shard_count: int) -> DataFrame:
    city_to_tile_path = Path(OUTPUT_DIR) / "city_to_tile.csv"
    if not city_to_tile_path.exists():
        msg = f"Missing city to tile mapping: {city_to_tile_path}"
        raise FileNotFoundError(msg)

    city_lookup = (
        pd.read_csv(
            city_to_tile_path,
            usecols=["location_id", "lat", "lng", "tile_id"],
        )
        .sort_values("location_id")
        .reset_index(drop=True)
    )
    if len(city_lookup) % weather_shard_count != 0:
        msg = (
            f"Cannot evenly split {len(city_lookup)} cities across "
            f"{weather_shard_count} weather shards."
        )
        raise ValueError(msg)

    shard_size = len(city_lookup) // weather_shard_count
    city_lookup["city_shard_index"] = city_lookup.index // shard_size
    return city_lookup


def _weather_year_filter(year: int) -> object:
    range_start_date, range_end_date = build_year_date_bounds(year)
    range_start = pd.Timestamp(range_start_date)
    range_end = pd.Timestamp(range_end_date + timedelta(days=1))
    timestamp_field = dataset_module.field("timestamp")
    return (timestamp_field >= range_start) & (timestamp_field < range_end)


def _passthrough_era5_shard(
    shard_key: ShardKey,
    *,
    era5_root: str,
    era5_files: list[str],
    out_dir: str,
) -> None:
    """Copy an era5 parquet directly to the combined output path."""
    era5_df = read_parquet_files(era5_root, era5_files)
    if era5_df.empty:
        msg = f"Shard {shard_key.label}: era5 data is empty."
        raise RuntimeError(msg)

    output_dir = Path(out_dir) / shard_key.partition_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined.parquet"
    float_cols = era5_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        era5_df[float_cols] = era5_df[float_cols].astype("float32")
    era5_df.sort_values(["location_id", "time"]).reset_index(drop=True).to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )
    LOGGER.info("Wrote %s era5 rows to %s.", len(era5_df), output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine matching weather and MRT parquet shards into combined "
            "parquet shards."
        ),
    )
    parser.add_argument("--weather-root", default="weather_data_parquet")
    parser.add_argument("--mrt-root", default="utci_data_parquet")
    parser.add_argument(
        "--era5-root",
        default=None,
        help=(
            "Path to era5_data_parquet root. When era5 shards are found for the "
            "requested year/tile range, they are passed through directly to the "
            "combined output without merging weather and MRT."
        ),
    )
    parser.add_argument("--out-dir", default="combined_data_parquet")
    parser.add_argument("--year", type=int)
    parser.add_argument("--tile-id", dest="tile_ids", action="append", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def _select_requested_shards(
    shard_keys: Iterable[ShardKey],
    args: argparse.Namespace,
) -> list[ShardKey]:
    return select_shards(
        shard_keys,
        year=args.year,
        tile_ids=args.tile_ids,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )


def _process_era5_passthrough(args: argparse.Namespace) -> bool:
    if not args.era5_root:
        return False

    era5_shards = discover_parquet_shards(args.era5_root)
    if not era5_shards:
        return False

    selected_shards = _select_requested_shards(era5_shards.keys(), args)
    if not selected_shards:
        return False

    for shard_key in selected_shards:
        _passthrough_era5_shard(
            shard_key,
            era5_root=args.era5_root,
            era5_files=era5_shards[shard_key],
            out_dir=args.out_dir,
        )
    return True


def _process_standard_weather_shards(
    args: argparse.Namespace,
    cities_df: DataFrame,
    mrt_shards: dict[ShardKey, list[str]],
) -> bool:
    weather_shards = discover_parquet_shards(args.weather_root)
    if not weather_shards:
        return False

    common_shards = discover_common_shards(args.weather_root, args.mrt_root)
    selected_shards = _select_requested_shards(common_shards, args)

    if not selected_shards:
        LOGGER.warning("No matching shards found for the requested filters.")
        return True

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
    return True


def _weather_files_for_tile(
    tile_cities_df: DataFrame,
    weather_city_shards: dict[int, list[str]],
) -> list[str]:
    return sorted(
        {
            file_path
            for shard_index in tile_cities_df["city_shard_index"].unique()
            for file_path in weather_city_shards.get(int(shard_index), [])
        },
    )


def _process_city_sharded_weather(args: argparse.Namespace) -> None:
    mrt_shards = discover_parquet_shards(args.mrt_root)
    weather_city_shards = _discover_weather_city_shards(args.weather_root)
    if not weather_city_shards:
        LOGGER.warning("No weather parquet shards found under %s.", args.weather_root)
        return

    selected_shards = _select_requested_shards(mrt_shards.keys(), args)
    if not selected_shards:
        LOGGER.warning("No MRT shards found for the requested filters.")
        return

    city_lookup = _load_city_lookup(len(weather_city_shards))
    for shard_key in selected_shards:
        tile_cities_df = city_lookup[city_lookup["tile_id"] == shard_key.tile_id][
            ["location_id", "lat", "lng", "city_shard_index"]
        ].copy()
        if tile_cities_df.empty:
            msg = f"No cities found for tile {shard_key.tile_id}."
            raise RuntimeError(msg)

        weather_files = _weather_files_for_tile(tile_cities_df, weather_city_shards)
        if not weather_files:
            msg = f"No weather shard files found for tile {shard_key.tile_id}."
            raise RuntimeError(msg)

        combine_shard(
            shard_key=shard_key,
            weather_root=args.weather_root,
            weather_files=weather_files,
            mrt_root=args.mrt_root,
            mrt_files=mrt_shards[shard_key],
            cities_df=tile_cities_df[["location_id", "lat", "lng"]],
            out_dir=args.out_dir,
            weather_columns=WEATHER_COLUMNS_WITH_LOCATION_ID,
            weather_filters=_weather_year_filter(shard_key.year),
        )


def main() -> None:
    """Combine the selected weather and MRT parquet shards."""
    args = _parse_args()
    cities_df = pd.read_csv(
        Path("cities.csv"),
        usecols=["location_id", "lat", "lng"],
    )

    if _process_era5_passthrough(args):
        return

    mrt_shards = discover_parquet_shards(args.mrt_root)
    if _process_standard_weather_shards(args, cities_df, mrt_shards):
        return

    _process_city_sharded_weather(args)


if __name__ == "__main__":
    main()
