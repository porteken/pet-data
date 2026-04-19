"""Pull Google ARCO ERA5 weather data using tile-bounded spatial windows."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

from tqdm.auto import tqdm

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from google_era5 import (
    ERA5_TIME_ORIGIN,
    ERA5_WEATHER_VARIABLES,
    GCS_BATCH_MAX_RETRIES,
    GCS_BATCH_RETRY_DELAY_SECONDS,
    _configure_concurrency,
    _open_zarr_store,
)
from pull_cds_shared import LOGGER, DataFrame, pa, partition_file_max_timestamp, pd, pq
from shards import resolve_filesystem
from shared_config import build_year_date_bounds

WEATHER_THREAD_LOCAL = threading.local()
WEATHER_PARQUET_NAME = "weather.parquet"
_B = 17.625
_C = 243.04

DataFrameLike: TypeAlias = Any


def _normalize_longitude(longitude: float) -> float:
    return longitude % 360.0


def _ensure_tile_outputs() -> None:
    required_paths = [
        Path(OUTPUT_DIR) / "tile_boxes.csv",
        Path(OUTPUT_DIR) / "city_to_tile.csv",
    ]
    if all(path.exists() for path in required_paths):
        return
    LOGGER.info("Tile metadata missing. Regenerating under %s.", OUTPUT_DIR)
    generate_tile_outputs()


def _load_selected_tiles(
    *,
    tile_ids: list[int] | None,
    tile_shard_index: int,
    tile_shard_count: int,
) -> tuple[DataFrame, DataFrame]:
    _ensure_tile_outputs()
    tile_boxes_df = pd.read_csv(Path(OUTPUT_DIR) / "tile_boxes.csv")
    city_to_tile_df = pd.read_csv(
        Path(OUTPUT_DIR) / "city_to_tile.csv",
        usecols=["location_id", "grid_lat", "grid_lon", "tile_id"],
    )

    available_tile_ids = sorted(
        int(tile_id) for tile_id in tile_boxes_df["tile_id"].tolist()
    )
    allowed_tile_ids = (
        available_tile_ids
        if tile_ids is None
        else sorted({int(tile_id) for tile_id in tile_ids})
    )
    selected_tile_ids = [
        tile_id
        for position, tile_id in enumerate(allowed_tile_ids)
        if position % tile_shard_count == tile_shard_index
    ]
    selected_tiles = tile_boxes_df[
        tile_boxes_df["tile_id"].isin(selected_tile_ids)
    ].copy()
    selected_cities = city_to_tile_df[
        city_to_tile_df["tile_id"].isin(selected_tile_ids)
    ].copy()
    return selected_tiles.sort_values("tile_id"), selected_cities.sort_values(
        ["tile_id", "location_id"],
    )


def _datetime_to_hour_index(timestamp: object) -> int:
    dt = pd.Timestamp(timestamp)
    epoch = pd.Timestamp(ERA5_TIME_ORIGIN)
    return int((dt - epoch).total_seconds() // 3600)


def _resolve_weather_window(
    *,
    year: int,
    month: int | None,
    start_date: str | None,
    end_date: str | None,
) -> tuple[int, int, date, date]:
    if (start_date is None) != (end_date is None):
        msg = "--start-date and --end-date must be provided together."
        raise ValueError(msg)

    if start_date is not None and end_date is not None:
        parsed_start = date.fromisoformat(start_date)
        parsed_end = date.fromisoformat(end_date)
        if parsed_start > parsed_end:
            msg = "--start-date must be on or before --end-date."
            raise ValueError(msg)
        allowed_start, allowed_end = build_year_date_bounds(year, month=month)
        if parsed_start < allowed_start or parsed_end > allowed_end:
            msg = (
                "Explicit weather date overrides fall outside the supported pull window "
                f"{allowed_start.isoformat()} to {allowed_end.isoformat()}."
            )
            raise ValueError(msg)
        start_hour = _datetime_to_hour_index(
            datetime.combine(parsed_start, datetime.min.time()),
        )
        end_hour = _datetime_to_hour_index(
            datetime.combine(parsed_end, datetime.max.time()),
        )
        return start_hour, end_hour, parsed_start, parsed_end

    range_start, range_end = build_year_date_bounds(year, month=month)
    start_hour = _datetime_to_hour_index(
        datetime.combine(range_start, datetime.min.time()),
    )
    end_hour = _datetime_to_hour_index(datetime.combine(range_end, datetime.max.time()))
    return start_hour, end_hour, range_start, range_end


def _weather_partition_path(year: int, tile_id: int, *, month: int | None) -> str:
    if month is not None:
        return f"year={year}/month={month:02d}/tile_id={tile_id}"
    return f"year={year}/tile_id={tile_id}"


def _weather_partition_is_current(
    weather_root: str,
    partition_path: str,
    *,
    expected_end_date: date,
) -> bool:
    max_date = partition_file_max_timestamp(
        weather_root,
        partition_path,
        WEATHER_PARQUET_NAME,
    )
    return max_date is not None and max_date >= expected_end_date


def _slice_tile_region(
    ds: object,
    tile_cities: DataFrame,
    *,
    start_h: int,
    end_h: int,
) -> object:
    xr = cast("Any", importlib.import_module("xarray"))
    dataset = cast("Any", ds)[ERA5_WEATHER_VARIABLES].sel(time=slice(start_h, end_h))

    north = float(tile_cities["grid_lat"].max()) + GRID_DEG / 2
    south = float(tile_cities["grid_lat"].min()) - GRID_DEG / 2
    selection_lons = tile_cities["grid_lon"].map(_normalize_longitude)
    east = float(selection_lons.max()) + GRID_DEG / 2
    west = float(selection_lons.min()) - GRID_DEG / 2

    lat_values = cast("Any", dataset.latitude.values)
    latitude_slice = (
        slice(north, south) if lat_values[0] > lat_values[-1] else slice(south, north)
    )
    longitude_slice = slice(west, east)
    return dataset.sel(
        latitude=latitude_slice,
        longitude=longitude_slice,
    ).sel(
        latitude=xr.DataArray(
            tile_cities["grid_lat"].to_numpy(dtype=float),
            dims="location",
        ),
        longitude=xr.DataArray(selection_lons.to_numpy(dtype=float), dims="location"),
        method="nearest",
    )


def _compute_weather_tile_frame(
    ds: object,
    tile_cities: DataFrame,
    *,
    start_h: int,
    end_h: int,
    compute_workers: int,
) -> DataFrame:
    np = cast("Any", importlib.import_module("numpy"))
    dask = cast("Any", importlib.import_module("dask"))

    loc_ids = tile_cities["location_id"].to_numpy()
    lats = tile_cities["grid_lat"].to_numpy(dtype=float)
    lons = tile_cities["grid_lon"].to_numpy(dtype=float)

    tile_selection = _slice_tile_region(
        ds,
        tile_cities,
        start_h=start_h,
        end_h=end_h,
    )
    with dask.config.set(scheduler="threads", num_workers=compute_workers):
        city_data = cast("Any", tile_selection).compute()

    city_data = city_data.assign_coords(
        time=pd.to_datetime(city_data.time.values, unit="h", origin=ERA5_TIME_ORIGIN),
    )

    u10 = city_data["10u"].values.T
    v10 = city_data["10v"].values.T
    t2m = city_data["2t"].values.T
    d2m = city_data["2d"].values.T
    times = city_data.time.values

    n_locs, n_times = len(lats), len(times)
    temp_c = t2m - 273.15
    dew_c = d2m - 273.15
    wind_speed = np.sqrt(u10**2 + v10**2)
    gamma_t = _B * temp_c / (_C + temp_c)
    gamma_td = _B * dew_c / (_C + dew_c)
    rh = np.exp(gamma_td - gamma_t) * 100.0

    return pd.DataFrame(
        {
            "location_id": np.repeat(loc_ids, n_times),
            "lat": np.repeat(lats, n_times),
            "lng": np.repeat(lons, n_times),
            "timestamp": np.tile(times, n_locs),
            "temperature_c": temp_c.ravel(),
            "wind_speed": wind_speed.ravel(),
            "relative_humidity": rh.ravel(),
        },
    ).dropna()


def _write_weather_partition(
    weather_root: str,
    partition_path: str,
    frame: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(weather_root)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/{WEATHER_PARQUET_NAME}"
    table = pa.Table.from_pandas(
        frame.sort_values(["location_id", "timestamp"]).reset_index(drop=True),
        preserve_index=False,
    )
    tmp_output_path = output_path + ".tmp"
    with filesystem.open_output_stream(tmp_output_path) as out_stream:
        pq.write_table(table, out_stream, compression="snappy")
    filesystem.move(tmp_output_path, output_path)


def _dask_worker_count(concurrency_profile: str) -> int:
    if concurrency_profile == "aggressive":
        return 8
    if concurrency_profile == "conservative":
        return 2
    return 4


def _process_tile_job(
    *,
    ds: object,
    tile_id: int,
    tile_cities: DataFrame,
    weather_root: str,
    partition_path: str,
    start_h: int,
    end_h: int,
    compute_workers: int,
) -> None:
    frame = _compute_weather_tile_frame(
        ds,
        tile_cities,
        start_h=start_h,
        end_h=end_h,
        compute_workers=compute_workers,
    )
    if frame.empty:
        msg = f"Weather tile {tile_id} produced no rows."
        raise RuntimeError(msg)
    _write_weather_partition(weather_root, partition_path, frame)


def _process_tile_job_with_fresh_dataset(
    *,
    tile_id: int,
    tile_cities: DataFrame,
    weather_root: str,
    partition_path: str,
    start_h: int,
    end_h: int,
    compute_workers: int,
) -> None:
    ds: object | None = None
    for attempt in range(1, GCS_BATCH_MAX_RETRIES + 1):
        try:
            _configure_concurrency(
                os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced"),
            )
            ds = _open_zarr_store()
            _process_tile_job(
                ds=ds,
                tile_id=tile_id,
                tile_cities=tile_cities,
                weather_root=weather_root,
                partition_path=partition_path,
                start_h=start_h,
                end_h=end_h,
                compute_workers=compute_workers,
            )
        except Exception:  # noqa: PERF203
            if attempt == GCS_BATCH_MAX_RETRIES:
                raise
            time.sleep(GCS_BATCH_RETRY_DELAY_SECONDS * attempt)
        else:
            return
        finally:
            del ds
            asyncio.set_event_loop(None)


def process_weather_tiles(
    *,
    year: int,
    out_dir: str,
    month: int | None,
    start_date: str | None,
    end_date: str | None,
    tile_ids: list[int] | None,
    tile_shard_index: int,
    tile_shard_count: int,
    max_workers: int,
    concurrency_profile: str,
) -> None:
    """Download and process ERA5 weather data for a set of tiles."""
    os.environ["ERA5_CONCURRENCY_PROFILE"] = concurrency_profile
    weather_root = f"{out_dir}/weather_data_parquet"
    compute_workers = _dask_worker_count(concurrency_profile)
    start_h, end_h, _, expected_end_date = _resolve_weather_window(
        year=year,
        month=month,
        start_date=start_date,
        end_date=end_date,
    )
    selected_tiles, selected_cities = _load_selected_tiles(
        tile_ids=tile_ids,
        tile_shard_index=tile_shard_index,
        tile_shard_count=tile_shard_count,
    )
    tile_jobs: list[tuple[int, DataFrame, str]] = []
    for tile in selected_tiles.itertuples(index=False):
        partition_path = _weather_partition_path(year, int(tile.tile_id), month=month)
        if _weather_partition_is_current(
            weather_root,
            partition_path,
            expected_end_date=expected_end_date,
        ):
            LOGGER.info(
                "Weather tile %s already current for %s.",
                tile.tile_id,
                partition_path,
            )
            continue
        tile_cities = selected_cities[selected_cities["tile_id"] == tile.tile_id].copy()
        tile_jobs.append((int(tile.tile_id), tile_cities, partition_path))

    if not tile_jobs:
        LOGGER.info("No weather tiles selected for processing.")
        return

    worker_count = max(1, min(max_workers, len(tile_jobs)))
    LOGGER.info(
        "Processing %s weather tile(s) for year=%s%s with %s worker(s).",
        len(tile_jobs),
        year,
        "" if month is None else f", month={month:02d}",
        worker_count,
    )
    if worker_count == 1:
        for tile_id, tile_cities, partition_path in tqdm(
            tile_jobs,
            desc=f"Weather tiles {year}",
        ):
            _process_tile_job_with_fresh_dataset(
                tile_id=tile_id,
                tile_cities=tile_cities,
                weather_root=weather_root,
                partition_path=partition_path,
                start_h=start_h,
                end_h=end_h,
                compute_workers=compute_workers,
            )
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _process_tile_job_with_fresh_dataset,
                tile_id=tile_id,
                tile_cities=tile_cities,
                weather_root=weather_root,
                partition_path=partition_path,
                start_h=start_h,
                end_h=end_h,
                compute_workers=compute_workers,
            )
            for tile_id, tile_cities, partition_path in tile_jobs
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Weather tiles {year}",
        ):
            future.result()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull Google ARCO weather data by tile.",
    )
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--month", type=int, choices=range(1, 13))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--tile-id", dest="tile_ids", action="append", type=int)
    parser.add_argument("--tile-shard-index", type=int, default=0)
    parser.add_argument("--tile-shard-count", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--concurrency-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="conservative",
    )
    return parser.parse_args()


def main() -> None:
    """Run the tile weather pipeline."""
    args = _parse_args()
    exit_code = 0
    try:
        process_weather_tiles(
            year=args.year,
            out_dir=args.out_dir,
            month=args.month,
            start_date=args.start_date,
            end_date=args.end_date,
            tile_ids=args.tile_ids,
            tile_shard_index=args.tile_shard_index,
            tile_shard_count=args.tile_shard_count,
            max_workers=args.max_workers,
            concurrency_profile=args.concurrency_profile,
        )
    except Exception:  # noqa: BLE001
        LOGGER.exception("Tile weather pipeline failed for year=%s", args.year)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    multiprocessing.freeze_support()
    main()
