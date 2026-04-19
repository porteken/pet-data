"""Download and shard ERA5 weather data from Google ARCO."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any, TypeAlias, cast

from tqdm.auto import tqdm

from google_era5 import (
    DEFAULT_BATCH_HOURS,
    ERA5_TIME_ORIGIN,
    ERA5_WEATHER_VARIABLES,
    GCS_BATCH_MAX_RETRIES,
    GCS_BATCH_RETRY_DELAY_SECONDS,
    GRID_DEG,
    _configure_concurrency,
    _iter_time_batches,
    _open_zarr_store,
    _resolve_era5_max_workers,
    _select_time_shard_batches,
)
from shards import resolve_filesystem
from shared_config import build_year_date_bounds

DataFrame: TypeAlias = Any

pa = cast("Any", importlib.import_module("pyarrow"))
pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

WEATHER_PARQUET_NAME = "weather.parquet"
_B = 17.625
_C = 243.04


def _normalize_longitude(longitude: float) -> float:
    return longitude % 360.0


def _load_weather_city_shard(city_shard_index: int, city_shard_count: int) -> DataFrame:
    cities_df = pd.read_csv("cities.csv", usecols=["location_id", "lat", "lng"])
    cities_df["lat"] = (pd.to_numeric(cities_df["lat"]) / GRID_DEG).round() * GRID_DEG
    cities_df["lng"] = (pd.to_numeric(cities_df["lng"]) / GRID_DEG).round() * GRID_DEG
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)
    shard_size = (len(cities_df) + city_shard_count - 1) // city_shard_count
    start = city_shard_index * shard_size
    return cities_df.iloc[start : min(start + shard_size, len(cities_df))].copy()


def _datetime_to_hour_index(timestamp: object) -> int:
    dt = pd.Timestamp(timestamp)
    epoch = pd.Timestamp(ERA5_TIME_ORIGIN)
    return int((dt - epoch).total_seconds() // 3600)


def _resolve_explicit_window(
    year: int,
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[int, int] | None:
    if (start_date is None) != (end_date is None):
        msg = "--start-date and --end-date must be provided together."
        raise ValueError(msg)
    if start_date is None or end_date is None:
        return None

    parsed_start = date.fromisoformat(start_date)
    parsed_end = date.fromisoformat(end_date)
    if parsed_start > parsed_end:
        msg = "--start-date must be on or before --end-date."
        raise ValueError(msg)
    if parsed_start.year != year or parsed_end.year != year:
        msg = "Explicit weather date overrides must stay within the requested --year."
        raise ValueError(msg)

    allowed_start, allowed_end = build_year_date_bounds(year)
    if parsed_start < allowed_start or parsed_end > allowed_end:
        msg = (
            "Explicit weather date overrides fall outside the supported pull window "
            f"{allowed_start.isoformat()} to {allowed_end.isoformat()}."
        )
        raise ValueError(msg)

    start_hour = _datetime_to_hour_index(pd.Timestamp(parsed_start))
    end_hour = _datetime_to_hour_index(pd.Timestamp(parsed_end + timedelta(days=1))) - 1
    return start_hour, end_hour


def _clamp_batches_to_window(
    batches: list[tuple[int, int, int]],
    *,
    start_hour: int,
    end_hour: int,
) -> list[tuple[int, int, int]]:
    clamped_batches: list[tuple[int, int, int]] = []
    for batch_index, batch_start, batch_end in batches:
        if batch_end < start_hour or batch_start > end_hour:
            continue
        clamped_batches.append(
            (batch_index, max(batch_start, start_hour), min(batch_end, end_hour)),
        )
    return clamped_batches


def _select_requested_batches(
    *,
    year: int,
    batch_hours: int,
    months: list[int] | None,
    start_date: str | None,
    end_date: str | None,
    time_shard_index: int,
    time_shard_count: int,
) -> list[tuple[int, int, int]]:
    batches = _iter_time_batches(year, batch_hours, months)
    explicit_window = _resolve_explicit_window(
        year,
        start_date=start_date,
        end_date=end_date,
    )
    if explicit_window is not None:
        start_hour, end_hour = explicit_window
        batches = _clamp_batches_to_window(
            batches,
            start_hour=start_hour,
            end_hour=end_hour,
        )
    return _select_time_shard_batches(
        batches,
        time_shard_index,
        time_shard_count,
    )


def _filter_frame_to_months(
    frame: DataFrame,
    allowed_months: list[int] | None,
) -> DataFrame:
    if not allowed_months or frame.empty:
        return frame

    allowed_month_set = set(allowed_months)
    return frame[
        pd.to_datetime(frame["timestamp"]).dt.month.isin(allowed_month_set)
    ].copy()


def _filter_frame_to_window(
    frame: DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> DataFrame:
    if frame.empty or start_date is None or end_date is None:
        return frame

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    timestamps = pd.to_datetime(frame["timestamp"])
    return frame[(timestamps >= start_ts) & (timestamps < end_ts)].copy()


def _compute_weather_frame(
    ds: object,
    selected_cities: DataFrame,
    start_h: int,
    end_h: int,
    compute_workers: int,
) -> DataFrame:
    np = cast("Any", importlib.import_module("numpy"))
    dask = cast("Any", importlib.import_module("dask"))
    xr = cast("Any", importlib.import_module("xarray"))

    lats = selected_cities["lat"].values.astype(float)
    lons = selected_cities["lng"].values.astype(float)
    selection_lons = _normalize_longitude(lons)
    loc_ids = selected_cities["location_id"].values

    city_selection = (
        cast("Any", ds)[ERA5_WEATHER_VARIABLES]
        .sel(time=slice(start_h, end_h))
        .sel(
            latitude=xr.DataArray(lats, dims="location"),
            longitude=xr.DataArray(selection_lons, dims="location"),
            method="nearest",
        )
    )
    with dask.config.set(scheduler="threads", num_workers=compute_workers):
        city_data = city_selection.compute()

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


def _weather_partition_path(
    year: int,
    city_shard_index: int,
    batch_index: int,
) -> str:
    return (
        f"city_shard_index={city_shard_index}/year={year}/batch_index={batch_index:04d}"
    )


def _weather_batch_exists(
    weather_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
) -> bool:
    filesystem, base_path = resolve_filesystem(weather_root)
    path = (
        f"{base_path}/"
        f"{_weather_partition_path(year, city_shard_index, batch_index)}/"
        f"{WEATHER_PARQUET_NAME}"
    )
    try:
        return filesystem.get_file_info(path).type != 0
    except Exception:  # noqa: BLE001
        return False


def _write_weather_partition(
    weather_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    frame: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(weather_root)
    partition_path = _weather_partition_path(year, city_shard_index, batch_index)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/{WEATHER_PARQUET_NAME}"
    table = pa.Table.from_pandas(
        frame.sort_values(["location_id", "timestamp"]).reset_index(drop=True),
        preserve_index=False,
    )
    with filesystem.open_output_stream(output_path) as out_stream:
        pq.write_table(table, out_stream, compression="snappy")


def _process_weather_batch_job(
    *,
    ds: object,
    weather_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    compute_workers: int,
    allowed_months: list[int] | None,
    start_date: str | None,
    end_date: str | None,
) -> int:
    LOGGER.info(
        "Computing Google weather for year %s batch %s over %s cities.",
        year,
        batch_index,
        len(pending_batch_df),
    )
    frame = _compute_weather_frame(
        ds,
        pending_batch_df,
        start_h=start_h,
        end_h=end_h,
        compute_workers=compute_workers,
    )
    frame = _filter_frame_to_months(frame, allowed_months)
    frame = _filter_frame_to_window(
        frame,
        start_date=start_date,
        end_date=end_date,
    )

    if frame.empty:
        LOGGER.info(
            "Weather frame is empty for year %s batch %s after filtering.",
            year,
            batch_index,
        )
        return batch_index
    _write_weather_partition(
        weather_root,
        year,
        city_shard_index,
        batch_index,
        frame,
    )
    return batch_index


def _process_weather_batch_with_thread_dataset(
    weather_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    compute_workers: int,
    allowed_months: list[int] | None,
    start_date: str | None,
    end_date: str | None,
) -> int:
    for attempt in range(1, GCS_BATCH_MAX_RETRIES + 1):
        ds: object | None = None
        try:
            _configure_concurrency(
                os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced"),
            )
            ds = _open_zarr_store()
            return _process_weather_batch_job(
                ds=ds,
                weather_root=weather_root,
                year=year,
                city_shard_index=city_shard_index,
                batch_index=batch_index,
                start_h=start_h,
                end_h=end_h,
                pending_batch_df=pending_batch_df,
                compute_workers=compute_workers,
                allowed_months=allowed_months,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception:
            if attempt == GCS_BATCH_MAX_RETRIES:
                raise
            time.sleep(GCS_BATCH_RETRY_DELAY_SECONDS * attempt)
        finally:
            del ds
            asyncio.set_event_loop(None)
    return batch_index


def _dask_worker_count(concurrency_profile: str) -> int:
    if concurrency_profile == "aggressive":
        return 8
    if concurrency_profile == "conservative":
        return 2
    return 4


def _run_weather_batch_jobs(  # noqa: PLR0913
    *,
    selected_batches: list[tuple[int, int, int]],
    shard_df: DataFrame,
    weather_root: str,
    year: int,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    max_workers: int,
    allowed_months: list[int] | None,
    start_date: str | None,
    end_date: str | None,
    force: bool = False,
) -> bool:
    concurrency_profile = os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced")
    dask_workers = _dask_worker_count(concurrency_profile)

    batch_jobs: list[tuple[int, int, int, DataFrame]] = []
    for batch_index, start_h, end_h in selected_batches:
        pending_df = shard_df.copy()
        if not force and _weather_batch_exists(
            weather_root,
            year,
            city_shard_index,
            batch_index,
        ):
            pending_df = pd.DataFrame(columns=shard_df.columns)
        if not pending_df.empty:
            batch_jobs.append((batch_index, start_h, end_h, pending_df))

    if not batch_jobs:
        return False

    LOGGER.info(
        "Google weather year=%d city_shard=%d/%d time_shard=%d/%d: %d batch(es)",
        year,
        city_shard_index,
        city_shard_count,
        time_shard_index,
        time_shard_count,
        len(batch_jobs),
    )

    worker_count = max(1, min(_resolve_era5_max_workers(max_workers), len(batch_jobs)))

    if worker_count == 1:
        for b_idx, s_h, e_h, df in tqdm(batch_jobs, desc=f"Weather {year}"):
            _process_weather_batch_with_thread_dataset(
                weather_root=weather_root,
                year=year,
                city_shard_index=city_shard_index,
                batch_index=b_idx,
                start_h=s_h,
                end_h=e_h,
                pending_batch_df=df,
                compute_workers=dask_workers,
                allowed_months=allowed_months,
                start_date=start_date,
                end_date=end_date,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_weather_batch_with_thread_dataset,
                    weather_root=weather_root,
                    year=year,
                    city_shard_index=city_shard_index,
                    batch_index=b,
                    start_h=s,
                    end_h=e,
                    pending_batch_df=df,
                    compute_workers=dask_workers,
                    allowed_months=allowed_months,
                    start_date=start_date,
                    end_date=end_date,
                )
                for b, s, e, df in batch_jobs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Weather {year}",
            ):
                future.result()

    return True


def process_weather(
    year: int,
    out_dir: str,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    max_workers: int,
    batch_hours: int,
    concurrency_profile: str,
    months: list[int] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Download Google ARCO weather variables and write hourly parquet shards."""
    _configure_concurrency(concurrency_profile)
    os.environ["ERA5_CONCURRENCY_PROFILE"] = concurrency_profile
    weather_root = f"{out_dir}/weather_data_parquet"

    shard_df = _load_weather_city_shard(city_shard_index, city_shard_count)
    if shard_df.empty:
        return

    selected_batches = _select_requested_batches(
        year=year,
        batch_hours=batch_hours,
        months=months,
        start_date=start_date,
        end_date=end_date,
        time_shard_index=time_shard_index,
        time_shard_count=time_shard_count,
    )
    if not selected_batches:
        LOGGER.warning("No Google weather batches selected for year=%s.", year)
        return

    _run_weather_batch_jobs(
        selected_batches=selected_batches,
        shard_df=shard_df,
        weather_root=weather_root,
        year=year,
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
        time_shard_index=time_shard_index,
        time_shard_count=time_shard_count,
        max_workers=max_workers,
        allowed_months=months,
        start_date=start_date,
        end_date=end_date,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull weather data from Google ARCO.")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        help="Only process specific months (1-12)",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--city-shard-index", type=int, default=0)
    parser.add_argument("--city-shard-count", type=int, default=1)
    parser.add_argument("--time-shard-index", type=int, default=0)
    parser.add_argument("--time-shard-count", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=-1)
    parser.add_argument("--batch-hours", type=int, default=DEFAULT_BATCH_HOURS)
    parser.add_argument(
        "--concurrency-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    exit_code = 0
    try:
        process_weather(
            args.year,
            args.out_dir,
            args.city_shard_index,
            args.city_shard_count,
            args.time_shard_index,
            args.time_shard_count,
            args.max_workers,
            args.batch_hours,
            args.concurrency_profile,
            args.months,
            args.start_date,
            args.end_date,
        )
    except Exception:
        LOGGER.exception("Weather pipeline failed for year=%s", args.year)
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
