"""Download ERA5 weather + MRT data (2000-2025) from the Google ARCO Zarr store."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import importlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias, TypeVar, cast

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from pull_cds_shared import LOGGER, DataFrame, pa, partition_file_exists, pd, pq
from shards import resolve_filesystem

SAFE_STABLE_DATA_MONTH = 4


def _arco_stable_end_year() -> int:
    """Return the latest year whose full data is available in the ARCO store.

    Google ARCO ERA5 publishes stable (final) data on a ~3-month rolling lag.
    ERA5T (near-real-time) has a ~5-day lag but is subject to revision.
    A full year is only safe once its December has cleared the lag
    (i.e., March of the following year). We use April as a safe threshold.
    """
    now = datetime.now(tz=timezone.utc)
    if now.month < SAFE_STABLE_DATA_MONTH:
        return now.year - 2
    return now.year - 1


Dataset: TypeAlias = Any
ArrayLike: TypeAlias = Any
ZarrMetadata: TypeAlias = dict[str, object]


ERA5_ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ERA5_START_YEAR = 2000
ERA5_END_YEAR = (
    _arco_stable_end_year()
)  # Dynamic: all fully-complete years available in ARCO


ERA5_VARIABLE_CANDIDATES: dict[str, list[str]] = {
    "10u": [
        "10u",
        "10m_u_component_of_wind",
    ],
    "10v": [
        "10v",
        "10m_v_component_of_wind",
    ],
    "2t": [
        "2t",
        "2m_temperature",
    ],
    "2d": [
        "2d",
        "2m_dewpoint_temperature",
    ],
    "ssrd": [
        "ssrd",
        "surface_solar_radiation_downwards",
    ],
    "strd": [
        "strd",
        "surface_thermal_radiation_downwards",
    ],
    "ssr": [
        "ssr",
        "surface_net_solar_radiation",
    ],
    "str": [
        "str",
        "surface_net_thermal_radiation",
    ],
    "fdir": [
        "fdir",
        "total_sky_direct_solar_radiation_at_surface",
        "clear_sky_direct_solar_radiation_at_surface",
    ],
}

ERA5_WEATHER_VARIABLES = ["10u", "10v", "2t", "2d"]
ERA5_RADIATION_VARIABLES = ["ssrd", "strd", "ssr", "str", "fdir"]
ERA5_ALL_ARCO_VARIABLES = [*ERA5_WEATHER_VARIABLES, *ERA5_RADIATION_VARIABLES]


RADIATION_SCALE = 1.0 / 3600.0

_B = 17.625
_C = 243.04

EXPECTED_LOCATION_COUNT = 500
THREE_DIMENSIONAL_ARRAY_NDIMS = 3
ERA5_TIME_ORIGIN = "1959-01-01"
DEFAULT_BATCH_HOURS = 24 * 30  # 720 hours (~1 month)
ERA5_THREAD_LOCAL = threading.local()

GCS_BATCH_MAX_RETRIES = 3
GCS_BATCH_RETRY_DELAY_SECONDS = 10
OPEN_GCS_FILESYSTEMS: list[Any] = []

TrackedFilesystemT = TypeVar("TrackedFilesystemT")


def _track_gcs_filesystem(fs: TrackedFilesystemT) -> TrackedFilesystemT:
    """Track a gcsfs filesystem so its session can be closed at shutdown."""
    OPEN_GCS_FILESYSTEMS.append(fs)
    return fs


def _coerce_int_tuple(raw_value: object) -> tuple[int, ...] | None:
    """Return an integer tuple when the metadata value is a list of integers."""
    if not isinstance(raw_value, list):
        return None
    raw_dims = cast("list[object]", raw_value)
    if not all(isinstance(dim, int) for dim in raw_dims):
        return None
    return tuple(cast("list[int]", raw_dims))


def _load_arco_store_metadata() -> ZarrMetadata:
    """Return consolidated Zarr metadata for the public ARCO ERA5 store."""
    gcsfs = cast("Any", importlib.import_module("gcsfs"))
    fs = _track_gcs_filesystem(
        gcsfs.GCSFileSystem(token="anon", skip_instance_cache=True),  # noqa: S106
    )
    metadata_path = ERA5_ARCO_STORE.removeprefix("gs://") + "/.zmetadata"
    with fs.open(metadata_path, "r") as metadata_file:
        return cast("ZarrMetadata", json.load(metadata_file))


def _problematic_arco_chunks() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return any variables chunked as one full global field per hour."""
    metadata = _load_arco_store_metadata()
    metadata_entries = cast("ZarrMetadata", metadata.get("metadata", {}))

    problematic_chunks: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for canonical_name, candidates in ERA5_VARIABLE_CANDIDATES.items():
        for candidate in candidates:
            zarray_key = f"{candidate}/.zarray"
            raw_entry = metadata_entries.get(zarray_key)
            if not isinstance(raw_entry, dict):
                continue

            entry = cast("dict[str, object]", raw_entry)
            chunks = _coerce_int_tuple(entry.get("chunks"))
            shape = _coerce_int_tuple(entry.get("shape"))
            if chunks is None or shape is None:
                continue

            if (
                len(chunks) == THREE_DIMENSIONAL_ARRAY_NDIMS
                and len(shape) == THREE_DIMENSIONAL_ARRAY_NDIMS
                and chunks[0] == 1
                and chunks[1:] == shape[1:]
            ):
                problematic_chunks[canonical_name] = (chunks, shape)
            break

    return problematic_chunks


def _warn_if_full_globe_chunks() -> None:
    """Warn when the configured ARCO store requires hourly full-globe reads."""
    problematic_chunks = _problematic_arco_chunks()
    if not problematic_chunks:
        return

    formatted_chunks = ", ".join(
        f"{name}: chunks={chunks} shape={shape}"
        for name, (chunks, shape) in sorted(problematic_chunks.items())
    )
    LOGGER.warning(
        "Google ARCO ERA5 variables are chunked as one full global "
        "latitude/longitude field per hour (%s). This script now reads "
        "resumable time batches for each requested shard and splits each "
        "in-memory result by tile to avoid repeated full-globe fetches, but "
        "each batch read can still be slow.",
        formatted_chunks,
    )


def _resolve_store_variables(array_names: set[str]) -> dict[str, str]:
    """Map canonical ERA5 variable keys to actual Zarr array names."""
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical_name, candidates in ERA5_VARIABLE_CANDIDATES.items():
        actual_name = next((name for name in candidates if name in array_names), None)
        if actual_name is None:
            missing.append(canonical_name)
            continue
        resolved[canonical_name] = actual_name

    if missing:
        available_sample = sorted(array_names)[:80]
        msg = (
            "Required ARCO ERA5 variables could not be resolved: "
            f"{missing}. Variable candidates: "
            f"{ {name: ERA5_VARIABLE_CANDIDATES[name] for name in missing} }. "
            f"Sample available arrays: {available_sample}"
        )
        raise KeyError(msg)

    return resolved


def _open_zarr_store() -> Dataset:
    """Open a minimal dask-backed Dataset for only the required ERA5 arrays."""
    gcsfs = cast("Any", importlib.import_module("gcsfs"))
    xr = cast("Any", importlib.import_module("xarray"))
    dask_array = cast("Any", importlib.import_module("dask.array"))
    zarr = cast("Any", importlib.import_module("zarr"))

    # Create GCS filesystem with tuned connection/block settings.
    # skip_instance_cache=True ensures each call gets a fresh GCSFileSystem
    # bound to the current event loop, preventing the shutdown RuntimeError
    # where gcsfs tries to close an aiohttp session on a different loop.
    fs = _track_gcs_filesystem(
        gcsfs.GCSFileSystem(
            token="anon",  # noqa: S106
            default_block_size=8 * 1024 * 1024,  # 8MB blocks
            skip_instance_cache=True,
        ),
    )
    root = zarr.open_group(fs.get_mapper(ERA5_ARCO_STORE), mode="r")
    resolved_names = _resolve_store_variables(set(root.array_keys()))
    return xr.Dataset(
        data_vars={
            canonical_name: (
                ("time", "latitude", "longitude"),
                dask_array.from_zarr(root[actual_name]),
            )
            for canonical_name, actual_name in resolved_names.items()
        },
        coords={
            "time": root["time"][:],
            "latitude": root["latitude"][:],
            "longitude": root["longitude"][:],
        },
    )


def _configure_concurrency(profile: str) -> None:
    """Tune Zarr and Dask concurrency based on the selected profile."""
    zarr = cast("Any", importlib.import_module("zarr"))
    dask = cast("Any", importlib.import_module("dask"))

    configs = {
        "conservative": {"zarr_concurrency": 16, "dask_workers": 2},
        "balanced": {"zarr_concurrency": 64, "dask_workers": 4},
        "aggressive": {"zarr_concurrency": 128, "dask_workers": 8},
    }
    profile_cfg = configs.get(profile, configs["balanced"])
    zarr_io_limit: int | str = profile_cfg["zarr_concurrency"]

    # Zarr async concurrency (default is 10, only available in V3+)
    if hasattr(zarr, "config"):
        zarr.config.set({"async.concurrency": profile_cfg["zarr_concurrency"]})
    else:
        zarr_io_limit = "default (v2)"

    # Dask global settings
    dask.config.set(
        {
            "array.slicing.split_large_chunks": False,
        },
    )

    LOGGER.info(
        "Concurrency profile %r: Zarr I/O limit=%s, Dask workers/batch=%s",
        profile,
        zarr_io_limit,
        profile_cfg["dask_workers"],
    )


def _resolve_era5_max_workers(requested_max_workers: int) -> int:
    """Resolve the requested ERA5 worker count into a concrete thread count."""
    if requested_max_workers == -1:
        return max(1, os.cpu_count() or 1)
    if requested_max_workers < 1:
        msg = "max_workers must be >= 1 or exactly -1 to use all available cores"
        raise ValueError(msg)
    return requested_max_workers


def _year_time_slice(year: int) -> tuple[int, int]:
    """Return (start_h, end_h_inclusive) as integer hours since 1959-01-01."""
    pd_module = cast("Any", importlib.import_module("pandas"))
    epoch = pd_module.Timestamp(ERA5_TIME_ORIGIN)

    start_h: int = int(
        (pd_module.Timestamp(f"{year}-01-01") - epoch).total_seconds() // 3600,
    )
    end_h: int = (
        int((pd_module.Timestamp(f"{year + 1}-01-01") - epoch).total_seconds() // 3600)
        - 1
    )
    return start_h, end_h


def _iter_time_batches(
    year: int,
    *,
    batch_hours: int,
) -> list[tuple[int, int, int]]:
    """Return deterministic (batch_index, start_h, end_h) windows for one year."""
    if batch_hours < 1:
        msg = "batch_hours must be >= 1"
        raise ValueError(msg)

    start_h, end_h = _year_time_slice(year)
    batches: list[tuple[int, int, int]] = []
    batch_index = 0
    batch_start = start_h
    while batch_start <= end_h:
        batch_end = min(batch_start + batch_hours - 1, end_h)
        batches.append((batch_index, batch_start, batch_end))
        batch_index += 1
        batch_start = batch_end + 1
    return batches


def _select_time_shard_batches(
    batches: list[tuple[int, int, int]],
    *,
    time_shard_index: int,
    time_shard_count: int,
) -> list[tuple[int, int, int]]:
    """Return only the batches assigned to one time shard."""
    if time_shard_count < 1:
        msg = "time_shard_count must be >= 1"
        raise ValueError(msg)
    if time_shard_index < 0 or time_shard_index >= time_shard_count:
        msg = f"time_shard_index must be between 0 and {time_shard_count - 1}"
        raise ValueError(msg)

    return [
        batch for batch in batches if batch[0] % time_shard_count == time_shard_index
    ]


def _ensure_tile_outputs() -> None:
    required = [
        Path(OUTPUT_DIR) / "city_to_tile.csv",
        Path(OUTPUT_DIR) / "unique_grid_cells.csv",
    ]
    if all(p.exists() for p in required):
        return

    LOGGER.info("Tile metadata missing. Regenerating under %s.", OUTPUT_DIR)
    generate_tile_outputs()


def _load_era5_city_shard(
    *,
    city_shard_index: int,
    city_shard_count: int,
    expected_location_count: int | None = EXPECTED_LOCATION_COUNT,
) -> DataFrame:
    """Return the cities and their tile_ids for the requested shard."""
    if city_shard_count < 1:
        msg = "city_shard_count must be >= 1"
        raise ValueError(msg)

    if city_shard_index < 0 or city_shard_index >= city_shard_count:
        msg = f"city_shard_index must be between 0 and {city_shard_count - 1}"
        raise ValueError(msg)

    cities_path = Path("cities.csv")
    if not cities_path.exists():
        msg = "cities.csv is required before pulling ERA5 data."
        raise FileNotFoundError(msg)

    _ensure_tile_outputs()

    cities_df = pd.read_csv(cities_path, usecols=["location_id", "lat", "lng"])
    city_tile_df = pd.read_csv(
        Path(OUTPUT_DIR) / "city_to_tile.csv",
        usecols=["location_id", "tile_id"],
    )

    cities_df["lat"] = (pd.to_numeric(cities_df["lat"]) / GRID_DEG).round() * GRID_DEG
    cities_df["lng"] = (pd.to_numeric(cities_df["lng"]) / GRID_DEG).round() * GRID_DEG

    merged = (
        cities_df.merge(city_tile_df, on="location_id", how="inner")
        .sort_values("location_id")
        .reset_index(drop=True)
    )

    if expected_location_count is not None and len(merged) != expected_location_count:
        msg = (
            f"Expected {expected_location_count} locations for ERA5, "
            f"found {len(merged)} after city/tile merge."
        )
        raise ValueError(msg)

    if len(merged) % city_shard_count != 0:
        msg = (
            f"Cannot evenly split {len(merged)} cities across "
            f"{city_shard_count} ERA5 shards."
        )
        raise ValueError(msg)

    shard_size = len(merged) // city_shard_count
    start = city_shard_index * shard_size
    shard = merged.iloc[start : start + shard_size].copy()

    LOGGER.info(
        "ERA5 shard %s/%s selected location_ids %s-%s (%s cities, %s tiles).",
        city_shard_index + 1,
        city_shard_count,
        int(shard["location_id"].min()),
        int(shard["location_id"].max()),
        len(shard),
        shard["tile_id"].nunique(),
    )
    return shard


def _era5_partition_path(
    year: int,
    tile_id: int,
    *,
    batch_index: int | None = None,
) -> str:
    base_path = f"year={year}/tile_id={tile_id}"
    if batch_index is None:
        return base_path
    return f"{base_path}/batch_index={batch_index:04d}"


def _era5_batch_exists(
    era5_root: str,
    year: int,
    tile_id: int,
    *,
    batch_index: int,
) -> bool:
    return partition_file_exists(
        era5_root,
        _era5_partition_path(year, tile_id, batch_index=batch_index),
        "era5.parquet",
    )


def _calc_cossza(
    *,
    lat: float,
    lon: float,
    times: ArrayLike,
) -> ArrayLike:
    """Return hourly cosine solar zenith angle for one city/time series."""
    np = cast("Any", importlib.import_module("numpy"))
    timestamps = pd.to_datetime(times)

    day_of_year = timestamps.dayofyear.to_numpy(dtype="float64")
    hour = (
        timestamps.hour.to_numpy(dtype="float64")
        + timestamps.minute.to_numpy(dtype="float64") / 60.0
        + timestamps.second.to_numpy(dtype="float64") / 3600.0
    )

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma)
        + 0.00148 * np.sin(3.0 * gamma)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )
    true_solar_time_minutes = hour * 60.0 + equation_of_time + 4.0 * lon
    hour_angle = np.deg2rad(true_solar_time_minutes / 4.0 - 180.0)
    lat_radians = np.deg2rad(lat)

    cossza = np.sin(lat_radians) * np.sin(decl) + np.cos(lat_radians) * np.cos(
        decl,
    ) * np.cos(hour_angle)
    return np.clip(cossza, 0.0, 1.0)


def _compute_location_frame(
    ds: Dataset,
    selected_cities: DataFrame,
    *,
    start_h: int,
    end_h: int,
    compute_workers: int,
) -> DataFrame:
    """Fetch one batch of ERA5 data for a set of city points.

    Return a combined DataFrame with weather variables and MRT already in the
    final combined schema.
    """
    np = cast("Any", importlib.import_module("numpy"))
    dask = cast("Any", importlib.import_module("dask"))
    xr = cast("Any", importlib.import_module("xarray"))
    tf = cast("Any", importlib.import_module("thermofeel"))

    lats = selected_cities["lat"].values.astype(float)
    lons = selected_cities["lng"].values.astype(float)
    location_ids = selected_cities["location_id"].values
    tile_ids = selected_cities["tile_id"].values

    lats_da = xr.DataArray(lats, dims="location")
    lons_da = xr.DataArray(lons, dims="location")

    year_slice: Any = ds[ERA5_ALL_ARCO_VARIABLES].sel(time=slice(start_h, end_h))
    city_selection: Any = year_slice.sel(
        latitude=lats_da,
        longitude=lons_da,
        method="nearest",
    )
    with dask.config.set(scheduler="threads", num_workers=compute_workers):
        fetch_start = time.time()
        city_data: Any = city_selection.compute()
        fetch_elapsed = time.time() - fetch_start

    n_hours = len(city_data.time)
    n_chunks = n_hours * len(ERA5_ALL_ARCO_VARIABLES)
    LOGGER.info(
        "Fetched %s chunks (%s hours x %s vars) in %.1f seconds (%.1f chunks/sec).",
        n_chunks,
        n_hours,
        len(ERA5_ALL_ARCO_VARIABLES),
        fetch_elapsed,
        n_chunks / max(fetch_elapsed, 0.001),
    )

    n_time_steps: int = len(city_data.time)
    pd_module = cast("Any", importlib.import_module("pandas"))
    proper_times: Any = pd_module.to_datetime(
        city_data.time.values,
        unit="h",
        origin=ERA5_TIME_ORIGIN,
    )
    city_data = city_data.assign_coords(time=proper_times)

    u10: Any = city_data["10u"].values
    v10: Any = city_data["10v"].values
    t2m: Any = city_data["2t"].values
    d2m: Any = city_data["2d"].values

    ssrd: Any = city_data["ssrd"].values * RADIATION_SCALE
    strd: Any = city_data["strd"].values * RADIATION_SCALE
    ssr: Any = city_data["ssr"].values * RADIATION_SCALE
    strr: Any = city_data["str"].values * RADIATION_SCALE
    fdir: Any = city_data["fdir"].values * RADIATION_SCALE

    times: ArrayLike = city_data.time.values
    n_locations = len(lats)

    cossza: ArrayLike = np.empty((n_locations, n_time_steps), dtype="float64")
    for i, (lat, lon) in enumerate(zip(lats, lons, strict=False)):
        cossza[i, :] = _calc_cossza(lat=float(lat), lon=float(lon), times=times)

    wind_speed: Any = np.sqrt(u10**2 + v10**2)
    temperature_c: Any = t2m - 273.15
    dewpoint_c: Any = d2m - 273.15
    gamma_t: Any = _B * temperature_c / (_C + temperature_c)
    gamma_td: Any = _B * dewpoint_c / (_C + dewpoint_c)
    relative_humidity: Any = np.exp(gamma_td - gamma_t) * 100.0

    try:
        mrt_k: Any = tf.calculate_mean_radiant_temperature(
            ssrd=ssrd.ravel(),
            ssr=ssr.ravel(),
            dsrp=fdir.ravel(),
            fdir=fdir.ravel(),
            strd=strd.ravel(),
            strr=strr.ravel(),
            cossza=cossza.ravel(),
        )
    except TypeError:
        mrt_k = tf.calculate_mean_radiant_temperature(
            ssrd=ssrd.ravel(),
            fdir=fdir.ravel(),
            strd=strd.ravel(),
            cossza=cossza.ravel(),
        )

    mrt_c: Any = (mrt_k - 273.15).reshape(n_locations, n_time_steps)

    times_tiled: Any = np.tile(times, n_locations)
    loc_ids_rep: Any = np.repeat(location_ids, n_time_steps)
    tile_ids_rep: Any = np.repeat(tile_ids, n_time_steps)
    lats_rep: Any = np.repeat(lats, n_time_steps)
    lons_rep: Any = np.repeat(lons, n_time_steps)

    frame = pd.DataFrame(
        {
            "location_id": loc_ids_rep,
            "tile_id": tile_ids_rep,
            "lat": lats_rep,
            "lng": lons_rep,
            "time": times_tiled,
            "t": temperature_c.ravel(),
            "v": wind_speed.ravel(),
            "rh": relative_humidity.ravel(),
            "mrt": mrt_c.ravel(),
        },
    )
    return (
        frame.dropna(subset=["t", "v", "rh", "mrt"])
        .sort_values(["tile_id", "location_id", "time"])
        .reset_index(drop=True)
    )


def _write_era5_partition(
    era5_root: str,
    year: int,
    tile_id: int,
    frame: DataFrame,
    *,
    batch_index: int,
) -> None:
    filesystem, base_path = resolve_filesystem(era5_root)
    partition_dir = (
        f"{base_path}/{_era5_partition_path(year, tile_id, batch_index=batch_index)}"
    )
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/era5.parquet"

    float_cols = frame.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        frame[float_cols] = frame[float_cols].astype("float32")
    table: Any = pa.Table.from_pandas(frame, preserve_index=False)
    with filesystem.open_output_stream(output_path) as out_stream:
        pq.write_table(table, out_stream, compression="ZSTD")


def _pending_batch_tiles(
    shard_df: DataFrame,
    *,
    era5_root: str,
    year: int,
    batch_index: int,
    force: bool = False,
) -> DataFrame:
    """Return only the cities for tiles that still need this batch written."""
    pending_frames: list[DataFrame] = []
    tile_ids = sorted(int(raw_tile_id) for raw_tile_id in shard_df["tile_id"].unique())
    for tile_id in tile_ids:
        tile_cities = shard_df[shard_df["tile_id"] == tile_id].copy()
        if not force and _era5_batch_exists(
            era5_root,
            year,
            tile_id,
            batch_index=batch_index,
        ):
            LOGGER.info(
                "ERA5 tile %s year %s batch %s already exists. Skipping.",
                tile_id,
                year,
                batch_index,
            )
            continue

        LOGGER.info(
            "Queueing ERA5 tile %s year %s batch %s (%s cities).",
            tile_id,
            year,
            batch_index,
            len(tile_cities),
        )
        pending_frames.append(tile_cities)

    if not pending_frames:
        return shard_df.iloc[0:0].copy()
    return pd.concat(pending_frames, ignore_index=True)


def _write_shard_tiles(
    *,
    era5_root: str,
    year: int,
    frame: DataFrame,
    batch_index: int,
) -> None:
    """Split a shard-wide frame by tile and write each partition."""
    if frame.empty:
        LOGGER.warning(
            "No ERA5 rows returned for year %s batch %s.",
            year,
            batch_index,
        )
        return

    for tile_id, tile_frame in frame.groupby("tile_id", sort=True):
        output_frame = tile_frame.drop(columns=["tile_id"]).reset_index(drop=True)
        _write_era5_partition(
            era5_root,
            year,
            int(tile_id),
            output_frame,
            batch_index=batch_index,
        )
        LOGGER.info(
            "Wrote %s ERA5 rows for tile %s year %s batch %s.",
            len(output_frame),
            int(tile_id),
            year,
            batch_index,
        )


def _get_thread_dataset() -> Dataset:
    """Return a thread-local ERA5 dataset handle."""
    ds = getattr(ERA5_THREAD_LOCAL, "ds", None)
    if ds is None:
        _configure_concurrency(os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced"))
        ds = _open_zarr_store()
        ERA5_THREAD_LOCAL.ds = ds
    return ds


def _process_era5_batch_job(
    *,
    ds: Dataset,
    era5_root: str,
    year: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    compute_workers: int,
) -> int:
    """Compute and write one ERA5 batch."""
    batch_start = pd.to_datetime(start_h, unit="h", origin=ERA5_TIME_ORIGIN)
    batch_end = pd.to_datetime(end_h, unit="h", origin=ERA5_TIME_ORIGIN)
    LOGGER.info(
        "Reading ERA5 city shard %s/%s time shard %s/%s year %s batch %s "
        "covering %s to %s across %s cities in %s pending tiles.",
        city_shard_index + 1,
        city_shard_count,
        time_shard_index + 1,
        time_shard_count,
        year,
        batch_index,
        batch_start.isoformat(),
        batch_end.isoformat(),
        len(pending_batch_df),
        pending_batch_df["tile_id"].nunique(),
    )
    frame = _compute_location_frame(
        ds,
        pending_batch_df,
        start_h=start_h,
        end_h=end_h,
        compute_workers=compute_workers,
    )
    _write_shard_tiles(
        era5_root=era5_root,
        year=year,
        frame=frame,
        batch_index=batch_index,
    )
    return batch_index


def _process_era5_batch_with_thread_dataset(
    *,
    era5_root: str,
    year: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    compute_workers: int,
) -> int:
    """Process one ERA5 batch using a thread-local dataset."""
    for attempt in range(1, GCS_BATCH_MAX_RETRIES + 1):
        try:
            return _process_era5_batch_job(
                ds=_get_thread_dataset(),
                era5_root=era5_root,
                year=year,
                batch_index=batch_index,
                start_h=start_h,
                end_h=end_h,
                pending_batch_df=pending_batch_df,
                city_shard_index=city_shard_index,
                city_shard_count=city_shard_count,
                time_shard_index=time_shard_index,
                time_shard_count=time_shard_count,
                compute_workers=compute_workers,
            )
        except Exception as exc:  # noqa: PERF203
            if attempt == GCS_BATCH_MAX_RETRIES:
                raise
            delay = GCS_BATCH_RETRY_DELAY_SECONDS * attempt
            LOGGER.warning(
                "ERA5 batch %s attempt %s/%s failed: %s. Retrying in %s seconds.",
                batch_index,
                attempt,
                GCS_BATCH_MAX_RETRIES,
                exc,
                delay,
            )
            time.sleep(delay)

    return batch_index


def _run_era5_batch_jobs(
    *,
    selected_batches: list[tuple[int, int, int]],
    shard_df: DataFrame,
    era5_root: str,
    year: int,
    city_shard_index: int,
    city_shard_count: int,
    time_shard_index: int,
    time_shard_count: int,
    max_workers: int,
    force: bool = False,
) -> bool:
    """Run the pending ERA5 batch jobs, using worker threads when beneficial."""
    batch_jobs: list[tuple[int, int, int, DataFrame]] = []
    for batch_index, start_h, end_h in selected_batches:
        pending_batch_df = _pending_batch_tiles(
            shard_df,
            era5_root=era5_root,
            year=year,
            batch_index=batch_index,
            force=force,
        )
        if pending_batch_df.empty:
            continue
        batch_jobs.append((batch_index, start_h, end_h, pending_batch_df))

    if not batch_jobs:
        return False

    resolved_max_workers = _resolve_era5_max_workers(max_workers)

    # Map profile to dask internal workers
    # aggressive = 8, balanced = 4, conservative = 2
    # This is passed into _compute_location_frame
    dask_workers = 4
    if os.environ.get("ERA5_CONCURRENCY_PROFILE") == "aggressive":
        dask_workers = 8
    elif os.environ.get("ERA5_CONCURRENCY_PROFILE") == "conservative":
        dask_workers = 2

    worker_count = max(
        1,
        min(
            resolved_max_workers,
            len(batch_jobs),
            os.cpu_count() or 1,
        ),
    )
    if worker_count == 1:
        ds = _open_zarr_store()
        for batch_index, start_h, end_h, pending_batch_df in batch_jobs:
            _process_era5_batch_job(
                ds=ds,
                era5_root=era5_root,
                year=year,
                batch_index=batch_index,
                start_h=start_h,
                end_h=end_h,
                pending_batch_df=pending_batch_df,
                city_shard_index=city_shard_index,
                city_shard_count=city_shard_count,
                time_shard_index=time_shard_index,
                time_shard_count=time_shard_count,
                compute_workers=dask_workers,
            )
        return True

    LOGGER.info(
        "Processing %s ERA5 batches with %s worker threads (%s Dask threads/batch).",
        len(batch_jobs),
        worker_count,
        dask_workers,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_batch_index = {
            executor.submit(
                _process_era5_batch_with_thread_dataset,
                era5_root=era5_root,
                year=year,
                batch_index=batch_index,
                start_h=start_h,
                end_h=end_h,
                pending_batch_df=pending_batch_df,
                city_shard_index=city_shard_index,
                city_shard_count=city_shard_count,
                time_shard_index=time_shard_index,
                time_shard_count=time_shard_count,
                compute_workers=dask_workers,
            ): batch_index
            for batch_index, start_h, end_h, pending_batch_df in batch_jobs
        }
        for future in as_completed(future_to_batch_index):
            batch_index = future_to_batch_index[future]
            future.result()
            LOGGER.info("ERA5 batch %s completed for %s.", batch_index, year)
    return True


def process_era5(
    year: int,
    out_dir: str,
    *,
    city_shard_index: int = 0,
    city_shard_count: int = 1,
    time_shard_index: int = 0,
    time_shard_count: int = 1,
    max_workers: int = -1,
    batch_hours: int = DEFAULT_BATCH_HOURS,
    concurrency_profile: str = "aggressive",
    expected_location_count: int | None = EXPECTED_LOCATION_COUNT,
    force: bool = False,
) -> None:
    """Download ERA5 weather + MRT for one year and one city shard."""
    _configure_concurrency(concurrency_profile)
    os.environ["ERA5_CONCURRENCY_PROFILE"] = concurrency_profile

    if year < ERA5_START_YEAR or year > ERA5_END_YEAR:
        msg = (
            f"ERA5 ARCO store covers {ERA5_START_YEAR}-{ERA5_END_YEAR} "
            f"(dynamically computed from today's date minus the stable-data lag). "
            f"Requested year {year} is outside this range."
        )
        raise ValueError(msg)

    era5_root = f"{out_dir}/era5_data_parquet"

    shard_df = _load_era5_city_shard(
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
        expected_location_count=expected_location_count,
    )
    if shard_df.empty:
        LOGGER.warning("No ERA5 cities selected for shard %s.", city_shard_index)
        return

    _warn_if_full_globe_chunks()

    selected_batches = _select_time_shard_batches(
        _iter_time_batches(
            year,
            batch_hours=batch_hours,
        ),
        time_shard_index=time_shard_index,
        time_shard_count=time_shard_count,
    )
    LOGGER.info(
        "ERA5 year %s assigned %s batch(es) to time shard %s/%s.",
        year,
        len(selected_batches),
        time_shard_index + 1,
        time_shard_count,
    )

    wrote_any_batches = _run_era5_batch_jobs(
        selected_batches=selected_batches,
        shard_df=shard_df,
        era5_root=era5_root,
        year=year,
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
        time_shard_index=time_shard_index,
        time_shard_count=time_shard_count,
        max_workers=max_workers,
        force=force,
    )

    if not wrote_any_batches:
        LOGGER.info(
            "ERA5 processing complete for year=%s city shard %s/%s time shard "
            "%s/%s with no pending batches.",
            year,
            city_shard_index + 1,
            city_shard_count,
            time_shard_index + 1,
            time_shard_count,
        )
        return

    LOGGER.info(
        "ERA5 processing complete for year=%s city shard %s/%s time shard %s/%s.",
        year,
        city_shard_index + 1,
        city_shard_count,
        time_shard_index + 1,
        time_shard_count,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull ERA5 weather + MRT data from Google ARCO Zarr store.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help=f"Year to download ({ERA5_START_YEAR}-{ERA5_END_YEAR}; upper bound is dynamic).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output base path or S3 URI (e.g., s3://my-bucket/data).",
    )
    parser.add_argument(
        "--city-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for splitting cities across jobs.",
    )
    parser.add_argument(
        "--city-shard-count",
        type=int,
        default=1,
        help="Total number of city shards.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=-1,
        help="Maximum ERA5 batch worker threads; use -1 for all available cores.",
    )
    parser.add_argument(
        "--time-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for splitting ERA5 time batches across jobs.",
    )
    parser.add_argument(
        "--time-shard-count",
        type=int,
        default=1,
        help="Total number of ERA5 time shards.",
    )
    parser.add_argument(
        "--batch-hours",
        type=int,
        default=DEFAULT_BATCH_HOURS,
        help=f"Hours to fetch per resumable batch (default: {DEFAULT_BATCH_HOURS}).",
    )
    parser.add_argument(
        "--concurrency-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="aggressive",
        help=(
            "I/O concurrency profile. 'aggressive' maximizes GCS throughput "
            "(128 concurrent chunk reads, 8 dask workers). 'balanced' uses "
            "moderate settings (64/4). 'conservative' uses minimal settings (16/2)."
        ),
    )
    parser.add_argument(
        "--expected-location-count",
        type=int,
        default=EXPECTED_LOCATION_COUNT,
        help=(
            f"Expected number of locations after city/tile merge "
            f"(default: {EXPECTED_LOCATION_COUNT}). Use 0 to disable the check."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing partitions even if they already exist on disk.",
    )
    return parser.parse_args()


def _close_gcsfs_sessions() -> None:
    """Close all open gcsfs sessions before interpreter shutdown.

    Python's weakref finalizers close gcsfs aiohttp sessions, but they run
    after the main event loop has been torn down, causing a RuntimeError
    ("Future attached to a different loop").  By closing all cached
    GCSFileSystem instances explicitly on a fresh loop here — before the
    weakref finalizers fire — we prevent that race condition.
    """
    try:
        gcsfs_mod = importlib.import_module("gcsfs")
    except ImportError:
        return

    # Collect all live GCSFileSystem instances from the instance cache.
    instances = list(OPEN_GCS_FILESYSTEMS)
    cache: dict[Any, Any] | None = getattr(gcsfs_mod.GCSFileSystem, "_cache", None)
    if cache:
        instances.extend(cache.values())
    # Also handle the newer fsspec AbstractFileSystem._cache structure.
    fsspec_cache: dict[Any, Any] | None = None
    with contextlib.suppress(ImportError):
        fsspec_mod = importlib.import_module("fsspec")
        fsspec_cache = getattr(fsspec_mod.AbstractFileSystem, "_cache", None)
    if fsspec_cache:
        instances.extend(
            v for v in fsspec_cache.values() if isinstance(v, gcsfs_mod.GCSFileSystem)
        )

    if not instances:
        return

    seen_instance_ids: set[int] = set()
    for inst in instances:
        if id(inst) in seen_instance_ids:
            continue
        seen_instance_ids.add(id(inst))

        session = getattr(inst, "_session", None)
        if session is None:
            continue

        connector = getattr(session, "_connector", None)
        with contextlib.suppress(Exception):
            if connector is not None:
                connector._close()  # type: ignore[attr-defined]  # noqa: SLF001
        with contextlib.suppress(Exception):
            session._connector = None  # type: ignore[attr-defined]  # noqa: SLF001
        with contextlib.suppress(Exception):
            inst._session = None  # type: ignore[attr-defined]  # noqa: SLF001

    OPEN_GCS_FILESYSTEMS.clear()


def main() -> None:
    """Download and write ERA5 weather + MRT parquet shards."""
    # Register a shutdown backstop, but also close sessions explicitly in the
    # main control flow while tracked filesystems are still strongly referenced.
    atexit.register(_close_gcsfs_sessions)
    args = _parse_args()
    try:
        process_era5(
            args.year,
            args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            time_shard_index=args.time_shard_index,
            time_shard_count=args.time_shard_count,
            max_workers=args.max_workers,
            batch_hours=args.batch_hours,
            concurrency_profile=args.concurrency_profile,
            expected_location_count=args.expected_location_count or None,
            force=args.force,
        )
    except Exception as exc:
        LOGGER.exception("ERA5 pipeline failed for year=%s.", args.year)
        raise SystemExit(1) from exc
    finally:
        _close_gcsfs_sessions()


if __name__ == "__main__":
    main()
