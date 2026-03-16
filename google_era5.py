"""Download ERA5 weather + MRT data (2000-2023) from the Google ARCO Zarr store."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, TypeAlias, cast

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from pull_cds_shared import LOGGER, DataFrame, pa, partition_file_exists, pd, pq
from shards import resolve_filesystem

Dataset: TypeAlias = Any
ArrayLike: TypeAlias = Any


ERA5_ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ERA5_START_YEAR = 2000
ERA5_END_YEAR = 2023


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
    "fdir": [
        "fdir",
        "total_sky_direct_solar_radiation_at_surface",
        "clear_sky_direct_solar_radiation_at_surface",
    ],
}

ERA5_WEATHER_VARIABLES = ["10u", "10v", "2t", "2d"]
ERA5_MRT_ARCO_VARIABLES = ["ssrd", "strd", "fdir"]
ERA5_ALL_ARCO_VARIABLES = [*ERA5_WEATHER_VARIABLES, *ERA5_MRT_ARCO_VARIABLES]


RADIATION_SCALE = 1.0 / 3600.0

_B = 17.625
_C = 243.04

EXPECTED_LOCATION_COUNT = 500
THREE_DIMENSIONAL_ARRAY_NDIMS = 3


def _coerce_int_tuple(raw_value: object) -> tuple[int, ...] | None:
    """Return an integer tuple when the metadata value is a list of integers."""
    if not isinstance(raw_value, list):
        return None
    raw_dims = cast("list[object]", raw_value)
    if not all(isinstance(dim, int) for dim in raw_dims):
        return None
    return tuple(cast("list[int]", raw_dims))


def _load_arco_store_metadata() -> dict[str, object]:
    """Return consolidated Zarr metadata for the public ARCO ERA5 store."""
    gcsfs = cast("Any", importlib.import_module("gcsfs"))
    fs = gcsfs.GCSFileSystem(token="anon")  # noqa: S106
    metadata_path = ERA5_ARCO_STORE.removeprefix("gs://") + "/.zmetadata"
    with fs.open(metadata_path, "r") as metadata_file:
        return cast("dict[str, object]", json.load(metadata_file))


def _problematic_arco_chunks() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return any variables chunked as one full global field per hour."""
    metadata = _load_arco_store_metadata()
    metadata_entries = cast("dict[str, object]", metadata.get("metadata", {}))

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
        "latitude/longitude field per hour (%s). This script now reads each "
        "requested shard/year once and splits the in-memory result by tile to "
        "avoid repeated full-globe fetches, but the initial shard read can "
        "still be slow.",
        formatted_chunks,
    )


def _resolve_dataset_variables(ds: Dataset) -> dict[str, str]:
    """Map canonical ERA5 variable keys to actual dataset variable names."""
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical_name, candidates in ERA5_VARIABLE_CANDIDATES.items():
        actual_name = next((name for name in candidates if name in ds.data_vars), None)
        if actual_name is None:
            missing.append(canonical_name)
            continue
        resolved[canonical_name] = actual_name

    if missing:
        available_sample = sorted(ds.data_vars)[:80]
        msg = (
            "Required ARCO ERA5 variables could not be resolved: "
            f"{missing}. Variable candidates: "
            f"{ {name: ERA5_VARIABLE_CANDIDATES[name] for name in missing} }. "
            f"Sample available variables: {available_sample}"
        )
        raise KeyError(msg)

    return resolved


def _open_zarr_store(max_workers: int) -> Dataset:
    """Open the Google ARCO ERA5 Zarr store with dask-backed lazy loading.

    The store's time axis is encoded as integer hours since 1959-01-01. The
    full 1959-2023 span overflows pandas Timedelta during dataset open, so we
    pass decode_times=False and decode manually after subsetting to a single
    year.
    """
    gcsfs = cast("Any", importlib.import_module("gcsfs"))
    xr = cast("Any", importlib.import_module("xarray"))
    dask = cast("Any", importlib.import_module("dask"))

    dask.config.set(scheduler="threads", num_workers=max_workers)

    fs = gcsfs.GCSFileSystem(token="anon")  # noqa: S106
    store = fs.get_mapper(ERA5_ARCO_STORE)
    ds = xr.open_zarr(store, consolidated=True, decode_times=False)

    resolved_names = _resolve_dataset_variables(ds)
    rename_map = {
        actual_name: canonical_name
        for canonical_name, actual_name in resolved_names.items()
        if actual_name != canonical_name
    }
    return ds.rename_vars(rename_map) if rename_map else ds


def _year_time_slice(year: int) -> tuple[int, int]:
    """Return (start_h, end_h_inclusive) as integer hours since 1959-01-01."""
    pd_module = cast("Any", importlib.import_module("pandas"))
    epoch = pd_module.Timestamp("1959-01-01")

    start_h: int = int(
        (pd_module.Timestamp(f"{year}-01-01") - epoch).total_seconds() // 3600,
    )
    end_h: int = (
        int((pd_module.Timestamp(f"{year + 1}-01-01") - epoch).total_seconds() // 3600)
        - 1
    )
    return start_h, end_h


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

    if len(merged) != EXPECTED_LOCATION_COUNT:
        msg = (
            f"Expected {EXPECTED_LOCATION_COUNT} locations for ERA5, "
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


def _era5_partition_path(year: int, tile_id: int) -> str:
    return f"year={year}/tile_id={tile_id}"


def _era5_shard_exists(era5_root: str, year: int, tile_id: int) -> bool:
    return partition_file_exists(
        era5_root,
        _era5_partition_path(year, tile_id),
        "era5.parquet",
    )


def _calc_cossza(
    *,
    lat: float,
    lon: float,
    times: ArrayLike,
) -> ArrayLike:
    """Return hourly cosine solar zenith angle for one city/time series.

    Uses thermofeel helper functions instead of expecting cossza to exist
    in the ARCO dataset.
    """
    np = cast("Any", importlib.import_module("numpy"))
    tf = cast("Any", importlib.import_module("thermofeel"))

    integrated_fn = cast(
        "Any",
        getattr(tf, "calculate_cos_solar_zenith_angle_integrated", None),
    )
    instant_fn = cast("Any", getattr(tf, "calculate_cos_solar_zenith_angle", None))

    if integrated_fn is None and instant_fn is None:
        msg = (
            "thermofeel does not expose either "
            "'calculate_cos_solar_zenith_angle_integrated' or "
            "'calculate_cos_solar_zenith_angle'."
        )
        raise AttributeError(msg)

    out = np.empty(len(times), dtype="float64")

    for i, raw_ts in enumerate(times):
        timestamp = pd.Timestamp(raw_ts)

        if integrated_fn is not None:
            try:
                val = integrated_fn(
                    lat=lat,
                    lon=lon,
                    y=timestamp.year,
                    m=timestamp.month,
                    d=timestamp.day,
                    h=timestamp.hour,
                    base=0,
                    step=1,
                )
            except TypeError:
                val = integrated_fn(
                    lat=lat,
                    lon=lon,
                    y=timestamp.year,
                    m=timestamp.month,
                    d=timestamp.day,
                    h=timestamp.hour,
                    tbegin=0,
                    tend=1,
                )
        else:
            val = instant_fn(
                lat=lat,
                lon=lon,
                y=timestamp.year,
                m=timestamp.month,
                d=timestamp.day,
                h=timestamp.hour,
            )

        out[i] = float(val)

    return np.clip(out, 0.0, 1.0)


def _compute_location_frame(
    ds: Dataset,
    year: int,
    selected_cities: DataFrame,
) -> DataFrame:
    """Fetch one year of ERA5 data for a set of city points.

    Return a combined DataFrame with weather variables and MRT already in the
    final combined schema.
    """
    np = cast("Any", importlib.import_module("numpy"))
    xr = cast("Any", importlib.import_module("xarray"))
    tf = cast("Any", importlib.import_module("thermofeel"))

    lats = selected_cities["lat"].values.astype(float)
    lons = selected_cities["lng"].values.astype(float)
    location_ids = selected_cities["location_id"].values
    tile_ids = selected_cities["tile_id"].values

    lats_da = xr.DataArray(lats, dims="location")
    lons_da = xr.DataArray(lons, dims="location")

    start_h, end_h = _year_time_slice(year)
    year_slice: Any = ds[ERA5_ALL_ARCO_VARIABLES].sel(time=slice(start_h, end_h))
    city_selection: Any = year_slice.sel(
        latitude=lats_da,
        longitude=lons_da,
        method="nearest",
    )
    city_data: Any = city_selection.compute()

    n_time_steps: int = len(city_data.time)
    pd_module = cast("Any", importlib.import_module("pandas"))
    proper_times: Any = pd_module.date_range(
        f"{year}-01-01",
        periods=n_time_steps,
        freq="1h",
    )
    city_data = city_data.assign_coords(time=proper_times)

    u10: Any = city_data["10u"].values
    v10: Any = city_data["10v"].values
    t2m: Any = city_data["2t"].values
    d2m: Any = city_data["2d"].values

    ssrd: Any = city_data["ssrd"].values * RADIATION_SCALE
    strd: Any = city_data["strd"].values * RADIATION_SCALE
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
            fdir=fdir.ravel(),
            strd=strd.ravel(),
            cossza=cossza.ravel(),
        )
    except TypeError:
        mrt_k = tf.mrt.mean_radiant_temperature(
            ssrd=ssrd.ravel(),
            strd=strd.ravel(),
            fdir=fdir.ravel(),
            cos_projection=cossza.ravel(),
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
) -> None:
    filesystem, base_path = resolve_filesystem(era5_root)
    partition_dir = f"{base_path}/{_era5_partition_path(year, tile_id)}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/era5.parquet"

    table: Any = pa.Table.from_pandas(frame, preserve_index=False)
    with filesystem.open_output_stream(output_path) as out_stream:
        pq.write_table(table, out_stream)


def _pending_shard_tiles(
    shard_df: DataFrame,
    *,
    era5_root: str,
    year: int,
) -> DataFrame:
    """Return only the cities for tiles that still need an output shard."""
    pending_frames: list[DataFrame] = []
    tile_ids = sorted(int(raw_tile_id) for raw_tile_id in shard_df["tile_id"].unique())
    for tile_id in tile_ids:
        tile_cities = shard_df[shard_df["tile_id"] == tile_id].copy()
        if _era5_shard_exists(era5_root, year, tile_id):
            LOGGER.info(
                "ERA5 tile %s year %s already exists. Skipping.",
                tile_id,
                year,
            )
            continue

        LOGGER.info(
            "Queueing ERA5 tile %s year %s (%s cities).",
            tile_id,
            year,
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
) -> None:
    """Split a shard-wide frame by tile and write each partition."""
    if frame.empty:
        LOGGER.warning("No ERA5 rows returned for year %s.", year)
        return

    for tile_id, tile_frame in frame.groupby("tile_id", sort=True):
        output_frame = tile_frame.drop(columns=["tile_id"]).reset_index(drop=True)
        _write_era5_partition(era5_root, year, int(tile_id), output_frame)
        LOGGER.info(
            "Wrote %s ERA5 rows for tile %s year %s.",
            len(output_frame),
            int(tile_id),
            year,
        )


def process_era5(
    year: int,
    out_dir: str,
    *,
    city_shard_index: int = 0,
    city_shard_count: int = 1,
    max_workers: int = 4,
) -> None:
    """Download ERA5 weather + MRT for one year and one city shard."""
    if year < ERA5_START_YEAR or year > ERA5_END_YEAR:
        msg = (
            f"ERA5 ARCO store covers {ERA5_START_YEAR}-{ERA5_END_YEAR}. "
            f"Requested year {year} is outside this range."
        )
        raise ValueError(msg)

    era5_root = f"{out_dir}/era5_data_parquet"

    shard_df = _load_era5_city_shard(
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
    )
    if shard_df.empty:
        LOGGER.warning("No ERA5 cities selected for shard %s.", city_shard_index)
        return

    pending_shard_df = _pending_shard_tiles(shard_df, era5_root=era5_root, year=year)
    if pending_shard_df.empty:
        LOGGER.info(
            "ERA5 processing complete for year=%s shard %s/%s with no pending tiles.",
            year,
            city_shard_index + 1,
            city_shard_count,
        )
        return

    _warn_if_full_globe_chunks()
    ds = _open_zarr_store(max_workers=max_workers)
    LOGGER.info(
        "Reading ERA5 shard %s/%s for year %s across %s cities in %s pending tiles.",
        city_shard_index + 1,
        city_shard_count,
        year,
        len(pending_shard_df),
        pending_shard_df["tile_id"].nunique(),
    )
    frame = _compute_location_frame(ds, year, pending_shard_df)
    _write_shard_tiles(era5_root=era5_root, year=year, frame=frame)

    LOGGER.info(
        "ERA5 processing complete for year=%s shard %s/%s.",
        year,
        city_shard_index + 1,
        city_shard_count,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull ERA5 weather + MRT data from Google ARCO Zarr store.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=int,
        help=f"Year to download ({ERA5_START_YEAR}-{ERA5_END_YEAR}).",
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
        default=4,
        help="Dask thread-pool size for parallel GCS reads.",
    )
    return parser.parse_args()


def main() -> None:
    """Download and write ERA5 weather + MRT parquet shards."""
    args = _parse_args()
    try:
        process_era5(
            args.year,
            args.out_dir,
            city_shard_index=args.city_shard_index,
            city_shard_count=args.city_shard_count,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        LOGGER.exception("ERA5 pipeline failed for year=%s.", args.year)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
