"""Download ERA5 weather + MRT data (2000-2023) from the Google ARCO Zarr store."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from pull_cds_shared import LOGGER, DataFrame, pa, partition_file_exists, pd, pq
from shards import resolve_filesystem

if TYPE_CHECKING:
    from xarray import Dataset


# ARCO analysis-ready ERA5 store.
ERA5_ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ERA5_START_YEAR = 2000
ERA5_END_YEAR = 2023

# ARCO short variable names, not CDS long names.
ERA5_WEATHER_VARIABLES = [
    "10u",  # 10m_u_component_of_wind
    "10v",  # 10m_v_component_of_wind
    "2t",  # 2m_temperature
    "2d",  # 2m_dewpoint_temperature
]

# MRT needs these thermofeel-style inputs.
ERA5_MRT_VARIABLES = [
    "ssrd",  # surface_solar_radiation_downwards
    "strd",  # surface_thermal_radiation_downwards
    "fdir",  # total_sky_direct_solar_radiation_at_surface
    "cossza",  # cosine_solar_zenith_angle
]

ERA5_ALL_VARIABLES = ERA5_WEATHER_VARIABLES + ERA5_MRT_VARIABLES

RADIATION_SCALE = 1.0 / 3600.0

_B = 17.625
_C = 243.04

EXPECTED_LOCATION_COUNT = 500


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

    fs = gcsfs.GCSFileSystem()
    store = fs.get_mapper(ERA5_ARCO_STORE)
    ds = xr.open_zarr(store, consolidated=True, decode_times=False)

    missing = [name for name in ERA5_ALL_VARIABLES if name not in ds.data_vars]
    if missing:
        available_sample = sorted(ds.data_vars)[:50]
        msg = (
            "Required ARCO ERA5 variables are missing from the opened dataset: "
            f"{missing}. Sample available variables: {available_sample}"
        )
        raise KeyError(msg)

    return ds


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


def _compute_tile_frame(
    ds: Dataset,
    year: int,
    tile_cities: DataFrame,
) -> DataFrame:
    """Fetch one year of ERA5 data for a tile's city points.

    Return a combined DataFrame with weather variables and MRT already in the
    final combined schema.
    """
    xr = cast("Any", importlib.import_module("xarray"))
    np = cast("Any", importlib.import_module("numpy"))
    tf = cast("Any", importlib.import_module("thermofeel"))

    lats = tile_cities["lat"].values.astype(float)
    lons = tile_cities["lng"].values.astype(float)
    location_ids = tile_cities["location_id"].values

    lats_da = xr.DataArray(lats, dims="location")
    lons_da = xr.DataArray(lons, dims="location")

    # Time axis is integer hours since 1959-01-01 (decode_times=False on open).
    # Slice by integer offsets to avoid overflow when decoding the full range.
    start_h, end_h = _year_time_slice(year)
    city_selection: Any = ds[ERA5_ALL_VARIABLES].sel(
        time=slice(start_h, end_h),
        latitude=lats_da,
        longitude=lons_da,
        method="nearest",
    )
    city_data = city_selection.compute()

    # Replace integer hour coordinate with proper naive datetimes.
    n_time_steps: int = len(city_data.time)
    pd_module = cast("Any", importlib.import_module("pandas"))
    proper_times: Any = pd_module.date_range(
        f"{year}-01-01",
        periods=n_time_steps,
        freq="1h",
    )
    city_data = city_data.assign_coords(time=proper_times)

    # Weather variables
    u10: Any = city_data["10u"].values
    v10: Any = city_data["10v"].values
    t2m: Any = city_data["2t"].values
    d2m: Any = city_data["2d"].values

    # Radiation variables used by thermofeel MRT.
    # ERA5 accumulated radiation fields are commonly converted from J/m^2
    # per time step to W/m^2 by dividing by 3600 for hourly data.
    ssrd: Any = city_data["ssrd"].values * RADIATION_SCALE
    strd: Any = city_data["strd"].values * RADIATION_SCALE
    fdir: Any = city_data["fdir"].values * RADIATION_SCALE
    cossza: Any = city_data["cossza"].values

    wind_speed: Any = np.sqrt(u10**2 + v10**2)
    temperature_c: Any = t2m - 273.15
    dewpoint_c: Any = d2m - 273.15
    gamma_t: Any = _B * temperature_c / (_C + temperature_c)
    gamma_td: Any = _B * dewpoint_c / (_C + dewpoint_c)
    relative_humidity: Any = np.exp(gamma_td - gamma_t) * 100.0

    n_locations, n_times = wind_speed.shape

    # thermofeel returns MRT in Kelvin.
    mrt_k: Any = tf.mrt.mean_radiant_temperature(
        ssrd=ssrd.ravel(),
        strd=strd.ravel(),
        fdir=fdir.ravel(),
        cos_projection=cossza.ravel(),
    )
    mrt_c: Any = (mrt_k - 273.15).reshape(n_locations, n_times)

    times = city_data.time.values
    times_tiled: Any = np.tile(times, n_locations)
    loc_ids_rep: Any = np.repeat(location_ids, n_times)
    lats_rep: Any = np.repeat(lats, n_times)
    lons_rep: Any = np.repeat(lons, n_times)

    frame = pd.DataFrame(
        {
            "location_id": loc_ids_rep,
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
        .sort_values(["location_id", "time"])
        .reset_index(drop=True)
    )


def _process_era5_tile(
    ds: Dataset,
    year: int,
    era5_root: str,
    tile_id: int,
    tile_cities: DataFrame,
) -> None:
    if _era5_shard_exists(era5_root, year, tile_id):
        LOGGER.info(
            "ERA5 tile %s year %s already exists. Skipping.",
            tile_id,
            year,
        )
        return

    LOGGER.info(
        "Processing ERA5 tile %s year %s (%s cities).",
        tile_id,
        year,
        len(tile_cities),
    )

    frame = _compute_tile_frame(ds, year, tile_cities)
    if frame.empty:
        LOGGER.warning(
            "No ERA5 rows returned for tile %s year %s.",
            tile_id,
            year,
        )
        return

    _write_era5_partition(era5_root, year, tile_id, frame)
    LOGGER.info(
        "Wrote %s ERA5 rows for tile %s year %s.",
        len(frame),
        tile_id,
        year,
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

    ds = _open_zarr_store(max_workers=max_workers)

    for tile_id in sorted(shard_df["tile_id"].unique()):
        tile_cities = shard_df[shard_df["tile_id"] == tile_id].copy()
        _process_era5_tile(ds, year, era5_root, int(tile_id), tile_cities)

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
