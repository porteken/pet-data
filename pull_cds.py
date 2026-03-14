"""Download and shard ERA5 weather plus UTCI MRT data by tile."""

from __future__ import annotations

import argparse
import calendar
import importlib
import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeAlias, cast
from zipfile import ZipFile, is_zipfile

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from shards import resolve_filesystem
from shared_config import SHARED_MONTHS

if TYPE_CHECKING:
    from collections.abc import Collection

DataFrame: TypeAlias = Any
SeriesLike: TypeAlias = Any


class CDSClient(Protocol):
    """Typed subset of the CDS API client used by this module."""

    retrieve: Callable[[str, object, str], object]


class WeatherSelection(Protocol):
    """Typed subset of xarray selection objects used for weather extraction."""

    def to_dataframe(self) -> DataFrame:
        """Convert the selected weather cube to a DataFrame."""
        ...


class WeatherDataset(Protocol):
    """Typed subset of xarray.Dataset used by the weather path."""

    coords: Collection[str]
    dims: Collection[str]

    def load(self) -> WeatherDataset:
        """Eagerly load the dataset."""
        ...

    def rename(self, names: dict[str, str]) -> WeatherDataset:
        """Rename coordinates or dimensions."""
        ...

    def assign_coords(
        self,
        coords: dict[str, object] | None = None,
        **coords_kwargs: object,
    ) -> WeatherDataset:
        """Assign updated coordinate values."""
        ...

    def sortby(self, variables: str) -> WeatherDataset:
        """Sort the dataset by one coordinate."""
        ...

    def reset_coords(self, names: str, *, drop: bool = False) -> WeatherDataset:
        """Drop scalar coordinates before merging."""
        ...

    def squeeze(self, dim: str, *, drop: bool = False) -> WeatherDataset:
        """Remove length-1 dimensions."""
        ...

    def sel(self, *args: object, **kwargs: object) -> WeatherSelection:
        """Select coordinates from the dataset."""
        ...

    def __getitem__(self, key: str) -> object:
        """Return one coordinate or variable by name."""
        ...


class WeatherDatasetContext(Protocol):
    """Typed context manager returned by xarray.open_dataset."""

    def __enter__(self) -> WeatherDataset:
        """Open the dataset context."""
        ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None:
        """Close the dataset context."""
        ...


create_cds_client = cast(
    "Callable[[], CDSClient]",
    importlib.import_module("cdsapi").Client,
)
pa = cast("Any", importlib.import_module("pyarrow"))
pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
xr = cast("Any", importlib.import_module("xarray"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

B = 17.625
C = 243.04

WEATHER_DATASET = "reanalysis-era5-land"
WEATHER_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
]
WEATHER_VARIABLE_ALIASES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_dewpoint_temperature": "d2m",
    "2m_temperature": "t2m",
    "u10": "u10",
    "v10": "v10",
    "d2m": "d2m",
    "t2m": "t2m",
}


def partition_exists(base_uri: str, partition_path: str) -> bool:
    """Check whether a partition directory already exists."""
    try:
        filesystem, base_path = resolve_filesystem(base_uri)
        file_info = filesystem.get_file_info(f"{base_path}/{partition_path}")
    except (OSError, pa.ArrowException) as exc:
        LOGGER.warning(
            "Could not verify partition existence for %s: %s",
            partition_path,
            exc,
        )
        return False

    return bool(file_info.type != pa.fs.FileType.NotFound)


def process_weather(
    client: CDSClient,
    year: str,
    out_dir: str,
    *,
    tile_ids: list[int] | None = None,
    tile_shard_index: int = 0,
    tile_shard_count: int = 1,
    max_workers: int = 4,
) -> None:
    """Download and process ERA5 weather data grouped by tile."""
    weather_root = f"{out_dir}/weather_data_parquet"
    selected_tiles, selected_cells = _load_selected_tiles(
        tile_ids=tile_ids,
        tile_shard_index=tile_shard_index,
        tile_shard_count=tile_shard_count,
    )
    tile_jobs = [
        (
            tile._asdict(),
            selected_cells[selected_cells["tile_id"] == tile.tile_id][
                ["grid_lat", "grid_lon"]
            ].reset_index(drop=True),
        )
        for tile in selected_tiles.itertuples(index=False)
    ]
    if not tile_jobs:
        LOGGER.warning("No weather tiles selected for processing.")
        return

    worker_count = max(1, min(max_workers, len(tile_jobs)))
    if worker_count == 1:
        for tile_row, tile_cells in tile_jobs:
            _process_weather_tile(
                client=client,
                year=year,
                weather_root=weather_root,
                tile_row=tile_row,
                tile_cells=tile_cells,
            )
    else:
        LOGGER.info(
            "Processing %s weather tiles with %s worker threads.",
            len(tile_jobs),
            worker_count,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_tile_id = {
                executor.submit(
                    _process_weather_tile_with_new_client,
                    year=year,
                    weather_root=weather_root,
                    tile_row=tile_row,
                    tile_cells=tile_cells,
                ): int(tile_row["tile_id"])
                for tile_row, tile_cells in tile_jobs
            }
            for future in as_completed(future_to_tile_id):
                tile_id = future_to_tile_id[future]
                future.result()
                LOGGER.info(
                    "Weather tile %s completed for %s.",
                    tile_id,
                    year,
                )

    LOGGER.info(
        "Weather processing complete for %s selected tiles.",
        len(selected_tiles),
    )


def process_mrt(
    client: CDSClient,
    year: str,
    month: str,
    out_dir: str,
    *,
    tile_ids: list[int] | None = None,
    tile_shard_index: int = 0,
    tile_shard_count: int = 1,
) -> None:
    """Download and process UTCI mean radiant temperature data grouped by tile."""
    mrt_root = f"{out_dir}/utci_data_parquet"
    selected_tiles, selected_cells = _load_selected_tiles(
        tile_ids=tile_ids,
        tile_shard_index=tile_shard_index,
        tile_shard_count=tile_shard_count,
    )
    filesystem, base_path = resolve_filesystem(mrt_root)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        for tile in selected_tiles.itertuples(index=False):
            partition_path = (
                f"year={year}/month={int(month)}/tile_id={int(tile.tile_id)}"
            )
            if partition_exists(mrt_root, partition_path):
                LOGGER.info(
                    "MRT tile %s already exists for %s-%s. Skipping.",
                    tile.tile_id,
                    year,
                    month,
                )
                continue

            tile_cells = selected_cells[selected_cells["tile_id"] == tile.tile_id][
                ["grid_lat", "grid_lon"]
            ].rename(columns={"grid_lat": "lat", "grid_lon": "lng"})

            zip_path = tmpdir / f"mrt_{year}_{month}_tile_{tile.tile_id}.zip"
            LOGGER.info("Starting MRT tile %s for %s-%s.", tile.tile_id, year, month)
            client.retrieve(
                "derived-utci-historical",
                {
                    "variable": ["mean_radiant_temperature"],
                    "version": "1_1",
                    "product_type": "consolidated_dataset",
                    "year": year,
                    "month": month,
                    "day": [f"{day:02d}" for day in range(1, 32)],
                    "area": [
                        float(tile.north),
                        float(tile.west),
                        float(tile.south),
                        float(tile.east),
                    ],
                },
                str(zip_path),
            )

            mrt_df = _load_mrt_frame(zip_path)
            mrt_df = mrt_df.merge(
                tile_cells,
                how="inner",
                on=["lat", "lng"],
                validate="many_to_one",
            )
            if mrt_df.empty:
                LOGGER.warning(
                    "No MRT rows remained after grid-cell filtering for tile %s.",
                    tile.tile_id,
                )
                continue

            partition_dir = f"{base_path}/{partition_path}"
            filesystem.create_dir(partition_dir, recursive=True)
            output_path = f"{partition_dir}/mrt.parquet"
            table: Any = pa.Table.from_pandas(mrt_df, preserve_index=False)
            with filesystem.open_output_stream(output_path) as output_stream:
                pq.write_table(table, output_stream)

    LOGGER.info("MRT processing complete for %s selected tiles.", len(selected_tiles))


def _load_selected_tiles(
    *,
    tile_ids: list[int] | None,
    tile_shard_index: int,
    tile_shard_count: int,
) -> tuple[DataFrame, DataFrame]:
    _ensure_tile_outputs()
    tile_boxes_df = pd.read_csv(Path(OUTPUT_DIR) / "tile_boxes.csv")
    unique_cells_df = pd.read_csv(Path(OUTPUT_DIR) / "unique_grid_cells.csv")
    unique_cells_df = unique_cells_df.merge(
        tile_boxes_df[["tile_id", "tile_lat_min", "tile_lon_min"]],
        how="inner",
        on=["tile_lat_min", "tile_lon_min"],
        validate="many_to_one",
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
    selected_cells = unique_cells_df[
        unique_cells_df["tile_id"].isin(selected_tile_ids)
    ].copy()
    return selected_tiles.sort_values("tile_id"), selected_cells.sort_values(
        ["tile_id", "grid_lat", "grid_lon"],
    )


def _weather_tile_exists(base_uri: str, year: str, tile_id: int) -> bool:
    return all(
        _weather_month_exists(base_uri, year, month_value, tile_id)
        for month_value in SHARED_MONTHS
    )


def _ensure_tile_outputs() -> None:
    required_paths = [
        Path(OUTPUT_DIR) / "tile_boxes.csv",
        Path(OUTPUT_DIR) / "unique_grid_cells.csv",
    ]
    if all(path.exists() for path in required_paths):
        LOGGER.info("Using existing tile metadata under %s.", OUTPUT_DIR)
        return

    LOGGER.info("Tile metadata missing. Regenerating under %s.", OUTPUT_DIR)
    generate_tile_outputs()


def _process_weather_tile(
    *,
    client: CDSClient,
    year: str,
    weather_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    tile_id = int(tile_row["tile_id"])
    if _weather_tile_exists(weather_root, year, tile_id):
        LOGGER.info("Weather tile %s already exists for %s. Skipping.", tile_id, year)
        return

    LOGGER.info(
        "Starting weather tile %s for %s with %s snapped cells.",
        tile_id,
        year,
        len(tile_cells),
    )
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        for month_value in SHARED_MONTHS:
            if _weather_month_exists(weather_root, year, month_value, tile_id):
                LOGGER.info(
                    "Weather tile %s already exists for %s-%02d. Skipping.",
                    tile_id,
                    year,
                    month_value,
                )
                continue

            download_path = (
                tmpdir / f"weather_{year}_{month_value:02d}_tile_{tile_id}.grib"
            )
            client.retrieve(
                WEATHER_DATASET,
                _build_weather_request(tile_row, year, month_value),
                str(download_path),
            )

            weather_df = _load_weather_grib_frame(download_path, tile_cells)
            if weather_df.empty:
                LOGGER.warning(
                    (
                        "No weather rows remained after local cell selection "
                        "for tile %s %s-%02d."
                    ),
                    tile_id,
                    year,
                    month_value,
                )
                continue
            finalized_df = _finalize_weather_frame(weather_df)
            _write_weather_partition(
                weather_root=weather_root,
                year=year,
                month=month_value,
                tile_id=tile_id,
                weather_df=finalized_df,
            )


def _process_weather_tile_with_new_client(
    *,
    year: str,
    weather_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    _process_weather_tile(
        client=create_cds_client(),
        year=year,
        weather_root=weather_root,
        tile_row=tile_row,
        tile_cells=tile_cells,
    )


def _weather_month_exists(base_uri: str, year: str, month: int, tile_id: int) -> bool:
    return partition_exists(
        base_uri,
        _weather_partition_path(year, month, tile_id),
    )


def _weather_partition_path(year: str, month: int, tile_id: int) -> str:
    return f"year={year}/month={month}/tile_id={tile_id}"


def _build_weather_request(
    tile_row: dict[str, Any],
    year: str,
    month: int,
) -> dict[str, object]:
    day_count = calendar.monthrange(int(year), month)[1]
    return {
        "variable": WEATHER_VARIABLES,
        "year": year,
        "month": f"{month:02d}",
        "day": [f"{day:02d}" for day in range(1, day_count + 1)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": [
            float(tile_row["north"]),
            float(tile_row["west"]),
            float(tile_row["south"]),
            float(tile_row["east"]),
        ],
        "format": "grib",
    }


def _load_weather_grib_frame(download_path: Path, tile_cells: DataFrame) -> DataFrame:
    datasets: list[WeatherDataset] = []
    for level_value in (2, 10):
        dataset_context = cast(
            "WeatherDatasetContext",
            xr.open_dataset(
                download_path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {
                        "typeOfLevel": "heightAboveGround",
                        "level": level_value,
                    },
                },
            ),
        )
        with dataset_context as dataset:
            normalized_dataset = _normalize_weather_dataset(dataset.load())
        if "heightAboveGround" in normalized_dataset.coords:
            normalized_dataset = normalized_dataset.reset_coords(
                "heightAboveGround",
                drop=True,
            )
        if "heightAboveGround" in normalized_dataset.dims:
            normalized_dataset = normalized_dataset.squeeze(
                "heightAboveGround",
                drop=True,
            )
        datasets.append(normalized_dataset)

    merged_dataset = cast(
        "WeatherDataset",
        xr.merge(datasets, compat="override", join="exact"),
    )
    selected_df = _select_weather_cells(merged_dataset, tile_cells)

    return _normalize_weather_columns(selected_df)


def _normalize_weather_dataset(dataset: WeatherDataset) -> WeatherDataset:
    normalized_dataset = dataset
    rename_map = {
        source_name: target_name
        for source_name, target_name in {
            "valid_time": "time",
            "longitude": "longitude",
            "latitude": "latitude",
            "lon": "longitude",
            "lat": "latitude",
        }.items()
        if source_name in normalized_dataset.coords
        or source_name in normalized_dataset.dims
    }
    if rename_map:
        normalized_dataset = normalized_dataset.rename(rename_map)

    if "longitude" in normalized_dataset.coords:
        normalized_dataset = normalized_dataset.assign_coords(
            longitude=_normalize_longitude_values(normalized_dataset["longitude"]),
        ).sortby("longitude")
    if "latitude" in normalized_dataset.coords:
        normalized_dataset = normalized_dataset.sortby("latitude")
    return normalized_dataset


def _select_weather_cells(dataset: WeatherDataset, tile_cells: DataFrame) -> DataFrame:
    requested_cells = tile_cells.reset_index(drop=True).copy()
    requested_cells["cell"] = requested_cells.index
    requested_cells["grid_lat"] = pd.to_numeric(
        requested_cells["grid_lat"],
        errors="coerce",
    )
    requested_cells["grid_lon"] = pd.to_numeric(
        requested_cells["grid_lon"],
        errors="coerce",
    )

    selection = dataset.sel(
        latitude=xr.DataArray(
            requested_cells["grid_lat"].to_numpy(),
            dims="cell",
            coords={"cell": requested_cells["cell"].to_numpy()},
        ),
        longitude=xr.DataArray(
            _normalize_longitude_values(requested_cells["grid_lon"].to_numpy()),
            dims="cell",
            coords={"cell": requested_cells["cell"].to_numpy()},
        ),
        method="nearest",
    )
    frame = selection.to_dataframe().reset_index()
    frame = frame.merge(
        requested_cells[["cell", "grid_lat", "grid_lon"]],
        how="inner",
        on="cell",
        validate="many_to_one",
    )
    frame["lat"] = frame.pop("grid_lat")
    frame["lng"] = frame.pop("grid_lon")
    return frame.drop(
        columns=["cell", "latitude", "longitude", "surface", "number", "step"],
        errors="ignore",
    )


def _normalize_longitude_values(values: object) -> object:
    longitude_values = cast("Any", values)
    return ((longitude_values + 180) % 360) - 180


def _write_weather_partition(
    *,
    weather_root: str,
    year: str,
    month: int,
    tile_id: int,
    weather_df: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(weather_root)
    partition_path = _weather_partition_path(year, month, tile_id)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/weather.parquet"
    table: Any = pa.Table.from_pandas(
        weather_df.sort_values(["timestamp", "lat", "lng"]).reset_index(drop=True),
        preserve_index=False,
    )
    with filesystem.open_output_stream(output_path) as output_stream:
        pq.write_table(table, output_stream)


def _normalize_weather_columns(df: DataFrame) -> DataFrame:
    df = df.rename(
        columns={
            "date": "time",
            "datetime": "time",
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lng",
            **{
                column_name: WEATHER_VARIABLE_ALIASES[column_name]
                for column_name in df.columns
                if column_name in WEATHER_VARIABLE_ALIASES
            },
        },
    )

    if {"variable", "value"}.issubset(df.columns):
        df["variable"] = (
            df["variable"].map(WEATHER_VARIABLE_ALIASES).fillna(df["variable"])
        )
        index_columns = [
            column_name
            for column_name in ["time", "lat", "lng"]
            if column_name in df.columns
        ]
        df = df.pivot_table(
            index=index_columns,
            columns="variable",
            values="value",
            aggfunc="first",
        ).reset_index()
        df.columns.name = None

    required_columns = {"time", "u10", "v10", "t2m", "d2m"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        msg = (
            "Weather download missing expected columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        raise ValueError(msg)

    df["time"] = pd.to_datetime(df["time"])
    for column_name in ["u10", "v10", "t2m", "d2m"]:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    return df.dropna(subset=["time", "u10", "v10", "t2m", "d2m"]).reset_index(drop=True)


def _finalize_weather_frame(df: DataFrame) -> DataFrame:
    df = df.rename(columns={"time": "timestamp"})
    df["wind_speed"] = (df["u10"] ** 2 + df["v10"] ** 2) ** 0.5
    df["temperature_c"] = df["t2m"] - 273.15
    df["dewpoint_c"] = df["d2m"] - 273.15

    gamma_t: Any = B * df["temperature_c"] / (C + df["temperature_c"])
    gamma_td: Any = B * df["dewpoint_c"] / (C + df["dewpoint_c"])
    df["relative_humidity"] = (gamma_td - gamma_t).apply(math.exp) * 100

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    return df.drop(columns=["u10", "v10", "t2m", "d2m", "dewpoint_c"], errors="ignore")


def _load_mrt_frame(zip_path: Path) -> DataFrame:
    nc_files = _extract_files(zip_path, suffix=".nc")
    frames: list[DataFrame] = []
    for nc_file in nc_files:
        with xr.open_dataset(
            nc_file,
            engine="netcdf4",
            decode_timedelta=True,
        ) as dataset:
            frame = dataset.to_dataframe().reset_index()
        frames.append(_normalize_mrt_columns(frame))

    if not frames:
        msg = f"No NetCDF files found in MRT download: {zip_path}"
        raise FileNotFoundError(msg)

    return pd.concat(frames, ignore_index=True)


def _normalize_mrt_columns(df: DataFrame) -> DataFrame:
    df = df.rename(
        columns={
            "lon": "lng",
            "longitude": "lng",
            "latitude": "lat",
            "time": "timestamp",
            "mrt": "mean_radiant_temperature_c",
        },
    )

    required_columns = {"timestamp", "lat", "lng", "mean_radiant_temperature_c"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        msg = (
            "MRT download missing expected columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        raise ValueError(msg)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["lat"] = _snap_grid_series(df["lat"])
    df["lng"] = _snap_grid_series(df["lng"])
    df["mean_radiant_temperature_c"] = (
        pd.to_numeric(
            df["mean_radiant_temperature_c"],
            errors="coerce",
        )
        - 273.15
    )

    return (
        df.dropna(subset=["timestamp", "lat", "lng", "mean_radiant_temperature_c"])
        .drop(columns=["height"], errors="ignore")
        .reset_index(drop=True)
    )


def _snap_grid_series(series: SeriesLike) -> SeriesLike:
    return (pd.to_numeric(series, errors="coerce") / GRID_DEG).round() * GRID_DEG


def _extract_files(download_path: Path, *, suffix: str) -> list[Path]:
    with tempfile.TemporaryDirectory() as extract_dir_name:
        extract_dir = Path(extract_dir_name)
        if is_zipfile(download_path):
            with ZipFile(download_path, "r") as zip_file:
                zip_file.extractall(path=extract_dir)
            extracted_files = sorted(extract_dir.rglob(f"*{suffix}"))
            copied_files: list[Path] = []
            for extracted_file in extracted_files:
                copied_path = download_path.parent / extracted_file.name
                copied_path.write_bytes(extracted_file.read_bytes())
                copied_files.append(copied_path)
            return copied_files

    if download_path.suffix == suffix:
        return [download_path]
    return []


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull weather and MRT data from CDS.")
    parser.add_argument(
        "--year",
        required=True,
        type=str,
        help="Year to download (e.g., 2023)",
    )
    parser.add_argument(
        "--month",
        type=str,
        help="Month to download when pulling MRT (e.g., 07)",
    )
    parser.add_argument(
        "--dataset",
        choices=["weather", "mrt", "all"],
        default="all",
        help="Which dataset to pull.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory base path or S3 URI (e.g., s3://my-bucket/data)",
    )
    parser.add_argument(
        "--tile-id",
        dest="tile_ids",
        action="append",
        type=int,
        help="Optional tile_id filter. Pass multiple times to target specific tiles.",
    )
    parser.add_argument(
        "--tile-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for splitting tile work across CI jobs.",
    )
    parser.add_argument(
        "--tile-shard-count",
        type=int,
        default=1,
        help="Total number of tile shards across CI jobs.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent weather tile workers.",
    )
    args = parser.parse_args()
    if args.dataset in {"mrt", "all"} and not args.month:
        parser.error("--month is required when dataset includes mrt")
    return args


def main() -> None:
    """Run the requested CDS download and parquet conversion steps."""
    args = _parse_args()

    try:
        client = create_cds_client()
    except Exception as exc:
        LOGGER.exception(
            "Failed to initialize CDS API client. Ensure your credentials are set.",
        )
        raise SystemExit(1) from exc

    try:
        if args.dataset in {"weather", "all"}:
            process_weather(
                client,
                args.year,
                args.out_dir,
                tile_ids=args.tile_ids,
                tile_shard_index=args.tile_shard_index,
                tile_shard_count=args.tile_shard_count,
                max_workers=args.max_workers,
            )

        if args.dataset in {"mrt", "all"} and args.month:
            process_mrt(
                client,
                args.year,
                args.month,
                args.out_dir,
                tile_ids=args.tile_ids,
                tile_shard_index=args.tile_shard_index,
                tile_shard_count=args.tile_shard_count,
            )
    except Exception as exc:
        LOGGER.exception("Pipeline failed for year=%s month=%s", args.year, args.month)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
