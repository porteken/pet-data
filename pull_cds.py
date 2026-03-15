"""Download and shard ERA5 weather plus UTCI MRT data by tile."""

from __future__ import annotations

import argparse
import calendar
import csv
import importlib
import logging
import math
import multiprocessing
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast
from zipfile import ZipFile, is_zipfile

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from shards import resolve_filesystem
from shared_config import SHARED_MONTHS

DataFrame: TypeAlias = Any
SeriesLike: TypeAlias = Any

if TYPE_CHECKING:
    from collections.abc import Callable


class CDSResult(Protocol):
    """Typed subset of CDS result objects used by this module."""

    def download(self, target: str) -> object:
        """Download the result into a target path."""
        ...


class CDSClient(Protocol):
    """Typed subset of the CDS API client used by this module."""

    def retrieve(
        self,
        name: str,
        request: object,
        target: str | None = None,
    ) -> CDSResult:
        """Submit a dataset request and return a downloadable result."""
        ...


def create_cds_client() -> CDSClient:
    """Build a CDS API client with a stable static type."""
    cdsapi_module = cast("Any", importlib.import_module("cdsapi"))
    client_factory = cast("Callable[[], CDSClient]", cdsapi_module.Client)
    return client_factory()


pa = cast("Any", importlib.import_module("pyarrow"))
pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

B = 17.625
C = 243.04

WEATHER_DATASET = "reanalysis-era5-single-levels-timeseries"
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
WEATHER_CSV_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "latin-1",
)
QUEUE_LIMIT_REJECTION_TEXT = (
    "Number queued requests for this dataset is temporarily limited."
)
CDS_RETRY_ATTEMPTS = 6
CDS_RETRY_BASE_DELAY_SECONDS = 30
CDS_RETRY_MAX_DELAY_SECONDS = 300


def _retrieve_once(
    client: CDSClient,
    name: str,
    request: object,
    target: str | None = None,
) -> CDSResult:
    """Submit a single CDS retrieval request."""
    return client.retrieve(name, request, target)


def retrieve_with_retry(
    client: CDSClient,
    name: str,
    request: object,
    target: str | None = None,
) -> CDSResult:
    """Retry CDS retrievals when the service rejects jobs due to queue limits."""
    return _retrieve_with_retry_attempt(
        client=client,
        name=name,
        request=request,
        target=target,
        attempt=1,
    )


def _retrieve_with_retry_attempt(
    *,
    client: CDSClient,
    name: str,
    request: object,
    target: str | None,
    attempt: int,
) -> CDSResult:
    try:
        return _retrieve_once(client, name, request, target)
    except Exception as exc:
        if not _is_queue_limit_rejection(exc) or attempt == CDS_RETRY_ATTEMPTS:
            raise

        delay_seconds = min(
            CDS_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            CDS_RETRY_MAX_DELAY_SECONDS,
        )
        LOGGER.warning(
            "CDS queue limit hit for %s (attempt %s/%s). Retrying in %s seconds.",
            name,
            attempt,
            CDS_RETRY_ATTEMPTS,
            delay_seconds,
        )
        time.sleep(delay_seconds)
        return _retrieve_with_retry_attempt(
            client=client,
            name=name,
            request=request,
            target=target,
            attempt=attempt + 1,
        )


def _is_queue_limit_rejection(exc: Exception) -> bool:
    return QUEUE_LIMIT_REJECTION_TEXT in str(exc)


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

    _run_parallel_tile_jobs(
        tile_jobs=tile_jobs,
        max_workers=max_workers,
        dataset_label="weather",
        year=year,
        process_tile=lambda tile_row, tile_cells: _process_weather_tile(
            client=client,
            year=year,
            weather_root=weather_root,
            tile_row=tile_row,
            tile_cells=tile_cells,
        ),
        process_tile_with_new_client=lambda tile_row, tile_cells: _process_weather_tile_with_new_client(
            year=year,
            weather_root=weather_root,
            tile_row=tile_row,
            tile_cells=tile_cells,
        ),
    )

    LOGGER.info(
        "Weather processing complete for %s selected tiles.",
        len(selected_tiles),
    )


def process_mrt(
    client: CDSClient,
    year: str,
    out_dir: str,
    *,
    tile_ids: list[int] | None = None,
    tile_shard_index: int = 0,
    tile_shard_count: int = 1,
    max_workers: int = 4,
) -> None:
    """Download and process UTCI mean radiant temperature data grouped by tile."""
    mrt_root = f"{out_dir}/utci_data_parquet"
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
        LOGGER.warning("No MRT tiles selected for processing.")
        return

    _run_mrt_tile_jobs(
        tile_jobs=tile_jobs,
        max_workers=max_workers,
        mrt_root=mrt_root,
        year=year,
        client=client,
    )

    LOGGER.info(
        "MRT processing complete for %s selected tiles.",
        len(selected_tiles),
    )


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
    return _weather_year_exists(base_uri, year, tile_id)


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


def _run_parallel_tile_jobs(
    *,
    tile_jobs: list[tuple[dict[str, Any], DataFrame]],
    max_workers: int,
    dataset_label: str,
    year: str,
    process_tile: Callable[[dict[str, Any], DataFrame], None],
    process_tile_with_new_client: Callable[[dict[str, Any], DataFrame], None],
) -> None:
    worker_count = max(1, min(max_workers, len(tile_jobs)))
    if worker_count == 1:
        for tile_row, tile_cells in tile_jobs:
            process_tile(tile_row, tile_cells)
        return

    LOGGER.info(
        "Processing %s %s tiles with %s worker threads.",
        len(tile_jobs),
        dataset_label,
        worker_count,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_tile_id = {
            executor.submit(process_tile_with_new_client, tile_row, tile_cells): int(
                tile_row["tile_id"],
            )
            for tile_row, tile_cells in tile_jobs
        }
        for future in as_completed(future_to_tile_id):
            tile_id = future_to_tile_id[future]
            future.result()
            LOGGER.info(
                "%s tile %s completed for %s.",
                dataset_label,
                tile_id,
                year,
            )


def _run_mrt_tile_jobs(
    *,
    tile_jobs: list[tuple[dict[str, Any], DataFrame]],
    max_workers: int,
    mrt_root: str,
    year: str,
    client: CDSClient,
) -> None:
    worker_count = max(1, min(max_workers, len(tile_jobs)))
    if worker_count == 1:
        for tile_row, tile_cells in tile_jobs:
            _process_mrt_tile(
                client=client,
                year=year,
                mrt_root=mrt_root,
                tile_row=tile_row,
                tile_cells=tile_cells,
            )
        return

    LOGGER.info(
        "Processing %s MRT tiles with %s worker processes.",
        len(tile_jobs),
        worker_count,
    )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future_to_tile_id = {
            executor.submit(
                _process_mrt_tile_with_new_client,
                year=year,
                mrt_root=mrt_root,
                tile_row=tile_row,
                tile_cells=tile_cells,
            ): int(tile_row["tile_id"])
            for tile_row, tile_cells in tile_jobs
        }
        for future in as_completed(future_to_tile_id):
            tile_id = future_to_tile_id[future]
            future.result()
            LOGGER.info("MRT tile %s completed for %s.", tile_id, year)


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
        frames: list[DataFrame] = []
        for cell in tile_cells.itertuples(index=False):
            lat = float(cell.grid_lat)
            lng = float(cell.grid_lon)
            download_path = (
                tmpdir / f"weather_{year}_tile_{tile_id}_lat_{lat}_lng_{lng}.csv"
            )
            retrieve_with_retry(
                client,
                WEATHER_DATASET,
                _build_weather_request(lat, lng, year),
            ).download(str(download_path))

            weather_df = _load_weather_csv_frame(download_path, lat=lat, lng=lng)
            if weather_df.empty:
                LOGGER.warning(
                    "No weather rows returned for tile %s %s at (%s, %s).",
                    tile_id,
                    year,
                    lat,
                    lng,
                )
                continue
            frames.append(weather_df)

        if not frames:
            LOGGER.warning(
                "No weather rows were returned for tile %s in %s.",
                tile_id,
                year,
            )
            return

        finalized_df = _finalize_weather_frame(pd.concat(frames, ignore_index=True))
        _write_weather_partition(
            weather_root=weather_root,
            year=year,
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


def _process_mrt_tile(
    *,
    client: CDSClient,
    year: str,
    mrt_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    tile_id = int(tile_row["tile_id"])
    partition_path = _mrt_partition_path(year, tile_id)
    if partition_exists(mrt_root, partition_path):
        LOGGER.info("MRT tile %s already exists for %s. Skipping.", tile_id, year)
        return

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        zip_path = tmpdir / f"mrt_{year}_tile_{tile_id}.zip"
        LOGGER.info("Starting MRT tile %s for %s.", tile_id, year)
        retrieve_with_retry(
            client,
            "derived-utci-historical",
            {
                "variable": ["mean_radiant_temperature"],
                "version": "1_1",
                "product_type": "consolidated_dataset",
                "year": year,
                "month": [f"{month_value:02d}" for month_value in SHARED_MONTHS],
                "day": [f"{day:02d}" for day in range(1, 32)],
                "area": [
                    float(tile_row["north"]),
                    float(tile_row["west"]),
                    float(tile_row["south"]),
                    float(tile_row["east"]),
                ],
            },
            str(zip_path),
        )

        normalized_tile_cells = tile_cells.rename(
            columns={"grid_lat": "lat", "grid_lon": "lng"},
        )
        mrt_df = _load_mrt_frame(zip_path)
        mrt_df = mrt_df.merge(
            normalized_tile_cells,
            how="inner",
            on=["lat", "lng"],
            validate="many_to_one",
        )
        if mrt_df.empty:
            LOGGER.warning(
                "No MRT rows remained after grid-cell filtering for tile %s.",
                tile_id,
            )
            return

        _write_mrt_partition(
            mrt_root=mrt_root,
            partition_path=partition_path,
            mrt_df=mrt_df,
        )


def _process_mrt_tile_with_new_client(
    *,
    year: str,
    mrt_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    _process_mrt_tile(
        client=create_cds_client(),
        year=year,
        mrt_root=mrt_root,
        tile_row=tile_row,
        tile_cells=tile_cells,
    )


def _weather_year_exists(base_uri: str, year: str, tile_id: int) -> bool:
    return partition_exists(base_uri, _weather_partition_path(year, tile_id))


def _weather_partition_path(year: str, tile_id: int) -> str:
    return f"year={year}/tile_id={tile_id}"


def _mrt_partition_path(year: str, tile_id: int) -> str:
    return f"year={year}/tile_id={tile_id}"


def _build_weather_request(
    lat: float,
    lng: float,
    year: str,
) -> dict[str, object]:
    return {
        "variable": WEATHER_VARIABLES,
        "location": {"longitude": lng, "latitude": lat},
        "date": _build_weather_date_ranges(year),
        "data_format": "csv",
    }


def _build_weather_date_ranges(year: str) -> list[str]:
    if not SHARED_MONTHS:
        return []

    start_month = min(SHARED_MONTHS)
    end_month = max(SHARED_MONTHS)
    return [
        (
            f"{year}-{start_month:02d}-01/"
            f"{year}-{end_month:02d}-{calendar.monthrange(int(year), end_month)[1]:02d}"
        )
    ]


def _load_weather_csv_frame(
    download_path: Path,
    *,
    lat: float,
    lng: float,
) -> DataFrame:
    frame = _read_weather_csv(download_path)
    normalized_frame = _normalize_weather_columns(frame)
    if "lat" not in normalized_frame.columns:
        normalized_frame["lat"] = lat
    if "lng" not in normalized_frame.columns:
        normalized_frame["lng"] = lng
    return normalized_frame


def _read_weather_csv(download_path: Path) -> DataFrame:
    csv_path = _resolve_weather_csv_path(download_path)
    return _read_weather_csv_with_encoding(csv_path, encoding_index=0)


def _resolve_weather_csv_path(download_path: Path) -> Path:
    csv_files = _extract_files(download_path, suffix=".csv")
    if not csv_files:
        if is_zipfile(download_path):
            msg = f"No CSV files found in weather download: {download_path}"
            raise FileNotFoundError(msg)
        return download_path

    if len(csv_files) > 1:
        LOGGER.warning(
            "Weather download %s contained %s CSV files; using %s.",
            download_path,
            len(csv_files),
            csv_files[0].name,
        )
    return csv_files[0]


def _read_weather_csv_with_encoding(
    download_path: Path,
    *,
    encoding_index: int,
) -> DataFrame:
    encoding = WEATHER_CSV_ENCODINGS[encoding_index]
    try:
        header_row = _detect_weather_csv_header_row(download_path, encoding=encoding)
        return pd.read_csv(
            download_path,
            encoding=encoding,
            skiprows=header_row,
        )
    except (UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        if encoding_index + 1 < len(WEATHER_CSV_ENCODINGS):
            return _read_weather_csv_with_encoding(
                download_path,
                encoding_index=encoding_index + 1,
            )

        last_error = exc
    msg = (
        "Unable to decode weather CSV "
        f"{download_path} using encodings: {', '.join(WEATHER_CSV_ENCODINGS)}"
    )
    raise ValueError(msg) from last_error


def _detect_weather_csv_header_row(download_path: Path, *, encoding: str) -> int:
    with download_path.open("r", encoding=encoding, newline="") as weather_file:
        for line_number, line in enumerate(weather_file):
            normalized_columns = {
                column_name.strip().lower()
                for column_name in next(csv.reader([line]))
                if column_name.strip()
            }
            if "valid_time" in normalized_columns or "time" in normalized_columns:
                return line_number

    return 0


def _write_weather_partition(
    *,
    weather_root: str,
    year: str,
    tile_id: int,
    weather_df: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(weather_root)
    partition_path = _weather_partition_path(year, tile_id)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/weather.parquet"
    table: Any = pa.Table.from_pandas(
        weather_df.sort_values(["timestamp", "lat", "lng"]).reset_index(drop=True),
        preserve_index=False,
    )
    with filesystem.open_output_stream(output_path) as output_stream:
        pq.write_table(table, output_stream)


def _write_mrt_partition(
    *,
    mrt_root: str,
    partition_path: str,
    mrt_df: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(mrt_root)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/mrt.parquet"
    table: Any = pa.Table.from_pandas(mrt_df, preserve_index=False)
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
    return df.drop(columns=["u10", "v10", "t2m", "d2m", "dewpoint_c"], errors="ignore")


def _load_mrt_frame(zip_path: Path) -> DataFrame:
    xr = cast("Any", importlib.import_module("xarray"))
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
        help="Maximum number of concurrent tile workers for weather and MRT pulls.",
    )
    return parser.parse_args()


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

        if args.dataset in {"mrt", "all"}:
            process_mrt(
                client,
                args.year,
                args.out_dir,
                tile_ids=args.tile_ids,
                tile_shard_index=args.tile_shard_index,
                tile_shard_count=args.tile_shard_count,
                max_workers=args.max_workers,
            )
    except Exception as exc:
        LOGGER.exception("Pipeline failed for year=%s", args.year)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
