"""Download and shard ERA5 weather data from CDS."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import is_zipfile

from tqdm.auto import tqdm

from pull_cds_shared import (
    LOGGER,
    CDSClient,
    DataFrame,
    create_cds_client,
    extract_files,
    pa,
    partition_file_exists,
    pd,
    pq,
    retrieve_with_retry,
)
from shards import resolve_filesystem
from shared_config import build_full_date_range, build_year_date_range

B = 17.625
C = 243.04
EXPECTED_WEATHER_LOCATION_COUNT = 500
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
WEATHER_CDS_REQUEST_CONCURRENCY = 2

WEATHER_CDS_REQUEST_SEMAPHORE = threading.BoundedSemaphore(
    WEATHER_CDS_REQUEST_CONCURRENCY,
)
WEATHER_THREAD_LOCAL = threading.local()


def process_weather(
    client: CDSClient,
    out_dir: str,
    *,
    year: int | None = None,
    month: int | None = None,
    start_year: int | None = None,
    city_shard_index: int = 0,
    city_shard_count: int = 1,
    max_workers: int = 4,
) -> None:
    """Download and process ERA5 weather data grouped by city shards."""
    weather_root = f"{out_dir}/weather_data_parquet"
    date_range = _build_weather_date_range(
        year=year,
        month=month,
        start_year=start_year,
    )

    shard_city_cells = _load_weather_city_shard(
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
    )
    if shard_city_cells.empty:
        LOGGER.warning("No weather cities selected for processing.")
        return

    _process_weather_shard(
        client=client,
        weather_root=weather_root,
        city_cells=shard_city_cells,
        city_shard_index=city_shard_index,
        date_range=date_range,
        max_workers=max_workers,
    )

    LOGGER.info(
        "Weather processing complete for shard %s/%s with %s cities over %s.",
        city_shard_index + 1,
        city_shard_count,
        len(shard_city_cells),
        date_range,
    )


def _load_weather_city_shard(
    *,
    city_shard_index: int,
    city_shard_count: int,
) -> DataFrame:
    """Return the city rows assigned to the requested weather shard."""
    if city_shard_count < 1:
        msg = "city_shard_count must be >= 1"
        raise ValueError(msg)
    if city_shard_index < 0 or city_shard_index >= city_shard_count:
        msg = f"city_shard_index must be between 0 and {city_shard_count - 1}"
        raise ValueError(msg)

    cities_path = Path("cities.csv")
    if not cities_path.exists():
        msg = "cities.csv is required before pulling weather data."
        raise FileNotFoundError(msg)

    cities_df = pd.read_csv(cities_path)
    required_columns = {"location_id", "lat", "lng"}
    missing_columns = required_columns.difference(cities_df.columns)
    if missing_columns:
        msg = (
            "cities.csv is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        raise ValueError(msg)

    city_cells = (
        cities_df[["location_id", "lat", "lng"]]
        .sort_values("location_id")
        .reset_index(drop=True)
    )
    if len(city_cells) != EXPECTED_WEATHER_LOCATION_COUNT:
        msg = (
            "Expected weather sharding to cover exactly "
            f"{EXPECTED_WEATHER_LOCATION_COUNT} locations, found {len(city_cells)}."
        )
        raise ValueError(msg)

    total_cities = len(city_cells)
    if total_cities % city_shard_count != 0:
        msg = (
            f"Cannot evenly split {total_cities} cities across "
            f"{city_shard_count} weather shards."
        )
        raise ValueError(msg)

    shard_size = total_cities // city_shard_count
    start_index = city_shard_index * shard_size
    end_index = start_index + shard_size
    shard_city_cells = city_cells.iloc[start_index:end_index].copy()
    LOGGER.info(
        "Weather shard %s/%s selected location_ids %s-%s (%s cities).",
        city_shard_index + 1,
        city_shard_count,
        int(shard_city_cells["location_id"].min()),
        int(shard_city_cells["location_id"].max()),
        len(shard_city_cells),
    )
    return shard_city_cells


def _weather_shard_exists(
    base_uri: str,
    city_shard_index: int,
) -> bool:
    return partition_file_exists(
        base_uri,
        _weather_partition_path(city_shard_index),
        "weather.parquet",
    )


def _process_weather_shard(
    *,
    client: CDSClient,
    weather_root: str,
    city_cells: DataFrame,
    city_shard_index: int,
    date_range: str,
    max_workers: int,
) -> None:
    if _weather_shard_exists(weather_root, city_shard_index):
        LOGGER.info("Weather shard %s already exists. Skipping.", city_shard_index)
        return

    LOGGER.info(
        "Starting weather shard %s for %s cities over %s.",
        city_shard_index,
        len(city_cells),
        date_range,
    )
    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        city_jobs = [
            (
                int(cell.location_id),
                float(cell.lat),
                float(cell.lng),
                tmpdir
                / f"weather_shard_{city_shard_index}_location_{int(cell.location_id)}.csv",
            )
            for cell in city_cells.itertuples(index=False)
        ]
        frames = _run_weather_city_jobs(
            client=client,
            city_jobs=city_jobs,
            date_range=date_range,
            max_workers=max_workers,
        )

        if not frames:
            LOGGER.warning(
                "No weather rows were returned for shard %s.",
                city_shard_index,
            )
            return

        finalized_df = _finalize_weather_frame(pd.concat(frames, ignore_index=True))
        _write_weather_partition(
            weather_root=weather_root,
            city_shard_index=city_shard_index,
            weather_df=finalized_df,
        )


def _weather_partition_path(city_shard_index: int) -> str:
    return f"city_shard_index={city_shard_index}"


def _build_weather_date_range(
    *,
    year: int | None,
    month: int | None,
    start_year: int | None = None,
) -> str:
    if month is not None and year is None:
        msg = "--month requires --year for weather pulls."
        raise ValueError(msg)
    if year is None:
        full_range_kwargs: dict[str, int] = {}
        if start_year is not None:
            full_range_kwargs["start_year"] = start_year
        return build_full_date_range(**full_range_kwargs)
    return build_year_date_range(year, month=month)


def _run_weather_city_jobs(
    *,
    client: CDSClient,
    city_jobs: list[tuple[int, float, float, Path]],
    date_range: str,
    max_workers: int,
) -> list[DataFrame]:
    progress_desc = "Weather locations"
    worker_count = max(
        1,
        min(
            max_workers,
            len(city_jobs),
            WEATHER_CDS_REQUEST_CONCURRENCY,
            os.cpu_count() or 1,
        ),
    )
    if worker_count == 1:
        return [
            weather_df
            for weather_df in (
                _process_weather_location(
                    client=client,
                    location_id=location_id,
                    lat=lat,
                    lng=lng,
                    date_range=date_range,
                    download_path=download_path,
                )
                for location_id, lat, lng, download_path in tqdm(
                    city_jobs,
                    desc=progress_desc,
                    unit="location",
                )
            )
            if not weather_df.empty
        ]

    LOGGER.info(
        "Processing %s weather locations with %s worker threads.",
        len(city_jobs),
        worker_count,
    )
    frames: list[DataFrame] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _process_weather_location_with_thread_client,
                location_id=location_id,
                lat=lat,
                lng=lng,
                date_range=date_range,
                download_path=download_path,
            )
            for location_id, lat, lng, download_path in city_jobs
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=progress_desc,
            unit="location",
        ):
            weather_df = future.result()
            if not weather_df.empty:
                frames.append(weather_df)
    return frames


def _process_weather_location_with_thread_client(
    *,
    location_id: int,
    lat: float,
    lng: float,
    date_range: str,
    download_path: Path,
) -> DataFrame:
    client = getattr(WEATHER_THREAD_LOCAL, "client", None)
    if client is None:
        client = create_cds_client()
        WEATHER_THREAD_LOCAL.client = client
    return _process_weather_location(
        client=client,
        location_id=location_id,
        lat=lat,
        lng=lng,
        date_range=date_range,
        download_path=download_path,
    )


def _process_weather_location(
    *,
    client: CDSClient,
    location_id: int,
    lat: float,
    lng: float,
    date_range: str,
    download_path: Path,
) -> DataFrame:
    with WEATHER_CDS_REQUEST_SEMAPHORE:
        retrieve_with_retry(
            client,
            WEATHER_DATASET,
            _build_weather_request(lat, lng, date_range=date_range),
            str(download_path),
        )

    weather_df = _load_weather_csv_frame(download_path, lat=lat, lng=lng)
    if weather_df.empty:
        LOGGER.warning(
            "No weather rows returned for location_id %s at (%s, %s).",
            location_id,
            lat,
            lng,
        )
        return pd.DataFrame()

    weather_df["location_id"] = location_id
    return weather_df


def _build_weather_request(
    lat: float,
    lng: float,
    *,
    date_range: str,
) -> dict[str, object]:
    return {
        "variable": WEATHER_VARIABLES,
        "location": {"longitude": lng, "latitude": lat},
        "date": [date_range],
        "data_format": "csv",
    }


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
    csv_files = extract_files(download_path, suffix=".csv")
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
    city_shard_index: int,
    weather_df: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(weather_root)
    partition_path = _weather_partition_path(city_shard_index)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/weather.parquet"
    table: Any = pa.Table.from_pandas(
        weather_df.sort_values(["location_id", "timestamp"]).reset_index(drop=True),
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
    return df.drop(columns=["u10", "v10", "t2m", "d2m", "dewpoint_c"], errors="ignore")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull weather data from CDS.")
    parser.add_argument(
        "--year",
        type=int,
        help="Optional year to download instead of the full historical window.",
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Optional month to download within the requested weather year (1-12).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory base path or S3 URI (e.g., s3://my-bucket/data)",
    )
    parser.add_argument(
        "--weather-city-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for splitting weather work across cities.",
    )
    parser.add_argument(
        "--weather-city-shard-count",
        type=int,
        default=1,
        help=(
            "Total weather city shards. The selected city set must divide evenly "
            "across this shard count."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers for weather pulls.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=(
            "Earliest year to include in a full-range pull. "
            "Ignored when --year is specified."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the weather CDS download and parquet conversion steps."""
    args = _parse_args()

    try:
        client = create_cds_client()
    except Exception as exc:
        LOGGER.exception(
            "Failed to initialize CDS API client. Ensure your credentials are set.",
        )
        raise SystemExit(1) from exc

    try:
        process_weather(
            client,
            args.out_dir,
            year=args.year,
            month=args.month,
            start_year=args.start_year,
            city_shard_index=args.weather_city_shard_index,
            city_shard_count=args.weather_city_shard_count,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        LOGGER.exception("Weather pipeline failed.")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
