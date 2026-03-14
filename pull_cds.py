"""Download and process ERA5 weather and UTCI historical data."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Protocol, TypeAlias, cast
from zipfile import ZipFile, is_zipfile

from shared_config import SHARED_AREA


class CDSClient(Protocol):
    """Typed subset of the CDS API client used by this module."""

    retrieve: Callable[[str, object, str], object]


create_cds_client = cast(
    "Callable[[], CDSClient]",
    importlib.import_module("cdsapi").Client,
)
pa = cast("Any", importlib.import_module("pyarrow"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))
xr = cast("Any", importlib.import_module("xarray"))

DataFrame: TypeAlias = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

B = 17.625
C = 243.04

WEATHER_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
]


def check_partition_exists(base_uri: str, year: str, month: str) -> bool:
    """Check whether the target Parquet partition already exists."""
    month_int = int(month)

    try:
        filesystem, path = pa.fs.FileSystem.from_uri(base_uri)
        partition_path = f"{path}/year={year}/month={month_int}"
        file_info = filesystem.get_file_info(partition_path)
    except (OSError, pa.ArrowException) as exc:
        LOGGER.warning("Could not verify partition existence, proceeding: %s", exc)
        return False

    return bool(file_info.type != pa.fs.FileType.NotFound)


def process_weather(client: CDSClient, year: str, month: str, out_dir: str) -> None:
    """Download and process ERA5 weather data."""
    weather_root = f"{out_dir}/weather_data_parquet"
    if check_partition_exists(weather_root, year, month):
        LOGGER.info("Weather data for %s-%s already exists. Skipping.", year, month)
        return

    LOGGER.info("Starting weather (ERA5) processing for %s-%s...", year, month)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        grib_path = tmpdir / f"weather_{year}_{month}.grib"

        LOGGER.info("Downloading from CDS API...")
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "grib",
                "variable": WEATHER_VARIABLES,
                "year": year,
                "month": month,
                "day": [f"{day:02d}" for day in range(1, 32)],
                "time": [f"{hour:02d}:00" for hour in range(24)],
                "area": SHARED_AREA,
            },
            str(grib_path),
        )

        LOGGER.info("Converting GRIB to DataFrame...")
        with xr.open_dataset(
            grib_path,
            engine="cfgrib",
            decode_timedelta=True,
        ) as ds:
            df: DataFrame = (
                ds[["u10", "v10", "t2m", "d2m"]].to_dataframe().reset_index()
            )

        LOGGER.info("Calculating derived metrics...")
        df = df.rename(
            columns={
                "latitude": "lat",
                "longitude": "lng",
                "time": "timestamp",
            },
        )

        df["wind_speed"] = (df["u10"] ** 2 + df["v10"] ** 2) ** 0.5
        df["temperature_c"] = df["t2m"] - 273.15
        df["dewpoint_c"] = df["d2m"] - 273.15

        gamma_t: Any = B * df["temperature_c"] / (C + df["temperature_c"])
        gamma_td: Any = B * df["dewpoint_c"] / (C + df["dewpoint_c"])
        df["relative_humidity"] = (gamma_td - gamma_t).apply(math.exp) * 100

        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month

        df = df.drop(
            columns=[
                "u10",
                "v10",
                "t2m",
                "d2m",
                "valid_time",
                "number",
                "step",
                "surface",
                "dewpoint_c",
            ],
            errors="ignore",
        )

        LOGGER.info("Writing Parquet to %s...", weather_root)
        table: Any = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path=weather_root,
            partition_cols=["year", "month"],
            existing_data_behavior="overwrite_or_ignore",
        )
        LOGGER.info("Weather processing complete.")


def process_mrt(client: CDSClient, year: str, month: str, out_dir: str) -> None:
    """Download and process UTCI mean radiant temperature data."""
    mrt_root = f"{out_dir}/utci_data_parquet"
    if check_partition_exists(mrt_root, year, month):
        LOGGER.info("MRT data for %s-%s already exists. Skipping.", year, month)
        return

    LOGGER.info("Starting MRT (UTCI) processing for %s-%s...", year, month)

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        zip_path = tmpdir / f"mrt_{year}_{month}.zip"

        LOGGER.info("Downloading from CDS API...")
        client.retrieve(
            "derived-utci-historical",
            {
                "variable": ["mean_radiant_temperature"],
                "version": "1_1",
                "product_type": "consolidated_dataset",
                "year": year,
                "month": month,
                "day": [f"{day:02d}" for day in range(1, 32)],
                "area": SHARED_AREA,
            },
            str(zip_path),
        )

        if not is_zipfile(zip_path):
            msg = f"Downloaded file is not a valid ZIP: {zip_path}"
            raise ValueError(msg)

        LOGGER.info("Extracting ZIP...")
        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(path=tmpdir)

        nc_files = list(tmpdir.glob("*.nc"))
        if not nc_files:
            msg = "No NetCDF (.nc) files found in extracted ZIP."
            raise FileNotFoundError(msg)

        for nc_file in nc_files:
            LOGGER.info("Converting NetCDF %s to DataFrame...", nc_file.name)
            with xr.open_dataset(
                nc_file,
                engine="netcdf4",
                decode_timedelta=True,
            ) as ds:
                df: DataFrame = ds.to_dataframe().reset_index()

            df = df.rename(
                columns={
                    "lon": "lng",
                    "time": "timestamp",
                    "mrt": "mean_radiant_temperature_c",
                },
            )

            if "mean_radiant_temperature_c" in df.columns:
                df["mean_radiant_temperature_c"] -= 273.15

            df["year"] = df["timestamp"].dt.year
            df["month"] = df["timestamp"].dt.month
            df = df.drop(columns=["height"], errors="ignore")

            LOGGER.info("Writing Parquet to %s...", mrt_root)
            table: Any = pa.Table.from_pandas(df)
            pq.write_to_dataset(
                table,
                root_path=mrt_root,
                partition_cols=["year", "month"],
                existing_data_behavior="overwrite_or_ignore",
            )

    LOGGER.info("MRT processing complete.")


def main() -> None:
    """Parse arguments and run the requested CDS data pipeline."""
    parser = argparse.ArgumentParser(
        description="Pull weather and MRT data from CDS.",
    )
    parser.add_argument(
        "--year",
        required=True,
        type=str,
        help="Year to download (e.g., 2023)",
    )
    parser.add_argument(
        "--month",
        required=True,
        type=str,
        help="Month to download (e.g., 07)",
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

    args = parser.parse_args()

    try:
        client = create_cds_client()
    except Exception as exc:
        LOGGER.exception(
            "Failed to initialize CDS API client. Ensure your credentials are set.",
        )
        raise SystemExit(1) from exc

    try:
        if args.dataset in ["weather", "all"]:
            process_weather(client, args.year, args.month, args.out_dir)

        if args.dataset in ["mrt", "all"]:
            process_mrt(client, args.year, args.month, args.out_dir)
    except Exception as exc:
        LOGGER.exception("Pipeline failed for %s-%s", args.year, args.month)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
