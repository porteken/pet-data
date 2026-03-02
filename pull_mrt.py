"""Download and process UTCI Historical Data."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import cdsapi
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

CONFIG = {
    "start_date": "2000-05-01",
    "end_date": "2025-10-01",
    "months": list(range(5, 10)),
    "area": [49.25, -124.5, 24.25, -66.5],
    "variable": "mean_radiant_temperature",
    "dataset": "derived-utci-historical",
    "zip_download_dir": Path("raw_utci_zip"),
    "parquet_output_dir": Path("utci_data_parquet"),
    "n_jobs": 4,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MAX_MONTH = 12


def generate_date_list(start_iso: str, end_iso: str, months: list[int]) -> list[str]:
    """Generate a sorted list of unique 'YYYY-MM' strings for the given period."""
    start_date = date.fromisoformat(start_iso)
    end_date = date.fromisoformat(end_iso)

    date_set = set()
    current_date = start_date
    while current_date < end_date:
        if current_date.month in months:
            date_set.add(current_date.strftime("%Y-%m"))

        year, month = (
            (current_date.year, current_date.month + 1)
            if current_date.month < MAX_MONTH
            else (current_date.year + 1, 1)
        )
        current_date = date(year, month, 1)

    logger.info(
        "Generated %d unique year-month combinations to process.",
        len(date_set),
    )
    return sorted(date_set)


def download_utci_data(
    client: cdsapi.Client,
    year: str,
    month: str,
    target_file: Path,
) -> None:
    """Download one month of UTCI data as a zip if it doesn't already exist."""
    if target_file.exists():
        logger.info("Skipping download, file already exists: %s", target_file)
        return

    logger.info("Downloading data for %s-%s to %s...", year, month, target_file)
    try:
        client.retrieve(
            CONFIG["dataset"],
            {
                "variable": [CONFIG["variable"]],
                "version": "1_1",
                "product_type": "consolidated_dataset",
                "year": year,
                "month": month,
                "day": [str(i).zfill(2) for i in range(1, 32)],
                "area": CONFIG["area"],
            },
            str(target_file),
        )
        logger.info("Successfully downloaded data for %s-%s.", year, month)
    except Exception:
        logger.exception("Failed to download data for %s-%s.", year, month)

        raise


def convert_netcdf_to_parquet(netcdf_path: Path, parquet_root: Path) -> None:
    """Convert a single NetCDF file to a partitioned Parquet dataset."""
    logger.info("Converting %s to Parquet format...", netcdf_path)

    try:
        with xr.open_dataset(netcdf_path, engine="netcdf4") as ds:
            df = ds.to_dataframe().reset_index()
    except Exception:
        logger.exception("Failed to read NetCDF file %s.", netcdf_path)
        raise

    df = df.rename(
        columns={
            "lon": "lng",
            "time": "timestamp",
            "mrt": "mean_radiant_temperature_c",
        },
    )

    df["mean_radiant_temperature_c"] -= 273.15

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    df = df.drop(columns=["height"], errors="ignore")

    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=parquet_root,
        partition_cols=["year", "month"],
        existing_data_behavior="overwrite_or_ignore",
    )
    logger.info("Successfully wrote data from %s to %s", netcdf_path, parquet_root)


def process_date(date_str: str, zip_dir: Path, parquet_dir: Path) -> None:
    """Download, extract, convert, and clean up data for a given year-month."""
    year, month = date_str.split("-")
    zip_filename = zip_dir / f"{date_str}.zip"
    extract_dir = zip_dir / f"temp_{date_str}"

    try:

        client = cdsapi.Client()

        download_utci_data(client, year, month, zip_filename)

        if not is_zipfile(zip_filename):
            logger.warning("Downloaded file is not a zip file: %s", zip_filename)
            return

        with ZipFile(zip_filename, "r") as zf:
            zf.extractall(path=extract_dir)
            logger.info("Extracted %d files from %s", len(zf.namelist()), zip_filename)

        netcdf_files = list(extract_dir.glob("*.nc"))
        if not netcdf_files:
            logger.warning("No NetCDF files found in %s", extract_dir)
            return

        for nc_file in netcdf_files:
            convert_netcdf_to_parquet(nc_file, parquet_dir)

    except Exception:
        logger.exception("Processing failed for %s", date_str)
    finally:

        if extract_dir.exists():
            for temp_file in extract_dir.iterdir():
                temp_file.unlink()
            extract_dir.rmdir()

        if zip_filename.exists():
            try:
                zip_filename.unlink()
                logger.info("Removed temporary file: %s", zip_filename)
            except OSError:
                logger.exception("Error removing file %s", zip_filename)


def main() -> None:
    """Orchestrator function."""
    CONFIG["zip_download_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["parquet_output_dir"].mkdir(parents=True, exist_ok=True)

    dates_to_process = generate_date_list(
        CONFIG["start_date"],
        CONFIG["end_date"],
        CONFIG["months"],
    )

    if not dates_to_process:
        logger.warning(
            "No dates to process based on the current configuration. Exiting.",
        )
        return

    logger.info("Starting parallel processing for %d months...", len(dates_to_process))
    Parallel(n_jobs=CONFIG["n_jobs"])(
        delayed(process_date)(
            date_str,
            CONFIG["zip_download_dir"],
            CONFIG["parquet_output_dir"],
        )
        for date_str in tqdm(dates_to_process, desc="Processing Months")
    )

    try:
        if not any(CONFIG["zip_download_dir"].iterdir()):
            CONFIG["zip_download_dir"].rmdir()
            logger.info("Removed empty ZIP download directory.")
    except OSError:
        logger.exception(
            "Could not remove ZIP directory %s",
            CONFIG["zip_download_dir"],
        )

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
