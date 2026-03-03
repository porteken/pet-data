"""Download and process ERA5 Historical Data."""

from __future__ import annotations

import importlib
import logging
import math
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol, cast


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
joblib = cast("Any", importlib.import_module("joblib"))
Parallel = joblib.Parallel
delayed = joblib.delayed
tqdm = importlib.import_module("tqdm").tqdm

B = 17.625
C = 243.04

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MONTHS_IN_YEAR = 12


@dataclass
class Config:
    """All user-configurable parameters for the ERA5 download pipeline."""

    start_date: str = "2025-05-01"
    end_date: str = "2025-10-01"
    months: list[int] = field(default_factory=lambda: list(range(5, 10)))
    area: list[float] = field(default_factory=lambda: [49.25, -124.5, 24.25, -66.5])
    variables: list[str] = field(
        default_factory=lambda: [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
        ],
    )
    grib_download_dir: Path = Path("raw_weather_grib")
    parquet_output_dir: Path = Path("weather_data_parquet")
    n_jobs: int = 7


CONFIG = Config()


class _ClientState(threading.local):
    client: CDSClient | None

    def __init__(self) -> None:
        # threading.local state is per-thread, so initialize defaults here.
        self.client = None


_THREAD_LOCAL = _ClientState()


def _get_client() -> CDSClient:
    """Create one CDS API client lazily per worker thread."""
    client = _THREAD_LOCAL.client
    if client is None:
        client = create_cds_client()
        _THREAD_LOCAL.client = client
        logger.info("Initialized CDS client in worker thread.")
    return client


def _download_month(year: str, month: str, target_file: Path) -> Exception | None:
    """Attempt one ERA5 monthly download and return the raised error, if any."""
    try:
        logger.info("Downloading data for %s-%s to %s...", year, month, target_file)
        client = _get_client()
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "grib",
                "variable": CONFIG.variables,
                "year": year,
                "month": month,
                "day": [str(i).zfill(2) for i in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": CONFIG.area,
            },
            str(target_file),
        )
    except Exception as exc:
        logger.exception("Download attempt failed for %s-%s.", year, month)
        return exc
    return None


def generate_date_list(start_iso: str, end_iso: str, months: list[int]) -> list[str]:
    """Generate a sorted list of unique 'YYYY-MM' strings for the given period."""
    start_date = date.fromisoformat(start_iso)
    end_date = date.fromisoformat(end_iso)

    year_months: set[str] = set()
    current_date = start_date
    while current_date < end_date:
        if current_date.month in months:
            year_months.add(current_date.strftime("%Y-%m"))
        if current_date.month < MONTHS_IN_YEAR:
            current_date = date(current_date.year, current_date.month + 1, 1)
        else:
            current_date = date(current_date.year + 1, 1, 1)

    logger.info(
        "Generated %d unique year-month combinations to process.",
        len(year_months),
    )
    return sorted(year_months)


def download_era5_data(year: str, month: str, target_file: Path) -> None:
    """Download one month of ERA5 data if it doesn't already exist."""
    if target_file.exists():
        logger.info("Skipping download, file already exists: %s", target_file)
        return

    max_attempts = 3
    wait_seconds = 4
    for attempt in range(1, max_attempts + 1):
        error = _download_month(year, month, target_file)
        if error is None:
            logger.info("Successfully downloaded data for %s-%s.", year, month)
            return
        if attempt == max_attempts:
            logger.error(
                "Download failed for %s-%s after %d attempts.",
                year,
                month,
                max_attempts,
                exc_info=(type(error), error, error.__traceback__),
            )
            raise error
        logger.warning(
            "Download failed for %s-%s on attempt %d/%d. Retrying in %d seconds.",
            year,
            month,
            attempt,
            max_attempts,
            wait_seconds,
            exc_info=(type(error), error, error.__traceback__),
        )
        time.sleep(wait_seconds)
        wait_seconds = min(wait_seconds * 2, 60)


def convert_grib_to_parquet(grib_path: Path, parquet_root: Path) -> None:
    """Convert from GRIB to Parquet."""
    logger.info("Converting %s to Parquet format...", grib_path)

    try:
        with xr.open_dataset(grib_path, engine="cfgrib") as ds_raw:
            df: Any = ds_raw[["u10", "v10", "t2m", "d2m"]].to_dataframe().reset_index()
    except Exception:
        logger.exception("Failed to read GRIB file %s.", grib_path)
        raise

    df = df.rename(
        columns={"latitude": "lat", "longitude": "lng", "time": "timestamp"},
    )

    df["wind_speed"] = (df["u10"] ** 2 + df["v10"] ** 2) ** 0.5

    df["temperature_c"] = df["t2m"] - 273.15
    df["dewpoint_c"] = df["d2m"] - 273.15

    gamma_t = B * df["temperature_c"] / (C + df["temperature_c"])
    gamma_td = B * df["dewpoint_c"] / (C + df["dewpoint_c"])
    df["relative_humidity"] = (gamma_td - gamma_t).apply(math.exp) * 100

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    df = df.drop(
        columns=["u10", "v10", "t2m", "d2m", "valid_time", "number", "step", "surface"],
        errors="ignore",
    )

    table: Any = pa.Table.from_pandas(df)
    with tempfile.TemporaryDirectory() as tmp_dir:
        pq.write_to_dataset(
            table,
            root_path=tmp_dir,
            partition_cols=["year", "month"],
            existing_data_behavior="overwrite_or_ignore",
        )
        shutil.copytree(tmp_dir, parquet_root, dirs_exist_ok=True)

    logger.info("Successfully wrote data from %s to %s", grib_path, parquet_root)


def process_date(date_str: str, grib_dir: Path, parquet_dir: Path) -> str | None:
    """Download, convert, and clean up data for month."""
    year, month = date_str.split("-")
    grib_filename = grib_dir / f"{date_str}.grib"

    try:
        download_era5_data(year, month, grib_filename)
        convert_grib_to_parquet(grib_filename, parquet_dir)
    except Exception:
        logger.exception("Processing failed for %s", date_str)
        return date_str
    else:
        return None
    finally:
        if grib_filename.exists():
            try:
                grib_filename.unlink()
                logger.info("Removed temporary file: %s", grib_filename)
            except OSError:
                logger.exception("Error removing file %s", grib_filename)


def main() -> None:
    """Orchestrator function."""
    CONFIG.grib_download_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.parquet_output_dir.mkdir(parents=True, exist_ok=True)

    dates_to_process = generate_date_list(
        CONFIG.start_date,
        CONFIG.end_date,
        CONFIG.months,
    )

    if not dates_to_process:
        logger.warning(
            "No dates to process based on the current configuration. Exiting.",
        )
        return

    logger.info("Starting parallel processing for %d months...", len(dates_to_process))

    results: list[str | None] = list(
        Parallel(
            n_jobs=CONFIG.n_jobs,
            backend="threading",
        )(
            delayed(process_date)(
                date_str,
                CONFIG.grib_download_dir,
                CONFIG.parquet_output_dir,
            )
            for date_str in tqdm(dates_to_process, desc="Processing Months")
        ),
    )

    failed = [r for r in results if r is not None]
    if failed:
        logger.warning(
            "%d month(s) failed to process: %s",
            len(failed),
            ", ".join(failed),
        )
    else:
        logger.info("All months processed successfully.")

    try:
        if not any(CONFIG.grib_download_dir.iterdir()):
            CONFIG.grib_download_dir.rmdir()
            logger.info("Removed empty GRIB download directory.")
    except OSError:
        logger.exception("Could not remove GRIB directory %s", CONFIG.grib_download_dir)

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
