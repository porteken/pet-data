"""Download Google ARCO ERA5, calculate MRT, and compute daily maximum PET."""

from __future__ import annotations

import argparse
import importlib
import logging
import multiprocessing
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast
from zipfile import ZipFile, is_zipfile

from tqdm.auto import tqdm

from shards import resolve_filesystem

if TYPE_CHECKING:
    from collections.abc import Callable

# --- Shared Helpers (formerly pull_cds_shared.py) ---
pa = cast("Any", importlib.import_module("pyarrow"))
pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

DataFrame: TypeAlias = Any
SeriesLike: TypeAlias = Any


class CDSResult(Protocol):
    """Typed subset of CDS result objects used by the pull scripts."""

    reply: dict[str, Any]

    def download(self, target: str) -> object:
        """Download the result into a target path."""
        ...

    def update(self, request_id: str | None = None) -> object:
        """Refresh the current request state."""
        ...

    def delete(self) -> object:
        """Delete the remote job when supported by CDS."""
        ...


class CDSClient(Protocol):
    """Typed subset of the CDS API client used by the pull scripts."""

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
    client_factory = cast("Callable[..., CDSClient]", cdsapi_module.Client)
    return client_factory(wait_until_complete=False)


QUEUE_LIMIT_REJECTION_TEXT = (
    "Number queued requests for this dataset is temporarily limited."
)
QUEUE_LIMIT_REJECTION_MARKERS = (
    QUEUE_LIMIT_REJECTION_TEXT.lower(),
    "rate limit exceeded",
    "too many requests",
    "request limit exceeded",
)
EMPTY_REJECTION_MARKERS = (
    "ended in state rejected: no reason provided",
    "ended in state failed: no reason provided",
)
CDS_RETRY_ATTEMPTS = 6
CDS_RETRY_BASE_DELAY_SECONDS = 30
CDS_RETRY_MAX_DELAY_SECONDS = 300
CDS_POLL_INITIAL_DELAY_SECONDS = 5
CDS_POLL_MAX_DELAY_SECONDS = 60


def _retrieve_once(
    client: CDSClient,
    name: str,
    request: object,
) -> CDSResult:
    """Submit a single CDS retrieval request."""
    return client.retrieve(name, request)


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
        result = _retrieve_once(client, name, request)
        completed_result = _wait_for_completion(result, name=name)
        if target is not None:
            completed_result.download(target)
    except Exception as exc:
        if not _is_retryable_rejection(exc) or attempt == CDS_RETRY_ATTEMPTS:
            raise

        delay_seconds = min(
            CDS_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            CDS_RETRY_MAX_DELAY_SECONDS,
        )
        LOGGER.warning(
            "Retryable CDS rejection for %s (attempt %s/%s). Retrying in %s seconds.",
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
    else:
        return completed_result


def _is_retryable_rejection(exc: Exception) -> bool:
    message = str(exc)
    return _is_queue_limit_message(message) or _is_empty_rejection_message(message)


def _is_queue_limit_message(message: str) -> bool:
    normalized_message = message.lower()
    return any(marker in normalized_message for marker in QUEUE_LIMIT_REJECTION_MARKERS)


def _is_empty_rejection_message(message: str) -> bool:
    normalized_message = message.lower()
    return any(marker in normalized_message for marker in EMPTY_REJECTION_MARKERS)


def _wait_for_completion(result: CDSResult, *, name: str) -> CDSResult:
    delay_seconds = CDS_POLL_INITIAL_DELAY_SECONDS
    last_state: str | None = None

    while True:
        reply = result.reply
        state = str(reply.get("state", "")).lower()
        if state != last_state:
            LOGGER.info(
                "CDS request %s for %s is %s.",
                reply.get("request_id", "<unknown>"),
                name,
                state or "<unknown>",
            )
            last_state = state

        if state == "completed":
            return result

        if state in {"accepted", "queued", "running"}:
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, CDS_POLL_MAX_DELAY_SECONDS)
            result.update()
            continue

        error_message = _result_error_message(reply)
        if state in {"failed", "rejected"} and _is_queue_limit_message(error_message):
            raise RuntimeError(error_message)

        if state in {"failed", "rejected"}:
            msg = (
                f"CDS request {reply.get('request_id', '<unknown>')} for {name} "
                f"ended in state {state}: {error_message or 'no reason provided'}"
            )
            raise RuntimeError(msg)

        msg = (
            f"CDS request {reply.get('request_id', '<unknown>')} for {name} "
            f"returned unknown state {state!r}."
        )
        raise RuntimeError(msg)


def _result_error_message(reply: dict[str, Any]) -> str:
    error_payload = reply.get("error")
    if isinstance(error_payload, dict):
        error_details = cast("dict[str, object]", error_payload)
        message_parts = [
            str(error_details.get("message", "")).strip(),
            str(error_details.get("reason", "")).strip(),
        ]
        return ". ".join(part for part in message_parts if part)

    return str(reply.get("reason", "")).strip()


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


def partition_file_exists(base_uri: str, partition_path: str, file_name: str) -> bool:
    """Check whether a specific partition file already exists."""
    try:
        filesystem, base_path = resolve_filesystem(base_uri)
        file_info = filesystem.get_file_info(
            f"{base_path}/{partition_path}/{file_name}",
        )
    except (OSError, pa.ArrowException) as exc:
        LOGGER.warning(
            "Could not verify partition file existence for %s/%s: %s",
            partition_path,
            file_name,
            exc,
        )
        return False

    return bool(file_info.type != pa.fs.FileType.NotFound)


def partition_file_max_timestamp(
    base_uri: str,
    partition_path: str,
    file_name: str,
    *,
    column_name: str = "timestamp",
) -> date | None:
    """Return the latest date stored in a parquet partition file."""
    filesystem, base_path = resolve_filesystem(base_uri)
    file_path = f"{base_path}/{partition_path}/{file_name}"
    max_timestamp: date | None = None

    try:
        with filesystem.open_input_file(file_path) as input_file:
            parquet_file = pq.ParquetFile(input_file)
            field_index = parquet_file.schema_arrow.get_field_index(column_name)
            if field_index != -1:
                max_value: object | None = None
                missing_stats = False
                for row_group_index in range(parquet_file.metadata.num_row_groups):
                    row_group = parquet_file.metadata.row_group(row_group_index)
                    statistics = row_group.column(field_index).statistics
                    if statistics is None or not statistics.has_min_max:
                        missing_stats = True
                        break

                    candidate = statistics.max
                    if max_value is None or candidate > max_value:
                        max_value = candidate

                if max_value is not None and not missing_stats:
                    max_timestamp = _coerce_partition_date(max_value)
    except (OSError, pa.ArrowException) as exc:
        LOGGER.warning(
            "Could not inspect max timestamp for %s/%s: %s",
            partition_path,
            file_name,
            exc,
        )
    else:
        if max_timestamp is not None:
            return max_timestamp

        try:
            table = pq.read_table(
                file_path,
                columns=[column_name],
                filesystem=filesystem,
            )
        except (OSError, pa.ArrowException) as exc:
            LOGGER.warning(
                "Could not read timestamps for %s/%s: %s",
                partition_path,
                file_name,
                exc,
            )
        else:
            if table.num_rows > 0 and column_name in table.column_names:
                values = table[column_name].to_pandas()
                if not values.empty:
                    max_timestamp = _coerce_partition_date(values.max())

    return max_timestamp


def _coerce_partition_date(value: object) -> date | None:
    """Convert parquet timestamp-like values into a concrete date."""
    converted = pd.to_datetime(value)
    if isinstance(converted, datetime):
        return converted.date()
    if isinstance(converted, date):
        return converted

    date_method = getattr(converted, "date", None)
    if callable(date_method):
        maybe_date = cast("Any", date_method())
        if isinstance(maybe_date, date):
            return maybe_date

    return None


def extract_files(download_path: Path, *, suffix: str) -> list[Path]:
    """Extract files of a specific suffix from a download path when needed."""
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

    if download_path.name.endswith(suffix):
        return [download_path]
    return []


# --- End of Shared Helpers ---

GRID_DEG: float = 0.25

SAFE_STABLE_DATA_MONTH = 4
PET_ROUNDING_FACTOR = 2.0
CHUNK_SIZE = 50000


# --- PET Corrected Import ---
class _PetCorrectedCallable(Protocol):
    def __call__(
        self,
        tair: object,
        t_mrt: object,
        v_air: object,
        rh: object,
        *,
        icl: float,
    ) -> Any:  # noqa: ANN401
        ...


try:
    pet_module: Any = importlib.import_module("pet_corrected")
    pet_corrected: _PetCorrectedCallable = cast(
        "_PetCorrectedCallable",
        pet_module.pet_corrected,
    )
except (ImportError, AttributeError) as error:
    LOGGER.exception("Could not import pet_corrected from pet_corrected.py.")
    raise SystemExit(1) from error


def _arco_stable_end_year() -> int:
    now = datetime.now(tz=timezone.utc)
    return now.year - 2 if now.month < SAFE_STABLE_DATA_MONTH else now.year - 1


Dataset: TypeAlias = Any
ArrayLike: TypeAlias = Any
ZarrMetadata: TypeAlias = dict[str, object]

ERA5_ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
ERA5_START_YEAR = 2000
ERA5_END_YEAR = _arco_stable_end_year()

ERA5_VARIABLE_CANDIDATES: dict[str, list[str]] = {
    "10u": ["10u", "10m_u_component_of_wind"],
    "10v": ["10v", "10m_v_component_of_wind"],
    "2t": ["2t", "2m_temperature"],
    "2d": ["2d", "2m_dewpoint_temperature"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "strd": ["strd", "surface_thermal_radiation_downwards"],
    "ssr": ["ssr", "surface_net_solar_radiation"],
    "str": ["str", "surface_net_thermal_radiation"],
    "fdir": [
        "fdir",
        "total_sky_direct_solar_radiation_at_surface",
        "clear_sky_direct_solar_radiation_at_surface",
    ],
    "msdrswrf": ["msdrswrf", "mean_surface_direct_short_wave_radiation_flux"],
}

ERA5_WEATHER_VARIABLES = ["10u", "10v", "2t", "2d"]
ERA5_RADIATION_VARIABLES = ["ssrd", "strd", "ssr", "str", "fdir", "msdrswrf"]
ERA5_ALL_ARCO_VARIABLES = [*ERA5_WEATHER_VARIABLES, *ERA5_RADIATION_VARIABLES]

RADIATION_SCALE = 1.0 / 3600.0
_B = 17.625
_C = 243.04

EXPECTED_LOCATION_COUNT = 0
THREE_DIMENSIONAL_ARRAY_NDIMS = 3
ERA5_TIME_ORIGIN = "1959-01-01"
DEFAULT_BATCH_HOURS = 24 * 30
ERA5_THREAD_LOCAL = threading.local()

GCS_BATCH_MAX_RETRIES = 3
GCS_BATCH_RETRY_DELAY_SECONDS = 10


def _resolve_store_variables(array_names: set[str]) -> dict[str, str]:
    """Map canonical names to actual store names."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical_name, candidates in ERA5_VARIABLE_CANDIDATES.items():
        actual_name = next((name for name in candidates if name in array_names), None)
        if actual_name is None:
            missing.append(canonical_name)
            continue
        resolved[canonical_name] = actual_name
    if missing:
        msg = f"Required ARCO variables missing: {missing}"
        raise KeyError(msg)
    return resolved


def _open_zarr_store() -> Dataset:
    """Initialize public ARCO ERA5 Zarr store."""
    gcsfs = cast("Any", importlib.import_module("gcsfs"))
    xr = cast("Any", importlib.import_module("xarray"))
    dask_array = cast("Any", importlib.import_module("dask.array"))
    zarr = cast("Any", importlib.import_module("zarr"))

    fs = gcsfs.GCSFileSystem(
        token="anon",  # noqa: S106
        default_block_size=8 * 1024 * 1024,
        skip_instance_cache=True,
    )
    root = zarr.open_group(fs.get_mapper(ERA5_ARCO_STORE), mode="r")
    resolved_names = _resolve_store_variables(set(root.array_keys()))
    return xr.Dataset(
        data_vars={
            c_name: (
                ("time", "latitude", "longitude"),
                dask_array.from_zarr(root[a_name]),
            )
            for c_name, a_name in resolved_names.items()
        },
        coords={
            "time": root["time"][:],
            "latitude": root["latitude"][:],
            "longitude": root["longitude"][:],
        },
    )


def _configure_concurrency(profile: str) -> None:
    zarr = cast("Any", importlib.import_module("zarr"))
    dask = cast("Any", importlib.import_module("dask"))
    configs = {
        "conservative": {"zarr_concurrency": 16, "dask_workers": 2},
        "balanced": {"zarr_concurrency": 64, "dask_workers": 4},
        "aggressive": {"zarr_concurrency": 128, "dask_workers": 8},
    }
    profile_cfg = configs.get(profile, configs["balanced"])
    if hasattr(zarr, "config"):
        zarr.config.set({"async.concurrency": profile_cfg["zarr_concurrency"]})
    dask.config.set({"array.slicing.split_large_chunks": False})


def _resolve_era5_max_workers(requested_max_workers: int) -> int:
    """Determine max workers for parallel execution."""
    if requested_max_workers == 0:
        msg = (
            f"max_workers must be positive or -1 for auto, got {requested_max_workers}"
        )
        raise ValueError(msg)
    if requested_max_workers == -1:
        return max(1, os.cpu_count() or 1)
    return requested_max_workers


def _year_time_slice(year: int) -> tuple[int, int]:
    """Return the inclusive (start_h, end_h) for the given year relative to ERA5_TIME_ORIGIN."""
    epoch = pd.Timestamp(ERA5_TIME_ORIGIN)
    start_h = int((pd.Timestamp(f"{year}-01-01") - epoch).total_seconds() // 3600)
    end_h = int((pd.Timestamp(f"{year + 1}-01-01") - epoch).total_seconds() // 3600) - 1
    return start_h, end_h


def _era5_partition_path(year: int, batch_index: int | None = None) -> str:
    """Return a standardized partition path for ERA5 output."""
    path = f"year={year}"
    if batch_index is not None:
        path += f"/batch_index={batch_index:04d}"
    return path


def _iter_time_batches(
    year: int,
    batch_hours: int,
    allowed_months: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    if batch_hours <= 0:
        msg = f"batch_hours must be positive, got {batch_hours}"
        raise ValueError(msg)
    start_h, end_h = _year_time_slice(year)

    batches: list[tuple[int, int, int]] = []
    batch_index = 0
    batch_start = start_h
    while batch_start <= end_h:
        batch_end = min(batch_start + batch_hours - 1, end_h)
        batch_start_ts = pd.to_datetime(batch_start, unit="h", origin=ERA5_TIME_ORIGIN)
        batch_end_ts = pd.to_datetime(batch_end, unit="h", origin=ERA5_TIME_ORIGIN)

        # Only include batch if it overlaps with allowed months
        if (
            not allowed_months
            or batch_start_ts.month in allowed_months
            or batch_end_ts.month in allowed_months
        ):
            batches.append((batch_index, batch_start, batch_end))

        batch_index += 1
        batch_start = batch_end + 1
    return batches


def _select_time_shard_batches(
    batches: list[tuple[int, int, int]],
    time_shard_index: int,
    time_shard_count: int,
) -> list[tuple[int, int, int]]:
    if time_shard_index >= time_shard_count:
        msg = f"time_shard_index {time_shard_index} must be less than time_shard_count {time_shard_count}"
        raise ValueError(msg)
    return [b for b in batches if b[0] % time_shard_count == time_shard_index]


def _load_era5_city_shard(city_shard_index: int, city_shard_count: int) -> DataFrame:
    cities_df = pd.read_csv("cities.csv", usecols=["location_id", "lat", "lng"])
    cities_df["lat"] = (pd.to_numeric(cities_df["lat"]) / GRID_DEG).round() * GRID_DEG
    cities_df["lng"] = (pd.to_numeric(cities_df["lng"]) / GRID_DEG).round() * GRID_DEG
    cities_df = cities_df.sort_values("location_id").reset_index(drop=True)
    shard_size = (len(cities_df) + city_shard_count - 1) // city_shard_count
    start = city_shard_index * shard_size
    return cities_df.iloc[start : min(start + shard_size, len(cities_df))].copy()


def _calc_cossza(lat: float, lon: float, times: ArrayLike) -> ArrayLike:
    np = cast("Any", importlib.import_module("numpy"))
    timestamps = pd.to_datetime(times)
    day_of_year = timestamps.dayofyear.to_numpy(dtype="float64")
    hour = (
        timestamps.hour.to_numpy(dtype="float64")
        + timestamps.minute.to_numpy(dtype="float64") / 60.0
    )
    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma)
        + 0.000907 * np.sin(2.0 * gamma)
    )
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
    )
    tst_min = hour * 60.0 + eq_time + 4.0 * lon
    ha = np.deg2rad(tst_min / 4.0 - 180.0)
    lat_r = np.deg2rad(lat)
    cossza = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    return np.clip(cossza, 0.0, 1.0)


def _compute_pet_chunk(df_chunk: DataFrame) -> DataFrame:
    t_vals, mrt_vals, v_vals, rh_vals = (
        df_chunk["t"].to_numpy(),
        df_chunk["mrt"].to_numpy(),
        df_chunk["v"].to_numpy(),
        df_chunk["rh"].to_numpy(),
    )
    pet_results = pet_corrected(t_vals, mrt_vals, v_vals, rh_vals, icl=0.5)
    df_result = df_chunk.copy()
    df_result["pet"] = (pet_results * PET_ROUNDING_FACTOR).round() / PET_ROUNDING_FACTOR
    return df_result


def _filter_frame_to_months(
    frame: DataFrame,
    allowed_months: list[int] | None,
) -> DataFrame:
    """Restrict computed daily PET rows to the requested months only."""
    if not allowed_months or frame.empty:
        return frame

    allowed_month_set = set(allowed_months)
    return frame[pd.to_datetime(frame["date"]).dt.month.isin(allowed_month_set)].copy()


def _compute_location_frame(
    ds: Dataset,
    selected_cities: DataFrame,
    start_h: int,
    end_h: int,
    compute_workers: int,
) -> DataFrame:
    np = cast("Any", importlib.import_module("numpy"))
    dask = cast("Any", importlib.import_module("dask"))
    xr = cast("Any", importlib.import_module("xarray"))
    tf = cast("Any", importlib.import_module("thermofeel"))

    lats, lons = (
        selected_cities["lat"].values.astype(float),
        selected_cities["lng"].values.astype(float),
    )
    loc_ids = selected_cities["location_id"].values

    city_selection = (
        ds[ERA5_ALL_ARCO_VARIABLES]
        .sel(time=slice(start_h, end_h))
        .sel(
            latitude=xr.DataArray(lats, dims="location"),
            longitude=xr.DataArray(lons, dims="location"),
            method="nearest",
        )
    )
    with dask.config.set(scheduler="threads", num_workers=compute_workers):
        city_data = city_selection.compute()

    city_data = city_data.assign_coords(
        time=pd.to_datetime(city_data.time.values, unit="h", origin=ERA5_TIME_ORIGIN),
    )

    # xarray returns shape (n_times, n_locs); transpose to (n_locs, n_times) so
    # that all arrays share the same (loc-first) memory layout used for ravel,
    # cossza construction, and the DataFrame row ordering below.
    u10 = city_data["10u"].values.T
    v10 = city_data["10v"].values.T
    t2m = city_data["2t"].values.T
    d2m = city_data["2d"].values.T
    ssrd = city_data["ssrd"].values.T * RADIATION_SCALE
    strd = city_data["strd"].values.T * RADIATION_SCALE
    ssr = city_data["ssr"].values.T * RADIATION_SCALE
    strr = city_data["str"].values.T * RADIATION_SCALE

    # msdrswrf is flux (W/m2), NO scale needed
    # But wait, ARCO mean_* might be in W/m2.
    msdrswrf = city_data["msdrswrf"].values.T

    times = city_data.time.values

    n_locs, n_times = len(lats), len(times)
    cossza = np.empty((n_locs, n_times), dtype="float64")
    # Shift times backwards by 30 mins to get middle of accumulation hour
    cossza_times = pd.to_datetime(times) - pd.Timedelta(minutes=30)
    for i, (lat, lon) in enumerate(zip(lats, lons, strict=False)):
        cossza[i, :] = _calc_cossza(
            lat=float(lat),
            lon=float(lon),
            times=cossza_times.values,
        )

    temp_c, dew_c = t2m - 273.15, d2m - 273.15
    wind_speed = np.sqrt(u10**2 + v10**2)
    gamma_t, gamma_td = _B * temp_c / (_C + temp_c), _B * dew_c / (_C + dew_c)
    rh = np.exp(gamma_td - gamma_t) * 100.0

    # Since msdrswrf is the flux, let's use it directly instead of fdir!
    mrt_k = tf.calculate_mean_radiant_temperature(
        ssrd=ssrd.ravel(),
        ssr=ssr.ravel(),
        dsrp=msdrswrf.ravel(),
        strd=strd.ravel(),
        fdir=msdrswrf.ravel(),
        strr=strr.ravel(),
        cossza=cossza.ravel(),
    )
    mrt_c = (mrt_k - 273.15).reshape(n_locs, n_times)

    frame = pd.DataFrame(
        {
            "location_id": np.repeat(loc_ids, n_times),
            "time": np.tile(times, n_locs),
            "t": temp_c.ravel(),
            "v": wind_speed.ravel(),
            "rh": rh.ravel(),
            "mrt": mrt_c.ravel(),
        },
    ).dropna()

    # --- IN-MEMORY PET AGGREGATION ---
    df = frame[(frame["rh"] >= 1) & (frame["v"] > 0)].copy()
    for col in ["v", "t", "rh", "mrt"]:
        df[col] = (df[col] * PET_ROUNDING_FACTOR).round() / PET_ROUNDING_FACTOR

    df_distinct = df[["v", "t", "rh", "mrt"]].drop_duplicates().reset_index(drop=True)
    chunks = [
        df_distinct.iloc[i : i + CHUNK_SIZE]
        for i in range(0, len(df_distinct), CHUNK_SIZE)
    ]
    results = [_compute_pet_chunk(c) for c in chunks]

    if results:
        df_pet_unique = pd.concat(results, ignore_index=True)
        df = df.merge(df_pet_unique, on=["v", "t", "rh", "mrt"], how="inner")
        df["date"] = pd.to_datetime(df["time"]).dt.date
        return df.groupby(["location_id", "date"])["pet"].max().reset_index()
    return pd.DataFrame(columns=["location_id", "date", "pet"])


def _write_pet_partition(
    pet_root: str,
    year: int,
    city_shard_index: int,
    frame: DataFrame,
    batch_index: int,
) -> None:
    filesystem, base_path = resolve_filesystem(pet_root)
    partition_dir = f"{base_path}/year={year}"
    filesystem.create_dir(partition_dir, recursive=True)

    output_path = (
        f"{partition_dir}/pet_batch_{batch_index:04d}_{city_shard_index:02d}.parquet"
    )
    float_cols = frame.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        frame[float_cols] = frame[float_cols].astype("float32")

    table = pa.Table.from_pandas(frame, preserve_index=False)
    with filesystem.open_output_stream(output_path) as out_stream:
        pq.write_table(table, out_stream, compression="snappy")


def _pet_batch_exists(
    pet_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
) -> bool:
    """Check if a PET batch already exists in the output directory."""
    filesystem, base_path = resolve_filesystem(pet_root)
    path = f"{base_path}/year={year}/pet_batch_{batch_index:04d}_{city_shard_index:02d}.parquet"
    try:
        return filesystem.get_file_info(path).type != 0
    except Exception:  # noqa: BLE001
        return False


def _pending_batch_tiles(
    shard_df: DataFrame,
    *,
    era5_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    force: bool = False,
) -> DataFrame:
    """Return the shard_df rows that still need processing for this batch."""
    if force:
        return shard_df.copy()
    if _pet_batch_exists(era5_root, year, city_shard_index, batch_index):
        return pd.DataFrame(columns=shard_df.columns)
    return shard_df.copy()


def _process_era5_batch_job(
    *,
    ds: Dataset,
    pet_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    compute_workers: int,
    allowed_months: list[int] | None = None,
) -> int:
    """Execute ERA5 to PET calculation for a specific time batch."""
    LOGGER.info(
        "Computing ERA5->PET for year %s batch %s over %s cities.",
        year,
        batch_index,
        len(pending_batch_df),
    )
    frame = _compute_location_frame(
        ds,
        pending_batch_df,
        start_h=start_h,
        end_h=end_h,
        compute_workers=compute_workers,
    )
    frame = _filter_frame_to_months(frame, allowed_months)

    if frame.empty:
        return batch_index
    _write_pet_partition(pet_root, year, city_shard_index, frame, batch_index)
    return batch_index


def _process_era5_batch_with_thread_dataset(
    *,
    pet_root: str,
    year: int,
    city_shard_index: int,
    batch_index: int,
    start_h: int,
    end_h: int,
    pending_batch_df: DataFrame,
    compute_workers: int,
    allowed_months: list[int] | None = None,
) -> int:
    """Process batch using thread-local Zarr store."""
    for attempt in range(1, GCS_BATCH_MAX_RETRIES + 1):
        try:
            ds = getattr(ERA5_THREAD_LOCAL, "ds", None)
            if ds is None:
                _configure_concurrency(
                    os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced"),
                )
                ds = ERA5_THREAD_LOCAL.ds = _open_zarr_store()
            return _process_era5_batch_job(
                ds=ds,
                pet_root=pet_root,
                year=year,
                city_shard_index=city_shard_index,
                batch_index=batch_index,
                start_h=start_h,
                end_h=end_h,
                pending_batch_df=pending_batch_df,
                compute_workers=compute_workers,
                allowed_months=allowed_months,
            )
        except Exception:  # noqa: PERF203
            if attempt == GCS_BATCH_MAX_RETRIES:
                raise
            time.sleep(GCS_BATCH_RETRY_DELAY_SECONDS * attempt)
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
    allowed_months: list[int] | None = None,
    force: bool = False,
) -> bool:
    """Execute ERA5-to-PET batch jobs, returning True if any batches were dispatched."""
    concurrency_profile = os.environ.get("ERA5_CONCURRENCY_PROFILE", "balanced")
    dask_workers = (
        8
        if concurrency_profile == "aggressive"
        else (2 if concurrency_profile == "conservative" else 4)
    )

    batch_jobs: list[tuple[int, int, int, DataFrame]] = []
    for batch_index, start_h, end_h in selected_batches:
        pending_df = _pending_batch_tiles(
            shard_df,
            era5_root=era5_root,
            year=year,
            city_shard_index=city_shard_index,
            batch_index=batch_index,
            force=force,
        )
        if not pending_df.empty:
            batch_jobs.append((batch_index, start_h, end_h, pending_df))

    if not batch_jobs:
        return False

    LOGGER.info(
        "ERA5->PET year=%d city_shard=%d/%d time_shard=%d/%d: %d batch(es)",
        year,
        city_shard_index,
        city_shard_count,
        time_shard_index,
        time_shard_count,
        len(batch_jobs),
    )

    worker_count = max(1, min(_resolve_era5_max_workers(max_workers), len(batch_jobs)))

    if worker_count == 1:
        ds = _open_zarr_store()
        for b_idx, s_h, e_h, df in tqdm(batch_jobs, desc=f"ERA5->PET {year}"):
            _process_era5_batch_job(
                ds=ds,
                pet_root=era5_root,
                year=year,
                city_shard_index=city_shard_index,
                batch_index=b_idx,
                start_h=s_h,
                end_h=e_h,
                pending_batch_df=df,
                compute_workers=dask_workers,
                allowed_months=allowed_months,
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_era5_batch_with_thread_dataset,
                    pet_root=era5_root,
                    year=year,
                    city_shard_index=city_shard_index,
                    batch_index=b,
                    start_h=s,
                    end_h=e,
                    pending_batch_df=df,
                    compute_workers=dask_workers,
                    allowed_months=allowed_months,
                )
                for b, s, e, df in batch_jobs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"ERA5->PET {year}",
            ):
                future.result()

    return True


def process_era5(
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
) -> None:
    """Download ERA5, calculate MRT, and compute daily maximum PET."""
    _configure_concurrency(concurrency_profile)
    os.environ["ERA5_CONCURRENCY_PROFILE"] = concurrency_profile
    era5_root = f"{out_dir}/pet_data_csv"

    shard_df = _load_era5_city_shard(city_shard_index, city_shard_count)
    if shard_df.empty:
        return

    selected_batches = _select_time_shard_batches(
        _iter_time_batches(year, batch_hours, months),
        time_shard_index,
        time_shard_count,
    )

    _run_era5_batch_jobs(
        selected_batches=selected_batches,
        shard_df=shard_df,
        era5_root=era5_root,
        year=year,
        city_shard_index=city_shard_index,
        city_shard_count=city_shard_count,
        time_shard_index=time_shard_index,
        time_shard_count=time_shard_count,
        max_workers=max_workers,
        allowed_months=months,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        help="Only process specific months (1-12)",
    )
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
        default="aggressive",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    try:
        process_era5(
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
        )
    finally:
        # HARD EXIT: Bypasses Python's buggy aiohttp/asyncio weakref teardown
        # completely since all our data is already safely flushed to disk.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
