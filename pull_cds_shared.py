"""Shared CDS download helpers for weather and MRT pull scripts."""

from __future__ import annotations

import importlib
import logging
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast
from zipfile import ZipFile, is_zipfile

from shards import resolve_filesystem

DataFrame: TypeAlias = Any
SeriesLike: TypeAlias = Any

if TYPE_CHECKING:
    from collections.abc import Callable


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


pa = cast("Any", importlib.import_module("pyarrow"))
pd = cast("Any", importlib.import_module("pandas"))
pq = cast("Any", importlib.import_module("pyarrow.parquet"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

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

    if download_path.suffix == suffix:
        return [download_path]
    return []
