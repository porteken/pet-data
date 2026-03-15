"""Cancel active CDS retrieval jobs before starting new pull batches."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, cast

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_CDS_API_URL = "https://cds.climate.copernicus.eu/api"
ACTIVE_JOB_STATES = {"accepted", "queued", "running"}
REQUEST_TIMEOUT_SECONDS = 30
HTTP_NO_CONTENT = 204
HTTP_NOT_FOUND = 404


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cancel active CDS jobs for the configured API token.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("CDSAPI_URL", DEFAULT_CDS_API_URL),
        help="CDS API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CDSAPI_KEY"),
        help="CDS API bearer token. Defaults to CDSAPI_KEY.",
    )
    return parser.parse_args()


def _authorization_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _list_jobs(api_url: str, *, api_key: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{api_url.rstrip('/')}/retrieve/v1/jobs",
        headers=_authorization_headers(api_key),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        msg = "Unexpected CDS jobs payload."
        raise TypeError(msg)

    typed_jobs = cast("list[Any]", jobs)
    return [job for job in typed_jobs if isinstance(job, dict)]


def _cancel_job(api_url: str, *, api_key: str, job_id: str) -> bool:
    response = requests.delete(
        f"{api_url.rstrip('/')}/retrieve/v1/jobs/{job_id}",
        headers=_authorization_headers(api_key),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == HTTP_NO_CONTENT:
        return True
    if response.status_code == HTTP_NOT_FOUND:
        LOGGER.info("CDS job %s was already gone.", job_id)
        return False
    response.raise_for_status()
    return False


def main() -> None:
    """Run the main entry point for the CDS job cancellation script."""
    args = _parse_args()
    if not args.api_key:
        msg = "CDS API key is required via --api-key or CDSAPI_KEY."
        raise SystemExit(msg)

    jobs = _list_jobs(args.api_url, api_key=args.api_key)
    active_jobs = [
        job for job in jobs if str(job.get("status", "")).lower() in ACTIVE_JOB_STATES
    ]
    if not active_jobs:
        LOGGER.info("No active CDS jobs found.")
        return

    LOGGER.info("Cancelling %s active CDS jobs.", len(active_jobs))
    cancelled_count = 0
    for job in active_jobs:
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            continue
        if _cancel_job(args.api_url, api_key=args.api_key, job_id=job_id):
            cancelled_count += 1
            LOGGER.info("Cancelled CDS job %s.", job_id)

    LOGGER.info("Cancelled %s CDS jobs.", cancelled_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
