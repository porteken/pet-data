"""Delete all objects under an S3 prefix using batched s3api calls."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DELETE_BATCH_SIZE = 1000
DEFAULT_MAX_WORKERS = 8
logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def _aws_executable() -> str:
    aws_path = shutil.which("aws")
    if aws_path is None:
        msg = "aws CLI is required but was not found on PATH."
        raise RuntimeError(msg)
    return aws_path


def _run_aws_json(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603
        [_aws_executable(), *args, "--output", "json"],
        check=True,
        capture_output=True,
        text=True,
    )
    if not completed.stdout.strip():
        return {}
    return json.loads(completed.stdout)


def _delete_batch(bucket: str, objects: list[dict[str, str]]) -> int:
    payload = {"Objects": objects, "Quiet": True}
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle)
        handle.flush()
        payload_path = handle.name

    try:
        subprocess.run(  # noqa: S603
            [
                _aws_executable(),
                "s3api",
                "delete-objects",
                "--bucket",
                bucket,
                "--delete",
                f"file://{payload_path}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        Path(payload_path).unlink()

    return len(objects)


def _chunked(items: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _list_versioned_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    key_marker: str | None = None
    version_id_marker: str | None = None

    while True:
        args = [
            "s3api",
            "list-object-versions",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
        ]
        if key_marker is not None:
            args.extend(["--key-marker", key_marker])
        if version_id_marker is not None:
            args.extend(["--version-id-marker", version_id_marker])

        response = _run_aws_json(args)
        objects.extend(
            {
                "Key": entry["Key"],
                "VersionId": entry["VersionId"],
            }
            for entry in response.get("Versions", [])
        )
        objects.extend(
            {
                "Key": entry["Key"],
                "VersionId": entry["VersionId"],
            }
            for entry in response.get("DeleteMarkers", [])
        )

        if not response.get("IsTruncated"):
            return objects

        key_marker = response.get("NextKeyMarker")
        version_id_marker = response.get("NextVersionIdMarker")


def _list_current_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    continuation_token: str | None = None

    while True:
        args = [
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            prefix,
        ]
        if continuation_token is not None:
            args.extend(["--continuation-token", continuation_token])

        response = _run_aws_json(args)
        objects.extend({"Key": entry["Key"]} for entry in response.get("Contents", []))

        if not response.get("IsTruncated"):
            return objects

        continuation_token = response.get("NextContinuationToken")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete all S3 objects under a prefix using batched delete-objects calls.",
    )
    parser.add_argument("--bucket", required=True, help="Bucket name.")
    parser.add_argument("--prefix", required=True, help="Prefix to clear.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent delete-objects calls.",
    )
    parser.add_argument(
        "--include-versions",
        action="store_true",
        help=(
            "Delete all object versions and delete markers under the prefix. "
            "By default, only current objects are deleted, matching "
            "`aws s3 rm --recursive` behavior on a versioned bucket."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Delete every object under the requested S3 prefix."""
    args = _parse_args()
    if args.max_workers < 1:
        msg = "--max-workers must be >= 1"
        raise SystemExit(msg)

    objects = (
        _list_versioned_objects(args.bucket, args.prefix)
        if args.include_versions
        else _list_current_objects(args.bucket, args.prefix)
    )
    if not objects:
        LOGGER.info("No S3 objects found under s3://%s/%s", args.bucket, args.prefix)
        return

    deleted_count = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(_delete_batch, args.bucket, batch)
            for batch in _chunked(objects, DELETE_BATCH_SIZE)
        ]
        for future in futures:
            deleted_count += future.result()

    LOGGER.info(
        "Deleted %s S3 entries under s3://%s/%s",
        deleted_count,
        args.bucket,
        args.prefix,
    )


if __name__ == "__main__":
    main()
