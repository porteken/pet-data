"""Tests for pull_cds_shared.py — CDS retry logic and partition helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pull_cds_shared import (
    QUEUE_LIMIT_REJECTION_TEXT,
    _is_empty_rejection_message,
    _is_queue_limit_message,
    _is_retryable_rejection,
    _wait_for_completion,
    extract_files,
    partition_exists,
    partition_file_exists,
)


class TestIsQueueLimitMessage:
    def test_recognizes_queue_limit(self) -> None:
        assert _is_queue_limit_message(QUEUE_LIMIT_REJECTION_TEXT)

    def test_recognizes_rate_limit(self) -> None:
        assert _is_queue_limit_message("Rate limit exceeded for this endpoint")

    def test_rejects_unrelated_message(self) -> None:
        assert not _is_queue_limit_message("Something else went wrong")


class TestIsEmptyRejectionMessage:
    def test_recognizes_no_reason(self) -> None:
        assert _is_empty_rejection_message(
            "ended in state rejected: no reason provided"
        )

    def test_recognizes_failed_no_reason(self) -> None:
        assert _is_empty_rejection_message("ended in state failed: no reason provided")

    def test_rejects_unrelated(self) -> None:
        assert not _is_empty_rejection_message("data not available")


class TestIsRetryableRejection:
    def test_queue_limit_is_retryable(self) -> None:
        exc = RuntimeError(QUEUE_LIMIT_REJECTION_TEXT)
        assert _is_retryable_rejection(exc)

    def test_generic_error_is_not_retryable(self) -> None:
        exc = RuntimeError("Unknown server error")
        assert not _is_retryable_rejection(exc)


class TestWaitForCompletion:
    def test_completed_immediately(self) -> None:
        mock_result = MagicMock()
        mock_result.reply = {"state": "completed", "request_id": "abc123"}
        result = _wait_for_completion(mock_result, name="test")
        assert result is mock_result

    def test_failed_state_raises(self) -> None:
        mock_result = MagicMock()
        mock_result.reply = {
            "state": "failed",
            "request_id": "abc123",
            "reason": "server error",
        }
        with pytest.raises(RuntimeError, match="failed"):
            _wait_for_completion(mock_result, name="test")

    def test_unknown_state_raises(self) -> None:
        mock_result = MagicMock()
        mock_result.reply = {"state": "exploded", "request_id": "abc123"}
        with pytest.raises(RuntimeError, match="unknown state"):
            _wait_for_completion(mock_result, name="test")

    def test_queue_limit_failure_raises_for_retry(self) -> None:
        mock_result = MagicMock()
        mock_result.reply = {
            "state": "failed",
            "request_id": "abc123",
            "error": {"message": QUEUE_LIMIT_REJECTION_TEXT},
        }
        with pytest.raises(RuntimeError, match="limited"):
            _wait_for_completion(mock_result, name="test")


class TestPartitionExists:
    def test_existing_partition(self, tmp_path: Path) -> None:
        part_dir = tmp_path / "year=2020" / "tile_id=1"
        part_dir.mkdir(parents=True)
        assert partition_exists(str(tmp_path), "year=2020/tile_id=1")

    def test_missing_partition(self, tmp_path: Path) -> None:
        assert not partition_exists(str(tmp_path), "year=2099/tile_id=999")


class TestPartitionFileExists:
    def test_existing_file(self, tmp_path: Path) -> None:
        part_dir = tmp_path / "year=2020"
        part_dir.mkdir(parents=True)
        (part_dir / "data.parquet").write_text("fake")
        assert partition_file_exists(str(tmp_path), "year=2020", "data.parquet")

    def test_missing_file(self, tmp_path: Path) -> None:
        assert not partition_file_exists(str(tmp_path), "year=2020", "data.parquet")


class TestExtractFiles:
    def test_returns_download_path_if_matching_suffix(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv.gz"
        csv_file.write_text("a,b\n1,2\n")
        result = extract_files(csv_file, suffix=".csv.gz")
        assert len(result) == 1
        assert result[0] == csv_file

    def test_returns_empty_for_wrong_suffix(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv.gz"
        csv_file.write_text("a,b\n1,2\n")
        result = extract_files(csv_file, suffix=".nc")
        assert result == []
