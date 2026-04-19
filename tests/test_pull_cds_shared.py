"""Tests for pull_cds_shared.py."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pull_cds_shared import partition_file_max_timestamp

if TYPE_CHECKING:
    import pytest


class TestPartitionFileMaxTimestamp:
    def test_missing_file_returns_none_without_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)

        result = partition_file_max_timestamp(
            str(tmp_path),
            "year=2024/month=05/tile_id=7",
            "weather.parquet",
        )

        assert result is None
        assert not caplog.records
