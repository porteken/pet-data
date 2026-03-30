"""Tests for load.py — CSV discovery and DB loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from load import (
    TABLE_NAMES,
    _discover_csv_inputs,
    _extract_partition_marker,
    _validate_load_shard_args,
)


class TestValidateLoadShardArgs:
    def test_valid_args(self) -> None:
        _validate_load_shard_args(0, 1)
        _validate_load_shard_args(0, 5)
        _validate_load_shard_args(4, 5)

    def test_invalid_shard_count(self) -> None:
        with pytest.raises(ValueError, match="shard_count"):
            _validate_load_shard_args(0, 0)

    def test_invalid_shard_index(self) -> None:
        with pytest.raises(ValueError, match="shard_index"):
            _validate_load_shard_args(5, 5)

        with pytest.raises(ValueError, match="shard_index"):
            _validate_load_shard_args(-1, 5)


class TestExtractPartitionMarker:
    def test_extracts_value(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "shard_count=00020" / "shard_index=00003" / "data.csv.gz"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        result = _extract_partition_marker(csv_path, tmp_path, "shard_count")
        assert result == "00020"

    def test_returns_none_for_missing_key(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "other=5" / "data.csv.gz"
        result = _extract_partition_marker(csv_path, tmp_path, "shard_count")
        assert result is None


class TestDiscoverCsvInputs:
    def test_direct_path_fallback(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "pet.csv.gz"
        csv_file.write_text("location_id,date,pet\n1,2020-01-01,10\n")
        result = _discover_csv_inputs(csv_file)
        assert len(result) == 1
        assert result[0] == csv_file

    def test_shard_root_discovery(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "analytics" / "shard_count=00020" / "shard_index=00000"
        shard_dir.mkdir(parents=True)
        (shard_dir / "percentiles.csv.gz").write_text("year,location_id,p10,p90\n")

        result = _discover_csv_inputs(
            "percentiles.csv.gz",
            shard_root=str(tmp_path / "analytics"),
            shard_file_name="percentiles.csv.gz",
            shard_count=20,
            shard_partition_key="shard_count",
        )
        assert len(result) == 1

    def test_no_files_found(self, tmp_path: Path) -> None:
        result = _discover_csv_inputs(tmp_path / "nonexistent.csv.gz")
        assert result == []


class TestTableNames:
    def test_expected_tables(self) -> None:
        assert "locations" in TABLE_NAMES
        assert "pet" in TABLE_NAMES
        assert "pet_percentiles" in TABLE_NAMES
        assert "pet_forecast" in TABLE_NAMES
        assert "pet_change" in TABLE_NAMES
