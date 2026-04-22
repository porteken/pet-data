"""Tests for CSV discovery and DB loading."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from load import (
    TABLE_NAMES,
    _discover_csv_inputs,
    _discover_pet_csv_paths,
    _extract_partition_marker,
    _filter_paths_by_partition_value,
    _select_partition_shard_paths,
    _validate_load_shard_args,
)


class TestFilterPathsByPartitionValue:
    def test_filters_partitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "analytics"
        p1 = root / "shard_index=00000" / "data.parquet"
        p2 = root / "shard_index=00001" / "data.parquet"
        p1.parent.mkdir(parents=True)
        p2.parent.mkdir(parents=True)
        p1.touch()
        p2.touch()

        result = _filter_paths_by_partition_value(
            [p1, p2],
            root_path=root,
            partition_key="shard_index",
            partition_value="00000",
        )
        assert result == [p1]

    def test_handles_unpartitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "analytics"
        root.mkdir()
        p1 = tmp_path / "data.parquet"
        p1.touch()

        result0 = _filter_paths_by_partition_value(
            [p1], root_path=root, partition_key="shard_index", partition_value="00000"
        )
        assert result0 == [p1]

        result1 = _filter_paths_by_partition_value(
            [p1], root_path=root, partition_key="shard_index", partition_value="00001"
        )
        assert result1 == []


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
        from typing import Any, cast

        import pandas as pd

        pd.DataFrame(
            columns=cast("Any", ["year", "location_id", "p10", "p90"])
        ).to_parquet(shard_dir / "percentiles.parquet", index=False)

        result = _discover_csv_inputs(
            "percentiles.parquet",
            shard_root=str(tmp_path / "analytics"),
            shard_file_name="percentiles.parquet",
            shard_count=20,
            shard_partition_key="shard_count",
        )
        assert len(result) == 1

    def test_no_files_found(self, tmp_path: Path) -> None:
        result = _discover_csv_inputs(tmp_path / "nonexistent.parquet")
        assert result == []


class TestDiscoverPetCsvPaths:
    def test_prefers_direct_csv_when_requested(self, tmp_path: Path) -> None:
        pet_root = tmp_path / "pet_data_csv"
        shard_dir = pet_root / "year=2024"
        shard_dir.mkdir(parents=True)
        (shard_dir / "pet_batch_0000_00.parquet").touch()

        pet_csv = tmp_path / "pet_full.csv"
        pet_csv.write_text("location_id,date,pet\n1,2024-05-01,25.0\n")

        args = argparse.Namespace(
            pet_root=str(pet_root),
            pet_csv=str(pet_csv),
            load_shard_index=0,
            load_shard_count=1,
            prefer_pet_csv=True,
        )

        assert _discover_pet_csv_paths(args) == [pet_csv]


class TestSelectPartitionShardPaths:
    def test_shards_partitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pet_data"
        p1 = root / "year=2020" / "data.csv"
        p2 = root / "year=2021" / "data.csv"
        p1.parent.mkdir(parents=True)
        p2.parent.mkdir(parents=True)
        p1.touch()
        p2.touch()

        shard0 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=0, shard_count=2
        )
        assert shard0 == [p1]

        shard1 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=1, shard_count=2
        )
        assert shard1 == [p2]

    def test_handles_unpartitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pet_data"
        root.mkdir()
        p1 = tmp_path / "pet.csv"
        p1.touch()

        shard0 = _select_partition_shard_paths(
            [p1], root_path=root, partition_key="year", shard_index=0, shard_count=2
        )
        assert shard0 == [p1]

        shard1 = _select_partition_shard_paths(
            [p1], root_path=root, partition_key="year", shard_index=1, shard_count=2
        )
        assert shard1 == []

    def test_distributes_multiple_unpartitioned_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pet_data"
        root.mkdir()
        p1 = tmp_path / "pet_1.csv"
        p2 = tmp_path / "pet_2.csv"
        p1.touch()
        p2.touch()

        shard0 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=0, shard_count=2
        )
        shard1 = _select_partition_shard_paths(
            [p1, p2], root_path=root, partition_key="year", shard_index=1, shard_count=2
        )

        assert len(shard0) == 1
        assert len(shard1) == 1
        assert shard0 != shard1


class TestTableNames:
    def test_expected_tables(self) -> None:
        assert "locations" in TABLE_NAMES
        assert "pet" in TABLE_NAMES
        assert "pet_forecast" in TABLE_NAMES
        assert "pet_percentiles" not in TABLE_NAMES
        assert "pet_change" not in TABLE_NAMES
