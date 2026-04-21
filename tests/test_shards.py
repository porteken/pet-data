"""Tests for shard management."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shards import (
    ShardKey,
    discover_common_shards,
    discover_parquet_shards,
    read_parquet_files,
    resolve_filesystem,
    select_shards,
)


class TestShardKey:
    def test_partition_path_no_month(self) -> None:
        key = ShardKey(year=2020, tile_id=5)
        assert key.partition_path == "year=2020/tile_id=5"

    def test_partition_path_with_month(self) -> None:
        key = ShardKey(year=2020, tile_id=5, month=3)
        assert key.partition_path == "year=2020/month=03/tile_id=5"

    def test_label_no_month(self) -> None:
        key = ShardKey(year=2020, tile_id=5)
        assert key.label == "2020-tile-005"

    def test_label_with_month(self) -> None:
        key = ShardKey(year=2020, tile_id=5, month=3)
        assert key.label == "2020-03-tile-005"

    def test_ordering(self) -> None:
        a = ShardKey(year=2020, tile_id=1)
        b = ShardKey(year=2020, tile_id=2)
        c = ShardKey(year=2021, tile_id=1)
        assert sorted([c, b, a]) == [a, b, c]

    def test_frozen(self) -> None:
        key = ShardKey(year=2020, tile_id=5)
        with pytest.raises(AttributeError):
            key.year = 2021  # type: ignore[misc]


class TestResolveFilesystem:
    def test_local_path(self, tmp_path: Path) -> None:
        _, root = resolve_filesystem(tmp_path)
        assert root == str(tmp_path.resolve())


class TestDiscoverParquetShards:
    def test_empty_directory(self, tmp_path: Path) -> None:
        result = discover_parquet_shards(tmp_path)
        assert result == {}

    def test_discovers_year_tile_shards(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(shard_dir / "data.parquet")

        result = discover_parquet_shards(tmp_path)
        assert ShardKey(year=2020, tile_id=1) in result
        assert len(result[ShardKey(year=2020, tile_id=1)]) == 1

    def test_discovers_year_month_tile_shards(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "month=03" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(shard_dir / "data.parquet")

        result = discover_parquet_shards(tmp_path)
        assert ShardKey(year=2020, tile_id=1, month=3) in result

    def test_ignores_non_parquet_files(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "year=2020" / "tile_id=1"
        shard_dir.mkdir(parents=True)
        (shard_dir / "notes.txt").write_text("hello")

        result = discover_parquet_shards(tmp_path)
        assert result == {}


class TestDiscoverCommonShards:
    def test_returns_intersection(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        for root in [root_a, root_b]:
            shard_dir = root / "year=2020" / "tile_id=1"
            shard_dir.mkdir(parents=True)
            pd.DataFrame({"x": [1]}).to_parquet(shard_dir / "data.parquet")

        shard_dir_a_only = root_a / "year=2020" / "tile_id=2"
        shard_dir_a_only.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(shard_dir_a_only / "data.parquet")

        common = discover_common_shards(root_a, root_b)
        assert len(common) == 1
        assert common[0] == ShardKey(year=2020, tile_id=1)

    def test_empty_when_no_roots(self) -> None:
        assert discover_common_shards() == []


class TestSelectShards:
    @pytest.fixture()
    def sample_keys(self) -> list[ShardKey]:
        return [
            ShardKey(year=2020, tile_id=1),
            ShardKey(year=2020, tile_id=2),
            ShardKey(year=2020, tile_id=3),
            ShardKey(year=2021, tile_id=1),
        ]

    def test_no_filters(self, sample_keys: list[ShardKey]) -> None:
        result = select_shards(sample_keys)
        assert len(result) == 4

    def test_filter_by_year(self, sample_keys: list[ShardKey]) -> None:
        result = select_shards(sample_keys, year=2020)
        assert all(k.year == 2020 for k in result)
        assert len(result) == 3

    def test_filter_by_tile_ids(self, sample_keys: list[ShardKey]) -> None:
        result = select_shards(sample_keys, tile_ids=[1])
        assert all(k.tile_id == 1 for k in result)
        assert len(result) == 2

    def test_sharding(self, sample_keys: list[ShardKey]) -> None:
        shard_0 = select_shards(sample_keys, shard_index=0, shard_count=2)
        shard_1 = select_shards(sample_keys, shard_index=1, shard_count=2)
        assert len(shard_0) + len(shard_1) == 4
        assert set(shard_0).isdisjoint(set(shard_1))

    def test_invalid_shard_count(self, sample_keys: list[ShardKey]) -> None:
        with pytest.raises(ValueError, match="shard_count"):
            select_shards(sample_keys, shard_count=0)

    def test_invalid_shard_index(self, sample_keys: list[ShardKey]) -> None:
        with pytest.raises(ValueError, match="shard_index"):
            select_shards(sample_keys, shard_index=2, shard_count=2)


class TestReadParquetFiles:
    def test_reads_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "year=2020" / "tile_id=1"
        path.mkdir(parents=True)
        file_path = path / "data.parquet"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(file_path)

        df = read_parquet_files(tmp_path, [str(file_path)], columns=["a"])
        assert list(df.columns) == ["a"]
        assert len(df) == 2
