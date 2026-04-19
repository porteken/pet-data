"""Utilities for discovering and selecting partitioned dataset shards."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast, overload

if TYPE_CHECKING:
    from collections.abc import Iterable


class FileInfo(Protocol):
    """Protocol for file info objects."""

    type: int
    path: str


class FileSystem(Protocol):
    """Protocol for filesystem objects."""

    @overload
    def get_file_info(self, paths: str) -> FileInfo: ...

    @overload
    def get_file_info(self, paths: list[str] | object) -> list[FileInfo]: ...

    def get_file_info(
        self,
        paths: str | list[str] | object,
    ) -> FileInfo | list[FileInfo]:
        """Get info for one or more files."""
        ...

    def open_input_file(self, path: str) -> Any:
        """Open a file for reading."""
        ...

    def open_output_stream(self, path: str) -> Any:
        """Open a file for writing."""
        ...

    def create_dir(self, path: str, recursive: bool = True) -> None:
        """Create a directory."""
        ...

    def move(self, src: str, dest: str) -> None:
        """Move a file or directory."""
        ...


pa = cast("Any", import_module("pyarrow"))
dataset_module = cast("Any", import_module("pyarrow.dataset"))
fs_module = cast("Any", import_module("pyarrow.fs"))

DataFrame: TypeAlias = Any
LOGGER = logging.getLogger(__name__)
ShardMapping: TypeAlias = dict["ShardKey", list[str]]
READ_PARQUET_MAX_ATTEMPTS = 3
READ_PARQUET_RETRY_DELAY_SECONDS = 1.0


@dataclass(frozen=True, order=True)
class ShardKey:
    """A logical year/month/tile shard key."""

    year: int
    tile_id: int
    month: int = 0

    @property
    def partition_path(self) -> str:
        """Return the canonical partition path for this shard."""
        if self.month:
            return f"year={self.year}/month={self.month:02d}/tile_id={self.tile_id}"
        return f"year={self.year}/tile_id={self.tile_id}"

    @property
    def label(self) -> str:
        """Return a human-readable shard label."""
        if self.month:
            return f"{self.year}-{self.month:02d}-tile-{self.tile_id:03d}"
        return f"{self.year}-tile-{self.tile_id:03d}"


def resolve_filesystem(base_uri: str | Path) -> tuple[FileSystem, str]:
    """Return a filesystem object plus normalized root path."""
    uri_text = str(base_uri)
    if "://" in uri_text:
        return fs_module.FileSystem.from_uri(uri_text)

    root_path = str(Path(uri_text).expanduser().resolve())
    return fs_module.LocalFileSystem(), root_path


def discover_parquet_shards(base_uri: str | Path) -> ShardMapping:
    """Map each discovered shard to the parquet files it contains."""
    filesystem, root_path = resolve_filesystem(base_uri)
    try:
        file_infos = filesystem.get_file_info(
            fs_module.FileSelector(root_path, recursive=True),
        )
    except (OSError, pa.ArrowException) as exc:
        LOGGER.warning("Could not list parquet files under %s: %s", base_uri, exc)
        return {}

    shards: ShardMapping = {}
    for file_info in file_infos:
        if file_info.type != fs_module.FileType.File or not file_info.path.endswith(
            ".parquet",
        ):
            continue

        shard_key = _parse_shard_key(root_path, file_info.path)
        if shard_key is None:
            continue
        shards.setdefault(shard_key, []).append(file_info.path)

    for file_paths in shards.values():
        file_paths.sort()

    return shards


def discover_common_shards(*roots: str | Path) -> list[ShardKey]:
    """Return shard keys that exist under every provided parquet root."""
    discovered: list[set[ShardKey]] = [
        set(discover_parquet_shards(root)) for root in roots
    ]
    if not discovered:
        return []

    common_shards = set(discovered[0])
    for shard_keys in discovered[1:]:
        common_shards &= shard_keys

    return sorted(common_shards)


def select_shards(
    shard_keys: Iterable[ShardKey],
    *,
    year: int | None = None,
    tile_ids: Iterable[int] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[ShardKey]:
    """Filter and evenly shard a list of shard keys."""
    if shard_count < 1:
        msg = "shard_count must be >= 1"
        raise ValueError(msg)
    if shard_index < 0 or shard_index >= shard_count:
        msg = f"shard_index must be between 0 and {shard_count - 1}"
        raise ValueError(msg)

    allowed_tiles = None if tile_ids is None else {int(tile_id) for tile_id in tile_ids}
    filtered = [
        shard_key
        for shard_key in sorted(shard_keys)
        if (year is None or shard_key.year == year)
        and (allowed_tiles is None or shard_key.tile_id in allowed_tiles)
    ]
    return [
        shard_key
        for position, shard_key in enumerate(filtered)
        if position % shard_count == shard_index
    ]


def read_parquet_files(
    base_uri: str | Path,
    file_paths: list[str],
    *,
    columns: list[str] | None = None,
    filters: object | None = None,
) -> DataFrame:
    """Read a set of parquet files from a shared filesystem into a DataFrame."""
    filesystem, _ = resolve_filesystem(base_uri)
    for attempt in range(1, READ_PARQUET_MAX_ATTEMPTS + 1):
        try:
            dataset = dataset_module.dataset(
                file_paths,
                filesystem=filesystem,
                format="parquet",
            )
            table = dataset.to_table(columns=columns, filter=filters)
            return table.to_pandas()
        except FileNotFoundError:  # noqa: PERF203
            if attempt == READ_PARQUET_MAX_ATTEMPTS:
                raise

            missing_paths = _missing_file_paths(filesystem, file_paths)
            LOGGER.warning(
                "Parquet read hit a transient missing file under %s (attempt %s/%s). "
                "Missing paths: %s. Retrying in %.1f seconds.",
                base_uri,
                attempt,
                READ_PARQUET_MAX_ATTEMPTS,
                missing_paths or file_paths,
                READ_PARQUET_RETRY_DELAY_SECONDS,
            )
            time.sleep(READ_PARQUET_RETRY_DELAY_SECONDS)

    msg = f"Unable to read parquet files from {base_uri}: {file_paths}"
    raise FileNotFoundError(msg)


def _missing_file_paths(filesystem: FileSystem, file_paths: list[str]) -> list[str]:
    file_infos = cast("list[FileInfo]", filesystem.get_file_info(file_paths))
    return [
        file_info.path
        for file_info in file_infos
        if file_info.type == fs_module.FileType.NotFound
    ]


def _parse_shard_key(root_path: str, file_path: str) -> ShardKey | None:
    relative_path = file_path.removeprefix(root_path).lstrip("/")
    partitions: dict[str, str] = {}
    for part in PurePosixPath(relative_path).parts:
        if "=" not in part:
            continue
        key, value = part.split("=", maxsplit=1)
        partitions[key] = value

    try:
        return ShardKey(
            year=int(partitions["year"]),
            month=int(partitions.get("month", "0")),
            tile_id=int(partitions["tile_id"]),
        )
    except (KeyError, ValueError):
        return None
