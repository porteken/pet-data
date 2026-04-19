"""Load generated parquet inputs into the database and manage views."""

from __future__ import annotations

import argparse
import io
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2
import pyarrow.parquet as pq
from psycopg2 import sql

if TYPE_CHECKING:
    from psycopg2.extensions import connection

LOGGER = logging.getLogger(__name__)
DB_URI = os.getenv("SUPABASE_DB_URI")
COPY_BATCH_SIZE = 50_000
TABLE_NAMES = [
    "locations",
    "pet",
    "pet_percentiles",
    "pet_forecast",
    "pet_change",
]


def execute_sql_file(conn: connection, file_path: str | Path) -> None:
    """Execute a raw SQL file."""
    path = Path(file_path)
    if not path.exists():
        LOGGER.warning("SQL file %s not found. Skipping.", file_path)
        return

    LOGGER.info("Executing SQL file: %s...", file_path)
    with conn.cursor() as cur, path.open("r", encoding="utf-8") as f:
        cur.execute(f.read())
    LOGGER.info("Successfully executed %s.", file_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load shard-aware PET CSV outputs into the database and recreate views."
        ),
    )
    parser.add_argument("--cities-csv", default="cities.csv")
    parser.add_argument("--pet-csv", default="pet.csv")
    parser.add_argument("--pet-root", default="pet_data_csv")
    parser.add_argument("--analytics-root", default="analytics_data_csv")
    parser.add_argument("--analytics-shard-count", type=int, default=20)
    parser.add_argument("--load-shard-index", type=int, default=0)
    parser.add_argument("--load-shard-count", type=int, default=1)
    parser.add_argument("--copy-batch-size", type=int, default=COPY_BATCH_SIZE)
    parser.add_argument(
        "--append-only",
        action="store_true",
        help="Append rows without truncating destination tables first.",
    )
    parser.add_argument(
        "--skip-drop-views",
        action="store_true",
        help="Leave existing views in place before loading data.",
    )
    parser.add_argument(
        "--skip-create-views",
        action="store_true",
        help="Do not recreate SQL views after loading data.",
    )
    parser.add_argument(
        "--truncate-table",
        dest="truncate_tables",
        action="append",
        choices=TABLE_NAMES,
        help=(
            "Table to truncate before loading. Defaults to truncating all tables "
            "to preserve full rebuild behavior."
        ),
    )
    parser.add_argument(
        "--skip-table",
        dest="skip_tables",
        action="append",
        choices=TABLE_NAMES,
        help="Table to leave untouched during this run.",
    )
    return parser.parse_args()


def _discover_csv_inputs(
    direct_path: str | Path,
    *,
    shard_root: str | Path | None = None,
    shard_file_name: str | None = None,
    shard_count: int | None = None,
    shard_partition_key: str | None = None,
) -> list[Path]:
    shard_paths = _discover_shard_csv_inputs(
        shard_root=shard_root,
        shard_file_name=shard_file_name,
        shard_count=shard_count,
        shard_partition_key=shard_partition_key,
    )
    if shard_paths:
        return shard_paths

    direct_csv_path = Path(direct_path)
    if direct_csv_path.exists():
        return [direct_csv_path]

    return []


def _discover_shard_csv_inputs(
    *,
    shard_root: str | Path | None,
    shard_file_name: str | None,
    shard_count: int | None,
    shard_partition_key: str | None,
) -> list[Path]:
    if shard_root is None or shard_file_name is None:
        return []

    root_path = Path(shard_root)
    if not root_path.exists():
        return []

    if shard_partition_key is None:
        return sorted(root_path.rglob(shard_file_name))

    if shard_count is not None:
        selected_root = root_path / f"shard_count={shard_count:05d}"
        return sorted(selected_root.rglob(shard_file_name))

    shard_groups: dict[str, list[Path]] = {}
    for csv_path in sorted(root_path.rglob(shard_file_name)):
        shard_group = _extract_partition_marker(
            csv_path,
            root_path,
            shard_partition_key,
        )
        if shard_group is None:
            continue
        shard_groups.setdefault(shard_group, []).append(csv_path)

    if not shard_groups:
        return []
    if len(shard_groups) > 1:
        group_labels = ", ".join(sorted(shard_groups))
        msg = (
            f"Found multiple analytics shard groups for {shard_file_name}: "
            f"{group_labels}. Pass --analytics-shard-count to select one."
        )
        raise RuntimeError(msg)

    return next(iter(shard_groups.values()))


def _extract_partition_marker(
    csv_path: Path,
    root_path: Path,
    partition_key: str,
) -> str | None:
    try:
        relative_parts = csv_path.relative_to(root_path).parts
    except ValueError:
        return None

    prefix = f"{partition_key}="
    for part in relative_parts:
        if part.startswith(prefix):
            return part.removeprefix(prefix)
    return None


def _validate_load_shard_args(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        msg = "load_shard_count must be >= 1"
        raise ValueError(msg)
    if shard_index < 0 or shard_index >= shard_count:
        msg = f"load_shard_index must be between 0 and {shard_count - 1}"
        raise ValueError(msg)


def _partition_sort_key(partition_value: str) -> tuple[int, object]:
    try:
        return (0, int(partition_value))
    except ValueError:
        return (1, partition_value)


def _select_partition_shard_paths(
    csv_paths: list[Path],
    *,
    root_path: Path,
    partition_key: str,
    shard_index: int,
    shard_count: int,
) -> list[Path]:
    _validate_load_shard_args(shard_index, shard_count)
    if shard_count == 1:
        return csv_paths

    grouped_paths: dict[str, list[Path]] = {}
    for csv_path in csv_paths:
        partition_value = _extract_partition_marker(csv_path, root_path, partition_key)
        if partition_value is None:
            # Fallback to a special value for non-partitioned files.
            # We include the filename to allow multiple non-partitioned files
            # to be distributed across shards if there are many of them.
            partition_value = f"__unpartitioned__:{csv_path.name}"
        grouped_paths.setdefault(partition_value, []).append(csv_path)

    selected_partition_values = {
        partition_value
        for position, partition_value in enumerate(
            sorted(grouped_paths, key=_partition_sort_key),
        )
        if position % shard_count == shard_index
    }
    return [
        csv_path
        for partition_value in sorted(grouped_paths, key=_partition_sort_key)
        if partition_value in selected_partition_values
        for csv_path in grouped_paths[partition_value]
    ]


def _filter_paths_by_partition_value(
    csv_paths: list[Path],
    *,
    root_path: Path,
    partition_key: str,
    partition_value: str,
) -> list[Path]:
    filtered_paths: list[Path] = []
    for csv_path in csv_paths:
        marker = _extract_partition_marker(csv_path, root_path, partition_key)
        if marker is None:
            # If the file is not partitioned, we treat it as belonging to
            # the first shard (index 0) of that partition key.
            # This allows single files to be loaded in a sharded environment.
            if partition_value in ("0", "00000"):
                filtered_paths.append(csv_path)
            continue
        if marker == partition_value:
            filtered_paths.append(csv_path)
    return filtered_paths


def _copy_parquet_file_in_batches(
    conn: connection,
    table_name: str,
    parquet_path: Path,
    *,
    batch_size: int,
) -> int:
    parquet_file = pq.ParquetFile(str(parquet_path))
    column_names = parquet_file.schema_arrow.names
    copy_statement = sql.SQL("COPY {} ({}) FROM STDIN WITH CSV").format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(sql.Identifier(col) for col in column_names),
    )

    total_rows = 0
    with conn.cursor() as cur:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_df = batch.to_pandas()
            csv_buffer = io.StringIO()
            batch_df.to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)
            cur.copy_expert(copy_statement.as_string(conn), csv_buffer)
            total_rows += cur.rowcount
    return total_rows


def _copy_csv_file_in_batches(
    conn: connection,
    table_name: str,
    csv_path: Path,
    *,
    batch_size: int,
) -> int:
    """Load a plain CSV file (no compression) into a table via COPY."""
    with conn.cursor() as cur, csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(f, None)
        if header is None:
            LOGGER.warning("CSV file %s is empty. Skipping.", csv_path)
            return 0
        column_names = [col.strip() for col in header.split(",")]
        copy_statement = sql.SQL("COPY {} ({}) FROM STDIN WITH CSV").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in column_names),
        )
        total_rows = 0
        while True:
            lines = list(f.readlines(batch_size))
            if not lines:
                break
            cur.copy_expert(copy_statement.as_string(conn), io.StringIO("".join(lines)))
            total_rows += cur.rowcount
    return total_rows


def bulk_insert_csv_files(
    conn: connection,
    csv_paths: list[Path],
    table_name: str,
    *,
    batch_size: int,
    truncate: bool,
) -> None:
    """Load one or more parquet (or CSV) files into a destination table."""
    if not csv_paths:
        LOGGER.warning("No data inputs found for %s. Skipping.", table_name)
        return

    if truncate:
        LOGGER.info(
            "Truncating and loading %s file(s) into %s...",
            len(csv_paths),
            table_name,
        )
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                    sql.Identifier(table_name),
                ),
            )
    else:
        LOGGER.info(
            "Appending %s file(s) into %s...",
            len(csv_paths),
            table_name,
        )

    total_rows: int = 0
    for file_path in csv_paths:
        LOGGER.info("Loading %s into %s...", file_path, table_name)
        if file_path.suffix == ".parquet":
            total_rows += _copy_parquet_file_in_batches(
                conn,
                table_name,
                file_path,
                batch_size=batch_size,
            )
        else:
            total_rows += _copy_csv_file_in_batches(
                conn,
                table_name,
                file_path,
                batch_size=batch_size,
            )

    if total_rows == 0:
        msg = f"Zero rows were loaded into table {table_name}. Continuing load."
        LOGGER.warning(msg)
        return

    LOGGER.info(
        "Successfully loaded %d rows into %s.",
        int(total_rows),
        str(table_name),
    )


def _discover_locations_csv_paths(args: argparse.Namespace) -> list[Path]:
    return [Path(args.cities_csv)]


def _discover_pet_csv_paths(args: argparse.Namespace) -> list[Path]:
    pet_paths = _discover_csv_inputs(
        args.pet_csv,
        shard_root=args.pet_root,
        shard_file_name="pet_batch_*.parquet",
        shard_partition_key=None,
    )
    return _select_partition_shard_paths(
        pet_paths,
        root_path=Path(args.pet_root),
        partition_key="year",
        shard_index=args.load_shard_index,
        shard_count=args.load_shard_count,
    )


def _discover_analytics_csv_paths(
    args: argparse.Namespace,
    shard_file_name: str,
) -> list[Path]:
    csv_paths = _discover_csv_inputs(
        shard_file_name,
        shard_root=args.analytics_root,
        shard_file_name=shard_file_name,
        shard_count=args.analytics_shard_count,
        shard_partition_key="shard_count",
    )
    if args.load_shard_count == 1:
        return csv_paths

    return _filter_paths_by_partition_value(
        csv_paths,
        root_path=Path(args.analytics_root),
        partition_key="shard_index",
        partition_value=f"{args.load_shard_index:05d}",
    )


def _discover_percentiles_csv_paths(args: argparse.Namespace) -> list[Path]:
    return _discover_analytics_csv_paths(args, "percentiles.parquet")


def _discover_forecast_csv_paths(args: argparse.Namespace) -> list[Path]:
    return _discover_analytics_csv_paths(args, "forecast.parquet")


def _discover_pet_change_csv_paths(args: argparse.Namespace) -> list[Path]:
    return _discover_analytics_csv_paths(args, "change_per_decade.parquet")


def _load_requested_tables(
    conn: connection,
    args: argparse.Namespace,
    *,
    truncate_tables: set[str],
    skip_tables: set[str],
) -> None:
    table_csv_resolvers = (
        ("locations", _discover_locations_csv_paths),
        ("pet", _discover_pet_csv_paths),
        ("pet_percentiles", _discover_percentiles_csv_paths),
        ("pet_forecast", _discover_forecast_csv_paths),
        ("pet_change", _discover_pet_change_csv_paths),
    )

    for table_name, csv_resolver in table_csv_resolvers:
        if table_name in skip_tables:
            continue

        bulk_insert_csv_files(
            conn,
            csv_resolver(args),
            table_name,
            batch_size=args.copy_batch_size,
            truncate=table_name in truncate_tables
            and not (table_name == "locations" and args.skip_drop_views),
        )


def main() -> None:
    """Load all generated CSV datasets and recreate views."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    _validate_load_shard_args(args.load_shard_index, args.load_shard_count)

    if args.append_only:
        truncate_table_names: list[str] = []
    elif args.truncate_tables is None:
        truncate_table_names = TABLE_NAMES
    else:
        truncate_table_names = args.truncate_tables

    truncate_tables: set[str] = set(truncate_table_names)
    skip_tables: set[str] = set(args.skip_tables or [])

    if not DB_URI:
        LOGGER.warning(
            "SUPABASE_DB_URI environment variable is not set. Skipping database operations.",
        )
        return

    LOGGER.info("Connecting to the database...")
    conn: connection = psycopg2.connect(DB_URI)
    conn.autocommit = True

    try:
        if not args.skip_drop_views:
            execute_sql_file(conn, "drop_views.sql")

        _load_requested_tables(
            conn,
            args,
            truncate_tables=truncate_tables,
            skip_tables=skip_tables,
        )

        if not args.skip_create_views:
            execute_sql_file(conn, "create_views.sql")

    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
