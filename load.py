"""Load generated CSV inputs into the database and manage views."""

from __future__ import annotations

import argparse
import logging
import os
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2
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
    parser.add_argument("--analytics-shard-count", type=int)
    parser.add_argument("--copy-batch-size", type=int, default=COPY_BATCH_SIZE)
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


def _copy_csv_file_in_batches(
    conn: connection,
    table_name: str,
    csv_path: Path,
    *,
    batch_size: int,
) -> int:
    copy_statement = sql.SQL("COPY {} FROM STDIN WITH CSV").format(
        sql.Identifier(table_name),
    )
    rows_loaded = 0

    with conn.cursor() as cur, csv_path.open("r", encoding="utf-8", newline="") as f:
        header = next(f, None)
        if header is None:
            LOGGER.warning("CSV file %s is empty. Skipping.", csv_path)
            return 0

        batch_rows: list[str] = []
        for row in f:
            batch_rows.append(row)
            if len(batch_rows) < batch_size:
                continue

            cur.copy_expert(
                copy_statement.as_string(conn),
                StringIO("".join(batch_rows)),
            )
            rows_loaded += len(batch_rows)
            batch_rows.clear()

        if batch_rows:
            cur.copy_expert(
                copy_statement.as_string(conn),
                StringIO("".join(batch_rows)),
            )
            rows_loaded += len(batch_rows)

    return rows_loaded


def bulk_insert_csv_files(
    conn: connection,
    csv_paths: list[Path],
    table_name: str,
    *,
    batch_size: int,
    truncate: bool,
) -> None:
    """Load one or more CSV files into a destination table."""
    if not csv_paths:
        LOGGER.warning("No CSV inputs found for %s. Skipping.", table_name)
        return

    if truncate:
        LOGGER.info(
            "Truncating and loading %s CSV inputs into %s...",
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
            "Appending %s CSV inputs into %s...",
            len(csv_paths),
            table_name,
        )

    total_rows = 0
    for csv_path in csv_paths:
        LOGGER.info("Loading %s into %s...", csv_path, table_name)
        total_rows += _copy_csv_file_in_batches(
            conn,
            table_name,
            csv_path,
            batch_size=batch_size,
        )

    LOGGER.info("Successfully loaded %s rows into %s.", total_rows, table_name)


def main() -> None:
    """Load all generated CSV datasets and recreate views."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    truncate_tables = set(
        TABLE_NAMES if args.truncate_tables is None else args.truncate_tables
    )
    skip_tables = set(args.skip_tables or [])

    if not DB_URI:
        msg = "SUPABASE_DB_URI environment variable is not set"
        raise RuntimeError(msg)

    LOGGER.info("Connecting to the database...")
    conn: connection = psycopg2.connect(DB_URI)
    conn.autocommit = True

    try:
        execute_sql_file(conn, "drop_views.sql")

        if "locations" not in skip_tables:
            bulk_insert_csv_files(
                conn,
                [Path(args.cities_csv)],
                "locations",
                batch_size=args.copy_batch_size,
                truncate="locations" in truncate_tables,
            )
        if "pet" not in skip_tables:
            bulk_insert_csv_files(
                conn,
                _discover_csv_inputs(
                    args.pet_csv,
                    shard_root=args.pet_root,
                    shard_file_name="pet.csv",
                    shard_partition_key=None,
                ),
                "pet",
                batch_size=args.copy_batch_size,
                truncate="pet" in truncate_tables,
            )
        if "pet_percentiles" not in skip_tables:
            bulk_insert_csv_files(
                conn,
                _discover_csv_inputs(
                    "percentiles.csv",
                    shard_root=args.analytics_root,
                    shard_file_name="percentiles.csv",
                    shard_count=args.analytics_shard_count,
                    shard_partition_key="shard_count",
                ),
                "pet_percentiles",
                batch_size=args.copy_batch_size,
                truncate="pet_percentiles" in truncate_tables,
            )
        if "pet_forecast" not in skip_tables:
            bulk_insert_csv_files(
                conn,
                _discover_csv_inputs(
                    "forecast.csv",
                    shard_root=args.analytics_root,
                    shard_file_name="forecast.csv",
                    shard_count=args.analytics_shard_count,
                    shard_partition_key="shard_count",
                ),
                "pet_forecast",
                batch_size=args.copy_batch_size,
                truncate="pet_forecast" in truncate_tables,
            )
        if "pet_change" not in skip_tables:
            bulk_insert_csv_files(
                conn,
                _discover_csv_inputs(
                    "change_per_decade.csv",
                    shard_root=args.analytics_root,
                    shard_file_name="change_per_decade.csv",
                    shard_count=args.analytics_shard_count,
                    shard_partition_key="shard_count",
                ),
                "pet_change",
                batch_size=args.copy_batch_size,
                truncate="pet_change" in truncate_tables,
            )

        execute_sql_file(conn, "create_views.sql")

    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
