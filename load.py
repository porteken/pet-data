"""Load generated CSV into the database and manage views."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg2
from psycopg2 import sql

if TYPE_CHECKING:
    from psycopg2.extensions import connection

LOGGER = logging.getLogger(__name__)
DB_URI = os.getenv("SUPABASE_DB_URI")


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


def bulk_insert_csv(conn: connection, csv_path: str | Path, table_name: str) -> None:
    """Load a CSV file into a destination table."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        LOGGER.warning("CSV file %s not found. Skipping %s.", csv_file, table_name)
        return

    LOGGER.info("Truncating and loading %s into %s...", csv_file, table_name)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE TABLE {} CASCADE").format(
                sql.Identifier(table_name),
            ),
        )

        with csv_file.open("r", encoding="utf-8", newline="") as csv_buffer:
            next(csv_buffer, None)
            copy_statement = sql.SQL("COPY {} FROM STDIN WITH CSV").format(
                sql.Identifier(table_name),
            )
            cur.copy_expert(copy_statement.as_string(conn), csv_buffer)

    LOGGER.info("Successfully loaded %s.", table_name)


def main() -> None:
    """Load all generated CSV datasets and recreate views."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not DB_URI:
        msg = "SUPABASE_DB_URI environment variable is not set"
        raise RuntimeError(msg)

    LOGGER.info("Connecting to the database...")
    conn: connection = psycopg2.connect(DB_URI)
    conn.autocommit = True

    try:
        execute_sql_file(conn, "drop_views.sql")

        bulk_insert_csv(conn, "cities.csv", "locations")
        bulk_insert_csv(conn, "pet.csv", "pet")
        bulk_insert_csv(conn, "percentiles.csv", "pet_percentiles")
        bulk_insert_csv(conn, "forecast.csv", "pet_forecast")
        bulk_insert_csv(conn, "change_per_decade.csv", "pet_change")

        execute_sql_file(conn, "create_views.sql")

    finally:
        conn.close()
        LOGGER.info("Database connection closed.")


if __name__ == "__main__":
    main()
