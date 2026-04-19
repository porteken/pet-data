"""Utility for exporting and merging PET data for updates."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql


def export_pet(window_start: str, window_end: str, output_path: str) -> None:
    """Export PET data outside the specified window to a CSV."""
    db_uri = os.environ.get("SUPABASE_DB_URI")
    if not db_uri:
        print("SUPABASE_DB_URI not set, skipping database export.")  # noqa: T201
        Path(output_path).write_text("location_id,date,pet\n", encoding="utf-8")
        return

    conn = psycopg2.connect(db_uri)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.pet')")
            regclass = cur.fetchone()
            if regclass is None or regclass[0] is None:
                print("Table 'pet' does not exist. Creating empty export.")  # noqa: T201
                Path(output_path).write_text("location_id,date,pet\n", encoding="utf-8")
                return

            copy_query = sql.SQL(
                """
                COPY (
                  SELECT location_id, date, pet
                  FROM pet
                  WHERE date < {window_start} OR date > {window_end}
                  ORDER BY date, location_id
                ) TO STDOUT WITH CSV HEADER
                """,
            ).format(
                window_start=sql.Literal(window_start),
                window_end=sql.Literal(window_end),
            )

            with Path(output_path).open("w", encoding="utf-8", newline="") as csv_file:
                cur.copy_expert(copy_query.as_string(conn), csv_file)
    finally:
        conn.close()


def merge_csvs(
    source_dirs: list[str],
    source_files: list[str],
    output_file: str,
) -> None:
    """Merge multiple PET CSV/parquet files/directories into one output CSV."""
    pd = importlib.import_module("pandas")

    print(f"Merging PET data into {output_file}...")  # noqa: T201
    frames = []

    # Explicit file list (CSV or parquet)
    for path_str in source_files:
        source_path = Path(path_str)
        if not source_path.exists():
            continue
        try:
            if source_path.suffix == ".parquet":
                frames.append(
                    pd.read_parquet(
                        source_path,
                        columns=["location_id", "date", "pet"],
                    ),
                )
            else:
                frames.append(
                    pd.read_csv(source_path, usecols=["location_id", "date", "pet"]),
                )
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {source_path}: {e}")  # noqa: T201

    # Directory scan: prefer parquet shards, fall back to CSV files
    for d in source_dirs:
        parquet_shards = sorted(Path(d).rglob("pet_batch_*.parquet"))
        if parquet_shards:
            for p in parquet_shards:
                try:
                    frames.append(
                        pd.read_parquet(p, columns=["location_id", "date", "pet"]),
                    )
                except Exception as e:  # noqa: BLE001, PERF203
                    print(f"Error processing {p}: {e}")  # noqa: T201
        else:
            for p in sorted(Path(d).rglob("pet.csv")):
                try:
                    frames.append(
                        pd.read_csv(p, usecols=["location_id", "date", "pet"]),
                    )
                except Exception as e:  # noqa: BLE001, PERF203
                    print(f"Error processing {p}: {e}")  # noqa: T201

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(output_file, index=False)
        print(f"Wrote {len(combined)} rows to {output_file}.")  # noqa: T201
    else:
        Path(output_file).write_text("location_id,date,pet\n", encoding="utf-8")
        print("No source data found; wrote empty CSV.")  # noqa: T201


def delete_window(window_start: str, window_end: str) -> None:
    """Delete PET rows within a specific date window."""
    db_uri = os.environ.get("SUPABASE_DB_URI")
    if not db_uri:
        print("SUPABASE_DB_URI not set, skipping database cleanup.")  # noqa: T201
        return

    conn = psycopg2.connect(db_uri)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            if Path("drop_views.sql").exists():
                cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))

            cur.execute("SELECT to_regclass('public.pet')")
            regclass = cur.fetchone()
            if regclass is not None and regclass[0] is not None:
                print(f"Deleting PET data in window [{window_start}, {window_end}]...")  # noqa: T201
                cur.execute(
                    """
                    DELETE FROM pet
                    WHERE date >= %(window_start)s AND date <= %(window_end)s
                    """,
                    {"window_start": window_start, "window_end": window_end},
                )

            print("Truncating analytics tables...")  # noqa: T201
            cur.execute(
                """
                TRUNCATE TABLE
                  pet_percentiles,
                  pet_forecast,
                  pet_change
                CASCADE
                """,
            )
    finally:
        conn.close()


def main() -> None:
    """Execute the historical PET update commands based on CLI arguments."""
    command = sys.argv[1]
    if command == "export":
        export_pet(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "merge":
        # merge output_file source_file1 source_file2 ... --dirs dir1 dir2 ...
        output_file = sys.argv[2]
        source_files = []
        source_dirs = []
        current_list = source_files
        for arg in sys.argv[3:]:
            if arg == "--dirs":
                current_list = source_dirs
            else:
                current_list.append(arg)
        merge_csvs(source_dirs, source_files, output_file)
    elif command == "delete-window":
        delete_window(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
