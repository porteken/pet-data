"""Utility for PET data export, merging, and incremental updates."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psycopg2
from psycopg2 import sql

if TYPE_CHECKING:
    from psycopg2.extensions import connection, cursor

PET_CSV_HEADER = "location_id,date,pet\n"
ANALYTICS_TABLES = ("pet_forecast",)


def _write_empty_pet_csv(output_path: str | Path) -> None:
    Path(output_path).write_text(PET_CSV_HEADER, encoding="utf-8")


def _build_export_copy_query(
    cur: cursor,
    *,
    window_start: str | None,
    window_end: str | None,
) -> str:
    if window_start is None or window_end is None:
        return (
            "COPY ("
            "SELECT location_id, date, pet "
            "FROM public.pet "
            "ORDER BY location_id, date"
            ") TO STDOUT WITH CSV HEADER"
        )

    return cur.mogrify(
        ""
        "COPY ("
        "SELECT location_id, date, pet "
        "FROM public.pet "
        "WHERE date < %s::date OR date > %s::date "
        "ORDER BY location_id, date"
        ") TO STDOUT WITH CSV HEADER"
        "",
        (window_start, window_end),
    ).decode("utf-8")


def export_pet(
    window_start: str | None,
    window_end: str | None,
    output_path: str,
) -> None:
    """Export PET data, optionally excluding a target date window."""
    db_uri = os.environ.get("SUPABASE_DB_URI")
    if not db_uri:
        print("SUPABASE_DB_URI not set, skipping database export.")  # noqa: T201
        _write_empty_pet_csv(output_path)
        return

    conn = psycopg2.connect(db_uri)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.pet')")
            regclass = cur.fetchone()
            if regclass is None or regclass[0] is None:
                print("Table 'pet' does not exist. Creating empty export.")  # noqa: T201
                _write_empty_pet_csv(output_path)
                return

            copy_query = _build_export_copy_query(
                cur,
                window_start=window_start,
                window_end=window_end,
            )

            with Path(output_path).open("w", encoding="utf-8", newline="") as csv_file:
                cur.copy_expert(copy_query, csv_file)
    finally:
        conn.close()


def _iter_source_paths(source_dirs: list[str], source_files: list[str]) -> list[Path]:
    source_paths = [Path(path_str) for path_str in source_files]

    for directory in source_dirs:
        directory_path = Path(directory)
        parquet_paths = sorted(directory_path.rglob("pet_batch_*.parquet"))
        if parquet_paths:
            source_paths.extend(parquet_paths)
            continue

        source_paths.extend(sorted(directory_path.rglob("pet.csv")))

    return source_paths


def _load_pet_frame_from_path(source_path: Path, pd: Any) -> Any:  # noqa: ANN401
    if source_path.suffix == ".parquet":
        return pd.read_parquet(source_path, columns=["location_id", "date", "pet"])
    return pd.read_csv(source_path, usecols=["location_id", "date", "pet"])


def merge_csvs(
    source_dirs: list[str],
    source_files: list[str],
    output_file: str,
) -> None:
    """Merge multiple PET CSV/parquet sources into one de-duplicated CSV."""
    pd = importlib.import_module("pandas")

    print(f"Merging PET data into {output_file}...")  # noqa: T201
    frames = []

    for source_order, source_path in enumerate(
        _iter_source_paths(source_dirs, source_files),
    ):
        if not source_path.exists():
            continue

        try:
            frame = _load_pet_frame_from_path(source_path, pd)
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {source_path}: {exc}")  # noqa: T201
            continue

        frame = frame[["location_id", "date", "pet"]].copy()
        frame["_source_order"] = source_order
        frames.append(frame)

    if not frames:
        _write_empty_pet_csv(output_file)
        print("No source data found; wrote empty CSV.")  # noqa: T201
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["location_id", "date", "pet"])
    combined = combined.sort_values(
        ["location_id", "date", "_source_order"],
        kind="stable",
    )
    combined = combined.drop_duplicates(
        subset=["location_id", "date"],
        keep="last",
    )
    combined = combined.sort_values(["location_id", "date"], kind="stable")
    combined = combined.drop(columns=["_source_order"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(output_file, index=False)
    print(f"Wrote {len(combined)} rows to {output_file}.")  # noqa: T201


def _existing_public_tables(
    conn: connection,
    table_names: tuple[str, ...],
) -> list[str]:
    existing_tables: list[str] = []
    with conn.cursor() as cur:
        for table_name in table_names:
            cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
            row = cur.fetchone()
            if row is not None and row[0] is not None:
                existing_tables.append(table_name)
    return existing_tables


def delete_window(window_start: str, window_end: str) -> None:
    """Delete PET rows within a specific date window and reset analytics tables."""
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
                    "DELETE FROM public.pet WHERE date BETWEEN %s::date AND %s::date",
                    (window_start, window_end),
                )

        analytics_tables = _existing_public_tables(conn, ANALYTICS_TABLES)
        if analytics_tables:
            print("Truncating analytics tables...")  # noqa: T201
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("TRUNCATE TABLE {}").format(
                        sql.SQL(", ").join(
                            sql.Identifier("public", table_name)
                            for table_name in analytics_tables
                        ),
                    ),
                )
    finally:
        conn.close()


def main() -> None:
    """Execute historical PET update commands."""
    if len(sys.argv) < 2:  # noqa: PLR2004
        msg = (
            "Usage: historical_pet_update.py "
            "[export|export-all|merge|delete-window] ..."
        )
        raise SystemExit(msg)

    command = sys.argv[1]
    if command == "export":
        export_pet(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "export-all":
        export_pet(None, None, sys.argv[2])
    elif command == "merge":
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
    else:
        msg = f"Unknown command: {command}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
