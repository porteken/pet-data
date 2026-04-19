"""Comparison script for PET data across different database environments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from comparison_metrics import (
    compute_pair_metrics,
    compute_year_metrics,
    merge_pet_frames,
    metrics_to_dict,
    sample_differences,
)

load_dotenv()

QUERY = """
SELECT location_id, date, pet
FROM pet
WHERE extract(month from date) BETWEEN 5 AND 9
  AND extract(year from date) IN (2024, 2025)
ORDER BY location_id, date
"""
DEV_EXPORT_PATH = Path("pet_db1_2024_2025.csv")
PRD_EXPORT_PATH = Path("pet_db2_2024_2025.csv")


def _load_pet_frame(uri: str) -> pd.DataFrame:
    with psycopg2.connect(uri, connect_timeout=10) as conn:
        return pd.read_sql_query(QUERY, conn)


def _export_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.sort_values(["location_id", "date"]).to_csv(output_path, index=False)


def main() -> None:
    """Compare PET data between dev and prd databases."""
    uri_dev = os.environ.get("SUPABASE_DB_URI")
    uri_prd = os.environ.get("SUPABASE_DB_URI_PRD")
    if not uri_dev or not uri_prd:
        msg = "SUPABASE_DB_URI and SUPABASE_DB_URI_PRD must be set."
        raise RuntimeError(msg)

    df_dev = _load_pet_frame(uri_dev)
    df_prd = _load_pet_frame(uri_prd)
    _export_csv(df_dev, DEV_EXPORT_PATH)
    _export_csv(df_prd, PRD_EXPORT_PATH)

    merged = merge_pet_frames(
        df_dev.rename(columns={"pet": "pet_dev"}),
        df_prd.rename(columns={"pet": "pet_prd"}),
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )
    metrics = compute_pair_metrics(
        merged,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )
    year_metrics = compute_year_metrics(
        merged,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )
    diff_sample = sample_differences(
        merged,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
        max_rows=10,
    )

    print("DEV vs PRD summary:")  # noqa: T201
    print(json.dumps(metrics_to_dict(metrics), indent=2, sort_keys=True))  # noqa: T201
    print("Per-year metrics:")  # noqa: T201
    print(  # noqa: T201
        json.dumps(
            {
                year: metrics_to_dict(year_metric)
                for year, year_metric in year_metrics.items()
            },
            indent=2,
            sort_keys=True,
        ),
    )
    if not diff_sample.empty:
        print("Top keyed PET differences:")  # noqa: T201
        print(diff_sample.to_string(index=False))  # noqa: T201


if __name__ == "__main__":
    main()
