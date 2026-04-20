"""Compare PET data between the development and production databases."""

from __future__ import annotations

import logging
import os
from typing import Final

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

QUERY: Final[str] = """
SELECT location_id, date, pet
FROM pet
WHERE extract(month from date) BETWEEN 5 AND 9
  AND extract(year from date) IN (2024, 2025)
ORDER BY location_id, date
"""

MAX_SAMPLE_ROWS: Final[int] = 10


def _load_database_frame(env_var: str) -> pd.DataFrame:
    uri = os.environ.get(env_var)
    if not uri:
        msg = f"{env_var} is not set."
        raise SystemExit(msg)

    with psycopg2.connect(uri, connect_timeout=10) as conn:
        frame = pd.read_sql_query(QUERY, conn)

    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(by=["location_id", "date"]).reset_index(drop=True)


def _print_comparison(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left_dup = int(left.duplicated(["location_id", "date"]).sum())
    right_dup = int(right.duplicated(["location_id", "date"]).sum())
    validate = "one_to_one" if left_dup == 0 and right_dup == 0 else "many_to_many"

    merged = left.rename(columns={"pet": "pet_dev"}).merge(
        right.rename(columns={"pet": "pet_prd"}),
        on=["location_id", "date"],
        how="outer",
        indicator=True,
        validate=validate,
    )

    overlap = merged[merged["_merge"] == "both"].copy()
    exact_mask = overlap["pet_dev"] == overlap["pet_prd"]
    rounded_mask = overlap["pet_dev"].round(4) == overlap["pet_prd"].round(4)
    differing_rows = overlap.loc[~rounded_mask].copy()
    abs_diff = np.abs(overlap["pet_dev"] - overlap["pet_prd"])

    logger.info("=== Database Comparison: dev vs prd ===")
    logger.info("Rows: dev=%d, prd=%d", len(left), len(right))
    logger.info("Duplicate keys: dev=%d, prd=%d", left_dup, right_dup)
    logger.info(
        "Overlap=%d, Only dev=%d, Only prd=%d",
        len(overlap),
        int((merged["_merge"] == "left_only").sum()),
        int((merged["_merge"] == "right_only").sum()),
    )
    logger.info("Exact matches=%d", int(exact_mask.sum()))
    logger.info("4-decimal matches=%d", int(rounded_mask.sum()))
    logger.info(
        "Differing rows=%d (%.2f%%)",
        len(differing_rows),
        (len(differing_rows) / len(overlap) * 100.0) if len(overlap) else 0.0,
    )
    logger.info("MAE=%.6f", float(abs_diff.mean()) if len(overlap) else float("nan"))
    logger.info(
        "Max abs diff=%.6f",
        float(abs_diff.max()) if len(overlap) else float("nan"),
    )

    if differing_rows.empty:
        logger.info("No differing rows found.")
        return

    logger.info("\nSample differing rows:")
    logger.info(
        "%s",
        differing_rows[["location_id", "date", "pet_dev", "pet_prd"]]
        .head(MAX_SAMPLE_ROWS)
        .to_string(index=False),
    )


def main() -> None:
    """Execute the main entry point for database comparison."""
    dev = _load_database_frame("SUPABASE_DB_URI")
    prd = _load_database_frame("SUPABASE_DB_URI_PRD")

    if dev.empty and prd.empty:
        logger.info("Both database queries returned zero rows.")
        return

    _print_comparison(dev, prd)


if __name__ == "__main__":
    main()
