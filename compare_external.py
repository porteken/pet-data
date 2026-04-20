"""Compare database PET values against the external pet-files reference."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final, cast

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

REFERENCE_CANDIDATES: Final[tuple[Path, ...]] = (
    Path.home() / "pet-files" / "pet.csv",
    Path.home() / "pet-files" / "pet.parquet",
    Path.home() / "pet_files" / "pet.csv",
    Path.home() / "pet_files" / "pet.parquet",
)

MAX_SAMPLE_ROWS: Final[int] = 10


def _load_database_frame(env_var: str) -> pd.DataFrame:
    uri = os.environ.get(env_var)
    if not uri:
        msg = f"{env_var} is not set."
        raise SystemExit(msg)

    with psycopg2.connect(uri, connect_timeout=10) as conn:
        frame = pd.read_sql_query(QUERY, conn)

    frame["date"] = pd.to_datetime(frame["date"])
    return cast(
        "pd.DataFrame",
        frame.sort_values(by=["location_id", "date"]).reset_index(drop=True),
    )  # pyright: ignore[reportCallIssue]


def _filter_reference_window(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    filtered = frame[
        frame["date"].dt.year.isin([2024, 2025]) & frame["date"].dt.month.between(5, 9)
    ].copy()
    return cast(
        "pd.DataFrame",
        filtered.sort_values(by=["location_id", "date"]).reset_index(drop=True),  # pyright: ignore[reportCallIssue]
    )


def _load_reference_csv(path: Path) -> pd.DataFrame:
    return _filter_reference_window(
        pd.read_csv(path, usecols=cast("Any", ["location_id", "date", "pet"])),
    )


def _load_reference_parquet_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "date" not in frame.columns:
        if "time" not in frame.columns:
            msg = f"Reference parquet {path} must contain 'date' or 'time'."
            raise SystemExit(msg)
        frame["date"] = pd.to_datetime(frame["time"]).dt.normalize()

    if "pet" not in frame.columns:
        msg = f"Reference parquet {path} must contain 'pet'."
        raise SystemExit(msg)

    filtered = _filter_reference_window(
        cast("pd.DataFrame", frame[cast("Any", ["location_id", "date", "pet"])]),
    )
    return cast(
        "pd.DataFrame",
        filtered.groupby(["location_id", "date"], as_index=False)["pet"].max(),
    )


def _load_reference() -> tuple[pd.DataFrame | None, pd.DataFrame | None, Path]:
    csv_frame: pd.DataFrame | None = None
    parquet_daily: pd.DataFrame | None = None
    chosen_path: Path | None = None

    for path in REFERENCE_CANDIDATES:
        if not path.exists():
            continue
        if path.suffix == ".csv" and csv_frame is None:
            csv_frame = _load_reference_csv(path)
            chosen_path = path
        elif path.suffix == ".parquet" and parquet_daily is None:
            parquet_daily = _load_reference_parquet_daily(path)
            chosen_path = chosen_path or path

    if csv_frame is None and parquet_daily is None:
        searched = ", ".join(str(path) for path in REFERENCE_CANDIDATES)
        msg = f"Could not find an external reference under: {searched}"
        raise SystemExit(msg)

    return csv_frame, parquet_daily, chosen_path or REFERENCE_CANDIDATES[0]


def _bucket_absolute_differences(diff_series: pd.Series) -> pd.Series:
    bins = [-0.000001, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
    labels = ["0", "(0,0.5]", "(0.5,1]", "(1,2]", "(2,5]", "(5,10]", ">10"]
    return cast(
        "pd.Series",
        pd.cut(diff_series, bins=bins, labels=labels, include_lowest=True)
        .value_counts(sort=False)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        .rename_axis("abs_diff_bucket"),
    )


def _print_reference_equivalence(
    csv_frame: pd.DataFrame,
    parquet_daily: pd.DataFrame,
) -> None:
    merged = csv_frame.rename(columns={"pet": "pet_csv"}).merge(
        parquet_daily.rename(columns={"pet": "pet_parquet_daily"}),
        on=["location_id", "date"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    overlap = merged[merged["_merge"] == "both"].copy()
    rounded_matches = overlap["pet_csv"].round(4) == overlap["pet_parquet_daily"].round(
        4,
    )

    logger.info("=== Reference Consistency: pet.csv vs daily-max pet.parquet ===")
    logger.info("Rows: csv=%d, parquet_daily=%d", len(csv_frame), len(parquet_daily))
    logger.info(
        "Overlap=%d, Only csv=%d, Only parquet_daily=%d",
        len(overlap),
        int((merged["_merge"] == "left_only").sum()),
        int((merged["_merge"] == "right_only").sum()),
    )
    logger.info("4-decimal matches=%d", int(rounded_matches.sum()))
    logger.info("Differing rows=%d", int((~rounded_matches).sum()))


def _print_comparison(left: pd.DataFrame, right: pd.DataFrame, left_name: str) -> None:
    left_dup = int(left.duplicated(["location_id", "date"]).sum())
    right_dup = int(right.duplicated(["location_id", "date"]).sum())
    validate = "one_to_one" if left_dup == 0 and right_dup == 0 else "many_to_many"

    merged = left.rename(columns={"pet": f"pet_{left_name}"}).merge(
        right.rename(columns={"pet": "pet_reference"}),
        on=["location_id", "date"],
        how="outer",
        indicator=True,
        validate=validate,
    )

    overlap = merged[merged["_merge"] == "both"].copy()
    p1 = overlap[f"pet_{left_name}"]
    p2 = overlap["pet_reference"]
    exact = int((p1 == p2).sum())
    rounded_matches = p1.round(4) == p2.round(4)
    differing_rows = overlap.loc[~rounded_matches].copy()
    abs_diff = np.abs(p1 - p2)

    logger.info("\n=== External Comparison: %s vs reference ===", left_name)
    logger.info("Rows: %s=%d, reference=%d", left_name, len(left), len(right))
    logger.info("Duplicate keys: %s=%d, reference=%d", left_name, left_dup, right_dup)
    logger.info(
        "Overlap=%d, Only %s=%d, Only reference=%d",
        len(overlap),
        left_name,
        int((merged["_merge"] == "left_only").sum()),
        int((merged["_merge"] == "right_only").sum()),
    )
    logger.info("Exact matches=%d", exact)
    logger.info("4-decimal matches=%d", int(rounded_matches.sum()))
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
    logger.info(
        "%s pet stats: min=%.2f, median=%.2f, max=%.2f",
        left_name,
        left["pet"].min(),
        left["pet"].median(),
        left["pet"].max(),
    )

    if differing_rows.empty:
        logger.info("No differing rows found.")
        return

    logger.info("Absolute-difference buckets:")
    logger.info(
        "%s",
        _bucket_absolute_differences(
            cast(
                "pd.Series",
                (
                    differing_rows[f"pet_{left_name}"] - differing_rows["pet_reference"]
                ).abs(),
            ),
        ).to_string(),
    )
    logger.info("\nSample differing rows:")
    logger.info(
        "%s",
        differing_rows[["location_id", "date", f"pet_{left_name}", "pet_reference"]]
        .head(MAX_SAMPLE_ROWS)
        .to_string(index=False),
    )


def main() -> None:
    """Execute the main entry point for external reference comparison."""
    reference, parquet_daily, reference_path = _load_reference()
    logger.info("Using external reference from %s", reference_path)

    if reference is None:
        if parquet_daily is None:
            msg = "Reference is unexpectedly None"
            raise ValueError(msg)
        reference = parquet_daily

    if parquet_daily is not None and reference_path.suffix == ".csv":
        _print_reference_equivalence(reference, parquet_daily)

    dev = _load_database_frame("SUPABASE_DB_URI")
    prd = _load_database_frame("SUPABASE_DB_URI_PRD")

    _print_comparison(dev, reference, "dev")
    _print_comparison(prd, reference, "prd")


if __name__ == "__main__":
    main()
