"""Comparison script for PET data across different external sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from comparison_metrics import (
    compute_pair_metrics,
    merge_pet_frames,
    metrics_to_dict,
    sample_differences,
)

DB1_PATH = Path("/home/kenneth-porter/pet-data/pet_db1_2024_2025.csv")
DB2_PATH = Path("/home/kenneth-porter/pet-data/pet_db2_2024_2025.csv")
BASE_PARQUET_PATH = Path("/home/kenneth-porter/pet_files/pet.parquet")
BASE_CSV_PATH = Path("/home/kenneth-porter/pet_files/pet.csv")


def _load_base_frame() -> pd.DataFrame:
    try:
        return pd.read_parquet(
            BASE_PARQUET_PATH,
            columns=["location_id", "date", "pet"],
        )
    except (OSError, ValueError):
        return pd.read_csv(
            BASE_CSV_PATH,
            usecols=["location_id", "date", "pet"],
        )


def _filtered_base_frame() -> pd.DataFrame:
    df_base = _load_base_frame()
    df_base["date"] = pd.to_datetime(df_base["date"])
    return cast(
        "pd.DataFrame",
        df_base[
            (df_base["date"].dt.year.isin([2024, 2025]))
            & (df_base["date"].dt.month.between(5, 9))
        ].copy(),
    )


def _report_pair(
    name: str,
    left: pd.DataFrame,
    left_name: str,
    right: pd.DataFrame,
    right_name: str,
) -> None:
    merged = merge_pet_frames(
        left.rename(columns={"pet": left_name}),
        right.rename(columns={"pet": right_name}),
        left_value_name=left_name,
        right_value_name=right_name,
    )
    metrics = compute_pair_metrics(
        merged,
        left_value_name=left_name,
        right_value_name=right_name,
    )
    print(f"{name}:")  # noqa: T201
    print(json.dumps(metrics_to_dict(metrics), indent=2, sort_keys=True))  # noqa: T201
    sample = sample_differences(
        merged,
        left_value_name=left_name,
        right_value_name=right_name,
        max_rows=5,
    )
    if not sample.empty:
        print(sample.to_string(index=False))  # noqa: T201


def main() -> None:
    """Compare PET data from CSVs and database exports against a base dataset."""
    df_db1 = pd.read_csv(DB1_PATH, usecols=["location_id", "date", "pet"])
    df_db2 = pd.read_csv(DB2_PATH, usecols=["location_id", "date", "pet"])
    df_base = _filtered_base_frame()

    _report_pair("DEV vs base", df_db1, "pet_dev", df_base, "pet_base")
    _report_pair("PRD vs base", df_db2, "pet_prd", df_base, "pet_base")
    _report_pair("DEV vs PRD", df_db1, "pet_dev", df_db2, "pet_prd")


if __name__ == "__main__":
    main()
