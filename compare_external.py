"""Comparison script for PET data across different external sources."""

from typing import Any

import pandas as pd


def main() -> None:
    """Compare PET data from CSVs and database exports against a base dataset."""
    df_db1: Any = pd.read_csv(
        "/home/kenneth-porter/pet-data/pet_db1_2024_2025.csv",
        usecols=["location_id", "date", "pet"],
    )

    df_db2: Any = pd.read_csv(
        "/home/kenneth-porter/pet-data/pet_db2_2024_2025.csv",
        usecols=["location_id", "date", "pet"],
    )

    try:
        df_base: Any = pd.read_parquet(
            "/home/kenneth-porter/pet_files/pet.parquet",
            columns=["location_id", "date", "pet"],
        )
    except (OSError, ValueError):
        df_base = pd.read_csv(
            "/home/kenneth-porter/pet_files/pet.csv",
            usecols=["location_id", "date", "pet"],
        )

    df_base["date"] = pd.to_datetime(df_base["date"])
    df_db1["date"] = pd.to_datetime(df_db1["date"])
    df_db2["date"] = pd.to_datetime(df_db2["date"])

    base_filtered: Any = df_base[
        (df_base["date"].dt.year.isin([2024, 2025]))
        & (df_base["date"].dt.month.between(5, 9))
    ]

    df_db1 = df_db1.rename(columns={"pet": "pet_db1"})
    df_db2 = df_db2.rename(columns={"pet": "pet_db2"})
    base_filtered = base_filtered.rename(columns={"pet": "pet_base"})

    merged = base_filtered.merge(df_db1, on=["location_id", "date"], how="outer")
    merged = merged.merge(df_db2, on=["location_id", "date"], how="outer")


if __name__ == "__main__":
    main()
