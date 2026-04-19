"""Compare location data between different CSV files."""

import pandas as pd


def compare(f1: str, f2: str) -> None:
    """Compare two CSV files containing location data."""
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)

    common_cols = [c for c in df1.columns if c in df2.columns]
    if "location_id" not in common_cols:
        return

    merged = df1[common_cols].merge(
        df2[common_cols],
        on="location_id",
        suffixes=("_data", "_files"),
    )

    diff_mask = pd.Series(data=False, index=merged.index)
    for col in common_cols:
        if col == "location_id":
            continue
        diff_mask |= merged[f"{col}_data"] != merged[f"{col}_files"]

    if not diff_mask.any():
        pass
    else:
        for col in common_cols:
            if col == "location_id":
                continue
            c_diff = (merged[f"{col}_data"] != merged[f"{col}_files"]).sum()
            if c_diff > 0:
                pass

        merged[diff_mask].head(10)

        display_cols = ["location_id"]
        for col in common_cols:
            if col == "location_id":
                continue
            display_cols.extend([f"{col}_data", f"{col}_files"])


compare(
    "/home/kenneth-porter/pet-data/cities.csv",
    "/home/kenneth-porter/pet_files/cities.csv",
)
compare(
    "/home/kenneth-porter/pet-data/cities.csv",
    "/home/kenneth-porter/pet_files/locations.csv",
)
