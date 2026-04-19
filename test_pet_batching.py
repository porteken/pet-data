"""Test PET batch computation vs scalar computation."""

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pet_corrected import pet_corrected


def main() -> None:
    """Run PET batching test."""
    # 1. Load mismatches
    try:
        ref = pd.read_csv("pet_smoke_reference.csv")
        cur = pd.read_csv("pet_sorted.csv")
    except FileNotFoundError:
        return

    # Align columns and rows
    ref = ref.sort_values(["location_id", "date"]).reset_index(drop=True)
    cur = cur.sort_values(["location_id", "date"]).reset_index(drop=True)

    # Using 'pet' column
    diff_mask = ~np.isclose(ref["pet"], cur["pet"], atol=1e-2)
    mismatched_rows = ref[diff_mask]

    # 2. Find the hourly data for these mismatches
    mismatched_keys = set(
        zip(mismatched_rows["location_id"], mismatched_rows["date"], strict=False),
    )

    combined_path = Path("combined_data_parquet")
    all_files = list(combined_path.rglob("*.parquet"))

    relevant_hourly = []
    for f in all_files:
        df = pd.read_parquet(f)
        if "time" in df.columns:
            df["date"] = df["time"].dt.strftime("%Y-%m-%d")

        mask = df.apply(
            lambda row: (row["location_id"], row["date"]) in mismatched_keys,
            axis=1,
        )
        relevant_hourly.append(df[mask])

    if not relevant_hourly:
        sys.exit()

    hourly_df = pd.concat(relevant_hourly)

    # 3. Identify distinct (v, t, rh, mrt) rounded combos
    # Use ONLY 20 combos to be really fast
    hourly_df["v_r"] = hourly_df["v"].round(4)
    hourly_df["t_r"] = hourly_df["t"].round(4)
    hourly_df["rh_r"] = hourly_df["rh"].round(4)
    hourly_df["mrt_r"] = hourly_df["mrt"].round(4)

    combos = hourly_df[["v_r", "t_r", "rh_r", "mrt_r"]].drop_duplicates()

    test_combos = combos.head(20).copy()

    # 4. Compute PET two ways
    v = cast("Any", test_combos["v_r"]).to_numpy()
    t = cast("Any", test_combos["t_r"]).to_numpy()
    rh = cast("Any", test_combos["rh_r"]).to_numpy()
    mrt = cast("Any", test_combos["mrt_r"]).to_numpy()

    # (a) Vectorized
    pet_vec = pet_corrected(v, t, rh, mrt)

    # (b) Scalar
    pet_scalar = np.array(
        [pet_corrected(v[i], t[i], rh[i], mrt[i]) for i in range(len(v))],
    )

    # 5. Compare
    _diffs = ~np.isclose(pet_vec, pet_scalar, atol=1e-5)


if __name__ == "__main__":
    main()
