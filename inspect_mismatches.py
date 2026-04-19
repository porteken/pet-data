"""Inspect and compare PET data mismatches between current and legacy datasets."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from pet_corrected import pet_corrected

# Configuration
mismatch_pairs = [
    (128, "2025-05-04"),
    (77, "2025-05-02"),
    (420, "2025-05-04"),
    (163, "2025-05-04"),
    (317, "2024-05-02"),
]

cities_cur = pd.read_csv("cities.csv")
cities_leg = pd.read_csv("/home/kenneth-porter/pet_files/cities.csv")


def get_city_info(loc_id: int) -> tuple[str, str]:
    """Get city information (name, lat, lon) for current and legacy datasets."""
    cur_row = cities_cur[cities_cur["location_id"] == loc_id]
    leg_row = cities_leg[cities_leg["location_id"] == loc_id]

    cur_info = (
        f"{cast('Any', cur_row['city'])[0]} ({cast('Any', cur_row['lat'])[0]}, {cast('Any', cur_row['lng'])[0]})"
        if not cur_row.empty
        else "N/A"
    )
    leg_info = (
        f"{cast('Any', leg_row['city'])[0]} ({cast('Any', leg_row['lat'])[0]}, {cast('Any', leg_row['lng'])[0]})"
        if not leg_row.empty
        else "N/A"
    )
    return cur_info, leg_info


def round_half(x: Any) -> Any:
    """Round value to the nearest half-integer."""
    if hasattr(x, "round"):
        return (x * 2).round() / 2
    return round(x * 2) / 2


def get_max_pet(df: pd.DataFrame) -> float | None:
    """Calculate and return the maximum PET value for a dataframe."""
    if df.empty:
        return None
    pets = df.apply(
        lambda r: pet_corrected(
            r["t_c_r"],
            r["v_r"],
            r["rh_r"],
            r["mrt_c_r"],
            icl=0.5,
        ),
        axis=1,
    )
    return float(round_half(pets).max())


def main() -> None:
    """Run the mismatch inspection script."""
    # Load data
    comb_cur = pd.read_parquet("combined_data_parquet")
    comb_leg = pd.read_parquet("/home/kenneth-porter/pet_files/combined_data.parquet")

    # Ensure time is datetime
    comb_cur["time"] = pd.to_datetime(comb_cur["time"])
    comb_leg["time"] = pd.to_datetime(comb_leg["time"])

    max_display_mismatches = 3

    for loc_id, date_str in mismatch_pairs:
        _cur_info, _leg_info = get_city_info(loc_id)

        date = pd.to_datetime(date_str).date()

        df_cur = comb_cur[
            (comb_cur["location_id"] == loc_id) & (comb_cur["time"].dt.date == date)
        ].copy()
        df_leg = comb_leg[
            (comb_leg["location_id"] == loc_id) & (comb_leg["time"].dt.date == date)
        ].copy()

        # Process current
        df_cur["t_c"] = df_cur["t"] - 273.15
        df_cur["mrt_c"] = df_cur["mrt"] - 273.15
        for col in ["t_c", "v", "rh", "mrt_c"]:
            df_cur[f"{col}_r"] = round_half(df_cur[col])

        # Process legacy
        df_leg["t_c"] = df_leg["t"] - 273.15
        df_leg["mrt_c"] = df_leg["mrt"] - 273.15
        for col in ["t_c", "v", "rh", "mrt_c"]:
            df_leg[f"{col}_r"] = round_half(df_leg[col])

        # Merge to compare
        merged = df_cur.merge(df_leg, on="time", suffixes=("_cur", "_leg"), how="outer")
        merged = merged.sort_values("time")

        mismatches = []
        for _, row in merged.iterrows():
            cols = ["t_c_r", "v_r", "rh_r", "mrt_c_r"]
            diff = False
            for col in cols:
                if row[f"{col}_cur"] != row[f"{col}_leg"]:
                    diff = True
                    break
            if diff:
                mismatches.append(row)

        if mismatches:
            for _m in mismatches[:max_display_mismatches]:  # Show first few
                pass
            if len(mismatches) > max_display_mismatches:
                pass

        get_max_pet(cast("pd.DataFrame", df_cur))
        get_max_pet(cast("pd.DataFrame", df_leg))


if __name__ == "__main__":
    main()
