"""Inspect weather and PET data for specific mismatching locations and dates."""

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
    """Get city information (name, lat, lng) for current and legacy datasets."""
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


def round_half(x: pd.Series) -> pd.Series:
    """Round values in a Series to the nearest 0.5."""
    return (x * 2).round() / 2


# Load only relevant location IDs to save memory/time
loc_ids = [p[0] for p in mismatch_pairs]
comb_cur = pd.read_parquet(
    "combined_data_parquet",
    filters=[("location_id", "in", loc_ids)],
)
comb_leg = pd.read_parquet(
    "/home/kenneth-porter/pet_files/combined_data.parquet",
    filters=[("location_id", "in", loc_ids)],
)

comb_cur["time"] = pd.to_datetime(comb_cur["time"])
comb_leg["time"] = pd.to_datetime(comb_leg["time"])


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
    return round_half(cast("pd.Series", pets)).max()


for loc_id, date_str in mismatch_pairs:
    cur_info, leg_info = get_city_info(loc_id)

    date = pd.to_datetime(date_str).date()
    df_cur = comb_cur[
        (comb_cur["location_id"] == loc_id) & (comb_cur["time"].dt.date == date)
    ].copy()
    df_leg = comb_leg[
        (comb_leg["location_id"] == loc_id) & (comb_leg["time"].dt.date == date)
    ].copy()

    for df in [df_cur, df_leg]:
        if df.empty:
            continue
        df["t_c_r"] = round_half(cast("pd.Series", df["t"] - 273.15))
        df["mrt_c_r"] = round_half(cast("pd.Series", df["mrt"] - 273.15))
        df["v_r"] = round_half(cast("pd.Series", df["v"]))
        df["rh_r"] = round_half(cast("pd.Series", df["rh"]))

    merged = df_cur.merge(
        df_leg,
        on="time",
        suffixes=("_cur", "_leg"),
        how="outer",
    ).sort_values("time")

    mismatches = []
    cols = ["t_c_r", "v_r", "rh_r", "mrt_c_r"]
    for _, row in merged.iterrows():
        if any(row[f"{c}_cur"] != row[f"{c}_leg"] for c in cols):
            mismatches.append(row)

    if not mismatches:
        pass
    else:
        for _m in mismatches[:1]:
            for _c in cols:
                pass
