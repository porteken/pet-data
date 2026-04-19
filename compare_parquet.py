"""Compare weather data stored in Parquet format."""

import sys
from pathlib import Path

import pandas as pd


def load_data(
    base_path: str,
    year: int,
    month: str | int,
    day: int,
    limit_files: int = 1,
) -> pd.DataFrame:
    """Load parquet data for a specific year, month, and day."""
    base = Path(base_path)
    dfs = []

    # Check for both string and integer month formats
    month_str = str(month).zfill(2)
    month_int = int(month)

    search_dirs = [
        base / f"year={year}" / f"month={month_str}",
        base / f"year={year}" / f"month={month_int}",
    ]

    paths = set()
    for d in search_dirs:
        if d.exists():
            paths.update(d.rglob("*.parquet"))

    for path in sorted(paths)[:limit_files]:
        df = pd.read_parquet(
            path,
            columns=[
                "lat",
                "lng",
                "timestamp",
                "temperature_c",
                "wind_speed",
                "relative_humidity",
            ],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        mask = df["timestamp"].dt.day == day
        dfs.append(df[mask])

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# Just 2024-05-01
current_df = load_data("weather_data_parquet", 2024, "05", 1, limit_files=1)

legacy_df = load_data(
    "/home/kenneth-porter/pet_files/weather_data_parquet",
    2024,
    "05",
    1,
    limit_files=50,
)

if current_df.empty or legacy_df.empty:
    sys.exit(0)

current_df["lat"] = current_df["lat"].round(4)
current_df["lng"] = current_df["lng"].round(4)
legacy_df["lat"] = legacy_df["lat"].round(4)
legacy_df["lng"] = legacy_df["lng"].round(4)

merged = current_df.merge(
    legacy_df,
    on=["lat", "lng", "timestamp"],
    suffixes=("_curr", "_leg"),
)

if not merged.empty:
    for var in ["temperature_c", "wind_speed", "relative_humidity"]:
        diff = (merged[f"{var}_curr"] - merged[f"{var}_leg"]).abs()
        merged[f"{var}_diff"] = diff
        top5 = merged.sort_values(f"{var}_diff", ascending=False).head(5)
else:
    pass
