"""Compare current weather data with reference data."""

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def load_current_weather() -> pd.DataFrame | None:
    """Load current weather data from parquet files for May 2024 and 2025."""
    base = Path("weather_data_parquet")
    paths = list(base.glob("year=2024/month=05/*/*.parquet")) + list(
        base.glob("year=2025/month=05/*/*.parquet"),
    )
    if not paths:
        return None
    df = pd.concat([pd.read_parquet(p) for p in paths])
    # Map current column names to standard t, v, rh
    name_map = {
        "temperature_c": "t",
        "wind_speed": "v",
        "relative_humidity": "rh",
        "timestamp": "time",
    }
    df = df.rename(columns=name_map)
    df["time"] = pd.to_datetime(df["time"])
    df["location_id"] = df["location_id"].astype(str)
    return df


def main() -> None:
    """Compare current weather data with reference data."""
    df_curr = load_current_weather()
    if df_curr is None:
        return

    min_t, max_t = df_curr["time"].min(), df_curr["time"].max()

    # Use pyarrow dataset for efficient filtering before loading to pandas
    dataset = ds.dataset(
        "/home/kenneth-porter/pet_files/weather.parquet",
        format="parquet",
    )

    # Filter by time range
    # pyarrow expects timestamp objects for filtering
    pa_any: Any = pa
    filter_expr = (ds.field("time") >= pa_any.scalar(min_t)) & (
        ds.field("time") <= pa_any.scalar(max_t)
    )

    df_ref = dataset.to_table(
        columns=["location_id", "time", "t", "v", "rh"],
        filter=filter_expr,
    ).to_pandas()
    df_ref["location_id"] = df_ref["location_id"].astype(str)
    df_ref["time"] = pd.to_datetime(df_ref["time"])

    merged = df_curr.merge(
        df_ref,
        on=["location_id", "time"],
        suffixes=("_curr", "_ref"),
    )

    for v in ["t", "v", "rh"]:
        if f"{v}_curr" in merged.columns and f"{v}_ref" in merged.columns:
            (merged[f"{v}_curr"] - merged[f"{v}_ref"]).abs()

    if len(merged) > 0:
        # Check for systematic offset
        (merged["time"] - merged["time"]).unique()


if __name__ == "__main__":
    main()
