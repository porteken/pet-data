"""Compare time shifts between current and legacy weather data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    """Analyze different time shifts to find the best match with legacy data."""
    current_path = "weather_data_parquet"
    legacy_path = "/home/kenneth-porter/pet_files/weather.parquet"

    files = list(Path(current_path).rglob("*.parquet"))
    if not files:
        return

    dfs = [pd.read_parquet(f) for f in files]
    df_curr = pd.concat(dfs, ignore_index=True)
    if "timestamp" in df_curr.columns:
        df_curr = df_curr.rename(columns={"timestamp": "time"})
    df_curr["time"] = pd.to_datetime(df_curr["time"])

    try:
        # Define the filters for May 1-7 in 2024 and 2025
        # Pyarrow filters: (column, op, value)
        # We want (time >= 2024-05-01 AND time < 2024-05-08) OR (time >= 2025-05-01 AND time < 2025-05-08)
        # pyarrow.parquet.read_table supports filters

        table = pq.read_table(
            legacy_path,
            columns=["location_id", "time", "t", "v", "rh"],
            filters=[
                [
                    ("time", ">=", pd.Timestamp("2024-05-01")),
                    ("time", "<", pd.Timestamp("2024-05-08")),
                ],
                [
                    ("time", ">=", pd.Timestamp("2025-05-01")),
                    ("time", "<", pd.Timestamp("2025-05-08")),
                ],
            ],
        )
        df_leg = table.to_pandas()
        df_leg["time"] = pd.to_datetime(df_leg["time"])
    except Exception:  # noqa: BLE001
        df_leg = pd.read_parquet(
            legacy_path,
            columns=["location_id", "time", "t", "v", "rh"],
        )
        df_leg["time"] = pd.to_datetime(df_leg["time"])
        mask = ((df_leg["time"] >= "2024-05-01") & (df_leg["time"] < "2024-05-08")) | (
            (df_leg["time"] >= "2025-05-01") & (df_leg["time"] < "2025-05-08")
        )
        df_leg = df_leg[mask].copy()

    mask_curr = (
        (df_curr["time"] >= "2024-05-01") & (df_curr["time"] < "2024-05-08")
    ) | ((df_curr["time"] >= "2025-05-01") & (df_curr["time"] < "2025-05-08"))
    df_curr = df_curr[mask_curr].copy()

    if df_curr.empty or df_leg.empty:
        return

    shifts = [-24, -12, -6, -3, -1, 0, 1, 3, 6, 12, 24]
    results = []
    for h in shifts:
        df_shifted = df_curr.copy()
        df_shifted["time"] = df_shifted["time"] + pd.Timedelta(hours=h)
        merged = df_shifted.merge(
            df_leg,
            on=["location_id", "time"],
            suffixes=("_c", "_l"),
        )
        if not merged.empty:
            t_mae = np.mean(np.abs(merged["temperature_c"] - merged["t"]))
            v_mae = np.mean(np.abs(merged["wind_speed"] - merged["v"]))
            rh_mae = np.mean(np.abs(merged["relative_humidity"] - merged["rh"]))
            results.append(
                {
                    "shift": h,
                    "overlap": len(merged),
                    "t_mae": t_mae,
                    "v_mae": v_mae,
                    "rh_mae": rh_mae,
                },
            )
        else:
            results.append(
                {
                    "shift": h,
                    "overlap": 0,
                    "t_mae": np.nan,
                    "v_mae": np.nan,
                    "rh_mae": np.nan,
                },
            )

    res_df = pd.DataFrame(results)
    if not bool(res_df["t_mae"].isna().all()):
        res_df.loc[res_df["t_mae"].idxmin()]


if __name__ == "__main__":
    main()
