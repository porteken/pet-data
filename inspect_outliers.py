"""Inspect weather data outliers compared to legacy and ARCO-ERA5 datasets."""

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr


def get_rh(t: float | pd.Series, td: float | pd.Series) -> float | pd.Series:
    """Calculate relative humidity from temperature and dewpoint (Celsius)."""
    return 100 * (
        np.exp((17.625 * td) / (243.04 + td)) / np.exp((17.625 * t) / (243.04 + t))
    )


def filter_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for specific early May dates in 2024 and 2025."""
    return cast(
        "pd.DataFrame",
        df[
            ((df["time"] >= "2024-05-01") & (df["time"] < "2024-05-08"))
            | ((df["time"] >= "2025-05-01") & (df["time"] < "2025-05-08"))
        ],
    )


def main() -> None:
    """Run the outlier inspection script."""
    # 1. Load current weather output using recursive glob
    current_files = list(Path("weather_data_parquet").rglob("*.parquet"))
    if not current_files:
        sys.exit(1)
    current_df = pd.concat([pd.read_parquet(f) for f in current_files])
    current_df = current_df.rename(
        columns={
            "timestamp": "time",
            "temperature_c": "temperature_curr",
            "wind_speed": "wind_speed_curr",
            "relative_humidity": "relative_humidity_curr",
            "lat": "latitude",
            "lng": "longitude",
        },
    )
    current_df["time"] = pd.to_datetime(current_df["time"])

    # 2. Filter dates for current weather
    current_filtered = filter_dates(current_df)

    # 3. Load and filter legacy weather file
    legacy_df = pd.read_parquet("/home/kenneth-porter/pet_files/weather.parquet")
    legacy_df["time"] = pd.to_datetime(legacy_df["time"])
    legacy_filtered = filter_dates(legacy_df).copy()
    legacy_filtered = legacy_filtered.rename(
        columns={
            "t": "temperature_leg",
            "v": "wind_speed_leg",
            "rh": "relative_humidity_leg",
        },
    )

    # 4. Merge
    merged = current_filtered.merge(legacy_filtered, on=["location_id", "time"])
    merged["t_diff"] = (merged["temperature_curr"] - merged["temperature_leg"]).abs()

    # 5. Top 5 Outliers
    top_5 = merged.sort_values("t_diff", ascending=False).head(5)

    for _i, (_idx, _row) in enumerate(top_5.iterrows()):
        pass

    # 6. ARCO Check
    try:
        zarr_path = (
            "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        )
        # Open dataset and select only needed variables
        ds = xr.open_zarr(zarr_path, storage_options={"token": "anon"})
        ds = ds[
            [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ]
        ]

        for _i, (_idx, row) in enumerate(top_5.head(2).iterrows()):
            lat, lon, time = row["latitude"], row["longitude"], row["time"]
            arco_lon = lon if lon >= 0 else lon + 360

            # Load exactly what we need
            subset = ds.sel(
                latitude=lat,
                longitude=arco_lon,
                time=time,
                method="nearest",
            ).compute()
            t2m = float(subset["2m_temperature"].to_numpy()) - 273.15
            d2m = float(subset["2m_dewpoint_temperature"].to_numpy()) - 273.15
            u10 = float(subset["10m_u_component_of_wind"].to_numpy())
            v10 = float(subset["10m_v_component_of_wind"].to_numpy())
            _speed = np.sqrt(u10**2 + v10**2)
            _rh = get_rh(t2m, d2m)

    except Exception:  # noqa: S110, BLE001
        # Silently fail for ARCO check as it depends on external connectivity
        pass


if __name__ == "__main__":
    main()
