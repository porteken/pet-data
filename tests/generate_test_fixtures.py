"""Generate minimal synthetic weather + MRT parquet fixtures for act integration testing.

Creates ERA5-style combined parquet for the era5 passthrough path, which
exercises the full combine → calculate_pet → generate_analytics pipeline
with the minimum amount of data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ERA5_OUT = Path("test_era5_parquet")
YEAR = 2000
DAYS = 3
HOURS = DAYS * 24

# Two cities from the real city_to_tile mapping
CITIES = [
    {"location_id": 7, "lat": 32.75, "lng": -117.25, "tile_id": 11},
    {"location_id": 116, "lat": 26.0, "lng": -97.5, "tile_id": 1},
]


def _generate_era5_combined() -> None:
    """Write ERA5-style combined parquet with all columns pre-merged."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(f"{YEAR}-01-01", periods=HOURS, freq="h")

    for city in CITIES:
        records = []
        for ts in timestamps:
            records.append(
                {
                    "location_id": city["location_id"],
                    "lat": city["lat"],
                    "lng": city["lng"],
                    "time": ts,
                    "t": round(float(rng.uniform(5, 35)), 5),
                    "v": round(float(rng.uniform(0.5, 8)), 5),
                    "rh": round(float(rng.uniform(20, 90)), 5),
                    "mrt": round(float(rng.uniform(10, 55)), 5),
                }
            )

        df = pd.DataFrame(records)
        out_dir = ERA5_OUT / f"year={YEAR}" / f"tile_id={city['tile_id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "combined.parquet", index=False)

    logger.info("Wrote ERA5 combined parquet for %d tiles to %s", len(CITIES), ERA5_OUT)


if __name__ == "__main__":
    _generate_era5_combined()
    logger.info("Done — synthetic test fixtures generated.")
