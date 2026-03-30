"""Generate minimal synthetic weather + MRT parquet fixtures for act integration testing.

Creates ERA5-style combined parquet for the era5 passthrough path and
separate weather + MRT parquet files for the CDS merge path, exercising
the full combine → calculate_pet → generate_analytics pipeline with
the minimum amount of data.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ERA5_OUT = Path("test_era5_parquet")
WEATHER_OUT = Path("weather_data_parquet")
MRT_OUT = Path("utci_data_parquet")
YEAR = 2000
DAYS = 3
HOURS = DAYS * 24

# Two cities from the real city_to_tile mapping
CITIES = [
    {"location_id": 7, "lat": 32.75, "lng": -117.25, "tile_id": 11},
    {"location_id": 116, "lat": 26.0, "lng": -97.5, "tile_id": 1},
]


def _generate_era5_combined(out_root: Path | None = None) -> None:
    """Write ERA5-style combined parquet with all columns pre-merged."""
    out = out_root or ERA5_OUT
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
        out_dir = out / f"year={YEAR}" / f"tile_id={city['tile_id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / "combined.parquet", index=False)

    logger.info("Wrote ERA5 combined parquet for %d tiles to %s", len(CITIES), out)


def generate_weather_parquet(
    out_root: Path | None = None,
    *,
    year: int = YEAR,
    month: int = 1,
    days: int = DAYS,
    cities: list[dict] | None = None,
) -> None:
    """Write CDS-style weather parquet files partitioned by city_shard_index."""
    out = out_root or WEATHER_OUT
    rng = np.random.default_rng(99)
    city_list = cities or CITIES
    hours = days * 24
    timestamps = pd.date_range(f"{year}-{month:02d}-01", periods=hours, freq="h")

    records = []
    for city in city_list:
        for ts in timestamps:
            records.append(
                {
                    "location_id": city["location_id"],
                    "timestamp": ts,
                    "lat": city["lat"],
                    "lng": city["lng"],
                    "temperature_c": round(float(rng.uniform(5, 35)), 5),
                    "wind_speed": round(float(rng.uniform(0.5, 8)), 5),
                    "relative_humidity": round(float(rng.uniform(20, 90)), 5),
                }
            )

    df = pd.DataFrame(records)
    shard_dir = out / "city_shard_index=0"
    shard_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(shard_dir / f"weather_{year}_{month:02d}.parquet", index=False)
    logger.info("Wrote %d weather rows to %s", len(df), shard_dir)


def generate_mrt_parquet(
    out_root: Path | None = None,
    *,
    year: int = YEAR,
    month: int = 1,
    days: int = DAYS,
    cities: list[dict] | None = None,
) -> None:
    """Write CDS-style MRT parquet files partitioned by year/month/tile_id."""
    out = out_root or MRT_OUT
    rng = np.random.default_rng(77)
    city_list = cities or CITIES
    hours = days * 24
    timestamps = pd.date_range(f"{year}-{month:02d}-01", periods=hours, freq="h")

    tiles: dict[int, list[dict]] = {}
    for city in city_list:
        tiles.setdefault(city["tile_id"], []).append(city)

    for tile_id, tile_cities in tiles.items():
        records = []
        for city in tile_cities:
            for ts in timestamps:
                records.append(
                    {
                        "timestamp": ts,
                        "lat": city["lat"],
                        "lng": city["lng"],
                        "mean_radiant_temperature_c": round(
                            float(rng.uniform(10, 55)), 5
                        ),
                    }
                )

        df = pd.DataFrame(records)
        tile_dir = out / f"year={year}" / f"month={month:02d}" / f"tile_id={tile_id}"
        tile_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tile_dir / "mrt.parquet", index=False)
        logger.info("Wrote %d MRT rows for tile %d to %s", len(df), tile_id, tile_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic test fixtures")
    parser.add_argument(
        "--mode",
        choices=["era5", "weather-mrt", "all"],
        default="all",
        help="Which fixtures to generate (default: all)",
    )
    parser.add_argument("--era5-out", type=Path, default=ERA5_OUT)
    parser.add_argument("--weather-out", type=Path, default=WEATHER_OUT)
    parser.add_argument("--mrt-out", type=Path, default=MRT_OUT)
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--days", type=int, default=DAYS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.mode in ("era5", "all"):
        _generate_era5_combined(out_root=args.era5_out)
    if args.mode in ("weather-mrt", "all"):
        generate_weather_parquet(
            out_root=args.weather_out,
            year=args.year,
            month=args.month,
            days=args.days,
        )
        generate_mrt_parquet(
            out_root=args.mrt_out,
            year=args.year,
            month=args.month,
            days=args.days,
        )
    logger.info("Done — synthetic test fixtures generated.")
