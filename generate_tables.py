"""Generate PET Analytics: Percentiles, Forecasts, and Decade Changes."""

from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ForecastRecord(TypedDict):
    """A single PET forecast row."""

    location_id: object
    forecast_year: int
    forecast_pet: float


def _quantile(values: object, quantile: float) -> float:
    """Calculate a quantile from a pandas Series-like object."""
    return float(cast("Any", values).quantile(quantile))


def _p10(values: object) -> float:
    """Calculate the 10th percentile from a pandas Series-like object."""
    return _quantile(values, 0.1)


def _p90(values: object) -> float:
    """Calculate the 90th percentile from a pandas Series-like object."""
    return _quantile(values, 0.9)


def generate_percentiles(df: pd.DataFrame) -> None:
    """Calculate the 10th and 90th percentile of PET per year per location."""
    logger.info("Generating percentiles.csv...")
    df_any = cast("Any", df)
    agg_df = (
        df_any.groupby(["year", "location_id"])["pet"]
        .agg(p10=_p10, p90=_p90)
        .reset_index()
    )

    rounded_df = agg_df.round(1)
    rounded_df.to_csv("percentiles.csv", index=False)


def generate_forecast(df: pd.DataFrame) -> None:
    """Generate linear trend forecasts for future decades (2030, 2040, 2050)."""
    logger.info("Generating forecast.csv...")
    df_any = cast("Any", df)
    yearly_avg = df_any.groupby(["location_id", "year"])["pet"].mean().reset_index()

    forecast_records: list[ForecastRecord] = []
    for loc_id, group in yearly_avg.groupby("location_id"):
        if len(group) > 1:
            x = np.asarray(group["year"].to_numpy(), dtype=float)
            y = np.asarray(group["pet"].to_numpy(), dtype=float)

            slope, intercept = (float(value) for value in np.polyfit(x, y, 1))

            for future_year in [2030, 2040, 2050]:
                projected_pet = (slope * future_year) + intercept
                forecast_records.append(
                    {
                        "location_id": loc_id,
                        "forecast_year": future_year,
                        "forecast_pet": round(projected_pet, 2),
                    },
                )

    forecast_df = pd.DataFrame(forecast_records)
    cast("Any", forecast_df).to_csv("forecast.csv", index=False)


def generate_change_per_decade(df: pd.DataFrame) -> None:
    """Calculate the change in average PET between decades."""
    logger.info("Generating change_per_decade.csv...")
    df_any = cast("Any", df)
    yearly_avg = df_any.groupby(["location_id", "year"])["pet"].mean().reset_index()

    yearly_avg["decade_start"] = (yearly_avg["year"] // 10) * 10

    decade_avg = (
        yearly_avg.groupby(["location_id", "decade_start"])["pet"]
        .mean()
        .reset_index()
    )

    decade_avg = decade_avg.sort_values(["location_id", "decade_start"])
    decade_avg["change_value"] = decade_avg.groupby("location_id")["pet"].diff()

    decade_avg["decade"] = decade_avg["decade_start"].astype(str) + "s"

    decade_avg = decade_avg.dropna(subset=["change_value"])

    final_df = decade_avg[["location_id", "decade", "change_value"]].round(2)
    final_df.to_csv("change_per_decade.csv", index=False)


def main() -> None:
    """Load PET data and generate all analytical CSVs."""
    logger.info("Loading pet.csv into memory...")
    try:
        df = cast("pd.DataFrame", cast("Any", pd).read_csv("pet.csv"))
    except FileNotFoundError:
        logger.exception("pet.csv not found. Ensure calculate_pet.py has run.")
        return

    df_any = cast("Any", df)
    df_any["date"] = cast("Any", pd).to_datetime(df_any["date"])
    df_any["year"] = df_any["date"].dt.year

    generate_percentiles(df)
    generate_forecast(df)
    generate_change_per_decade(df)

    logger.info("Analytics generation complete.")


if __name__ == "__main__":
    main()
