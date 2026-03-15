"""Get cities used in application and format for Supabase."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, TypeAlias, cast

DataFrame: TypeAlias = Any
np: Any = cast("Any", import_module("numpy"))
pd: Any = cast("Any", import_module("pandas"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(url: str) -> DataFrame:
    """Load and normalize data from Plotly."""
    df = pd.read_csv(url)
    rename_map = {
        "City": "city",
        "name": "city",
        "State": "state",
        "Population": "population",
        "pop": "population",
        "lon": "lng",
    }
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.lower()
    return df


def filter_bounding_box(df: DataFrame) -> DataFrame:
    """Filter cities within continental US."""
    min_lat, max_lat = 24.25, 49.25
    min_lng, max_lng = -124.5, -66.5
    return df[
        (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
        & (df["lng"] >= min_lng)
        & (df["lng"] <= max_lng)
    ]


def process_cities(df: DataFrame) -> DataFrame:
    """Keep highest population city per .25' and return top 500 cities."""
    df = df.copy()

    df[["real_lat", "real_lng"]] = df[["lat", "lng"]].round(2)
    df[["lat", "lng"]] = np.round(df[["lat", "lng"]] * 4) / 4

    idx = df.groupby(["lat", "lng"])["population"].idxmax()
    df = df.loc[idx]

    df = df.nlargest(500, "population").sort_values("population", ascending=False)
    df = df.reset_index(drop=True).reset_index(names="location_id")

    return df[
        [
            "location_id",
            "city",
            "state",
            "population",
            "lat",
            "lng",
            "real_lat",
            "real_lng",
        ]
    ]


def main() -> None:
    """Orchestration function."""
    logger.info("Generating locations dataset...")
    url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
    )
    df = load_data(url)
    df = filter_bounding_box(df)
    df = process_cities(df)

    df.to_csv("cities.csv", index=False)
    logger.info("Successfully saved %d cities to cities.csv", len(df))


if __name__ == "__main__":
    main()
