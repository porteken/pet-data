"""Get cities used in application."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_data(url: str) -> pd.DataFrame:
    """Load and normalize data."""
    df = pd.read_csv(url)
    df = df.rename(columns={"lon": "lng"})
    df.columns = df.columns.str.lower()
    return df


def filter_bounding_box(df: pd.DataFrame) -> pd.DataFrame:
    """Filter cities within continental US."""
    min_lat = 24.25
    max_lat = 49.25
    min_lng = -124.5
    max_lng = -66.5
    return df[
        (df["lat"] >= min_lat)
        & (df["lat"] <= max_lat)
        & (df["lng"] >= min_lng)
        & (df["lng"] <= max_lng)
    ]


def process_cities(df: pd.DataFrame) -> pd.DataFrame:
    """Keep highest population city per .25' and return top 500 cities."""
    df = df.copy()

    df[["real_lat", "real_lng"]] = df[["lat", "lng"]].round(2)

    df[["lat", "lng"]] = np.round(df[["lat", "lng"]] * 4) / 4

    df = df.loc[df.groupby(["lat", "lng"])["population"].idxmax()]

    df = df.nlargest(500, "population").sort_values("population", ascending=False)

    return df.reset_index(drop=True).reset_index(names="location_id")


def main() -> None:
    """Orchestration function."""
    df = load_data(
        "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv",
    )
    df = filter_bounding_box(df)
    df = process_cities(df)
    df.to_csv("cities.csv", index=False)


if __name__ == "__main__":
    main()
