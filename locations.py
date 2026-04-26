"""Generate the database-ready locations CSV."""

from __future__ import annotations

import logging

from cities import (
    CITY_COORD_DECIMALS,
    DataFrame,
    filter_bounding_box,
    load_data,
    process_cities,
)

LOGGER = logging.getLogger(__name__)
OUTPUT_FILE = "locations.csv"


def build_locations_frame(url: str) -> DataFrame:
    """Return processed city rows with the database column name for the key."""
    city_frame = process_cities(filter_bounding_box(load_data(url)))
    return city_frame.rename(columns={"location_id": "id"})


def main() -> None:
    """Write the database-ready locations.csv file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    url = (
        "https://raw.githubusercontent.com/plotly/datasets/master/us-cities-top-1k.csv"
    )
    locations_frame = build_locations_frame(url)
    locations_frame.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format=f"%.{CITY_COORD_DECIMALS}f",
    )
    LOGGER.info(
        "Successfully saved %d locations to %s",
        len(locations_frame),
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
