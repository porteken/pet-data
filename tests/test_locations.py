"""Tests for database-ready locations CSV generation."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pandas as pd

import locations

if TYPE_CHECKING:
    import pytest


def test_main_saves_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    sample = pd.DataFrame(
        {
            "id": [0],
            "city": ["Testville"],
            "state": ["TS"],
            "lat": [30.123],
            "lng": [-90.456],
        }
    )

    monkeypatch.setattr(locations, "build_locations_frame", lambda _url: sample)
    monkeypatch.chdir(tmp_path)
    output_file = tmp_path / "locations.csv"

    locations.main()

    assert output_file.exists()


def test_build_locations_frame_renames_location_id_to_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = pd.DataFrame(
        {
            "location_id": [0],
            "city": ["Testville"],
            "state": ["TS"],
            "lat": [30.123],
            "lng": [-90.456],
        }
    )

    monkeypatch.setattr(locations, "load_data", lambda _url: pd.DataFrame())
    monkeypatch.setattr(locations, "filter_bounding_box", lambda df: df)
    monkeypatch.setattr(locations, "process_cities", lambda _df: sample)

    result = locations.build_locations_frame("https://example.test/cities.csv")

    assert list(result.columns) == ["id", "city", "state", "lat", "lng"]
    assert result.iloc[0]["id"] == 0
