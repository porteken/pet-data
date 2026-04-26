"""Tests for historical PET export and merge helpers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

import historical_pet_update as hpu


class FakeCopy:
    def __init__(self, cursor: FakeCursor) -> None:
        """Initialize the fake COPY context manager."""
        self.cursor = cursor

    def __enter__(self) -> FakeCopy:  # noqa: PYI034
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)

    def __iter__(self) -> Iterator[str]:
        """Yield exported CSV lines."""
        return iter(["location_id,date,pet\n1,2000-05-01,20.0\n"])


class FakeCursor:
    def __init__(self) -> None:
        """Initialize the fake cursor."""
        self.copy_query = ""
        self.copy_params: tuple[str, str] | None = None
        self._fetchone_values = [("public.pet",)]

    def execute(self, query: object, params: object | None = None) -> None:
        _ = (query, params)

    def fetchone(self) -> tuple[str] | None:
        if not self._fetchone_values:
            return None
        return self._fetchone_values.pop(0)

    def copy(self, query: str, params: tuple[str, str] | None = None) -> FakeCopy:
        self.copy_query = query
        self.copy_params = params
        return FakeCopy(self)

    def __enter__(self) -> FakeCursor:  # noqa: PYI034
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Exit context manager."""
        _ = (exc_type, exc, tb)


class FakeConnection:
    def __init__(self) -> None:
        """Initialize fake connection."""
        self.cursor_instance = FakeCursor()

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        return None


def test_export_all_writes_pet_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_conn = FakeConnection()
    monkeypatch.setattr(hpu.psycopg, "connect", lambda _: fake_conn)
    monkeypatch.setenv("SUPABASE_DB_URI", "postgresql://example")

    output_path = tmp_path / "existing_pet.csv"
    hpu.export_pet(None, None, str(output_path))

    assert output_path.read_text() == "location_id,date,pet\n1,2000-05-01,20.0\n"
    assert "FROM public.pet" in fake_conn.cursor_instance.copy_query
    assert "ORDER BY location_id, date" in fake_conn.cursor_instance.copy_query
    assert fake_conn.cursor_instance.copy_params is None


def test_merge_csvs_prefers_later_sources_for_duplicates(tmp_path: Path) -> None:
    existing_pet = tmp_path / "existing_pet.csv"
    existing_pet.write_text(
        "location_id,date,pet\n1,2024-05-01,20.0\n2,2023-05-01,21.0\n",
    )

    pet_root = tmp_path / "pet_data_csv" / "year=2024"
    pet_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"location_id": 1, "date": "2024-05-01", "pet": 25.0},
            {"location_id": 3, "date": "2024-05-02", "pet": 30.0},
        ],
    ).to_parquet(pet_root / "pet_batch_0000_00.parquet", index=False)

    output_path = tmp_path / "pet_full.csv"
    hpu.merge_csvs(
        [str(tmp_path / "pet_data_csv")], [str(existing_pet)], str(output_path)
    )

    merged = pd.read_csv(output_path)
    assert len(merged) == 3
    assert float(
        merged.loc[
            (merged["location_id"] == 1) & (merged["date"] == "2024-05-01"),
            "pet",
        ].iloc[0],
    ) == pytest.approx(25.0)
