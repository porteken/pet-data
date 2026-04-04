"""Partial integration tests for the pull_all pipeline DB load.

Verify that the pipeline loads data into the expected tables and
materialized views.  Tests are skipped when SUPABASE_DB_URI is not set
or the database is unreachable.

Run with:
    pytest tests/test_pull_all_db.py -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from psycopg2.extensions import connection

DB_URI = os.getenv("SUPABASE_DB_URI")

REQUIRED_TABLES = [
    "locations",
    "pet",
    "pet_change",
    "pet_forecast",
    "pet_percentiles",
]

REQUIRED_MATERIALIZED_VIEWS = [
    "pet_year",
    "pet_year_avg",
    "pet_year_max",
]


@pytest.fixture(scope="module")
def db_conn() -> Generator[connection, None, None]:
    """Yield a psycopg2 connection, skip if unavailable."""
    if not DB_URI:
        pytest.skip("SUPABASE_DB_URI not set")

    try:
        import psycopg2
    except ImportError:
        pytest.skip("psycopg2 not installed")

    try:
        conn = psycopg2.connect(DB_URI)
        conn.autocommit = True
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Cannot connect to database: {exc}")

    yield conn
    conn.close()


def _row_count(conn: connection, relation: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM pg_class WHERE relname = %s",
            (relation,),
        )
        row = cur.fetchone()
        assert row is not None
        if row[0] == 0:
            pytest.fail(f"Relation {relation!r} does not exist")
        cur.execute(f"SELECT COUNT(*) FROM {relation}")  # noqa: S608
        count_row = cur.fetchone()
        assert count_row is not None
        return count_row[0]


# ── table existence & data checks ────────────────────────────────────


@pytest.mark.parametrize("table", REQUIRED_TABLES)
def test_table_has_rows(db_conn: connection, table: str) -> None:
    count = _row_count(db_conn, table)
    assert count > 0, f"Table {table!r} is empty (0 rows)"


# ── materialized view existence & data checks ────────────────────────


@pytest.mark.parametrize("view", REQUIRED_MATERIALIZED_VIEWS)
def test_materialized_view_has_rows(db_conn: connection, view: str) -> None:
    count = _row_count(db_conn, view)
    assert count > 0, f"Materialized view {view!r} is empty (0 rows)"


# ── year‑range coverage check ────────────────────────────────────────


def test_pet_covers_year_range(db_conn: connection) -> None:
    """The pet table should have data spanning 2000 to at least 2025."""
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(EXTRACT(YEAR FROM date))::int, "
            "MAX(EXTRACT(YEAR FROM date))::int FROM pet"
        )
        result = cur.fetchone()
        assert result is not None
        min_year, max_year = result
    assert min_year is not None, "pet table has no date data"
    assert min_year <= 2000, f"Earliest year in pet is {min_year}, expected <= 2000"
    assert max_year >= 2025, f"Latest year in pet is {max_year}, expected >= 2025"


def test_locations_not_empty(db_conn: connection) -> None:
    count = _row_count(db_conn, "locations")
    assert count > 0, "locations table is empty"


def test_pet_year_avg_has_seasons(db_conn: connection) -> None:
    """pet_year_avg should contain the expected season labels."""
    with db_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT season FROM pet_year_avg ORDER BY season")
        seasons = {row[0] for row in cur.fetchall()}
    expected = {"Jan-Dec", "Feb", "March-May", "June-August", "September-November"}
    assert expected.issubset(seasons), f"Missing seasons: {expected - seasons}"


def test_pet_year_max_has_seasons(db_conn: connection) -> None:
    """pet_year_max should contain the expected season labels."""
    with db_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT season FROM pet_year_max ORDER BY season")
        seasons = {row[0] for row in cur.fetchall()}
    expected = {"Jan-Dec", "Feb", "March-May", "June-August", "September-November"}
    assert expected.issubset(seasons), f"Missing seasons: {expected - seasons}"
