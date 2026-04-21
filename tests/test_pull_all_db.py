"""Integration tests for the pull_all pipeline database load."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from psycopg2.extensions import connection

pytestmark = pytest.mark.db

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


@pytest.mark.parametrize("table", REQUIRED_TABLES)
def test_table_has_rows(db_conn: connection, table: str) -> None:
    count = _row_count(db_conn, table)
    assert count > 0, f"Table {table!r} is empty (0 rows)"


@pytest.mark.parametrize("view", REQUIRED_MATERIALIZED_VIEWS)
def test_materialized_view_has_rows(db_conn: connection, view: str) -> None:
    count = _row_count(db_conn, view)
    assert count > 0, f"Materialized view {view!r} is empty (0 rows)"


def test_pet_covers_year_range(db_conn: connection) -> None:
    """The pet table should have data for the years it contains."""
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT MIN(EXTRACT(YEAR FROM date))::int, "
            "MAX(EXTRACT(YEAR FROM date))::int FROM pet"
        )
        result = cur.fetchone()
        assert result is not None
        min_year, max_year = result
    assert min_year is not None, "pet table has no date data"

    assert min_year >= 2000, f"Earliest year {min_year} is before supported start 2000"
    assert max_year <= 2026, f"Latest year {max_year} is beyond expected 2026"


def test_locations_not_empty(db_conn: connection) -> None:
    count = _row_count(db_conn, "locations")
    assert count > 0, "locations table is empty"


def _get_relation_columns(conn: connection, relation: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.attname "
            "FROM pg_catalog.pg_attribute AS a "
            "JOIN pg_catalog.pg_class AS c ON c.oid = a.attrelid "
            "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' "
            "AND c.relname = %s "
            "AND a.attnum > 0 "
            "AND NOT a.attisdropped "
            "ORDER BY a.attnum",
            (relation,),
        )
        return [row[0] for row in cur.fetchall()]


def _assert_year_season_pet_view(
    db_conn: connection, view: str, aggregate: str
) -> None:
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM ("  # noqa: S608
            "SELECT location_id, year, season, COUNT(*) AS row_count "
            f"FROM {view} "
            "GROUP BY location_id, year, season "
            "HAVING COUNT(*) > 1"
            ") AS duplicates"
        )  # noqa: RUF100, S608
        duplicate_count_row = cur.fetchone()
        assert duplicate_count_row is not None

        cur.execute(  # noqa: RUF100, S608
            "WITH pet_with_seasons AS ("  # noqa: S608
            "SELECT location_id, EXTRACT(YEAR FROM date)::int AS year, 'Annual'::text AS season, pet "
            "FROM pet "
            "UNION ALL "
            "SELECT "
            "location_id, "
            "EXTRACT(YEAR FROM date)::int AS year, "
            "CASE "
            "WHEN EXTRACT(MONTH FROM date)::int IN (12, 1, 2) THEN 'Winter' "
            "WHEN EXTRACT(MONTH FROM date)::int IN (3, 4, 5) THEN 'Spring' "
            "WHEN EXTRACT(MONTH FROM date)::int IN (6, 7, 8) THEN 'Summer' "
            "ELSE 'Fall' "
            "END AS season, "
            "pet "
            "FROM pet"
            "), expected AS ("
            "SELECT location_id, year, season, "
            f"ROUND({aggregate}(pet)::numeric, 1) AS pet "
            "FROM pet_with_seasons "
            "GROUP BY location_id, year, season"
            "), actual AS ("
            f"SELECT location_id, year, season, pet FROM {view}"
            "), missing AS ("
            "SELECT * FROM expected "
            "EXCEPT "
            "SELECT * FROM actual"
            "), extra AS ("
            "SELECT * FROM actual "
            "EXCEPT "
            "SELECT * FROM expected"
            ") "
            "SELECT "
            "(SELECT COUNT(*) FROM missing), "
            "(SELECT COUNT(*) FROM extra)"
        )
        diff_count_row = cur.fetchone()
        assert diff_count_row is not None

        cur.execute(
            f"SELECT DISTINCT season FROM {view} ORDER BY season"  # noqa: S608
        )
        seasons = {row[0] for row in cur.fetchall()}

    columns = _get_relation_columns(db_conn, view)
    assert columns == ["location_id", "year", "season", "pet"]

    duplicate_count = duplicate_count_row[0]
    missing_count, extra_count = diff_count_row
    assert duplicate_count == 0
    assert seasons
    assert "Annual" in seasons
    assert seasons.issubset({"Annual", "Winter", "Spring", "Summer", "Fall"})
    assert missing_count == 0
    assert extra_count == 0


def test_pet_year_avg_has_annual_and_seasons(db_conn: connection) -> None:
    """pet_year_avg should expose Annual plus Winter/Spring/Summer/Fall rows."""
    _assert_year_season_pet_view(db_conn, "pet_year_avg", "AVG")


def test_pet_year_max_has_annual_and_seasons(db_conn: connection) -> None:
    """Verify pet_year_max contains annual and seasonal rows."""
    _assert_year_season_pet_view(db_conn, "pet_year_max", "MAX")
