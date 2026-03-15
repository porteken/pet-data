"""Shared configuration values used by weather and MRT pull scripts."""

from __future__ import annotations

from datetime import date, timedelta

PULL_START_DATE = date(2000, 1, 1)
SHARED_AREA: tuple[float, float, float, float] = (49.25, -124.5, 24.25, -66.5)


def pull_end_date(current_date: date | None = None) -> date:
    """Return the last day of the month before the current date."""
    reference_date = date.today() if current_date is None else current_date
    return reference_date.replace(day=1) - timedelta(days=1)


def build_year_date_range(year: int) -> str:
    """Return the supported pull window for a specific year."""
    end_date = pull_end_date()
    year_start = max(PULL_START_DATE, date(year, 1, 1))
    year_end = min(end_date, date(year, 12, 31))
    if year_start > year_end:
        msg = (
            f"Year {year} falls outside the supported pull window "
            f"{PULL_START_DATE.isoformat()} to {end_date.isoformat()}."
        )
        raise ValueError(msg)

    return f"{year_start.isoformat()}/{year_end.isoformat()}"


def build_year_months(year: int) -> list[str]:
    """Return the supported month list for a specific year."""
    end_date = pull_end_date()
    year_start = max(PULL_START_DATE, date(year, 1, 1))
    year_end = min(end_date, date(year, 12, 31))
    if year_start > year_end:
        msg = (
            f"Year {year} falls outside the supported pull window "
            f"{PULL_START_DATE.isoformat()} to {end_date.isoformat()}."
        )
        raise ValueError(msg)

    return [f"{month:02d}" for month in range(year_start.month, year_end.month + 1)]


def shared_area() -> list[float]:
    """Return a fresh geographic bounding box list for config defaults."""
    return list(SHARED_AREA)
