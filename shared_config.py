"""Shared configuration values used by weather and MRT pull scripts."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc

PULL_START_DATE = date(2000, 1, 1)
SHARED_AREA: tuple[float, float, float, float] = (49.25, -124.5, 24.25, -66.5)
DECEMBER = 12


def pull_end_date(current_date: date | None = None) -> date:
    """Return the last day of the month before the current date."""
    reference_date = (
        datetime.now(tz=UTC).date() if current_date is None else current_date
    )
    return reference_date.replace(day=1) - timedelta(days=1)


def _month_end(year: int, month: int) -> date:
    """Return the last day for the requested month."""
    if month == DECEMBER:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def build_year_date_range(year: int, *, month: int | None = None) -> str:
    """Return the supported pull window for a specific year or month."""
    end_date = pull_end_date()
    range_start = max(PULL_START_DATE, date(year, 1, 1))
    range_end = min(end_date, date(year, 12, 31))

    if month is not None:
        month_start = date(year, month, 1)
        range_start = max(range_start, month_start)
        range_end = min(range_end, _month_end(year, month))

    if range_start > range_end:
        msg = (
            f"Requested window for year {year}"
            f"{'' if month is None else f', month {month:02d}'} "
            f"falls outside the supported pull window "
            f"{PULL_START_DATE.isoformat()} to {end_date.isoformat()}."
        )
        raise ValueError(msg)

    return f"{range_start.isoformat()}/{range_end.isoformat()}"


def build_year_months(year: int, *, month: int | None = None) -> list[str]:
    """Return the supported month list for a specific year or month."""
    end_date = pull_end_date()
    range_start = max(PULL_START_DATE, date(year, 1, 1))
    range_end = min(end_date, date(year, 12, 31))

    if month is not None:
        month_start = date(year, month, 1)
        range_start = max(range_start, month_start)
        range_end = min(range_end, _month_end(year, month))

    if range_start > range_end:
        msg = (
            f"Requested window for year {year}"
            f"{'' if month is None else f', month {month:02d}'} "
            f"falls outside the supported pull window "
            f"{PULL_START_DATE.isoformat()} to {end_date.isoformat()}."
        )
        raise ValueError(msg)

    return [
        f"{month_value:02d}"
        for month_value in range(range_start.month, range_end.month + 1)
    ]


def shared_area() -> list[float]:
    """Return a fresh geographic bounding box list for config defaults."""
    return list(SHARED_AREA)
