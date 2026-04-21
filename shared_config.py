"""Shared configuration for weather and MRT scripts."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc

PULL_START_DATE = date(2000, 1, 1)
SHARED_AREA: tuple[float, float, float, float] = (49.25, -124.5, 24.25, -66.5)
DECEMBER = 12
MRT_INTERMEDIATE_LAG_DAYS = 5
MRT_CONSOLIDATED_LAG_MONTHS = 3


def _resolve_current_date(current_date: date | None = None) -> date:
    """Return the current UTC date or the provided override."""
    return datetime.now(tz=UTC).date() if current_date is None else current_date


def _month_start(current_date: date) -> date:
    """Return the first day of the month for a date."""
    return current_date.replace(day=1)


def _shift_month_start(current_date: date, months: int) -> date:
    """Shift a month-start date by a signed month delta."""
    month_index = current_date.year * 12 + (current_date.month - 1) + months
    shifted_year, shifted_month_index = divmod(month_index, 12)
    return date(shifted_year, shifted_month_index + 1, 1)


def pull_end_date(current_date: date | None = None) -> date:
    """Return the last day of the month before the current date."""
    reference_date = _resolve_current_date(current_date)
    return reference_date.replace(day=1) - timedelta(days=1)


def _month_end(year: int, month: int) -> date:
    """Return the last day for the requested month."""
    if month == DECEMBER:
        return date(year + 1, 1, 1) - timedelta(days=1)
    return date(year, month + 1, 1) - timedelta(days=1)


def build_year_date_bounds(year: int, *, month: int | None = None) -> tuple[date, date]:
    """Return the supported inclusive start/end dates for a year or month."""
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

    return range_start, range_end


def mrt_available_end_date(current_date: date | None = None) -> date:
    """Return the latest month-end safely available from the MRT dataset."""
    reference_date = _resolve_current_date(current_date) - timedelta(
        days=MRT_INTERMEDIATE_LAG_DAYS,
    )
    return _month_start(reference_date) - timedelta(days=1)


def mrt_consolidated_end_date(current_date: date | None = None) -> date:
    """Return the latest month-end safely available in consolidated MRT data."""
    reference_date = _resolve_current_date(current_date)
    consolidated_month_start = _shift_month_start(
        _month_start(reference_date),
        -MRT_CONSOLIDATED_LAG_MONTHS,
    )
    return _month_end(
        consolidated_month_start.year,
        consolidated_month_start.month,
    )


def build_mrt_date_bounds(year: int, *, month: int | None = None) -> tuple[date, date]:
    """Return the supported inclusive MRT start/end dates for a year or month."""
    end_date = mrt_available_end_date()
    range_start = max(PULL_START_DATE, date(year, 1, 1))
    range_end = min(end_date, date(year, 12, 31))

    if month is not None:
        month_start = date(year, month, 1)
        range_start = max(range_start, month_start)
        range_end = min(range_end, _month_end(year, month))

    if range_start > range_end:
        msg = (
            f"Requested MRT window for year {year}"
            f"{'' if month is None else f', month {month:02d}'} "
            f"falls outside the supported pull window "
            f"{PULL_START_DATE.isoformat()} to {end_date.isoformat()}."
        )
        raise ValueError(msg)

    return range_start, range_end


def build_mrt_months(year: int, *, month: int | None = None) -> list[str]:
    """Return the supported MRT month list for a specific year or month."""
    range_start, range_end = build_mrt_date_bounds(year, month=month)
    return [
        f"{month_value:02d}"
        for month_value in range(range_start.month, range_end.month + 1)
    ]


def build_mrt_days(year: int, *, month: int) -> list[str]:
    """Return the supported MRT day list for a specific month."""
    range_start, range_end = build_mrt_date_bounds(year, month=month)
    return [
        f"{day_value:02d}" for day_value in range(range_start.day, range_end.day + 1)
    ]


def mrt_product_type(year: int, *, month: int | None = None) -> str:
    """Return the MRT dataset variant to request for a year or month."""
    _, range_end = build_mrt_date_bounds(year, month=month)
    if range_end <= mrt_consolidated_end_date():
        return "consolidated_dataset"
    return "intermediate_dataset"


def build_year_date_range(year: int, *, month: int | None = None) -> str:
    """Return the supported pull window for a specific year or month."""
    range_start, range_end = build_year_date_bounds(year, month=month)
    return f"{range_start.isoformat()}/{range_end.isoformat()}"


def build_year_months(year: int, *, month: int | None = None) -> list[str]:
    """Return the supported month list for a specific year or month."""
    range_start, range_end = build_year_date_bounds(year, month=month)
    return [
        f"{month_value:02d}"
        for month_value in range(range_start.month, range_end.month + 1)
    ]


def build_full_date_range(*, start_year: int | None = None) -> str:
    """Return the full supported pull window, optionally clamped to a start year."""
    end_date = pull_end_date()
    effective_start = PULL_START_DATE
    if start_year is not None:
        effective_start = max(PULL_START_DATE, date(start_year, 1, 1))
    return f"{effective_start.isoformat()}/{end_date.isoformat()}"


def shared_area() -> list[float]:
    """Return geographic bounding box for defaults."""
    return list(SHARED_AREA)
