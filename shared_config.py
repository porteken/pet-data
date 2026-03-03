"""Shared configuration values used by weather and MRT pull scripts."""

from __future__ import annotations

SHARED_START_DATE = "2000-05-01"
SHARED_END_DATE = "2025-10-01"
SHARED_MONTHS: tuple[int, ...] = (5, 6, 7, 8, 9)
SHARED_AREA: tuple[float, float, float, float] = (49.25, -124.5, 24.25, -66.5)


def shared_months() -> list[int]:
    """Return a fresh month list for config defaults."""
    return list(SHARED_MONTHS)


def shared_area() -> list[float]:
    """Return a fresh geographic bounding box list for config defaults."""
    return list(SHARED_AREA)
