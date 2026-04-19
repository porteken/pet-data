"""Tests for pull_mrt.py request window helpers."""

from __future__ import annotations

from datetime import date

from pull_mrt import _resolve_mrt_request_window


class TestResolveMrtRequestWindow:
    def test_explicit_smoke_window_returns_requested_end_date(self) -> None:
        months, days, expected_end = _resolve_mrt_request_window(
            year=2025,
            month=5,
            start_date="2025-05-01",
            end_date="2025-05-07",
        )

        assert months == ["05"]
        assert days == [f"{day:02d}" for day in range(1, 8)]
        assert expected_end == date(2025, 5, 7)

    def test_default_month_window_returns_month_end(self) -> None:
        months, days, expected_end = _resolve_mrt_request_window(
            year=2025,
            month=5,
        )

        assert months == ["05"]
        assert days[0] == "01"
        assert days[-1] == "31"
        assert expected_end == date(2025, 5, 31)
