"""Tests for compare_csvs.py."""

from __future__ import annotations

import pandas as pd

from compare_csvs import ComparisonSummary, compare_dataframes


class TestCompareDataFrames:
    def test_identical_frames_have_no_differences(self) -> None:
        df = pd.DataFrame(
            {
                "location_id": [1, 2],
                "date": ["2024-05-01", "2024-05-02"],
                "pet": [10.0, 11.5],
            },
        )

        summary = compare_dataframes(df, df.copy(), summary_only=True)

        assert summary == ComparisonSummary(
            old_rows=2,
            new_rows=2,
            overlap_rows=2,
            differing_rows=0,
            differing_cells=0,
        )

    def test_different_frames_report_rows_and_cells(self) -> None:
        df_old = pd.DataFrame(
            {
                "location_id": [1, 2],
                "date": ["2024-05-01", "2024-05-02"],
                "pet": [10.0, 11.5],
            },
        )
        df_new = pd.DataFrame(
            {
                "location_id": [1, 2],
                "date": ["2024-05-01", "2024-05-02"],
                "pet": [10.0, 12.0],
            },
        )

        summary = compare_dataframes(df_old, df_new, summary_only=True)

        assert summary.differing_rows == 1
        assert summary.differing_cells == 1
