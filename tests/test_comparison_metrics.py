"""Tests for comparison_metrics.py."""

from __future__ import annotations

import math

import pandas as pd

from comparison_metrics import (
    compute_pair_metrics,
    compute_year_metrics,
    merge_pet_frames,
)


def test_compute_pair_metrics_handles_overlap_and_missing() -> None:
    left = pd.DataFrame(
        {
            "location_id": [1, 2],
            "date": ["2024-05-01", "2024-05-02"],
            "pet_dev": [10.0, 20.0],
        },
    )
    right = pd.DataFrame(
        {
            "location_id": [1, 3],
            "date": ["2024-05-01", "2024-05-03"],
            "pet_prd": [10.0, 30.0],
        },
    )

    merged = merge_pet_frames(
        left,
        right,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )
    metrics = compute_pair_metrics(
        merged,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )

    assert metrics.left_rows == 2
    assert metrics.right_rows == 2
    assert metrics.overlap_rows == 1
    assert metrics.missing_in_right == 1
    assert metrics.missing_in_left == 1
    assert metrics.exact_match_count == 1
    assert math.isclose(metrics.exact_match_ratio, 1.0)
    assert math.isclose(metrics.mae or 0.0, 0.0)


def test_compute_year_metrics_splits_by_year() -> None:
    left = pd.DataFrame(
        {
            "location_id": [1, 1],
            "date": ["2024-05-01", "2025-05-01"],
            "pet_dev": [10.0, 12.0],
        },
    )
    right = pd.DataFrame(
        {
            "location_id": [1, 1],
            "date": ["2024-05-01", "2025-05-01"],
            "pet_prd": [11.0, 12.0],
        },
    )

    merged = merge_pet_frames(
        left,
        right,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )
    metrics = compute_year_metrics(
        merged,
        left_value_name="pet_dev",
        right_value_name="pet_prd",
    )

    assert math.isclose(metrics[2024].mae or 0.0, 1.0)
    assert metrics[2024].exact_match_count == 0
    assert math.isclose(metrics[2025].mae or 0.0, 0.0)
    assert metrics[2025].exact_match_count == 1
