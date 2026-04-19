"""Shared keyed comparison helpers for PET datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

KEY_COLUMNS = ["location_id", "date"]


@dataclass(frozen=True)
class PairMetrics:
    """Summary metrics for a keyed pairwise PET comparison."""

    left_rows: int
    right_rows: int
    overlap_rows: int
    missing_in_right: int
    missing_in_left: int
    exact_match_count: int
    exact_match_ratio: float
    mae: float | None


def normalize_pet_frame(df: pd.DataFrame, *, value_name: str) -> pd.DataFrame:
    """Return a normalized PET frame with stable key typing and sort order."""
    normalized = df.copy()
    normalized["location_id"] = cast(
        "pd.Series",
        pd.to_numeric(
            normalized["location_id"],
            errors="coerce",
        ),
    ).astype("Int64")
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized[value_name] = pd.to_numeric(normalized[value_name], errors="coerce")
    return normalized.sort_values(KEY_COLUMNS).reset_index(drop=True)


def merge_pet_frames(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_value_name: str,
    right_value_name: str,
) -> pd.DataFrame:
    """Outer-merge two PET frames on location/date."""
    return normalize_pet_frame(left_df, value_name=left_value_name).merge(
        normalize_pet_frame(right_df, value_name=right_value_name),
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
    )


def compute_pair_metrics(
    merged: pd.DataFrame,
    *,
    left_value_name: str,
    right_value_name: str,
) -> PairMetrics:
    """Compute keyed overlap/missing/exact/MAE metrics."""
    left_series = merged[left_value_name]
    right_series = merged[right_value_name]
    overlap_mask = left_series.notna() & right_series.notna()
    exact_mask = overlap_mask & (left_series.round(4) == right_series.round(4))
    mae = (
        float(
            cast("Any", (left_series[overlap_mask] - right_series[overlap_mask]))
            .abs()
            .mean(),
        )
        if overlap_mask.any()
        else None
    )

    overlap_rows = int(cast("Any", overlap_mask).sum())
    exact_match_count = int(cast("Any", exact_mask).sum())
    exact_match_ratio = exact_match_count / overlap_rows if overlap_rows else 0.0

    return PairMetrics(
        left_rows=int(cast("Any", left_series.notna()).sum()),
        right_rows=int(cast("Any", right_series.notna()).sum()),
        overlap_rows=overlap_rows,
        missing_in_right=int(cast("Any", left_series.notna()).sum() - overlap_rows),
        missing_in_left=int(cast("Any", right_series.notna()).sum() - overlap_rows),
        exact_match_count=exact_match_count,
        exact_match_ratio=exact_match_ratio,
        mae=mae,
    )


def compute_year_metrics(
    merged: pd.DataFrame,
    *,
    left_value_name: str,
    right_value_name: str,
) -> dict[int, PairMetrics]:
    """Compute pair metrics grouped by year."""
    year_metrics: dict[int, PairMetrics] = {}
    dated = merged.copy()
    dated["year"] = pd.to_datetime(dated["date"]).dt.year
    for year, year_df in dated.groupby("year", dropna=True):
        year_metrics[int(cast("Any", year))] = compute_pair_metrics(
            year_df,
            left_value_name=left_value_name,
            right_value_name=right_value_name,
        )
    return year_metrics


def sample_differences(
    merged: pd.DataFrame,
    *,
    left_value_name: str,
    right_value_name: str,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Return a small keyed sample of overlapping PET differences."""
    overlap_mask = merged[left_value_name].notna() & merged[right_value_name].notna()
    diff_mask = overlap_mask & (
        merged[left_value_name].round(4) != merged[right_value_name].round(4)
    )
    sampled = merged.loc[
        diff_mask,
        [*KEY_COLUMNS, left_value_name, right_value_name],
    ].copy()
    if sampled.empty:
        return sampled
    sampled["abs_diff"] = (sampled[left_value_name] - sampled[right_value_name]).abs()
    return sampled.sort_values(
        ["abs_diff", *KEY_COLUMNS],
        ascending=[False, True, True],
    ).head(max_rows)


def metrics_to_dict(metrics: PairMetrics) -> dict[str, Any]:
    """Serialize metrics for easy printing or JSON export."""
    return {
        "left_rows": metrics.left_rows,
        "right_rows": metrics.right_rows,
        "overlap_rows": metrics.overlap_rows,
        "missing_in_right": metrics.missing_in_right,
        "missing_in_left": metrics.missing_in_left,
        "exact_match_count": metrics.exact_match_count,
        "exact_match_ratio": metrics.exact_match_ratio,
        "mae": metrics.mae,
    }
