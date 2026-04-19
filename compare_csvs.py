"""Script to compare two CSV files and identify differing values.

Usage:
    python compare_csvs.py old.csv new.csv
"""

import argparse
import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Configure logging to behave like print for this CLI tool
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logging.raiseExceptions = False
logger = logging.getLogger(__name__)

DEFAULT_MAX_DIFFS = 100


@dataclass(frozen=True)
class ComparisonSummary:
    """Summary statistics for a CSV comparison run."""

    old_rows: int
    new_rows: int
    overlap_rows: int
    differing_rows: int
    differing_cells: int


def _comparison_mask(
    df_old_subset: pd.DataFrame,
    df_new_subset: pd.DataFrame,
) -> pd.DataFrame:
    comparison_mask = df_old_subset != df_new_subset
    return comparison_mask & ~(df_old_subset.isna() & df_new_subset.isna())


def _log_shape_warnings(df_old: pd.DataFrame, df_new: pd.DataFrame) -> None:
    if df_old.shape == df_new.shape:
        return

    logger.warning(
        "Warning: Shapes differ. Old: %s, New: %s",
        df_old.shape,
        df_new.shape,
    )


def _common_columns(df_old: pd.DataFrame, df_new: pd.DataFrame) -> list[str]:
    common_cols = [col for col in df_old.columns if col in df_new.columns]
    if len(common_cols) != len(df_old.columns) or len(common_cols) != len(
        df_new.columns,
    ):
        logger.warning("Warning: Columns differ. Common: %s", common_cols)
    return common_cols


def _iter_differing_cells(
    comparison_mask: pd.DataFrame,
    common_cols: list[str],
    df_old_subset: pd.DataFrame,
    df_new_subset: pd.DataFrame,
) -> tuple[int, int, list[str]]:
    differing_rows = int(comparison_mask.any(axis=1).sum())
    differing_cells = int(comparison_mask.to_numpy().sum())
    diff_messages: list[str] = []
    diff_indices = comparison_mask.any(axis=1)
    for idx in comparison_mask.index[diff_indices]:
        row_diffs = comparison_mask.loc[idx]
        for col in common_cols:
            if not row_diffs[col]:
                continue
            old_val = df_old_subset.loc[idx, col]
            new_val = df_new_subset.loc[idx, col]
            diff_messages.append(f"Row {idx}, Col '{col}': {old_val} -> {new_val}")
    return differing_rows, differing_cells, diff_messages


def _log_summary(summary: ComparisonSummary) -> None:
    logger.info(
        "Summary: old_rows=%d new_rows=%d overlap_rows=%d differing_rows=%d differing_cells=%d",
        summary.old_rows,
        summary.new_rows,
        summary.overlap_rows,
        summary.differing_rows,
        summary.differing_cells,
    )


def _log_diff_messages(
    diff_messages: list[str],
    *,
    max_diffs: int,
    summary_only: bool,
) -> None:
    if not diff_messages:
        logger.info("No differences found in overlapping data.")
        return

    if summary_only or max_diffs == 0:
        return

    logger.info("Showing up to %d differing cell(s):", max_diffs)
    for message in diff_messages[:max_diffs]:
        logger.info(message)

    remaining_diffs = len(diff_messages) - min(len(diff_messages), max_diffs)
    if remaining_diffs > 0:
        logger.info("... %d additional differing cell(s) not shown.", remaining_diffs)


def compare_dataframes(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    *,
    max_diffs: int = DEFAULT_MAX_DIFFS,
    summary_only: bool = False,
) -> ComparisonSummary:
    """Compare two DataFrames and emit a bounded human-readable report."""
    if max_diffs < 0:
        msg = "max_diffs must be >= 0"
        raise ValueError(msg)

    _log_shape_warnings(df_old, df_new)
    common_cols = _common_columns(df_old, df_new)

    min_rows = min(len(df_old), len(df_new))
    df_old_subset = df_old.iloc[:min_rows][common_cols]
    df_new_subset = df_new.iloc[:min_rows][common_cols]

    comparison_mask = _comparison_mask(df_old_subset, df_new_subset)
    differing_rows, differing_cells, diff_messages = _iter_differing_cells(
        comparison_mask,
        common_cols,
        df_old_subset,
        df_new_subset,
    )
    summary = ComparisonSummary(
        old_rows=len(df_old),
        new_rows=len(df_new),
        overlap_rows=min_rows,
        differing_rows=differing_rows,
        differing_cells=differing_cells,
    )

    _log_summary(summary)
    _log_diff_messages(
        diff_messages,
        max_diffs=max_diffs,
        summary_only=summary_only,
    )
    return summary


def compare_csvs(
    old_path: Path,
    new_path: Path,
    *,
    max_diffs: int = DEFAULT_MAX_DIFFS,
    summary_only: bool = False,
) -> ComparisonSummary | None:
    """Compare two CSV files and print the differences.

    Args:
        old_path: Path to the original CSV file.
        new_path: Path to the updated CSV file.
        max_diffs: Maximum number of differences to report.
        summary_only: If True, only return the summary without printing.

    Returns:
        A ComparisonSummary if the files were successfully compared, otherwise None.

    """
    if not old_path.exists():
        logger.error("Error: %s does not exist.", old_path)
        return None
    if not new_path.exists():
        logger.error("Error: %s does not exist.", new_path)
        return None

    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)
    return compare_dataframes(
        df_old,
        df_new,
        max_diffs=max_diffs,
        summary_only=summary_only,
    )


def _configure_sigpipe() -> None:
    if not hasattr(signal, "SIGPIPE"):
        return

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def main() -> None:
    """Entry point for the script."""
    _configure_sigpipe()
    parser = argparse.ArgumentParser(description="Compare two CSV files.")
    parser.add_argument("old", type=Path, help="Path to the old CSV file")
    parser.add_argument("new", type=Path, help="Path to the new CSV file")
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=DEFAULT_MAX_DIFFS,
        help="Maximum number of differing cells to print (default: 100).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary metrics, not individual cell diffs.",
    )
    args = parser.parse_args()

    compare_csvs(
        args.old,
        args.new,
        max_diffs=args.max_diffs,
        summary_only=args.summary_only,
    )


if __name__ == "__main__":
    main()
