"""Script to compare two CSV files and identify differing values.

Usage:
    python compare_csvs.py old.csv new.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Configure logging to behave like print for this CLI tool
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def compare_csvs(old_path: Path, new_path: Path) -> None:
    """Compare two CSV files and print the differences.

    Args:
        old_path: Path to the original CSV file.
        new_path: Path to the updated CSV file.

    """
    if not old_path.exists():
        logger.error("Error: %s does not exist.", old_path)
        return
    if not new_path.exists():
        logger.error("Error: %s does not exist.", new_path)
        return

    df_old = pd.read_csv(old_path)
    df_new = pd.read_csv(new_path)

    if df_old.shape != df_new.shape:
        logger.warning(
            "Warning: Shapes differ. Old: %s, New: %s",
            df_old.shape,
            df_new.shape,
        )

    # Align columns to compare common ones
    common_cols = [col for col in df_old.columns if col in df_new.columns]
    if len(common_cols) != len(df_old.columns) or len(common_cols) != len(
        df_new.columns,
    ):
        logger.warning("Warning: Columns differ. Common: %s", common_cols)

    # Only compare where shapes match for the common columns
    min_rows = min(len(df_old), len(df_new))
    df_old_subset = df_old.iloc[:min_rows][common_cols]
    df_new_subset = df_new.iloc[:min_rows][common_cols]

    # Find differences
    comparison_mask = df_old_subset != df_new_subset
    # Handle NaNs (since NaN != NaN is True)
    comparison_mask = comparison_mask & ~(df_old_subset.isna() & df_new_subset.isna())

    if not comparison_mask.any().any():
        logger.info("No differences found in overlapping data.")
        return

    logger.info("Found differences in %d rows:", min_rows)
    diff_indices = comparison_mask.any(axis=1)
    for idx in diff_indices[diff_indices].index:
        row_diffs = comparison_mask.loc[idx]
        for col in common_cols:
            if row_diffs[col]:
                old_val = df_old_subset.loc[idx, col]
                new_val = df_new_subset.loc[idx, col]
                logger.info("Row %s, Col '%s': %s -> %s", idx, col, old_val, new_val)


def main() -> None:
    """Entry point for the script."""
    parser = argparse.ArgumentParser(description="Compare two CSV files.")
    parser.add_argument("old", type=Path, help="Path to the old CSV file")
    parser.add_argument("new", type=Path, help="Path to the new CSV file")
    args = parser.parse_args()

    compare_csvs(args.old, args.new)


if __name__ == "__main__":
    main()
