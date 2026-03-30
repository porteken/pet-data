"""Update combine.py to downcast float columns to float32 before saving to parquet."""

from pathlib import Path

with Path("combine.py").open() as f:
    text = f.read()

bad_str_1 = (
    """    combined_df.to_parquet(output_path, index=False, compression="zstd")"""
)
good_str_1 = """    float_cols = combined_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        combined_df[float_cols] = combined_df[float_cols].astype("float32")
    combined_df.to_parquet(output_path, index=False, compression="zstd")"""

text = text.replace(bad_str_1, good_str_1)

bad_str_2 = """    era5_df.sort_values(["location_id", "time"]).reset_index(drop=True).to_parquet(
        output_path,
        index=False, compression="zstd",
    )"""

good_str_2 = """    float_cols = era5_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        era5_df[float_cols] = era5_df[float_cols].astype("float32")
    era5_df.sort_values(["location_id", "time"]).reset_index(drop=True).to_parquet(
        output_path,
        index=False, compression="zstd",
    )"""

text = text.replace(bad_str_2, good_str_2)
with Path("combine.py").open("w") as f:
    f.write(text)
