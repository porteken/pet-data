"""Downcast float columns to float32 in calculate_pet.py and generate_analytics.py."""

import re
from pathlib import Path

with Path("calculate_pet.py").open() as f:
    text = f.read()

bad_str = """        pet_df = calculate_pet_frame(combined_df)
        output_dir = Path(args.out_dir) / shard_key.partition_path"""

good_str = """        pet_df = calculate_pet_frame(combined_df)
        float_cols = pet_df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            pet_df[float_cols] = pet_df[float_cols].astype("float32")
        output_dir = Path(args.out_dir) / shard_key.partition_path"""
text = text.replace(bad_str, good_str)
with Path("calculate_pet.py").open("w") as f:
    f.write(text)

with Path("generate_analytics.py").open() as f:
    text = f.read()

# I will just write a function to replace `.to_csv` with downcasting and `.to_csv`
text = re.sub(
    r"(\s+)([a-zA-Z0-9_]+)\.to_csv\((.*)",
    r'\1float_cols = \2.select_dtypes(include=["float64"]).columns\1if len(float_cols) > 0:\1    \2[float_cols] = \2[float_cols].astype("float32")\1\2.to_csv(\3',
    text,
)
with Path("generate_analytics.py").open("w") as f:
    f.write(text)
