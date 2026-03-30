"""Update pull_weather.py to downcast float columns to float32 before saving to parquet."""

from pathlib import Path

with Path("pull_weather.py").open() as f:
    text = f.read()

bad_str = """    table: Any = pa.Table.from_pandas(
        float_cols = weather_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        weather_df[float_cols] = weather_df[float_cols].astype("float32")
    weather_df = weather_df.sort_values(["location_id", "timestamp"]).reset_index(drop=True),
        preserve_index=False,
    )"""

good_str = """    float_cols = weather_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        weather_df[float_cols] = weather_df[float_cols].astype("float32")
    table: Any = pa.Table.from_pandas(
        weather_df.sort_values(["location_id", "timestamp"]).reset_index(drop=True),
        preserve_index=False,
    )"""

text = text.replace(bad_str, good_str)
with Path("pull_weather.py").open("w") as f:
    f.write(text)
