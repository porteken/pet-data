"""Compare coordinates between tiles and cities data."""

import pandas as pd

df_tiles = pd.read_csv("/home/kenneth-porter/pet-data/output_tiles/city_to_tile.csv")
df_cities = pd.read_csv("/home/kenneth-porter/pet_files/cities.csv")

# Merge on location_id
merged = df_tiles[["location_id", "grid_lat", "grid_lon"]].merge(
    df_cities[["location_id", "lat", "lng"]],
    on="location_id",
)

# Compare current grid_lat/grid_lon (from city_to_tile) vs lat/lng (from cities.csv)
# Use a small tolerance for float comparison
tolerance = 1e-6
mismatches = merged[
    (abs(merged["grid_lat"] - merged["lat"]) > tolerance)
    | (abs(merged["grid_lon"] - merged["lng"]) > tolerance)
]

total_matched = len(merged)
diff_count = len(mismatches)

if diff_count == 0:
    pass
else:
    pass
