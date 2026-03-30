#!/bin/bash
set -e

export SUPABASE_DB_URI="postgresql://postgres:postgres@localhost:5432/postgres"

# Setup directories
mkdir -p output_tiles era5_data_parquet weather_data_parquet utci_data_parquet pet_data_csv analytics_data_csv combined_data_parquet

echo "Step 1: Generate reference locations"
python cities.py
python boxes.py --cities-csv cities.csv --cells-out output_tiles/unique_grid_cells.csv --boxes-out output_tiles/tile_boxes.csv --snapped-out output_tiles/snapped_cities.csv --city-tile-out output_tiles/city_to_tile.csv

echo "Step 2: Pull minimal data just to execute pipeline"
# 1 location / shard
python google_era5.py --year 2024 --out-dir ./era5_data_parquet --city-shard-count 10 --city-shard-index 0 --time-shard-count 10 --time-shard-index 0
python pull_weather.py --year 2024 --out-dir ./weather_data_parquet --city-shard-count 10 --city-shard-index 0
python pull_mrt.py --year 2024 --month 7 --out-dir ./utci_data_parquet --shard-count 128 --shard-index 0

echo "Step 3: Combine ERA5, Weather, and MRT"
python combine.py --year 2024 \
    --era5-root ./era5_data_parquet \
    --weather-root ./weather_data_parquet \
    --mrt-root ./utci_data_parquet \
    --out-dir ./combined_data_parquet \
    --boxes-csv output_tiles/tile_boxes.csv

echo "Step 4: Calculate PET"
python calculate_pet.py --year 2024 \
    --combined-root ./combined_data_parquet \
    --out-dir ./pet_data_csv

echo "Step 5: Generate Analytics (percentiles, forecast, change)"
python generate_analytics.py \
    --pet-root ./pet_data_csv \
    --out-dir ./analytics_data_csv \
    --reference-years 2020 2021 2022

echo "Step 6: Load DB"
python load.py \
    --cities-csv cities.csv \
    --pet-root ./pet_data_csv \
    --analytics-root ./analytics_data_csv \
    --copy-batch-size 1000

echo "Step 7: Copy CSV files"
find ./pet_data_csv -type f -name '*.csv' | head -n 1 | xargs -I {} cp {} ./pet.csv
find ./analytics_data_csv -type f -name '*forecast.csv' | head -n 1 | xargs -I {} cp {} ./forecast.csv
find ./analytics_data_csv -type f -name '*change_per_decade.csv' | head -n 1 | xargs -I {} cp {} ./change.csv
find ./analytics_data_csv -type f -name '*percentiles.csv' | head -n 1 | xargs -I {} cp {} ./percentiles.csv

echo "Pipeline finished!"
