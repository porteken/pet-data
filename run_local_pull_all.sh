#!/bin/bash
set -e

export SUPABASE_DB_URI="postgresql://postgres:WFqwZama6hcby3@db.nkyuotciejblfsifvoye.supabase.co:5432/postgres"

echo "====== Setting Up Local Directories ======"
mkdir -p output_tiles era5_data_parquet weather_data_parquet utci_data_parquet pet_data_csv analytics_data_csv combined_data_parquet

python cities.py
python boxes.py \
  --cities-csv cities.csv \
  --cells-out output_tiles/unique_grid_cells.csv \
  --boxes-out output_tiles/tile_boxes.csv \
  --snapped-out output_tiles/snapped_cities.csv \
  --city-tile-out output_tiles/city_to_tile.csv

TARGET_YEARS=$(seq 2000 $(($(date +%Y) - 1)))
# All months
TARGET_MONTHS="1 2 3 4 5 6 7 8 9 10 11 12"

for YEAR in $TARGET_YEARS; do
    echo "====== [Job: pull-google-era5] Year: $YEAR ======"
    # Provide one single batch. It expects expected-location-count for memory mapping
    python google_era5.py \
      --year $YEAR \
      --out-dir ./era5_data_parquet \
      --city-shard-count 500 \
      --city-shard-index 0 \
      --time-shard-count 120 \
      --time-shard-index 0 \
      --concurrency-profile conservative \
      --max-workers 1 || true # skip failure to ensure we continue

    echo "====== [Job: process-weather] Year: $YEAR ======"
    python pull_weather.py \
      --year $YEAR \
      --out-dir ./weather_data_parquet \
      --weather-city-shard-count 500 \
      --weather-city-shard-index 0 \
      --max-workers 1

    for MONTH in $TARGET_MONTHS; do
        echo "====== [Job: process-mrt] Year: $YEAR Month: $MONTH ======"
        python pull_mrt.py \
          --year $YEAR \
          --month $MONTH \
          --out-dir ./utci_data_parquet \
          --tile-shard-count 128 \
          --tile-shard-index 0 \
          --max-workers 1
    done

    echo "====== [Job: process-pet] Year: $YEAR ======"
    python combine.py \
      --year $YEAR \
      --era5-root ./era5_data_parquet \
      --weather-root ./weather_data_parquet \
      --mrt-root ./utci_data_parquet \
      --out-dir ./combined_data_parquet \


    python calculate_pet.py \
      --year $YEAR \
      --combined-root ./combined_data_parquet \
      --out-dir ./pet_data_csv \

done

echo "====== [Job: process-analytics] ======"
python generate_analytics.py \
  --pet-root ./pet_data_csv \
  --out-dir ./analytics_data_csv \


echo "====== [Job: load-results] ======"
python load.py \

  --cities-csv cities.csv \
  --pet-root ./pet_data_csv \
  --analytics-root ./analytics_data_csv \
  --copy-batch-size 1000

echo "====== Pipeline locally completed! ======"
