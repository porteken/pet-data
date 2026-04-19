#!/usr/bin/env bash
set -e

echo "Setting up container filesystem..."
mkdir -p output_tiles era5_data_parquet

echo "Generating cities and tiles..."
python cities.py

echo "Starting ERA5 pipeline..."
# The "$@" passes all the gcloud --args directly into the python script!
exec python google_era5.py "$@"