#!/usr/bin/env bash
set -e

echo "Setting up container filesystem..."
mkdir -p output_tiles pet_data_csv analytics_data_csv

echo "Generating cities and tiles..."
python cities.py

echo "Starting ERA5->PET pipeline..."
exec python google_era5.py "$@"