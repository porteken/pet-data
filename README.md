# PET Data Pipeline

This project pulls ERA5 weather data and UTCI mean radiant temperature (MRT) from CDS, matches them to US city grid points, computes PET (Physiological Equivalent Temperature), generates summary analytics, and loads the results into Supabase.

## Pipeline

1. `cities.py` creates `cities.csv` for the US city grid points.
2. `pull_cds.py` downloads ERA5 weather and MRT data and writes sharded parquets to `weather_data_parquet/` and `utci_data_parquet/`.
3. `combine.py` builds a year/month city-matched sharded parquet: `combined_data_<year>_<month>.parquet`.
4. `calculate_pet.py` uses `pet_corrected.py` to compute daily max PET for shards: `pet_<year>_<month>.csv`.
5. `generate_tables.py` combines PET results into `pet.csv`, `percentiles.csv`, `forecast.csv`, and `change_per_decade.csv`.
6. `load.py` loads those CSVs into Supabase and recreates SQL views from `create_views.sql`.

## Automation

`.github/workflows/data_pipeline.yml` runs the full pipeline in GitHub Actions, including S3-backed parquet storage via `template.yml`.