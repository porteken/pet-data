# PET Data Pipeline

This project pulls ERA5 weather data and UTCI mean radiant temperature (MRT) from CDS, matches them to US city grid points, computes PET (Physiological Equivalent Temperature), generates summary analytics, and loads the results into Supabase.

## Pipeline

1. `cities.py` creates `cities.csv` for the US city grid points.
2. `boxes.py` snaps cities to ERA5 grid cells, groups them into occupied 3x3 CDS tiles, and writes tile metadata to `output_tiles/`.
3. `pull_cds.py` uses those tiles to shard ERA5-Land weather and UTCI MRT pulls into parquet partitions under `year=<year>/month=<month>/tile_id=<tile_id>/`.
4. `combine.py` discovers matching weather and MRT parquet shards from local paths or S3 roots and writes combined parquet shards to `combined_data_parquet/`.
5. `calculate_pet.py` computes PET per combined shard and writes partitioned CSV outputs under `pet_data_csv/`.
6. `generate_tables.py` materializes `pet.csv` from PET shard outputs when needed, then generates `percentiles.csv`, `forecast.csv`, and `change_per_decade.csv`.
7. `load.py` loads those CSVs into Supabase and recreates SQL views from `create_views.sql`.