# PET Data Pipeline

This project pulls ERA5 weather data and UTCI mean radiant temperature (MRT) from CDS, matches them to US city grid points, computes PET (Physiological Equivalent Temperature), generates summary analytics, and loads the results into Supabase.

## Pipeline

1. `cities.py` creates `cities.csv` for the US city grid points.
2. `boxes.py` snaps cities to ERA5 grid cells, groups them into occupied 3x3 CDS tiles, and writes tile metadata to `output_tiles/`.
3. `pull_weather.py` and `pull_mrt.py` split the CDS ingestion by dataset, with shared CDS/retry/filesystem helpers in `pull_cds_shared.py`. Weather uses point-based ERA5 single-levels timeseries pulls from `2000-01-01` through the last day of the previous month at runtime and writes parquet partitions under `city_shard_index=<n>/`, while MRT uses tiled UTCI pulls over that same moving window and writes parquet partitions under `year=<year>/tile_id=<tile_id>/`.
4. `combine.py` reads the full-range weather city shards plus matching year/tile MRT shards and writes combined parquet shards to `combined_data_parquet/`.
5. `calculate_pet.py` computes PET per combined shard and writes partitioned CSV outputs under `pet_data_csv/`.
6. `generate_analytics.py` reads PET shard CSVs by tile, can split analytics work across `--shard-index/--shard-count`, and writes shard outputs under `analytics_data_csv/shard_count=<n>/shard_index=<i>/`.
7. `load.py` loads `pet_data_csv/` shard files plus analytics shard outputs into Supabase in bounded `COPY` batches and recreates SQL views from `create_views.sql`.
