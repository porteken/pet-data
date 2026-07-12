# PET Data Pipeline

This is the automated pipeline for processing climate data and loading it into the [Web Application](https://pet-app-ashen.vercel.app/). The repository for the web application itself is [porteken/pet-app](https://github.com/porteken/pet-app).

1. **Compute**: Google Cloud Run job tasks pull ARCO-ERA5 weather and radiation data and compute daily-max PET (Physiological Equivalent Temperature) and wet-bulb temperature per US city grid point in parallel.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are COPYed into a staging table and upserted into Northflank-hosted Postgres on the `(location_id, date)` unique key, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place (`refresh_views.py`); full rebuilds recreate them from `create_views.sql`.

## Entry points

- `pipeline.py` orchestrates the full pipeline locally (fetch → compute → load → views). `pull_all.sh`, `pull_pet.sh`, `pull_wetbulb.sh`, and `pull_both.sh` are thin wrappers around it, and the historical environment variables (`MODE`, `YEARS`, `PRODUCTS`, `USE_CLOUD_RUN`, ...) are still honored as defaults. Run `pipeline.py --help` for the full flag list; `--mode smoke` processes a single week for quick validation.
- The `yearly_update` GitHub workflow appends the previous year every early April (once ARCO's final data covers it — a pre-flight check in `build_year_range.py --check-arco` fails fast otherwise) and opens a GitHub issue if any job fails.
