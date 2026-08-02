# PET Data Pipeline

> **Archived:** This project is no longer actively maintained.

This is the automated pipeline for processing pet data and loading it into the [Web Application](https://pet-app-ashen.vercel.app/). The repository for the web application itself is [porteken/pet-app](https://github.com/porteken/pet-app).

1. **Compute**: Google Cloud Run job tasks pull ARCO-ERA5 weather data and compute daily PET (Physiological Equivalent Temperature) per US city grid point in parallel.
2. **Store**: Workers write parquet shards directly to an AWS S3 bucket, partitioned by year.
3. **Load**: Parquet shards are Copied into a staging table and upserted into Northflank-hosted Postgres, so every load is idempotent and safe to re-run.
4. **Analyze**: Materialized views in Postgres generate summary statistics, historical trend comparisons, and long-term forecasts. Yearly appends refresh the views in place.

PET city metadata is stored in `public.pet_locations`, separate from the
wet-bulb pipeline's `public.locations` catalog. This keeps PET's stable
location IDs independent and prevents either pipeline from relabeling the
other dataset's observations.
