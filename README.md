# PET Data Pipeline

This is the automated pipeline for processing climate data and loading it into the [Web Application](https://pet-app-ashen.vercel.app/). The repository for the web application itself is [porteken/pet-app](https://github.com/porteken/pet-app). Script pull_all.sh is used to pull data for all years while the GitHub workflow yearly_update pulls the previous year data every January 2nd.

1. **Compute**: Utilizes Google Cloud Run workers to pull ARCO-ERA5 weather and MRT data, computing PET (Physiological Equivalent Temperature) per US city grid point in parallel.
2. **Store**: Outputs the computed PET dataset directly into an AWS S3 bucket for scalable storage.
3. **Load**: GitHub Actions workflows shard and synchronize the processed data from S3, loading it into a Northflank-hosted Postgres database.
4. **Analyze**: Database views within Postgres generate summary statistics, historical trend comparisons, and long-term forecasts.
