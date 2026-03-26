# PET Data Pipeline

This is the automated pipeline for processing climate data and loading it into the [Web Application](https://pet-app-ashen.vercel.app/). The repository for the web application itself is [porteken/pet-app](https://github.com/porteken/pet-app).

1. **Ingest**: Pulls weather and MRT data from CDS for the full record (January 2000 to the end of the previous month).
2. **Process**: Merges city-level datasets to compute PET per US city grid point.
3. **Analyze**: Generates summary statistics and historical trend comparisons.
4. **Load**: Shards and loads analytics into Supabase via GitHub Actions.
