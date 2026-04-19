# PET Data Pipeline

This is the automated pipeline for processing climate data and loading it into the [Web Application](https://pet-app-ashen.vercel.app/). The repository for the web application itself is [porteken/pet-app](https://github.com/porteken/pet-app).

1. **Ingest**: Pulls core weather variables from Google ARCO ERA5 and MRT from CDS using tile bounding boxes.
2. **Process**: Merges weather city shards with MRT tile shards to compute PET per US city grid point.
3. **Analyze**: Generates summary statistics and historical trend comparisons.
4. **Load**: Shards and loads analytics into Supabase via GitHub Actions.
