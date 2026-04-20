# PET Data Pipeline

This is the automated pipeline for processing climate data and loading it into the [Web Application](https://pet-app-ashen.vercel.app/). The repository for the web application itself is [porteken/pet-app](https://github.com/porteken/pet-app).

1. **Ingest**: Pulls weather and MRT data from CDS for the full record (January 2000 to the end of the previous month).
2. **Process**: Merges city-level datasets to compute PET per US city grid point.
3. **Analyze**: Generates summary statistics and historical trend comparisons.
4. **Load**: Shards and loads analytics into Supabase via GitHub Actions.

## Development with uv

This repository now uses [`uv`](https://docs.astral.sh/uv/) as the source of truth for dependency management.

- Install project dependencies with `uv sync --extra gcs`
- Run project scripts with `uv run python <script>.py`
- Run the test suite with `uv run pytest`

The optional `gcs` extra installs the Google ARCO ERA5 / MRT stack used by `google_era5.py` and the Cloud Run worker image.

## CI/CD notes

- GitHub Actions now install dependencies with `uv sync --locked`
- The Cloud Run worker image is built from `pyproject.toml` + `uv.lock`
- `pull_all.yml` now provisions the Cloud Run `era5-worker` job before executing the ERA5 shard pipeline so the run always uses the current checkout
