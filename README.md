# PET Data Pipeline

This project pulls ERA5 weather data and UTCI mean radiant temperature (MRT) from CDS, matches them to US city grid points, computes PET (Physiological Equivalent Temperature), and exports `pet.csv`.

## Pipeline

1. `cities.py` creates `cities.csv` (top city points on the 0.25 degree grid).
2. `pull_weather.py` downloads ERA5 GRIB files to `raw_weather_grib/` and writes partitioned weather parquet to `weather_data_parquet/`.
3. `pull_mrt.py` downloads UTCI ZIP files to `raw_utci_zip/`, converts NetCDF to parquet in `utci_data_parquet/`, then cleans temp files.
4. `combine.py` builds city-matched hourly outputs:
   - `weather.parquet`
   - `mrt.parquet`
   - `combined_data.parquet`
5. `calculate_pet.py` uses `pet_corrected.py` to compute PET and writes daily max PET to `pet.csv`.
6. `pet.py` runs all pipeline steps above in sequence.

## Final Output

`pet.csv` contains daily max PET by `location_id` and `date` and is the file intended for database insert.
