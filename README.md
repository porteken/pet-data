# PET Data Pipeline

This project pulls ERA5 weather data and UTCI-derived mean radiant temperature (MRT), combines them for US city locations, calculates PET (Physiological Equivalent Temperature), and exports a final CSV for database ingestion.

## Pipeline

1. `cities.py` creates `cities.csv` (top city points on the 0.25 degree grid).
2. `pull_weather.py` downloads and processes ERA5 data into `weather_data_parquet/`.
3. `pull_mrt.py` downloads and processes UTCI MRT data into `utci_data_parquet/`.
4. `combine.py` builds city-matched hourly outputs:
   - `weather.parquet`
   - `mrt.parquet`
5. Weather + MRT are joined into `combined_data.parquet`.
6. `pet.R` (using `pet_corrected.R`) calculates PET and writes `pet.csv`.

## Final Output

`pet.csv` contains daily max PET by location and is the file intended for database insert.
 