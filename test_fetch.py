"""Test script for fetching data from the CDS API."""

import sys

sys.path.append(".")
import tempfile

from google_era5 import create_cds_client, retrieve_with_retry

client = create_cds_client()
request = {
    "variable": "mean_radiant_temperature",
    "version": "1_1",
    "product_type": "consolidated_dataset",
    "year": "2024",
    "month": "05",
    "day": "01",
    "time": "12:00",
    "area": [35, -95, 34, -94],
    "format": "netcdf",
}
with tempfile.NamedTemporaryFile(suffix=".nc") as tmp_nc:
    res = retrieve_with_retry(
        client,
        "derived-utci-historical",
        request,
        target=tmp_nc.name,
    )
