"""Debug script to inspect variables in the ERA5 Zarr store."""

import sys

sys.path.append("/home/kenneth-porter/pet-data")
from google_era5 import _open_zarr_store

ds = _open_zarr_store()
