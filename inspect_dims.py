"""Inspect dimensions of the Zarr store."""

import logging

import xarray as xr

from google_era5 import _open_zarr_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Check coordinates and dimensions of the Zarr store by attempting small slices."""
    try:
        ds = _open_zarr_store()

        # Check coordinates and first variable dims
        first_var = "10u"

        # Pull a tiny slice
        ds[first_var].isel(
            time=slice(0, 5),
            latitude=slice(0, 2),
            longitude=slice(0, 2),
        )

        # Point selection simulation
        lats = ds.latitude.values[0:2]
        lons = ds.longitude.values[0:2]
        lat_da = xr.DataArray(lats, dims="location")
        lon_da = xr.DataArray(lons, dims="location")

        (
            ds[first_var]
            .isel(time=slice(0, 5))
            .sel(latitude=lat_da, longitude=lon_da, method="nearest")
        )

    except Exception:
        logger.exception("Failed to inspect dimensions")


if __name__ == "__main__":
    main()
