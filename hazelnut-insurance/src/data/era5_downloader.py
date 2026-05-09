"""
ERA5 hourly 2m-temperature downloader for the Turkish Black Sea hazelnut region.

Requires a CDS account and ~/.cdsapirc with your personal token.
See: https://cds.climate.copernicus.eu/how-to-api

Usage:
    python -m src.data.era5_downloader --start 1990 --end 2024
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "era5"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Bounding box: [north, west, south, east]
BBOX = [42.0, 29.0, 40.0, 41.0]


def _get_cds_client():
    try:
        import cdsapi
    except ImportError:
        sys.exit(
            "cdsapi not installed. Run: pip install cdsapi\n"
            "Then set up ~/.cdsapirc per https://cds.climate.copernicus.eu/how-to-api"
        )
    return cdsapi.Client()


def download_year(year: int, force: bool = False) -> Path:
    """
    Download ERA5 hourly 2m_temperature for March–April of the given year.
    Saves to data/raw/era5/era5_temp_{year}.nc. Skips if file exists unless force=True.
    """
    out_path = RAW_DIR / f"era5_temp_{year}.nc"
    if out_path.exists() and not force:
        logger.info("Cache hit: %s", out_path)
        return out_path

    client = _get_cds_client()
    request = {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": str(year),
        "month": ["03", "04"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BBOX,
        "format": "netcdf",
        "grid": [0.25, 0.25],
    }

    logger.info("Requesting ERA5 temperature for year %d ...", year)
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    logger.info("Saved: %s", out_path)
    return out_path


def download_hail_year(year: int, force: bool = False) -> Path:
    """
    Download ERA5 hourly convective precipitation for June–August of the given year.
    Used as hail proxy (see src/triggers/hail.py for methodology and limitations).
    Saves to data/raw/era5/era5_hail_{year}.nc.

    Variable: 'cp' (convective precipitation, m accumulated per hour).
    Reference: Polat et al. (2016) MWR 144(1) for hail climatology calibration.
    """
    out_path = RAW_DIR / f"era5_hail_{year}.nc"
    if out_path.exists() and not force:
        logger.info("Cache hit: %s", out_path)
        return out_path

    client = _get_cds_client()
    request = {
        "product_type": "reanalysis",
        "variable": "convective_precipitation",
        "year": str(year),
        "month": ["06", "07", "08"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": BBOX,
        "format": "netcdf",
        "grid": [0.25, 0.25],
    }

    logger.info("Requesting ERA5 convective precipitation for year %d ...", year)
    client.retrieve("reanalysis-era5-single-levels", request, str(out_path))
    logger.info("Saved: %s", out_path)
    return out_path


def download_range(start: int, end: int, force: bool = False) -> None:
    """Download frost (March–April temperature) for a range of years."""
    for year in range(start, end + 1):
        try:
            download_year(year, force=force)
        except Exception as exc:
            logger.error("Failed for year %d: %s", year, exc)


def download_hail_range(start: int, end: int, force: bool = False) -> None:
    """Download hail proxy (June–August convective precipitation) for a range of years."""
    for year in range(start, end + 1):
        try:
            download_hail_year(year, force=force)
        except Exception as exc:
            logger.error("Failed for hail year %d: %s", year, exc)


def load_year(year: int) -> "xr.Dataset":
    """Load cached ERA5 NetCDF for the given year."""
    import xarray as xr

    path = RAW_DIR / f"era5_temp_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 data for {year} not found at {path}. "
            "Run era5_downloader.py first."
        )
    ds = xr.open_dataset(path)
    # ERA5 temperature is in Kelvin; convert to Celsius
    if "t2m" in ds:
        ds["t2m"] = ds["t2m"] - 273.15
        ds["t2m"].attrs["units"] = "°C"
    return ds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Download ERA5 data for hazelnut insurance triggers")
    parser.add_argument("--start", type=int, default=1990)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument(
        "--type",
        choices=["frost", "hail", "all"],
        default="all",
        help="Which ERA5 variable set to download",
    )
    args = parser.parse_args()
    if args.type in ("frost", "all"):
        download_range(args.start, args.end, force=args.force)
    if args.type in ("hail", "all"):
        download_hail_range(args.start, args.end, force=args.force)
