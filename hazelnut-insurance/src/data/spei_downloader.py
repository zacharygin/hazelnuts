"""
SPEI Global Drought Monitor downloader.

Downloads SPEIbase v2.10 (SPEI03 monthly, global NetCDF from 1901).
Source: https://spei.csic.es/database.html
No API key required.

Usage:
    python -m src.data.spei_downloader
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "spei"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# SPEIbase v2.10 - 3-month SPEI global NetCDF
# Primary: CSIC digital repository; fallback: Zenodo mirror
SPEI03_URL = "https://digital.csic.es/bitstream/10261/332007/5/spei03.nc"
SPEI03_URL_FALLBACK = "https://digital.csic.es/bitstream/10261/332007/2/spei03.nc"
SPEI03_PATH = RAW_DIR / "spei03.nc"

# Turkey hazelnut region bounds for subsetting
LAT_BOUNDS = (40.0, 42.0)
LON_BOUNDS = (29.0, 41.0)


def download(force: bool = False) -> Path:
    """Download SPEIbase SPEI03 NetCDF. Skips if cached."""
    if SPEI03_PATH.exists() and not force:
        logger.info("Cache hit: %s", SPEI03_PATH)
        return SPEI03_PATH

    for url in (SPEI03_URL, SPEI03_URL_FALLBACK):
        try:
            logger.info("Downloading SPEI03 from %s (~200MB) ...", url)
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            break
        except Exception as e:
            logger.warning("Failed from %s: %s — trying next", url, e)
    else:
        raise RuntimeError("All SPEI download URLs failed")

    total = int(resp.headers.get("content-length", 0))
    with open(SPEI03_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="SPEI03"
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Saved: %s", SPEI03_PATH)
    return SPEI03_PATH


def load_turkey_region() -> "xr.Dataset":
    """
    Load SPEI03 data subset to the Turkish hazelnut region.
    Downloads if not cached.
    """
    import xarray as xr

    if not SPEI03_PATH.exists():
        download()

    ds = xr.open_dataset(SPEI03_PATH, decode_times=True)

    # Subset to Turkey hazelnut region
    lat_slice = slice(LAT_BOUNDS[0], LAT_BOUNDS[1])
    lon_slice = slice(LON_BOUNDS[0], LON_BOUNDS[1])

    lat_dim = "lat" if "lat" in ds.dims else "latitude"
    lon_dim = "lon" if "lon" in ds.dims else "longitude"

    ds_sub = ds.sel({lat_dim: lat_slice, lon_dim: lon_slice})
    return ds_sub


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = download()
    print(f"Downloaded to: {path}")
    print(f"File size: {path.stat().st_size / 1e6:.1f} MB")
