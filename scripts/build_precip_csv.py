#!/usr/bin/env python3
"""
Rebuild era5_precip_monthly.csv from ERA5 monthly total precipitation.

Source:  data/raw/era5_monthly/data_stream-moda_stepType-avgad.nc
         Variable: tp (total precipitation, m/day mean daily rate)
         Coverage: 1950-01-01 – 2024-12-01, all months

Conversion: mm/month = tp_m_per_day × days_in_month × 1000

Province weighting (TÜİK production shares, same as frost/hail triggers):
  Ordu 31.1%, Giresun 16.3%, Samsun 14.1%, Sakarya 12.8%, Düzce 10.7%, Trabzon 6.1%

Output columns (all in mm/month, production-weighted):
  year        — crop year
  pollin_mm   — Feb + Mar total (pollination window; excess rain washes pollen)
  aug_mm      — August total
  sep_mm      — September total
  oct_mm      — October total
  harvest_mm  — Aug + Sep + Oct total (full harvest-damage window)
  summer_mm   — Jun + Jul + Aug total (warm-season reference, matches prior CSV)

Usage:
    python scripts/build_precip_csv.py
    python scripts/build_precip_csv.py --start 1990 --end 2024
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
NC_PATH    = ROOT / "data" / "raw" / "era5_monthly" / "data_stream-moda_stepType-avgad.nc"
OUT_CSV    = ROOT / "data" / "raw" / "era5_precip_monthly.csv"

PROVINCES = [
    {"name": "Ordu",    "lat": 40.98, "lon": 37.88, "w": 0.311},
    {"name": "Giresun", "lat": 40.91, "lon": 38.39, "w": 0.163},
    {"name": "Samsun",  "lat": 41.29, "lon": 36.33, "w": 0.141},
    {"name": "Sakarya", "lat": 40.74, "lon": 30.40, "w": 0.128},
    {"name": "Duzce",   "lat": 40.84, "lon": 31.16, "w": 0.107},
    {"name": "Trabzon", "lat": 41.00, "lon": 39.72, "w": 0.061},
]
TOTAL_W = sum(p["w"] for p in PROVINCES)


def build_precip_csv(start_year: int = 1950, end_year: int = 2024) -> pd.DataFrame:
    log.info("Opening %s ...", NC_PATH.name)
    ds = xr.open_dataset(NC_PATH, engine="h5netcdf")

    # Normalise time coordinate name
    time_dim = "valid_time" if "valid_time" in ds.coords else "time"
    times = pd.DatetimeIndex(ds[time_dim].values)

    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    da = ds["tp"]  # m/day

    # Build a weighted monthly precip series (mm/month) across provinces
    log.info("Extracting production-weighted precipitation for %d provinces ...", len(PROVINCES))
    weighted_mm = np.zeros(len(times))

    for prov in PROVINCES:
        w = prov["w"] / TOTAL_W
        cell = da.sel({lat_dim: prov["lat"], lon_dim: prov["lon"]}, method="nearest").values
        days = times.days_in_month.values.astype(float)
        weighted_mm += w * cell * days * 1000   # m/day → mm/month

    monthly = pd.DataFrame({"time": times, "precip_mm": weighted_mm})
    monthly["year"]  = monthly["time"].dt.year
    monthly["month"] = monthly["time"].dt.month

    rows = []
    for year in range(start_year, end_year + 1):
        yr = monthly[monthly["year"] == year].set_index("month")["precip_mm"]
        if yr.empty:
            continue

        def m(mon: int) -> float:
            return float(yr.get(mon, np.nan))

        rows.append({
            "year":       year,
            "pollin_mm":  round(m(2) + m(3), 2),              # Feb + Mar
            "aug_mm":     round(m(8), 2),                      # August
            "sep_mm":     round(m(9), 2),                      # September
            "oct_mm":     round(m(10), 2),                     # October
            "harvest_mm": round(m(8) + m(9) + m(10), 2),      # Aug–Oct
            "summer_mm":  round(m(6) + m(7) + m(8), 2),       # Jun–Aug (legacy column)
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    log.info("Saved %d rows → %s", len(df), OUT_CSV)
    log.info("\n%s", df.describe().round(1).to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1950)
    parser.add_argument("--end",   type=int, default=2024)
    args = parser.parse_args()
    build_precip_csv(args.start, args.end)
