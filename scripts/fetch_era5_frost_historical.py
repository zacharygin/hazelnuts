#!/usr/bin/env python3
"""
Download ERA5 hourly 2m temperature for March–April 1950–1989 and extend
the frost degree-hours record back from the current 1990–2024 series.

Methodology (matches hazelnut-insurance/src/triggers/frost.py exactly):
  - Production-weighted point extraction at 7 provinces
  - March 15-31: threshold -3.0°C  (cold-tolerant bud stage)
  - April 1-30:  threshold -1.5°C  (open catkins, highly sensitive)
  - Same bbox as existing files: lat 40-42°N, lon 29-41°E

Output:
  data/raw/era5/era5_temp_YYYY.nc   — per-year nc files (matching existing)
  data/raw/era5_frost_monthly.csv   — full record 1950–2024

Usage:
    python scripts/fetch_era5_frost_historical.py              # 1950-1989
    python scripts/fetch_era5_frost_historical.py --start 1940 --end 1989
    python scripts/fetch_era5_frost_historical.py --rebuild-csv-only
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
ERA5_DIR  = ROOT / 'data' / 'raw' / 'era5'
FROST_CSV = ROOT / 'data' / 'raw' / 'era5_frost_monthly.csv'

# ── Domain (matches existing files) ──────────────────────────────────────────
BBOX     = [42.0, 29.0, 40.0, 41.0]   # N, W, S, E
MONTHS   = ['02', '03', '04', '05']   # Feb–May: full phenological risk window
ALL_DAYS = [f'{d:02d}' for d in range(1, 32)]
ALL_HRS  = [f'{h:02d}:00' for h in range(24)]

# ── Province definitions — weights from TÜİK provincial production shares ─────
# Ordu 31.1%, Giresun 16.3%, Samsun 14.1%, Sakarya 12.8%, Düzce 10.7%, Trabzon 6.1%
# "Other" 8.9% excluded (no representative point). TOTAL_W normalises correctly.
PROVINCES = [
    {'name': 'Ordu',    'lat': 40.98, 'lon': 37.88, 'w': 0.311},
    {'name': 'Giresun', 'lat': 40.91, 'lon': 38.39, 'w': 0.163},
    {'name': 'Samsun',  'lat': 41.29, 'lon': 36.33, 'w': 0.141},
    {'name': 'Sakarya', 'lat': 40.74, 'lon': 30.40, 'w': 0.128},
    {'name': 'Duzce',   'lat': 40.84, 'lon': 31.16, 'w': 0.107},
    {'name': 'Trabzon', 'lat': 41.00, 'lon': 39.72, 'w': 0.061},
]
TOTAL_W = sum(p['w'] for p in PROVINCES)

# Thresholds by phenological phase (see hazelnut-insurance/config/trigger_params.yaml)
THRESH = {
    2: -5.0,   # Feb 1-28:   pollination / catkin emergence
    3: {
        'early': -5.0,   # Mar 1-14:  late pollination
        'late':  -3.0,   # Mar 15-31: post-pollination, fertilization
    },
    4: -2.3,   # Apr 1-30:  nut set (BIO Web Conf 2021: -2.3°C no-damage threshold)
    5: -2.3,   # May 1-31:  early nut development (same stage as April)
}


def _threshold(month: int, day: int) -> float:
    t = THRESH[month]
    if isinstance(t, dict):
        return t['early'] if day <= 14 else t['late']
    return t


# ── Download ──────────────────────────────────────────────────────────────────
def download_year(year: int, force: bool = False) -> Path:
    out = ERA5_DIR / f'era5_temp_{year}.nc'
    if out.exists() and not force:
        log.info('  %d: already on disk, skipping.', year)
        return out

    try:
        import cdsapi
    except ImportError:
        sys.exit('cdsapi not installed. Run: pip install cdsapi')

    log.info('  %d: requesting from CDS ...', year)
    c = cdsapi.Client(quiet=True)
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': '2m_temperature',
            'year': str(year),
            'month': MONTHS,
            'day': ALL_DAYS,
            'time': ALL_HRS,
            'area': BBOX,
            'data_format': 'netcdf',
            'download_format': 'unarchived',
        },
        str(out),
    )
    log.info('  %d: saved.', year)
    return out


# ── Frost DH computation ──────────────────────────────────────────────────────
def compute_frost_dh(nc_path: Path, year: int) -> dict:
    import xarray as xr

    ds  = xr.open_dataset(nc_path, engine='h5netcdf')
    key = 't2m' if 't2m' in ds else list(ds.data_vars)[0]
    t   = ds[key] - 273.15
    dt  = pd.DatetimeIndex(t.valid_time.values)
    months_in_file = set(dt.month)

    accum = {2: 0.0, '3e': 0.0, '3l': 0.0, 4: 0.0, 5: 0.0}

    for p in PROVINCES:
        cell  = t.sel(latitude=p['lat'], longitude=p['lon'], method='nearest')
        temps = cell.values
        w     = p['w'] / TOTAL_W
        for time_val, temp in zip(dt, temps):
            m, d = time_val.month, time_val.day
            if m not in THRESH:
                continue
            thr = _threshold(m, d)
            dh  = w * max(0.0, thr - float(temp))
            if   m == 2:          accum[2]   += dh
            elif m == 3 and d <= 14: accum['3e'] += dh
            elif m == 3 and d >= 15: accum['3l'] += dh
            elif m == 4:          accum[4]   += dh
            elif m == 5:          accum[5]   += dh

    ds.close()

    feb_dh        = round(accum[2],    4)
    emarch_dh     = round(accum['3e'], 4)
    lmarch_dh     = round(accum['3l'], 4)
    april_dh      = round(accum[4],    4)
    may_dh        = round(accum[5],    4)
    frost_dh      = round(feb_dh + emarch_dh + lmarch_dh + april_dh + may_dh, 4)

    missing = [m for m in (2, 3, 4, 5) if m not in months_in_file]
    if missing:
        log.debug('  %d: months %s not in NC file — DH set to 0', year, missing)

    return {
        'year':        year,
        'feb_dh':      feb_dh,      # Feb 1-28,  threshold -5.0°C
        'emarch_dh':   emarch_dh,   # Mar 1-14,  threshold -5.0°C
        'lmarch_dh':   lmarch_dh,   # Mar 15-31, threshold -3.0°C
        'april_dh':    april_dh,    # Apr 1-30,  threshold -2.3°C
        'may_dh':      may_dh,      # May 1-31,  threshold -2.3°C
        'frost_dh':    frost_dh,
    }


# ── Rebuild full CSV ──────────────────────────────────────────────────────────
def rebuild_frost_csv(all_years: range) -> pd.DataFrame:
    rows = []
    for yr in sorted(all_years):
        p = ERA5_DIR / f'era5_temp_{yr}.nc'
        if not p.exists():
            log.warning('  %d: nc file missing — skipping', yr)
            continue
        log.info('  Processing %d ...', yr)
        rows.append(compute_frost_dh(p, yr))
    df = pd.DataFrame(rows).set_index('year').sort_index()
    df.to_csv(FROST_CSV)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1940)
    parser.add_argument('--end',   type=int, default=1949)
    parser.add_argument('--rebuild-csv-only', action='store_true',
                        help='Skip downloads; recompute CSV from existing nc files')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if NC file already exists (use after adding new months)')
    args = parser.parse_args()

    ERA5_DIR.mkdir(parents=True, exist_ok=True)

    new_years = range(args.start, args.end + 1)
    all_years = range(min(args.start, 1940), 2025)

    if not args.rebuild_csv_only:
        log.info('Downloading ERA5 t2m for %d–%d (force=%s)', args.start, args.end, args.force)
        log.info('CDS queue: each year ~1-5 min\n')
        for yr in new_years:
            try:
                download_year(yr, force=args.force)
            except Exception as e:
                log.error('  ERROR %d: %s', yr, e)
                time.sleep(5)

    log.info('\nComputing frost DH for %d–%d ...', all_years.start, all_years.stop - 1)
    df = rebuild_frost_csv(all_years)

    log.info('\nSaved → %s', FROST_CSV)
    log.info('Years: %d–%d  (n=%d)', df.index.min(), df.index.max(), len(df))

    cols = ['feb_dh', 'emarch_dh', 'lmarch_dh', 'april_dh', 'may_dh', 'frost_dh']
    log.info('\nAll years:\n%s', df[cols].round(2).to_string())

    trigger = df[df['april_dh'] >= 3.0]
    log.info('\napril_dh >= 3.0  — %d/%d = %.1f%%:\n%s',
             len(trigger), len(df), len(trigger)/len(df)*100,
             trigger[cols].round(2).to_string())


if __name__ == '__main__':
    main()
