"""
Bad pollination trigger for hazelnut.

Mechanism:
  Hazelnuts are wind-pollinated. Male catkins release pollen January–March.
  Female flowers are receptive February–March. Two failure modes:

  1. Wet bloom: sustained rainfall during February–March washes pollen from
     catkins before dispersal and prevents airborne transfer. ERA5 monthly
     total precipitation for Feb + March is the proxy.

  2. Cold stall: sustained temperatures <0°C during catkin development in
     February delay or abort pollen shed. ERA5 mean Feb temperature proxy.

  Reference: Mehlenbacher (1991) "Hazelnuts (Corylus)" in Genetic Resources
  of Temperate Fruit and Nut Crops, ISHS Acta Horticulturae 290: 791–836.
  "Rain during pollen shed reduces fertilisation; cold below 0°C delays
  catkin development and can abort pollen viability."

Trigger metric:
  pollination_index = Feb_precip_mm + Mar_precip_mm
  Deductible: 120mm (background wet-season precip for Black Sea coast)
  Above 180mm total: partial payout (15% max)

Settlement: April — known before any field observation needed.
Data source: ERA5 monthly means (same download as drought/SPEI computation).

Note: this is a binary-ish trigger with a shallow payout cap (15%).
Pollination failure is rarely total at the basket level — some provinces
compensate for others — so this trigger adds frequency not severity.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"
_MONTHLY_CACHE = Path(__file__).parents[2] / "data" / "raw" / "era5_monthly"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("pollination", _default_cfg())


def _default_cfg() -> dict:
    return {
        "bloom_months": [2, 3],
        "deductible_mm": 120.0,
        "payout_bands": [
            [200.0, float("inf"), 0.15, 0.15],
            [160.0, 200.0, 0.15, 0.05],
            [120.0, 160.0, 0.05, 0.00],
            [0.0,   120.0, 0.00, 0.00],
        ],
    }


def _payout_from_bands(index: float, bands: list) -> float:
    for lo, hi, pay_lo, pay_hi in bands:
        lo = float(lo)
        hi = float("inf") if str(hi) in (".inf", "inf") else float(hi)
        if lo <= index < hi:
            if pay_lo == pay_hi:
                return pay_lo
            frac = (index - lo) / (hi - lo)
            return pay_lo + frac * (pay_hi - pay_lo)
    return 0.0


def compute_payout(pollination_index: float) -> float:
    """Return payout fraction given Feb+Mar accumulated precip (mm)."""
    cfg = _load_cfg()
    return _payout_from_bands(pollination_index, cfg["payout_bands"])


def metric_series() -> pd.DataFrame:
    """
    Return annual Feb+Mar total precipitation (mm) from ERA5 monthly data.
    Requires ERA5 monthly download (src/data/spei_from_era5.py).

    Returns DataFrame with columns [year, pollination_index].
    """
    nc_path = next(_MONTHLY_CACHE.glob("era5_monthly_*.nc"), None)
    if nc_path is None:
        raise FileNotFoundError(
            "ERA5 monthly data not found. "
            "Run: python -c \"from src.data.spei_from_era5 import _download_monthly_era5; _download_monthly_era5()\""
        )

    from src.data.spei_from_era5 import _open_monthly_dataset
    from src.utils.geo import load_provinces, province_weights

    ds = _open_monthly_dataset()

    precip_var = next((v for v in ds.data_vars if "precip" in v.lower() or v == "tp"), None)
    if precip_var is None:
        raise ValueError(f"No precip variable found in {list(ds.data_vars)}")

    provinces = load_provinces()
    weights   = province_weights()
    total_w   = sum(weights.values())

    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    cfg = _load_cfg()
    bloom_months = cfg.get("bloom_months", [2, 3])

    records = []
    # Filter to bloom months
    da = ds[precip_var]
    time_vals = pd.DatetimeIndex(da.time.values)

    years = sorted(set(time_vals.year))
    for year in years:
        bloom_mask = time_vals[(time_vals.year == year) & (time_vals.month.isin(bloom_months))]
        if len(bloom_mask) == 0:
            continue

        # days in each bloom month for this year (ERA5 tp is m/day mean rate)
        days_in_month = np.array([
            pd.Timestamp(year, m, 1).days_in_month for m in bloom_mask.month
        ])

        weighted_total = 0.0
        for prov in provinces:
            w = weights[prov["name"]] / total_w
            cell = da.sel({lat_dim: prov["lat"], lon_dim: prov["lon"]}, method="nearest")
            bloom_vals = cell.sel(time=bloom_mask).values  # m/day for each bloom month
            monthly_mm = float((bloom_vals * days_in_month * 1000).sum())  # m/day × days × 1000
            weighted_total += w * monthly_mm

        records.append({"year": int(year), "pollination_index": weighted_total})

    return pd.DataFrame(records)


def backtest(years: list[int]) -> pd.DataFrame:
    """Compute pollination index and payout for each year."""
    series = metric_series()
    series_idx = series.set_index("year")
    records = []
    for year in years:
        if year not in series_idx.index:
            logger.warning("Pollination: no ERA5 monthly data for %d — skipping", year)
            continue
        idx = float(series_idx.loc[year, "pollination_index"])
        payout = compute_payout(idx)
        records.append({"year": year, "pollination_index": idx, "payout": payout})
        logger.info("Pollination %d: index=%.1fmm, payout=%.3f", year, idx, payout)
    return pd.DataFrame(records)
