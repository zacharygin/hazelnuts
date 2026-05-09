"""
Hail trigger: ERA5 convective precipitation proxy, June–August.

KNOWN LIMITATION: ERA5 convective precipitation (cp) is a parameterized
output and significantly underestimates convective intensity at grid scale.
This is an acknowledged engineering approximation. The gold standard would
be ESWD point observations (eswd.eu) but ESWD has no programmatic API.

Methodology reference:
  Polat et al. (2016), "Severe Hail Climatology of Turkey,"
  Monthly Weather Review 144(1). https://doi.org/10.1175/MWR-D-15-0337.1
  — 1,489 severe hail cases on 1,107 days, 1925–2014.
  Hail = >60% of all weather-related insured agricultural losses in Turkey.

Metric: maximum 6-hour accumulated convective precipitation (mm) across
all province grid cells during June–August of the given year.
This captures peak-event intensity rather than season totals.

Payout: calibrated so ~15% of years trigger, matching Polat et al.'s
estimated Black Sea hail frequency (1–2 events/yr × ~15% basket-level
impact probability).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.utils.geo import extract_province_series, province_weights

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"
RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "era5"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["hail"]


def _payout_from_bands(metric: float, bands: list) -> float:
    for lo, hi, pay_lo, pay_hi in bands:
        lo = float(lo)
        hi = float("inf") if str(hi) in (".inf", "inf") else float(hi)
        if lo <= metric < hi:
            if pay_lo == pay_hi:
                return pay_lo
            frac = (metric - lo) / (hi - lo)
            return pay_lo + frac * (pay_hi - pay_lo)
    return 0.0


def _load_cp_year(year: int) -> xr.Dataset:
    """
    Load ERA5 convective precipitation for June–August of the given year.
    Expects file at data/raw/era5/era5_hail_{year}.nc.
    """
    path = RAW_DIR / f"era5_hail_{year}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 hail (cp) data for {year} not found at {path}. "
            "Run era5_downloader.download_hail_year() first."
        )
    ds = xr.open_dataset(path)
    # ERA5 cp is in metres per second (accumulated); convert to mm
    if "cp" in ds:
        ds["cp"] = ds["cp"] * 1000.0  # m → mm
        ds["cp"].attrs["units"] = "mm"
    return ds


def _max_6h_cp(da: xr.DataArray) -> float:
    """
    Compute maximum 6-hour accumulated convective precipitation (mm)
    from an hourly DataArray with dim (time,).
    """
    vals = da.values.astype(float)
    if len(vals) < 6:
        return 0.0
    # Rolling 6-hour sum
    rolling_sums = np.convolve(vals, np.ones(6), mode="valid")
    return float(np.nanmax(rolling_sums))


def compute_hail_metric(year: int) -> float:
    """
    Compute the hail trigger metric for a given year:
    production-weighted maximum 6-hour convective precipitation (mm)
    across all provinces during June–August.

    Returns
    -------
    float
        Weighted max 6-hour CP in mm. Higher = more severe convective activity.
    """
    cfg = _load_cfg()
    risk_months = cfg.get("risk_window_months", [6, 7, 8])

    ds = _load_cp_year(year)
    province_series = extract_province_series(ds, "cp")
    weights = province_weights()
    total_weight = sum(weights.values())

    weighted_max = 0.0
    for province, da in province_series.items():
        w = weights[province] / total_weight
        times = pd.DatetimeIndex(da.time.values)
        month_mask = times.month.isin(risk_months)
        da_season = da.isel(time=month_mask)
        province_max = _max_6h_cp(da_season)
        # Weighted sum of per-province maximum: gives more weight to
        # high-production provinces hit by severe storms
        weighted_max += w * province_max

    return weighted_max


def compute_payout(metric: float) -> float:
    """Return payout as fraction of notional (0.0–1.0) given hail metric."""
    cfg = _load_cfg()
    return _payout_from_bands(metric, cfg["payout_bands"])


def backtest(years: list[int]) -> pd.DataFrame:
    records = []
    for year in years:
        try:
            metric = compute_hail_metric(year)
            payout = compute_payout(metric)
            records.append({"year": year, "max_6h_cp_mm": metric, "payout": payout})
            logger.info("Hail %d: max_6h_cp=%.1fmm, payout=%.3f", year, metric, payout)
        except FileNotFoundError:
            logger.warning("ERA5 hail data missing for year %d — skipping", year)
    return pd.DataFrame(records)
