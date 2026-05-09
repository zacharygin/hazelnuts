"""
Frost trigger: production-weighted degree-hours below threshold during spring risk window.

Threshold logic:
  - March 15-31: -3.0°C  (buds partially closed, more cold-tolerant)
  - April 1-30:  -1.5°C  (open catkins / young leaves, highly sensitive)

This split matches phenological cold-tolerance curves for Corylus avellana
and avoids the calibration errors that a flat -2°C threshold would introduce
for moderate frost years.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

from src.utils.geo import extract_province_series, province_weights
from src.data.era5_downloader import load_year

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["frost"]


def _payout_from_bands(dh: float, bands: list) -> float:
    """Linearly interpolate payout within the matching degree-hour band."""
    for lo, hi, pay_lo, pay_hi in bands:
        lo = float(lo)
        hi = float("inf") if str(hi) in (".inf", "inf") else float(hi)
        if lo <= dh < hi:
            if lo == hi or pay_lo == pay_hi:
                return pay_lo
            frac = (dh - lo) / (hi - lo)
            return pay_lo + frac * (pay_hi - pay_lo)
    return 0.0


def compute_dh(
    year: int,
    threshold_march: float | None = None,
    threshold_april: float | None = None,
    split_threshold: bool | None = None,
) -> float:
    """
    Compute production-weighted total degree-hours below frost threshold
    during March 15 – April 30 for the given year.

    Parameters default to values in config/trigger_params.yaml.

    Returns
    -------
    float
        Total production-weighted degree-hours (≥0). Higher = more frost damage.
    """
    cfg = _load_cfg()
    if split_threshold is None:
        split_threshold = cfg.get("split_threshold", True)
    if threshold_march is None:
        threshold_march = cfg["threshold_march"] if split_threshold else cfg["threshold_flat"]
    if threshold_april is None:
        threshold_april = cfg["threshold_april"] if split_threshold else cfg["threshold_flat"]

    ds = load_year(year)
    var = "t2m" if "t2m" in ds else list(ds.data_vars)[0]

    province_series = extract_province_series(ds, var)
    weights = province_weights()

    window_start = pd.Timestamp(year=year, month=3, day=15)
    window_end = pd.Timestamp(year=year, month=4, day=30, hour=23)

    total_dh = 0.0
    total_weight = sum(weights.values())

    for province, da in province_series.items():
        w = weights[province] / total_weight
        times = pd.DatetimeIndex(da.time.values)
        temps = da.values.astype(float)

        mask = (times >= window_start) & (times <= window_end)
        t_masked = times[mask]
        temp_masked = temps[mask]

        for t, temp in zip(t_masked, temp_masked):
            threshold = threshold_march if t.month == 3 else threshold_april
            damage = max(0.0, threshold - temp)
            total_dh += w * damage

    return total_dh


def compute_payout(dh: float) -> float:
    """Return payout as fraction of notional (0.0–1.0) given degree-hours."""
    cfg = _load_cfg()
    return _payout_from_bands(dh, cfg["payout_bands"])


def backtest(years: list[int]) -> pd.DataFrame:
    """Compute DH and payout for each year. Returns DataFrame."""
    records = []
    for year in years:
        try:
            dh = compute_dh(year)
            payout = compute_payout(dh)
            records.append({"year": year, "dh": dh, "payout": payout})
            logger.info("Frost %d: DH=%.1f, payout=%.3f", year, dh, payout)
        except FileNotFoundError:
            logger.warning("ERA5 data missing for year %d — skipping", year)
    return pd.DataFrame(records)
