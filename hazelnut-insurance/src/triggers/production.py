"""
Production trigger: percentage decline vs. rolling baseline.

Baseline: 5-year rolling average of [year-5, year-1], excluding the
single worst year in the window (bottom-decile exclusion from a 5-year
window = 1 year excluded).

Note on calibration: The plan expects 2014 and 2025 in the -35% to -50%
band. Preliminary FAOSTAT review suggests both years may land closer to
-20% to -35%. Run notebook 03 to validate before treating the payout
schedule as final.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.faostat_downloader import load as load_faostat

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["production"]


def _payout_from_bands(shortfall: float, bands: list) -> float:
    """
    Linearly interpolate payout within the matching shortfall band.

    Shortfall is negative (e.g., -0.30 = 30% below baseline).
    Bands are ordered from most severe (lo most negative) to least severe.
    Within each band: payout_at_lo > payout_at_hi (worse shortfall → more payout).

    Shortfalls worse than the lowest band floor are capped at the floor's payout.
    """
    parsed = []
    for lo, hi, pay_lo, pay_hi in bands:
        lo = float("-inf") if str(lo) in ("-.inf", "-inf") else float(lo)
        hi = float("inf") if str(hi) in (".inf", "inf") else float(hi)
        parsed.append((lo, hi, float(pay_lo), float(pay_hi)))

    # Cap at most-severe band floor
    min_lo = min(lo for lo, _, _, _ in parsed)
    if shortfall < min_lo:
        return parsed[0][2]  # pay_lo of worst band = maximum payout

    for lo, hi, pay_lo, pay_hi in parsed:
        if lo <= shortfall < hi:
            if pay_lo == pay_hi:
                return pay_lo
            frac = (shortfall - lo) / (hi - lo)
            return pay_lo + frac * (pay_hi - pay_lo)
    return 0.0


def compute_baseline(year: int, df: pd.DataFrame | None = None) -> float | None:
    """
    5-year rolling baseline for the given year.
    Uses production from [year-5, year-1], excluding the single lowest year.
    Returns None if insufficient data.
    """
    cfg = _load_cfg()
    window = cfg.get("baseline_window", 5)
    exclude_n = cfg.get("exclude_worst_n", 1)

    if df is None:
        df = load_faostat()

    lookback = df[df["year"].between(year - window, year - 1)]["production_mt"]
    if len(lookback) < window - exclude_n:
        return None

    sorted_vals = lookback.sort_values()
    trimmed = sorted_vals.iloc[exclude_n:]
    return float(trimmed.mean())


def compute_shortfall(year: int, df: pd.DataFrame | None = None) -> float | None:
    """
    Returns fractional shortfall: (production - baseline) / baseline.
    Negative means production below baseline.
    Returns None if data unavailable.
    """
    if df is None:
        df = load_faostat()

    row = df[df["year"] == year]
    if row.empty:
        return None

    production = float(row["production_mt"].iloc[0])
    baseline = compute_baseline(year, df)
    if baseline is None or baseline == 0:
        return None

    return (production - baseline) / baseline


def compute_payout(shortfall: float) -> float:
    """Return payout as fraction of notional (0.0–1.0) given shortfall fraction."""
    cfg = _load_cfg()
    return _payout_from_bands(shortfall, cfg["payout_bands"])


def metric_series(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return the full historical shortfall series (all available years).
    Used by the distributional pricing engine as input to fit_and_integrate().

    Returns DataFrame with columns [year, shortfall].
    """
    if df is None:
        df = load_faostat()
    records = []
    for year in df["year"].values:
        sf = compute_shortfall(int(year), df)
        if sf is not None:
            records.append({"year": int(year), "shortfall": sf})
    return pd.DataFrame(records)


def backtest(years: list[int]) -> pd.DataFrame:
    """Compute shortfall and payout for each year. Returns DataFrame."""
    df_prod = load_faostat()
    records = []
    for year in years:
        shortfall = compute_shortfall(year, df_prod)
        if shortfall is None:
            logger.warning("Production data unavailable for year %d — skipping", year)
            continue
        payout = compute_payout(shortfall)
        records.append({
            "year": year,
            "shortfall": shortfall,
            "payout": payout,
        })
        logger.info(
            "Production %d: shortfall=%.1f%%, payout=%.3f",
            year, shortfall * 100, payout
        )
    return pd.DataFrame(records)
