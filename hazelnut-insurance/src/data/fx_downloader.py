"""
FX data downloader for TRY/USD exchange rate.

Source: Yahoo Finance (TRYUSD=X), daily close, 2005–present.
Used by the lira depreciation trigger.

Annual metric: calendar-year percentage change in TRY/USD rate.
Negative value = TRY weakened vs USD (depreciated).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "fx"
CACHE_FILE = RAW_DIR / "tryusd_annual.csv"


def _download_from_yahoo() -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fx = yf.download("TRYUSD=X", start="2000-01-01", end="2025-12-31", progress=False)
    if fx.empty:
        raise ValueError("yfinance returned empty dataframe for TRYUSD=X")

    close = fx["Close"].resample("YE").last()
    close.index = close.index.year
    close.name = "tryusd_close"
    return close


def load_annual_fx(force_download: bool = False) -> pd.DataFrame:
    """
    Return annual TRY/USD close price and YoY depreciation rate.

    Columns:
        year            : int
        tryusd_close    : year-end TRY per USD (higher = TRY weaker)
        depreciation    : fractional YoY change in TRY strength
                          negative = TRY weakened (e.g. -0.44 = lost 44% vs USD)

    Note: depreciation is from the TRY holder's perspective:
        -0.44 in 2021 means 1 TRY bought 44% fewer USD at year-end vs prior year.
    """
    if not force_download and CACHE_FILE.exists():
        df = pd.read_csv(CACHE_FILE)
        logger.info("Loaded FX from cache: %s", CACHE_FILE)
        return df

    logger.info("Downloading TRY/USD from Yahoo Finance")
    close = _download_from_yahoo()

    # pct_change: (close_t / close_{t-1}) - 1
    # TRYUSD = how many USD per 1 TRY; decline = TRY depreciation
    depr = close.pct_change()

    df = pd.DataFrame({
        "year": close.index.astype(int),
        "tryusd_close": close.values.flatten(),
        "depreciation": depr.values.flatten(),
    }).dropna(subset=["depreciation"]).reset_index(drop=True)

    df.to_csv(CACHE_FILE, index=False)
    logger.info("Saved FX cache: %s", CACHE_FILE)
    return df
