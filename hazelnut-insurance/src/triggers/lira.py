"""
Lira depreciation trigger.

Mechanism: Turkish hazelnut farmers buy inputs (fertilizers, pesticides,
machinery) priced in USD. When TRY depreciates sharply, input costs spike
in lira terms. Export revenue rises too (hazelnuts priced in EUR/USD), but
with a lag: the TMO price floor is set annually in August and does not
automatically adjust for mid-season FX moves. Farmers on forward contracts
or selling domestically are fully exposed.

Trigger metric: annual TRY/USD depreciation rate.
    -0.30 = TRY lost 30% of its USD value in the calendar year.

Deductible: -20% depreciation (normal background volatility for TRY).
Payout bands: ramp from -20% to -60%+ depreciation.
Max payout capped at 20% of notional (FX is a partial hedge — hazelnut
export prices rise in lira terms, partially offsetting input cost squeeze).

Payout schedule rationale:
  At -20% (deductible): 0% — TRY has lost 20%/yr on average; not a stress event.
  At -40%: 10% — significant input cost squeeze (2015-level event).
  At -60%: 20% — severe squeeze (2021-level event: TRY lost 44%).
  Above -60%: 20% cap — above this, hazelnut export prices typically
              rally enough in lira to offset further input squeeze.

Data source: Yahoo Finance TRYUSD=X, 2005–present.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fx_downloader import load_annual_fx

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("lira", _default_cfg())


def _default_cfg() -> dict:
    return {
        "deductible": -0.20,
        "payout_bands": [
            [-1.00, -0.60, 0.20, 0.20],
            [-0.60, -0.40, 0.20, 0.10],
            [-0.40, -0.20, 0.10, 0.00],
            [-0.20,  1.00, 0.00, 0.00],
        ],
    }


def _payout_from_bands(depr: float, bands: list) -> float:
    for lo, hi, pay_lo, pay_hi in bands:
        lo = float("-inf") if str(lo) in ("-.inf", "-inf") else float(lo)
        hi = float("inf") if str(hi) in (".inf", "inf") else float(hi)
        if lo <= depr < hi:
            if pay_lo == pay_hi or lo == hi:
                return pay_lo
            frac = (depr - lo) / (hi - lo)
            return pay_lo + frac * (pay_hi - pay_lo)
    return 0.0


def compute_payout(depreciation: float) -> float:
    """Return payout as fraction of notional given annual TRY depreciation rate."""
    cfg = _load_cfg()
    return _payout_from_bands(depreciation, cfg["payout_bands"])


def metric_series() -> pd.DataFrame:
    """
    Return annual TRY depreciation series.
    Returns DataFrame with columns [year, depreciation].
    """
    df = load_annual_fx()
    return df[["year", "depreciation"]].dropna().reset_index(drop=True)


def backtest(years: list[int]) -> pd.DataFrame:
    """Compute depreciation and payout for each year."""
    df = load_annual_fx()
    df_idx = df.set_index("year")
    records = []
    for year in years:
        if year not in df_idx.index:
            logger.warning("Lira: no FX data for %d — skipping", year)
            continue
        depr = float(df_idx.loc[year, "depreciation"])
        payout = compute_payout(depr)
        records.append({"year": year, "depreciation": depr, "payout": payout})
        logger.info("Lira %d: depreciation=%.1f%%, payout=%.3f", year, depr * 100, payout)
    return pd.DataFrame(records)
