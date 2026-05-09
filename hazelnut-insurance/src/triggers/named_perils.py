"""
Named perils trigger evaluator.

For EFB, export disruption, and logistics (Bosporus): binary triggers.
Expected loss is driven by assumed annual probabilities (not empirical
frequency) because historical events are too sparse.

Annual probabilities (see config/trigger_params.yaml for source notes):
  EFB outbreak:          0.5%  (phytosanitary risk; EFB spreading through EU)
  Export disruption:     3.0%  (government interventions; ~1 event per 33 years)
  Logistics disruption:  0.2%  (Bosporus 30+ day closure; extraordinarily rare)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.named_events import fired, severity

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["named_perils"]


def compute_historical_payout(peril: str, year: int) -> float:
    """
    Return payout for a peril in a given year based on historical event data.
    Uses severity from named_events.py; zero if no event recorded.
    """
    cfg = _load_cfg()
    if not fired(peril, year):
        return 0.0
    sev = severity(peril, year)
    return cfg[peril]["payout"] * sev


def expected_loss_assumed(peril: str) -> float:
    """
    Return assumed expected loss for a peril using annual probability × payout.
    Used when historical data is too sparse for empirical estimation.
    """
    cfg = _load_cfg()
    p = cfg[peril]["annual_probability"]
    payout = cfg[peril]["payout"]
    return p * payout


def backtest_named_perils(years: list[int]) -> pd.DataFrame:
    """
    Build historical payout series for all named perils.
    For years with no event, payout is 0. For priced EL, use expected_loss_assumed.
    """
    perils = ["efb_outbreak", "export_disruption", "logistics_disruption"]
    records = []
    for year in years:
        row = {"year": year}
        for peril in perils:
            row[f"{peril}_payout"] = compute_historical_payout(peril, year)
        records.append(row)
    return pd.DataFrame(records)


def assumed_el_table() -> pd.DataFrame:
    """Return assumed expected loss for each named peril."""
    perils = ["efb_outbreak", "export_disruption", "logistics_disruption"]
    cfg = _load_cfg()
    rows = []
    for peril in perils:
        p = cfg[peril]["annual_probability"]
        payout = cfg[peril]["payout"]
        rows.append({
            "peril": peril,
            "annual_probability": p,
            "payout_if_fired": payout,
            "expected_loss": p * payout,
            "note": cfg[peril].get("note", ""),
        })
    return pd.DataFrame(rows)
