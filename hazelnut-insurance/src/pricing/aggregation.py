"""
Aggregation logic for Component A payout in a given year.

Aggregation rules (from contract spec):
  1. Core payout = max(frost_payout, production_payout)   [substitutive]
  2. Total payout = core + drought + sum(named_perils)    [additive]
  3. Annual cap = min(total, 1.0)

Named perils use assumed annual probability for EL calculation, not
empirical frequency (insufficient history). The backtest series will
show near-zero named peril payouts, which is correct — the EL contribution
is added analytically in expected_loss.py.
"""
from __future__ import annotations

import pandas as pd


def aggregate_year(
    frost_payout: float,
    production_payout: float,
    drought_payout: float,
    named_perils_payout: float = 0.0,
    annual_cap: float = 1.0,
) -> dict:
    core = max(frost_payout, production_payout)
    total = core + drought_payout + named_perils_payout
    total_capped = min(total, annual_cap)
    return {
        "frost_payout": frost_payout,
        "production_payout": production_payout,
        "core_payout": core,
        "drought_payout": drought_payout,
        "named_perils_payout": named_perils_payout,
        "total_payout_uncapped": total,
        "total_payout": total_capped,
    }


def aggregate_backtest(
    frost_df: pd.DataFrame,
    production_df: pd.DataFrame,
    drought_df: pd.DataFrame,
    hail_df: pd.DataFrame,
    named_perils_df: pd.DataFrame,
    annual_cap: float = 1.0,
) -> pd.DataFrame:
    """
    Merge per-trigger backtest DataFrames on year and compute aggregated payout.

    Each input DataFrame must have a 'year' column and a 'payout' column
    (except named_perils_df which has per-peril columns).

    Aggregation rules:
      core = max(frost, production)       [substitutive — correlated events]
      total = core + drought + hail + named_perils   [additive]
      total = min(total, annual_cap)
    """
    df = (
        frost_df[["year", "payout"]].rename(columns={"payout": "frost_payout"})
        .merge(
            production_df[["year", "payout"]].rename(columns={"payout": "production_payout"}),
            on="year", how="outer",
        )
        .merge(
            drought_df[["year", "payout"]].rename(columns={"payout": "drought_payout"}),
            on="year", how="outer",
        )
        .merge(
            hail_df[["year", "payout"]].rename(columns={"payout": "hail_payout"}),
            on="year", how="outer",
        )
        .merge(named_perils_df, on="year", how="outer")
        .fillna(0.0)
        .sort_values("year")
        .reset_index(drop=True)
    )

    named_cols = [
        c for c in df.columns
        if c.endswith("_payout")
        and c not in ("frost_payout", "production_payout", "drought_payout", "hail_payout")
    ]
    df["named_perils_payout"] = df[named_cols].sum(axis=1)

    df["core_payout"] = df[["frost_payout", "production_payout"]].max(axis=1)
    df["total_payout_uncapped"] = (
        df["core_payout"]
        + df["drought_payout"]
        + df["hail_payout"]
        + df["named_perils_payout"]
    )
    df["total_payout"] = df["total_payout_uncapped"].clip(upper=annual_cap)

    return df
