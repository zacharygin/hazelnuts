"""
Expected loss calculation for Component A of the hazelnut insurance contract.

Uses the distributional approach:
  EL(trigger) = ∫ payout(x) · f(x; θ) dx

where f(x; θ) is a weighted distribution fitted to historical trigger metrics,
with exponential decay weighting so recent years matter more than distant history.

Named perils (EFB, export disruption, Bosporus) remain as analytical
P × payout estimates — too few historical events to fit a distribution.

Output: per-trigger EL, decomposed P(fires) × E[pay|fires], plus total
contract EL and an indicative premium range.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.pricing.distributions import fit_and_integrate, sensitivity_table, ELResult
from src.triggers.named_perils import assumed_el_table

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Per-trigger distributional EL
# ---------------------------------------------------------------------------

def production_el(half_life: float = 15.0) -> ELResult:
    """Fit weighted KDE to production shortfall history, integrate payout."""
    from src.triggers.production import metric_series, compute_payout
    series = metric_series()
    return fit_and_integrate(
        values=series["shortfall"].values,
        years=series["year"].values,
        payout_fn=compute_payout,
        trigger_threshold=-0.20,   # deductible: -20% shortfall
        dist_type="kde",
        half_life=half_life,
        integration_lower=-1.2,
        integration_upper=0.5,
    )


def drought_el(half_life: float = 15.0) -> ELResult:
    """
    Fit weighted Normal to August SPEI03 history, integrate payout.
    SPEI is approximately N(0,1) by construction; we verify then use KDE
    as the primary estimator for robustness.
    """
    from src.triggers.drought import metric_series as drought_series, compute_payout
    series = drought_series()
    return fit_and_integrate(
        values=series["spei"].values,
        years=series["year"].values,
        payout_fn=compute_payout,
        trigger_threshold=-1.5,    # deductible: SPEI < -1.5
        dist_type="kde",
        half_life=half_life,
        integration_lower=-4.0,
        integration_upper=4.0,
    )


def frost_el(half_life: float = 15.0) -> ELResult:
    """
    Fit zero-inflated Gamma to degree-hour history, integrate payout.
    Requires ERA5 data to have been downloaded.
    """
    from src.triggers.frost import backtest as frost_backtest, compute_payout
    from src.data.era5_downloader import RAW_DIR

    # Build metric series from cached ERA5 files
    years_available = sorted(
        int(p.stem.replace("era5_temp_", ""))
        for p in RAW_DIR.glob("era5_temp_*.nc")
    )
    if not years_available:
        raise FileNotFoundError(
            "No ERA5 temperature files found. "
            "Run: python -m src.data.era5_downloader --type frost"
        )

    df = frost_backtest(years_available)
    return fit_and_integrate(
        values=df["dh"].values,
        years=df["year"].values,
        payout_fn=compute_payout,
        trigger_threshold=25.0,    # deductible: 25 degree-hours
        dist_type="zero_inflated_gamma",
        half_life=half_life,
        integration_lower=0.0,
        integration_upper=500.0,
    )


def hail_el(half_life: float = 15.0) -> ELResult:
    """
    Fit zero-inflated Gamma to convective precip history, integrate payout.
    Requires ERA5 hail (cp) data to have been downloaded.
    """
    from src.triggers.hail import backtest as hail_backtest, compute_payout
    from src.data.era5_downloader import RAW_DIR

    years_available = sorted(
        int(p.stem.replace("era5_hail_", ""))
        for p in RAW_DIR.glob("era5_hail_*.nc")
    )
    if not years_available:
        raise FileNotFoundError(
            "No ERA5 hail (cp) files found. "
            "Run: python -m src.data.era5_downloader --type hail"
        )

    df = hail_backtest(years_available)
    return fit_and_integrate(
        values=df["max_6h_cp_mm"].values,
        years=df["year"].values,
        payout_fn=compute_payout,
        trigger_threshold=8.0,
        dist_type="zero_inflated_gamma",
        half_life=half_life,
        integration_lower=0.0,
        integration_upper=100.0,
    )


# ---------------------------------------------------------------------------
# Aggregate expected loss
# ---------------------------------------------------------------------------

def lira_el(half_life: float = 15.0) -> ELResult:
    """Fit weighted KDE to annual TRY/USD depreciation history, integrate payout."""
    from src.triggers.lira import metric_series as lira_series, compute_payout
    series = lira_series()
    return fit_and_integrate(
        values=series["depreciation"].values,
        years=series["year"].values,
        payout_fn=compute_payout,
        trigger_threshold=-0.20,
        dist_type="kde",
        half_life=half_life,
        integration_lower=-1.0,
        integration_upper=0.3,
    )


def compute_expected_loss(
    half_life: float = 15.0,
    include_frost: bool = True,
    include_hail: bool = True,
    include_drought: bool = True,
    include_lira: bool = True,
) -> dict:
    """
    Compute per-trigger and total contract expected loss using the
    distributional approach.

    Triggers requiring ERA5 data (frost, hail) are skipped with a warning
    if the data has not been downloaded yet.

    Parameters
    ----------
    half_life       : recency decay half-life in years (default 15)
    include_frost   : set False to skip if ERA5 not available
    include_hail    : set False to skip if ERA5 not available
    include_drought : set False to skip if SPEI not available

    Returns
    -------
    dict with keys:
        per_trigger  : DataFrame, one row per trigger
        total_el     : float, total contract EL as fraction of notional
        premium_low  : float, indicative premium (30% risk load, 1.5% capital)
        premium_high : float, indicative premium (50% risk load, 2% capital)
        half_life    : float, decay parameter used
    """
    cfg = _load_cfg()
    rows = []
    total_el = 0.0

    # --- Empirical triggers ---

    # Production: always available
    try:
        r = production_el(half_life)
        rows.append({
            "trigger": "Production",
            "el_pct": r.el * 100,
            "p_fires_pct": r.p_fires * 100,
            "cond_el_pct": r.cond_el * 100,
            "el_std_pct": np.sqrt(r.el_var) * 100,
            "eff_n": r.n_weighted_obs,
            "method": f"KDE, hl={half_life}yr",
        })
        total_el += r.el
        logger.info("Production EL: %.2f%%", r.el * 100)
    except Exception as e:
        logger.error("Production EL failed: %s", e)

    # Frost: requires ERA5
    if include_frost:
        try:
            r = frost_el(half_life)
            rows.append({
                "trigger": "Frost",
                "el_pct": r.el * 100,
                "p_fires_pct": r.p_fires * 100,
                "cond_el_pct": r.cond_el * 100,
                "el_std_pct": np.sqrt(r.el_var) * 100,
                "eff_n": r.n_weighted_obs,
                "method": f"ZI-Gamma, hl={half_life}yr",
            })
            # Frost and production are substitutive (max, not additive).
            # Adjusting: add frost EL minus the overlap with production.
            # Overlap = correlation adjustment; approximated below.
            # For now, add frost EL but note it will be adjusted for
            # the max() substitution in the final aggregation.
            total_el += r.el  # placeholder; aggregation handles double-counting
            logger.info("Frost EL: %.2f%%", r.el * 100)
        except FileNotFoundError as e:
            logger.warning("Frost EL skipped: %s", e)

    # Drought: requires SPEI
    if include_drought:
        try:
            r = drought_el(half_life)
            rows.append({
                "trigger": "Drought",
                "el_pct": r.el * 100,
                "p_fires_pct": r.p_fires * 100,
                "cond_el_pct": r.cond_el * 100,
                "el_std_pct": np.sqrt(r.el_var) * 100,
                "eff_n": r.n_weighted_obs,
                "method": f"KDE, hl={half_life}yr",
            })
            total_el += r.el
            logger.info("Drought EL: %.2f%%", r.el * 100)
        except FileNotFoundError as e:
            logger.warning("Drought EL skipped: %s", e)

    # Hail: requires ERA5 cp
    if include_hail:
        try:
            r = hail_el(half_life)
            rows.append({
                "trigger": "Hail",
                "el_pct": r.el * 100,
                "p_fires_pct": r.p_fires * 100,
                "cond_el_pct": r.cond_el * 100,
                "el_std_pct": np.sqrt(r.el_var) * 100,
                "eff_n": r.n_weighted_obs,
                "method": f"ZI-Gamma, hl={half_life}yr",
            })
            total_el += r.el
            logger.info("Hail EL: %.2f%%", r.el * 100)
        except FileNotFoundError as e:
            logger.warning("Hail EL skipped: %s", e)

    # Lira depreciation: always available (Yahoo Finance)
    if include_lira:
        try:
            r = lira_el(half_life)
            rows.append({
                "trigger": "Lira Depreciation",
                "el_pct": r.el * 100,
                "p_fires_pct": r.p_fires * 100,
                "cond_el_pct": r.cond_el * 100,
                "el_std_pct": np.sqrt(r.el_var) * 100,
                "eff_n": r.n_weighted_obs,
                "method": f"KDE, hl={half_life}yr",
            })
            total_el += r.el
            logger.info("Lira EL: %.2f%%", r.el * 100)
        except Exception as e:
            logger.warning("Lira EL skipped: %s", e)

    # --- Named perils (analytical) ---
    assumed = assumed_el_table()
    for _, row in assumed.iterrows():
        rows.append({
            "trigger": row["peril"].replace("_", " ").title(),
            "el_pct": row["expected_loss"] * 100,
            "p_fires_pct": row["annual_probability"] * 100,
            "cond_el_pct": row["payout_if_fired"] * 100,
            "el_std_pct": None,
            "eff_n": None,
            "method": "analytical P×payout",
        })
        total_el += row["expected_loss"]

    # Note: frost + production are substitutive (max), not additive.
    # A simple additive sum double-counts correlated years. The distributional
    # approach handles this correctly when frost data is available by computing
    # the joint distribution. Until then, flag it.
    frost_in = any(r["trigger"] == "Frost" for r in rows)
    prod_in  = any(r["trigger"] == "Production" for r in rows)
    if frost_in and prod_in:
        frost_el_val = next(r["el_pct"] for r in rows if r["trigger"] == "Frost") / 100
        prod_el_val  = next(r["el_pct"] for r in rows if r["trigger"] == "Production") / 100
        # Approximate substitution credit: in years where both fire, you pay max not sum.
        # Rough adjustment: reduce total by min(frost_el, prod_el) × correlation factor.
        # Placeholder until joint distribution is available.
        substitution_discount = min(frost_el_val, prod_el_val) * 0.6
        total_el -= substitution_discount
        logger.info(
            "Frost/production substitution discount: -%.2f%% (approx)",
            substitution_discount * 100,
        )

    per_trigger = pd.DataFrame(rows)

    # Add total row
    total_row = pd.DataFrame([{
        "trigger": "TOTAL",
        "el_pct": total_el * 100,
        "p_fires_pct": None,
        "cond_el_pct": None,
        "el_std_pct": None,
        "eff_n": None,
        "method": "—",
    }])
    per_trigger = pd.concat([per_trigger, total_row], ignore_index=True)

    premium_low  = total_el * 1.30 + 0.015
    premium_high = total_el * 1.50 + 0.020

    return {
        "per_trigger": per_trigger,
        "total_el": total_el,
        "premium_low": premium_low,
        "premium_high": premium_high,
        "half_life": half_life,
    }


def print_el_table(result: dict) -> None:
    df = result["per_trigger"].copy()
    hl = result["half_life"]
    print(f"\n{'='*78}")
    print(f"Component A — Expected Loss  (half-life={hl}yr recency weighting)")
    print(f"{'='*78}")
    fmt = {
        "el_pct":       lambda x: f"{x:.2f}%" if x is not None else "—",
        "p_fires_pct":  lambda x: f"{x:.1f}%"  if x is not None else "—",
        "cond_el_pct":  lambda x: f"{x:.1f}%"  if x is not None else "—",
        "el_std_pct":   lambda x: f"±{x:.2f}%" if x is not None else "—",
    }
    for col, fn in fmt.items():
        df[col] = df[col].apply(fn)
    print(df.to_string(index=False))
    print(f"\nTotal EL:       {result['total_el']*100:.2f}% of notional")
    print(f"Premium range:  {result['premium_low']*100:.2f}% – {result['premium_high']*100:.2f}% of notional")
    print(f"  (30–50% risk load + 1.5–2.0% capital charge)")
    print(f"{'='*78}\n")


def halflife_sensitivity(half_lives: list[float] | None = None) -> pd.DataFrame:
    """
    Compute total EL across a range of half-life assumptions.
    Shows how sensitive the premium is to the recency weighting choice.
    """
    if half_lives is None:
        half_lives = [5, 10, 15, 20, 30]

    rows = []
    for hl in half_lives:
        result = compute_expected_loss(
            half_life=hl, include_frost=False, include_hail=False, include_drought=False
        )
        rows.append({
            "half_life_yr": hl,
            "total_el_pct": result["total_el"] * 100,
            "premium_low_pct": result["premium_low"] * 100,
            "premium_high_pct": result["premium_high"] * 100,
        })
    return pd.DataFrame(rows)
