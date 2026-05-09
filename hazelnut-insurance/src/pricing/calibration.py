"""
Damage function calibration: map weather trigger metrics to production loss.

Regresses historical production shortfall on weather metrics (frost degree-hours,
drought SPEI, hail convective precip, pollination index) to calibrate what payout
each trigger should generate — grounded in actual crop damage rather than assumed bands.

Method:
  production_shortfall_t = β_frost × DH_t
                         + β_drought × max(0, -SPEI_t)
                         + β_hail × CP_t
                         + β_pollination × max(0, poll_t - deductible)
                         + ε_t

  Estimated via OLS on the overlapping historical period (1990–2024 as data lands).
  Coefficients give the marginal production loss per unit of each trigger metric.

  The payout for each trigger is then:
    payout(metric) = max(0, β × metric) × loading_factor
  where loading_factor ≥ 1 accounts for basis risk (weather trigger understates
  true farm-level loss due to spatial averaging).

Output:
  - Fitted coefficients β for each trigger
  - R² of the joint regression (how much production variance is explained)
  - Per-trigger implied payout bands (replacing the assumed bands)
  - Residual EL: production variance not explained by weather (basis risk)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_feature_matrix(
    years: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Assemble the joint weather + production dataset for available years.

    Returns a DataFrame with columns:
      year, shortfall, frost_dh, hail_cp, spei, pollination_index
    NaN where data not yet available.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from src.triggers.production import metric_series as prod_series
    from src.triggers.frost import compute_dh
    from src.triggers.hail import compute_payout as hail_payout
    from src.data.era5_downloader import RAW_DIR

    prod = prod_series().set_index("year")

    # Determine year range
    if years is None:
        years = list(range(1990, 2025))

    rows = []
    for year in years:
        row = {"year": year}

        # Production shortfall
        row["shortfall"] = float(prod.loc[year, "shortfall"]) if year in prod.index else np.nan

        # Frost: available if era5_temp_{year}.nc exists
        frost_path = RAW_DIR / f"era5_temp_{year}.nc"
        if frost_path.exists():
            try:
                row["frost_dh"] = compute_dh(year)
            except Exception as e:
                logger.warning("Frost DH failed for %d: %s", year, e)
                row["frost_dh"] = np.nan
        else:
            row["frost_dh"] = np.nan

        # Hail: available if era5_hail_{year}.nc exists
        hail_path = RAW_DIR / f"era5_hail_{year}.nc"
        if hail_path.exists():
            try:
                from src.triggers.hail import backtest as hail_backtest
                hdf = hail_backtest([year])
                row["hail_cp"] = float(hdf["max_6h_cp_mm"].iloc[0]) if len(hdf) else np.nan
            except Exception as e:
                logger.warning("Hail CP failed for %d: %s", year, e)
                row["hail_cp"] = np.nan
        else:
            row["hail_cp"] = np.nan

        # SPEI: from ERA5 monthly if computed
        spei_cache = Path(__file__).parents[2] / "data" / "raw" / "spei" / "spei03_era5.csv"
        if spei_cache.exists():
            try:
                spei_df = pd.read_csv(spei_cache).set_index("year")
                row["spei"] = float(spei_df.loc[year, "spei03"]) if year in spei_df.index else np.nan
            except Exception:
                row["spei"] = np.nan
        else:
            row["spei"] = np.nan

        # Pollination: from ERA5 monthly
        try:
            from src.triggers.pollination import metric_series as poll_series
            poll = poll_series().set_index("year")
            row["pollination_index"] = float(poll.loc[year, "pollination_index"]) if year in poll.index else np.nan
        except Exception:
            row["pollination_index"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        n_complete = df.dropna().shape[0]
        logger.info(
            "Feature matrix: %d years, %d complete rows (all features available)",
            len(df), n_complete
        )
    return df


def fit_damage_functions(df: pd.DataFrame) -> dict:
    """
    Fit OLS regression of production shortfall on weather metrics.

    Uses only rows where all features are available.
    Returns dict with coefficients, R², and per-trigger implied payout schedule.
    """
    from scipy import stats as scipy_stats

    features = ["frost_dh", "hail_cp", "spei", "pollination_index"]
    available_features = [f for f in features if f in df.columns and df[f].notna().sum() >= 5]

    complete = df[["shortfall"] + available_features].dropna()
    n = len(complete)

    if n < 5:
        logger.warning("Only %d complete rows — regression unreliable, need more ERA5 years", n)
        return {"n_obs": n, "coefficients": {}, "r_squared": None, "message": "insufficient data"}

    y = complete["shortfall"].values

    # Transform features for regression:
    # frost_dh: direct (higher DH = worse frost = more negative shortfall)
    # hail_cp: direct (higher CP = more hail damage)
    # spei: use negative SPEI below 0 (drought only; positive SPEI irrelevant)
    # pollination_index: use excess above deductible
    X_dict = {}
    if "frost_dh" in available_features:
        X_dict["frost_dh"] = complete["frost_dh"].values
    if "hail_cp" in available_features:
        X_dict["hail_cp"] = complete["hail_cp"].values
    if "spei" in available_features:
        X_dict["drought_deficit"] = np.maximum(0, -complete["spei"].values)
    if "pollination_index" in available_features:
        X_dict["poll_excess"] = np.maximum(0, complete["pollination_index"].values - 120)

    if not X_dict:
        return {"n_obs": n, "coefficients": {}, "r_squared": None}

    X = np.column_stack(list(X_dict.values()))
    X = np.hstack([np.ones((n, 1)), X])  # intercept

    # OLS via normal equations
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        logger.error("OLS failed: %s", e)
        return {"n_obs": n, "coefficients": {}, "r_squared": None}

    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coef_names = ["intercept"] + list(X_dict.keys())
    coefs = dict(zip(coef_names, beta))

    # Implied damage rates: how much shortfall per unit of each trigger
    # e.g. β_frost = -0.002 means 1 degree-hour → -0.2% production shortfall
    implied_rates = {k: v for k, v in coefs.items() if k != "intercept"}

    logger.info("Damage function R²=%.3f on %d obs, features: %s", r2, n, list(X_dict.keys()))
    for name, coef in implied_rates.items():
        logger.info("  β_%s = %.4f  (1 unit → %.2f%% shortfall)", name, coef, coef * 100)

    return {
        "n_obs": n,
        "coefficients": coefs,
        "implied_rates": implied_rates,
        "r_squared": r2,
        "feature_data": complete,
        "y_hat": y_hat,
        "residuals": y - y_hat,
    }


def calibrated_payout_bands(trigger: str, coef: float, n_percentiles: int = 5) -> list:
    """
    Generate payout bands from a fitted damage coefficient.

    Instead of assuming e.g. "25-50 DH → 25% payout", derive the bands
    from the regression: payout = max(0, coef × metric).

    Returns list of [lo, hi, payout_lo, payout_hi] bands for trigger_params.yaml.
    """
    # Typical metric ranges per trigger
    ranges = {
        "frost_dh":    (0, 200),
        "hail_cp":     (0, 50),
        "drought_deficit": (0, 3),
        "poll_excess": (0, 100),
    }
    lo, hi = ranges.get(trigger, (0, 100))
    points = np.linspace(lo, hi, n_percentiles + 1)
    bands = []
    for i in range(len(points) - 1):
        band_lo = points[i]
        band_hi = points[i + 1]
        pay_lo = float(np.clip(-coef * band_lo, 0, 1))
        pay_hi = float(np.clip(-coef * band_hi, 0, 1))
        bands.append([round(band_lo, 2), round(band_hi, 2), round(pay_lo, 3), round(pay_hi, 3)])
    return bands


def print_calibration_report(result: dict) -> None:
    print(f"\n{'='*70}")
    print("DAMAGE FUNCTION CALIBRATION REPORT")
    print(f"{'='*70}")
    print(f"Observations: {result['n_obs']}")
    if result.get("r_squared") is not None:
        print(f"R²:           {result['r_squared']:.3f}  "
              f"({'good fit' if result['r_squared'] > 0.4 else 'weak — need more years'})")

    if not result.get("coefficients"):
        print(f"\n{result.get('message', 'No coefficients fitted')}")
        return

    print(f"\nDamage coefficients (β = production shortfall per unit trigger):")
    for name, coef in result["coefficients"].items():
        if name == "intercept":
            print(f"  intercept:  {coef:+.4f}  ({coef*100:+.2f}% baseline drift)")
        else:
            print(f"  β_{name:<22} {coef:+.6f}  ({coef*100:+.4f}% per unit)")

    if "feature_data" in result:
        df = result["feature_data"].copy()
        df["predicted"] = result["y_hat"]
        df["residual"]  = result["residuals"]
        print(f"\nFit vs actual (available years):")
        print(f"  {'year':>4}  {'actual':>8}  {'predicted':>9}  {'residual':>9}")
        for _, row in df.iterrows():
            print(f"  {int(row['year']):>4}  {row['shortfall']:>+8.1%}  "
                  f"{row['predicted']:>+9.1%}  {row['residual']:>+9.1%}")

    print(f"\nResidual std (unexplained production variance): "
          f"{np.std(result['residuals'])*100:.1f}%")
    print(f"  → This is the basis risk: production shortfall NOT explained by weather triggers.")
    print(f"{'='*70}\n")
