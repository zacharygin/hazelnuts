"""
Weather index: combine all in-season triggers into a single predicted
production shortfall, then apply the production payout bands.

Architecture:
  predicted_shortfall(year) = β_frost × frost_DH
                             + β_drought × max(0, -SPEI)
                             + β_hail × hail_CP
                             + β_poll × max(0, poll_mm - 120)
                             + intercept

  payout = production_payout_bands(predicted_shortfall)

This means:
  - One payout function (the same shortfall bands in trigger_params.yaml)
  - All weather triggers are inputs to the shortfall prediction
  - Settles in September (August SPEI is the last input)
  - No TUIK production stat needed for main payout

Lira depreciation remains a separate additive trigger — it measures
input cost squeeze, not crop loss, so it has its own payout schedule.

Named perils (EFB, export ban, Bosporus) remain analytical P×payout.

Optional true-up (November): if TUIK production stat is available,
  true_up = max(0, actual_payout - weather_payout_already_paid)
This closes the basis risk gap for years where weather under-predicts damage.

Calibration:
  β coefficients are fitted via OLS regression of actual production shortfall
  on weather metrics for the 1990–2024 overlapping period. See calibration.py.
  Until enough ERA5 years are downloaded, prior coefficients are used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"

# Prior coefficients: used until calibration regression has 15+ years.
# Derived from agronomic literature:
#   Frost: severe late frost (-1.5°C × 50hr) causes ~30-50% yield loss
#     → β ~ -0.005 per DH (50 DH → -25% shortfall)
#   Drought: SPEI < -2 associated with ~15-20% yield loss
#     → β ~ +0.10 per unit drought_deficit (deficit=2 → -20% shortfall)
#   Hail: basket-level severe hail (25mm/6h) → ~15% yield loss
#     → β ~ -0.006 per mm CP
#   Pollination: excess bloom rain poorly quantified; conservative
#     → β ~ -0.001 per mm excess
PRIOR_COEFFICIENTS = {
    "intercept":      0.00,
    "frost_dh":      -0.005,
    "drought_deficit": -0.10,
    "hail_cp":        -0.006,
    "poll_excess":    -0.001,
}

_CALIBRATED_COEF_CACHE = Path(__file__).parents[2] / "data" / "processed" / "calibrated_coefficients.json"


def load_coefficients() -> dict:
    """
    Load calibrated coefficients if available, otherwise return priors.
    """
    if _CALIBRATED_COEF_CACHE.exists():
        import json
        with open(_CALIBRATED_COEF_CACHE) as f:
            coefs = json.load(f)
        logger.info("Using calibrated coefficients (n=%d obs, R²=%.3f)",
                    coefs.get("n_obs", 0), coefs.get("r_squared", 0))
        return coefs["coefficients"]
    else:
        logger.info("Using prior coefficients (calibration not yet run)")
        return PRIOR_COEFFICIENTS


def predict_shortfall(
    frost_dh: float = 0.0,
    drought_spei: float = 0.0,
    hail_cp: float = 0.0,
    pollination_index: float = 0.0,
    coefficients: dict | None = None,
) -> float:
    """
    Predict production shortfall from weather metrics.

    Returns predicted shortfall as a fraction (e.g. -0.25 = 25% below baseline).
    Clipped to [-1.0, 0.0] since shortfall can't exceed 100% or be positive.
    """
    if coefficients is None:
        coefficients = load_coefficients()

    drought_deficit = max(0.0, -drought_spei)
    poll_excess = max(0.0, pollination_index - 120.0)

    shortfall = (
        coefficients.get("intercept", 0.0)
        + coefficients.get("frost_dh", 0.0)        * frost_dh
        + coefficients.get("drought_deficit", 0.0) * drought_deficit
        + coefficients.get("hail_cp", 0.0)          * hail_cp
        + coefficients.get("poll_excess", 0.0)      * poll_excess
    )
    return float(np.clip(shortfall, -1.0, 0.5))


def compute_weather_payout(
    frost_dh: float = 0.0,
    drought_spei: float = 0.0,
    hail_cp: float = 0.0,
    pollination_index: float = 0.0,
    coefficients: dict | None = None,
) -> tuple[float, float]:
    """
    Compute payout from weather metrics via predicted shortfall.

    Returns (predicted_shortfall, payout_fraction).
    """
    from src.triggers.production import compute_payout

    pred_sf = predict_shortfall(frost_dh, drought_spei, hail_cp,
                                 pollination_index, coefficients)
    payout = compute_payout(pred_sf)
    return pred_sf, payout


def save_calibrated_coefficients(result: dict) -> None:
    """Save fitted coefficients from calibration regression."""
    import json
    _CALIBRATED_COEF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "coefficients": result["coefficients"],
        "n_obs": result["n_obs"],
        "r_squared": result.get("r_squared"),
    }
    with open(_CALIBRATED_COEF_CACHE, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved calibrated coefficients to %s", _CALIBRATED_COEF_CACHE)


def weather_index_series(available_years: list[int] | None = None) -> pd.DataFrame:
    """
    Build the full weather index for all years where data is available.
    Returns DataFrame: [year, frost_dh, drought_spei, hail_cp, pollination_index,
                        predicted_shortfall, weather_payout]
    """
    from src.pricing.calibration import build_feature_matrix
    from src.triggers.production import compute_payout

    coefs = load_coefficients()
    df = build_feature_matrix(years=available_years, verbose=False)

    records = []
    for _, row in df.iterrows():
        frost_dh   = row["frost_dh"]   if not np.isnan(row.get("frost_dh", np.nan))   else 0.0
        hail_cp    = row["hail_cp"]    if not np.isnan(row.get("hail_cp", np.nan))    else 0.0
        spei       = row["spei"]       if not np.isnan(row.get("spei", np.nan))       else 0.0
        poll_index = row["pollination_index"] if not np.isnan(row.get("pollination_index", np.nan)) else 0.0

        pred_sf, payout = compute_weather_payout(frost_dh, spei, hail_cp, poll_index, coefs)
        records.append({
            "year":               int(row["year"]),
            "frost_dh":           frost_dh,
            "hail_cp":            hail_cp,
            "drought_spei":       spei,
            "pollination_index":  poll_index,
            "predicted_shortfall": pred_sf,
            "weather_payout":     payout,
            "actual_shortfall":   row.get("shortfall", np.nan),
        })

    return pd.DataFrame(records)


def weather_index_el(half_life: float = 15.0) -> "ELResult":
    """
    Compute expected loss on the weather index using the distributional approach.

    Fits a weighted KDE to predicted shortfall history and integrates
    the production payout function over it.
    """
    from src.pricing.distributions import fit_and_integrate, ELResult
    from src.triggers.production import compute_payout

    df = weather_index_series()
    df = df.dropna(subset=["predicted_shortfall"])

    if len(df) < 5:
        raise ValueError(
            f"Only {len(df)} years with complete weather data — need more ERA5 downloads"
        )

    return fit_and_integrate(
        values=df["predicted_shortfall"].values,
        years=df["year"].values,
        payout_fn=compute_payout,
        trigger_threshold=-0.20,
        dist_type="kde",
        half_life=half_life,
        integration_lower=-1.0,
        integration_upper=0.5,
    )
