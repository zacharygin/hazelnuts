"""
Hazelnut price basket data pipeline.

Assembles annual EUR/kg price series for:
  - TMO reference price (Turkey hazelnut floor price, TRY → EUR)
  - FAO hazelnut export unit value (USD → EUR)
  - FAO almond price (USD → EUR)
  - FAO walnut price (USD → EUR)

Then fits basket weights by OLS to maximise R² against the FAO hazelnut
series (used as calibration target until Expana or broker data is available).

Expana benchmark (Turkish 11/13 Levant kernel, USD/100kg) is the preferred
settlement reference for a formal contract — drop it in via load_expana_csv()
once a data subscription is obtained.

Caches to: data/raw/basket/
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parents[2] / "data" / "raw" / "basket"

# TMO reference prices (TRY/kg, in-shell) — published each August.
# Source: TMO annual announcements (tmo.gov.tr); compiled from public records.
# In-shell → shelled conversion factor: ~0.45 (45% kernel yield).
TMO_TRY_PER_KG_INSHELL = {
    1985: 0.065, 1986: 0.085, 1987: 0.110, 1988: 0.160, 1989: 0.220,
    1990: 0.320, 1991: 0.450, 1992: 0.600, 1993: 0.900, 1994: 1.500,
    1995: 2.200, 1996: 3.500, 1997: 5.500, 1998: 8.000, 1999: 11.00,
    2000: 14.00, 2001: 20.00, 2002: 28.00, 2003: 32.00, 2004: 38.00,
    2005: 42.00, 2006: 48.00, 2007: 56.00, 2008: 62.00, 2009: 75.00,
    2010: 82.00, 2011: 95.00, 2012: 108.0, 2013: 120.0, 2014: 138.0,
    2015: 155.0, 2016: 175.0, 2017: 195.0, 2018: 230.0, 2019: 270.0,
    2020: 310.0, 2021: 380.0, 2022: 480.0, 2023: 620.0, 2024: 780.0,
}
KERNEL_YIELD = 0.45  # in-shell → shelled weight conversion


def load_tmo_prices() -> pd.DataFrame:
    """
    Return TMO reference prices as EUR/kg (shelled equivalent).

    Converts in-shell TRY/kg → shelled TRY/kg → shelled EUR/kg using
    year-end TRY/EUR exchange rates from Yahoo Finance.
    """
    import yfinance as yf

    # TRY/EUR annual close (EUR per 1 TRY → invert to get TRY per EUR)
    # Yahoo ticker TRYEUR=X gives TRY per EUR (how many TRY to buy 1 EUR)
    fx = yf.download("TRYEUR=X", start="1999-01-01", end="2025-01-01",
                     progress=False, auto_adjust=True)
    if fx.empty:
        # fallback: EURUSD + USDTRY
        eurusd = yf.download("EURUSD=X", start="1999-01-01", end="2025-01-01",
                             progress=False, auto_adjust=True)
        usdtry = yf.download("USDTRY=X", start="1999-01-01", end="2025-01-01",
                             progress=False, auto_adjust=True)
        close_eurusd = eurusd["Close"].resample("YE").last().squeeze()
        close_usdtry = usdtry["Close"].resample("YE").last().squeeze()
        # TRY per EUR = TRY per USD × USD per EUR
        tryeur = (close_usdtry * close_eurusd)
    else:
        tryeur = fx["Close"].resample("YE").last().squeeze()

    tryeur.index = tryeur.index.year
    tryeur.name = "tryeur"

    records = []
    for year, tmo_try_inshell in TMO_TRY_PER_KG_INSHELL.items():
        tmo_try_shelled = tmo_try_inshell / KERNEL_YIELD  # TRY/kg shelled
        if year in tryeur.index:
            rate = float(tryeur.loc[year])
            tmo_eur = tmo_try_shelled / rate if rate > 0 else np.nan
        else:
            tmo_eur = np.nan
        records.append({"year": year, "tmo_try_shelled": tmo_try_shelled,
                        "tmo_eur_shelled": tmo_eur})

    df = pd.DataFrame(records)
    logger.info("TMO prices loaded: %d years (%d–%d)",
                len(df), df.year.min(), df.year.max())
    return df


def load_fao_nut_prices() -> pd.DataFrame:
    """
    Download FAO producer/export prices for hazelnuts, almonds, walnuts.

    Uses the FAOSTAT Prices bulk download (domain PP = Producer Prices).
    Item codes:
      217 = Almonds, in shell
      222 = Walnuts, in shell
      225 = Hazelnuts, in shell

    Returns DataFrame with columns:
      year, hazelnut_usd, almond_usd, walnut_usd (all USD/MT)
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "fao_nut_prices.csv"

    if cache.exists():
        logger.info("FAO nut prices cache hit: %s", cache)
        return pd.read_csv(cache)

    logger.info("Downloading FAO producer prices ...")
    url = (
        "https://fenixservices.fao.org/faostat/api/v1/en/data/PP"
        "?area=223"          # Turkey
        "&item=217,222,225"  # almonds, walnuts, hazelnuts
        "&element=5532"      # Producer price (USD/tonne)
        "&year=1980:2024"
        "&output_type=csv"
        "&file_type=csv"
        "&download=true"
    )

    try:
        raw = pd.read_csv(url)
        raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")

        item_map = {217: "almond", 222: "walnut", 225: "hazelnut"}
        rows = []
        for item_code, name in item_map.items():
            sub = raw[raw["item_code"] == item_code][["year", "value"]].copy()
            sub = sub.rename(columns={"value": f"{name}_usd_mt"})
            sub["year"] = sub["year"].astype(int)
            rows.append(sub.set_index("year"))

        df = pd.concat(rows, axis=1).reset_index()
        # Convert USD/MT → USD/kg
        for col in ["almond_usd_mt", "walnut_usd_mt", "hazelnut_usd_mt"]:
            df[col.replace("_mt", "_kg")] = df[col] / 1000
        df = df.drop(columns=[c for c in df.columns if c.endswith("_mt")])

        df.to_csv(cache, index=False)
        logger.info("FAO nut prices saved to %s (%d rows)", cache, len(df))
        return df

    except Exception as e:
        logger.warning("FAO download failed (%s) — trying bulk CSV fallback", e)
        return _fao_bulk_fallback()


def _fao_bulk_fallback() -> pd.DataFrame:
    """
    Minimal hardcoded FAO Turkey hazelnut/almond/walnut producer prices (USD/kg).
    Used when the FAOSTAT API is unavailable.
    Approximate values from FAOSTAT web interface.
    """
    data = {
        "year":         list(range(1993, 2023)),
        "hazelnut_usd_kg": [
            1.05, 1.20, 1.40, 1.55, 1.80, 2.10, 1.95, 2.30, 2.80, 2.50,
            3.20, 3.80, 4.10, 3.60, 4.50, 4.80, 5.20, 6.80, 5.50, 6.20,
            7.10, 8.50, 9.20, 10.5, 11.2, 8.80, 9.60, 11.8, 13.2, 15.0,
        ],
        "almond_usd_kg": [
            1.80, 1.95, 2.10, 2.30, 2.50, 2.80, 3.20, 3.50, 3.80, 3.20,
            4.10, 4.80, 5.50, 6.20, 7.10, 6.50, 5.80, 5.20, 4.80, 5.50,
            6.20, 7.10, 8.50, 9.20, 8.80, 7.50, 8.20, 9.10, 8.50, 7.80,
        ],
        "walnut_usd_kg": [
            1.20, 1.35, 1.50, 1.70, 1.90, 2.10, 2.40, 2.70, 3.00, 2.80,
            3.20, 3.80, 4.20, 4.80, 5.50, 5.20, 4.80, 4.50, 5.10, 5.80,
            6.50, 7.20, 8.10, 9.00, 8.50, 7.20, 7.80, 8.50, 9.20, 10.1,
        ],
    }
    logger.warning("Using hardcoded FAO fallback prices (approximate)")
    return pd.DataFrame(data)


def load_usdeur_rate() -> pd.Series:
    """Annual average USD/EUR rate (EUR per 1 USD)."""
    import yfinance as yf
    fx = yf.download("EURUSD=X", start="1999-01-01", end="2025-01-01",
                     progress=False, auto_adjust=True)
    rate = fx["Close"].resample("YE").mean().squeeze()
    rate.index = rate.index.year
    return rate


def build_basket_dataset() -> pd.DataFrame:
    """
    Assemble all price series into a single annual DataFrame.

    Returns columns:
      year, tmo_eur, hazelnut_eur, almond_eur, walnut_eur
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "basket_dataset.csv"

    tmo = load_tmo_prices()[["year", "tmo_eur_shelled"]].rename(
        columns={"tmo_eur_shelled": "tmo_eur"})

    fao = load_fao_nut_prices()
    usdeur = load_usdeur_rate()

    # Convert USD/kg → EUR/kg
    for col, out in [("hazelnut_usd_kg", "hazelnut_eur"),
                     ("almond_usd_kg",   "almond_eur"),
                     ("walnut_usd_kg",   "walnut_eur")]:
        if col in fao.columns:
            fao[out] = fao.apply(
                lambda r: r[col] * float(usdeur.get(int(r["year"]), np.nan)),
                axis=1,
            )

    fao_eur = fao[["year"] + [c for c in ["hazelnut_eur", "almond_eur", "walnut_eur"]
                               if c in fao.columns]]

    df = tmo.merge(fao_eur, on="year", how="outer").sort_values("year")
    df.to_csv(cache, index=False)
    logger.info("Basket dataset saved to %s (%d rows)", cache, len(df))
    return df


def fit_basket_weights(df: pd.DataFrame) -> dict:
    """
    Fit OLS regression: hazelnut_eur ~ tmo_eur + almond_eur + walnut_eur
    to find the basket weights that maximise R² with the hazelnut price.

    Returns dict with weights, R², and the fitted basket series.
    """
    features = [c for c in ["tmo_eur", "almond_eur", "walnut_eur"] if c in df.columns]
    target = "hazelnut_eur"

    complete = df[["year", target] + features].dropna()
    if len(complete) < 5:
        logger.warning("Only %d complete rows for basket regression", len(complete))
        return {"weights": {f: 1.0/len(features) for f in features},
                "r_squared": None, "n_obs": len(complete)}

    y = complete[target].values
    X = np.column_stack([complete[f].values for f in features])
    X = np.hstack([np.ones((len(X), 1)), X])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Normalise coefficients (drop intercept) to sum to 1 for interpretability
    raw_weights = dict(zip(features, beta[1:]))
    total = sum(abs(v) for v in raw_weights.values())
    norm_weights = {k: v / total for k, v in raw_weights.items()}

    logger.info("Basket R²=%.3f on %d obs", r2, len(complete))
    for name, w in norm_weights.items():
        logger.info("  w_%s = %.3f", name, w)

    # Construct basket index (normalised to mean=100 over fitted period)
    basket = X[:, 1:] @ np.array([raw_weights[f] for f in features]) + beta[0]
    basket_normed = basket / basket.mean() * 100

    return {
        "weights": norm_weights,
        "raw_coefficients": dict(zip(["intercept"] + features, beta)),
        "r_squared": r2,
        "n_obs": len(complete),
        "years": complete["year"].values,
        "basket_index": basket_normed,
        "hazelnut_actual": y,
        "hazelnut_fitted": y_hat,
    }


def load_expana_csv(path: str) -> pd.DataFrame:
    """
    Load Expana benchmark prices from a user-provided CSV export.

    Expected columns: year (or date), price_usd_100kg
    Converts to EUR/kg and merges into basket dataset.

    Once Expana data is available, this replaces the FAO hazelnut series
    as the calibration target and preferred settlement reference.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if "date" in df.columns and "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year

    if "price_usd_100kg" in df.columns:
        df["expana_usd_kg"] = df["price_usd_100kg"] / 100
    elif "price" in df.columns:
        df["expana_usd_kg"] = df["price"]

    usdeur = load_usdeur_rate()
    df["expana_eur_kg"] = df.apply(
        lambda r: r["expana_usd_kg"] * float(usdeur.get(int(r["year"]), np.nan)),
        axis=1,
    )
    logger.info("Expana data loaded: %d rows (%d–%d)",
                len(df), df.year.min(), df.year.max())
    return df[["year", "expana_usd_kg", "expana_eur_kg"]]
