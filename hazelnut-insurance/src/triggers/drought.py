"""
Drought trigger: production-weighted SPEI03 for August (captures JJA window).

SPEI < -1.5: severe drought → partial payout
SPEI < -2.0: extreme drought → max payout (capped at 30% of notional)

Note: The plan uses August SPEI03 (captures June-July-August). This is
appropriate for nut-fill stress. An alternative worth testing is June SPEI03
(captures March-April-May) which would capture soil moisture deficit during
vegetative growth. Both are computed here; August is the default.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from src.data.spei_downloader import load_turkey_region
from src.utils.geo import load_provinces, province_weights

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "trigger_params.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["drought"]


def _payout_from_bands(spei: float, bands: list) -> float:
    for lo, hi, pay_lo, pay_hi in bands:
        lo = float("-inf") if str(lo) in ("-.inf", "-inf") else float(lo)
        hi = float("inf") if str(hi) in (".inf", "inf") else float(hi)
        if lo <= spei < hi:
            if lo == hi or pay_lo == pay_hi:
                return pay_lo
            frac = (spei - lo) / (hi - lo)
            return pay_lo + frac * (pay_hi - pay_lo)
    return 0.0


def _get_spei_for_month(ds: xr.Dataset, year: int, month: int) -> xr.DataArray:
    """Extract SPEI field for a specific year-month from the global dataset."""
    spei_var = next(
        (v for v in ds.data_vars if "spei" in v.lower()),
        list(ds.data_vars)[0],
    )
    da = ds[spei_var]
    time_dim = "time" if "time" in da.dims else da.dims[0]

    # Select the target month
    da_month = da.sel({time_dim: da[time_dim].dt.month == month})
    da_year = da_month.sel({time_dim: da_month[time_dim].dt.year == year})

    if da_year.sizes[time_dim] == 0:
        raise ValueError(f"No SPEI data found for {year}-{month:02d}")

    return da_year.isel({time_dim: 0})


def compute_spei(year: int, reference_month: int | None = None) -> float:
    """
    Compute production-weighted SPEI03 for the given year and reference month.
    reference_month defaults to config value (August = 8).

    Returns
    -------
    float
        Production-weighted mean SPEI03 across the 7-province basket.
        More negative = drier.
    """
    cfg = _load_cfg()
    if reference_month is None:
        reference_month = cfg.get("reference_month", 8)

    ds = load_turkey_region()
    da = _get_spei_for_month(ds, year, reference_month)

    provinces = load_provinces()
    weights = province_weights()
    total_weight = sum(weights.values())

    lat_dim = "lat" if "lat" in da.dims else "latitude"
    lon_dim = "lon" if "lon" in da.dims else "longitude"

    weighted_spei = 0.0
    for p in provinces:
        val = float(
            da.sel(
                {lat_dim: p["lat"], lon_dim: p["lon"]},
                method="nearest",
            ).values
        )
        if np.isnan(val):
            logger.warning("SPEI NaN for %s in %d-%02d — skipping province", p["name"], year, reference_month)
            continue
        weighted_spei += (weights[p["name"]] / total_weight) * val

    return weighted_spei


def compute_payout(spei: float) -> float:
    """Return payout as fraction of notional (0.0–1.0) given SPEI value."""
    cfg = _load_cfg()
    return _payout_from_bands(spei, cfg["payout_bands"])


def metric_series(reference_month: int | None = None) -> pd.DataFrame:
    """
    Return the full historical SPEI series for all available years.
    Used by the distributional pricing engine.

    Returns DataFrame with columns [year, spei].
    """
    ds = load_turkey_region()
    cfg = _load_cfg()
    if reference_month is None:
        reference_month = cfg.get("reference_month", 8)

    time_dim = "time"
    spei_var = next(
        (v for v in ds.data_vars if "spei" in v.lower()),
        list(ds.data_vars)[0],
    )
    da = ds[spei_var]
    years_available = sorted(set(int(y) for y in da[time_dim].dt.year.values))
    records = []
    for year in years_available:
        try:
            spei = compute_spei(year, reference_month)
            records.append({"year": year, "spei": spei})
        except Exception:
            pass
    return pd.DataFrame(records)


def backtest(years: list[int], reference_month: int | None = None) -> pd.DataFrame:
    """Compute SPEI and payout for each year. Returns DataFrame."""
    records = []
    for year in years:
        try:
            spei = compute_spei(year, reference_month)
            payout = compute_payout(spei)
            records.append({"year": year, "spei": spei, "payout": payout})
            logger.info("Drought %d: SPEI=%.3f, payout=%.3f", year, spei, payout)
        except Exception as exc:
            logger.warning("Drought %d: %s — skipping", year, exc)
    return pd.DataFrame(records)
