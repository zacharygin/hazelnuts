"""
Compute SPEI03 for the Turkish hazelnut region from ERA5 monthly data.

Replaces the CSIC SPEIbase download (which is unreliable) by computing
SPEI03 directly from ERA5 monthly total precipitation and 2m temperature.

Method:
  1. Download ERA5 monthly means: total_precipitation + 2m_temperature
     for the Turkey hazelnut bbox, 1950–2024.
  2. Compute Hargreaves PET from temperature (no radiation data needed).
  3. Compute D = P - PET (climatic water balance) monthly.
  4. Compute 3-month rolling D (D03) for each calendar month.
  5. Fit a log-logistic distribution to D03 and transform to SPEI03.
  6. Cache as a simple CSV: [year, month, spei03_weighted].

References:
  Vicente-Serrano et al. (2010) "A Multiscalar Drought Index Sensitive to
  Global Warming." J. Climate 23(7): 1696–1718.
  Hargreaves & Samani (1985) "Reference Crop Evapotranspiration from Ambient
  Air Temperature." Applied Engineering in Agriculture 1(2): 96–99.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR  = Path(__file__).parents[2] / "data" / "raw" / "era5_monthly"
CACHE    = Path(__file__).parents[2] / "data" / "raw" / "spei" / "spei03_era5.csv"
BBOX     = [42.0, 29.0, 40.0, 41.0]   # [N, W, S, E]


def _download_monthly_era5(start_year: int = 1950, end_year: int = 2024,
                            force: bool = False) -> Path:
    """Download ERA5 monthly total precip + 2m_temp for Turkey bbox."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"era5_monthly_{start_year}_{end_year}.nc"
    if out.exists() and not force:
        logger.info("Cache hit: %s", out)
        return out

    import cdsapi
    c = cdsapi.Client()
    years  = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    logger.info("Downloading ERA5 monthly means %d–%d ...", start_year, end_year)
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ["total_precipitation", "2m_temperature"],
            "year": years,
            "month": months,
            "time": "00:00",
            "area": BBOX,
            "format": "netcdf",
            "grid": [0.25, 0.25],
        },
        str(out),
    )
    logger.info("Saved: %s", out)
    return out


def _hargreaves_pet(t_mean_c: np.ndarray, t_max_c: np.ndarray,
                    t_min_c: np.ndarray, lat_deg: float,
                    doy: np.ndarray) -> np.ndarray:
    """
    Hargreaves-Samani PET (mm/month).
    Uses only temperature — no radiation or humidity data needed.
    """
    # Extraterrestrial radiation (Ra) from latitude and day-of-year
    lat_rad = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )  # MJ/m²/day
    # Hargreaves: ET0 = 0.0023 * Ra/λ * (T+17.8) * sqrt(Tmax-Tmin)
    lam = 2.45  # MJ/kg latent heat of vaporisation
    td = np.maximum(t_max_c - t_min_c, 0.0)
    et0_mm_day = 0.0023 * (Ra / lam) * (t_mean_c + 17.8) * np.sqrt(td)
    # Convert to monthly (approximate: multiply by days in month)
    return et0_mm_day  # returned as mm/day; caller multiplies by days


def _fit_log_logistic_lmom(x: np.ndarray) -> tuple[float, float, float]:
    """
    Fit 3-parameter log-logistic via L-moments (SPEI standard method).
    Returns (gamma, alpha, beta) — location, scale, shape in LL3.
    """
    x_sorted = np.sort(x)
    n = len(x_sorted)
    # Probability-weighted moments
    b0 = np.mean(x_sorted)
    b1 = np.mean(x_sorted * np.arange(n) / (n - 1))
    b2 = np.mean(x_sorted * np.arange(n) * np.maximum(np.arange(n) - 1, 0)
                 / ((n - 1) * (n - 2)))
    # L-moments
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    t3 = l3 / l2 if l2 != 0 else 0
    # LL3 parameter estimation from L-moments
    beta = (2 * t3) / (1 + 3 * t3) if abs(1 + 3 * t3) > 1e-10 else 0
    alpha = l2 * (1 - 2 * beta) / (
        np.math.gamma(1 + beta) * np.math.gamma(1 - beta)
    ) if abs(beta) < 0.9 else l2
    gamma = l1 - alpha * (
        np.math.gamma(1 + beta) * np.math.gamma(1 - beta) - 1
    ) / (1 - 2 * beta) if abs(1 - 2 * beta) > 1e-10 else l1
    return float(gamma), float(alpha), float(beta)


def _ll3_cdf(x: np.ndarray, gamma: float, alpha: float, beta: float) -> np.ndarray:
    """CDF of 3-parameter log-logistic."""
    z = (x - gamma) / alpha
    z = np.maximum(z, 1e-10)
    return 1 / (1 + (z ** (-1 / beta))) if beta != 0 else 1 / (1 + np.exp(-z))


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """Standard Normal quantile (inverse CDF)."""
    from scipy.stats import norm
    return norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))


def _open_monthly_dataset() -> "xr.Dataset":
    """
    Open the ERA5 monthly dataset. CDS returns a zip with separate files for
    temperature and precipitation — merge them into one Dataset.
    """
    import xarray as xr, zipfile
    from pathlib import Path

    raw = RAW_DIR
    nc_file = next(raw.glob("era5_monthly_*.nc"), None)
    if nc_file is None:
        raise FileNotFoundError("ERA5 monthly file not found. Run _download_monthly_era5().")

    # CDS sometimes returns a zip disguised as .nc
    with open(nc_file, "rb") as f:
        is_zip = f.read(2) == b"PK"

    if is_zip:
        with zipfile.ZipFile(nc_file) as z:
            z.extractall(raw)

    # Find extracted NetCDF files
    nc_files = [p for p in raw.glob("*.nc") if "era5_monthly" not in p.name]
    if not nc_files:
        raise FileNotFoundError(f"No extracted NetCDF files found in {raw}")

    # Floor each file's time coordinate to month-start before merging so that
    # temp (00:00) and precip (06:00) align on the same time index.
    aligned = []
    for p in nc_files:
        d = xr.open_dataset(p)
        if "valid_time" in d.dims:
            d = d.rename({"valid_time": "time"})
        # Floor to first of month
        import pandas as pd
        month_starts = pd.to_datetime(d["time"].values).to_period("M").to_timestamp()
        d = d.assign_coords(time=month_starts.values)
        aligned.append(d)

    ds = xr.merge(aligned, join="outer")
    # Drop any duplicate time steps that survived (shouldn't happen after floor)
    _, idx = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=idx)
    return ds


def compute_spei03_from_era5(start_year: int = 1950, end_year: int = 2024,
                              reference_month: int = 8,
                              force_download: bool = False) -> pd.DataFrame:
    """
    Compute production-weighted August SPEI03 for the Turkey hazelnut region.

    Returns DataFrame with columns [year, spei03].
    """
    import xarray as xr
    from src.utils.geo import load_provinces, province_weights

    if not any(RAW_DIR.glob("era5_monthly_*.nc")):
        _download_monthly_era5(start_year, end_year, force=force_download)
    ds = _open_monthly_dataset()

    # Variable names vary between CDS versions
    precip_var = next((v for v in ds.data_vars if "precip" in v.lower() or v == "tp"), None)
    temp_var   = next((v for v in ds.data_vars if "2m" in v.lower() or v == "t2m"), None)
    if precip_var is None or temp_var is None:
        raise ValueError(f"Could not find precip/temp vars in {list(ds.data_vars)}")

    provinces = load_provinces()
    weights   = province_weights()
    total_w   = sum(weights.values())

    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"

    records = []
    for prov in provinces:
        w = weights[prov["name"]] / total_w
        p_da = ds[precip_var].sel({lat_dim: prov["lat"], lon_dim: prov["lon"]},
                                   method="nearest")
        t_da = ds[temp_var].sel({lat_dim: prov["lat"], lon_dim: prov["lon"]},
                                 method="nearest")

        # ERA5 monthly tp is mean daily rate (m/day); convert to monthly total mm
        times = pd.to_datetime(ds["time"].values)
        days  = times.days_in_month.values.astype(float)
        p_mm  = p_da.values * days * 1000  # m/day × days × 1000 → mm/month
        t_c   = t_da.values - 273.15       # K → °C

        df_p  = pd.DataFrame({"time": times, "precip_mm": p_mm,
                               "temp_c": t_c, "province": prov["name"], "weight": w})
        records.append(df_p)

    df = pd.concat(records, ignore_index=True)
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # Production-weight precip and temp across provinces
    agg = df.groupby(["year", "month"]).apply(
        lambda g: pd.Series({
            "precip_mm": np.dot(g["weight"], g["precip_mm"]),
            "temp_c":    np.dot(g["weight"], g["temp_c"]),
        })
    ).reset_index()

    # Approximate PET via Hargreaves (simplified: use temp only, no Tmax/Tmin split)
    # ERA5 monthly means don't have Tmax/Tmin directly — use std proxy: Trange ≈ 10°C
    agg["pet_mm"] = 0.0023 * (agg["temp_c"] + 17.8) * np.sqrt(10.0) * 30  # mm/month approx

    # Climatic water balance D = P - PET
    agg["D"] = agg["precip_mm"] - agg["pet_mm"]

    # 3-month rolling sum of D (SPEI03)
    agg = agg.sort_values(["year", "month"])
    agg["D03"] = agg["D"].rolling(3, min_periods=3).sum()

    # Compute SPEI03 for reference month using log-logistic fit
    ref_data = agg[agg["month"] == reference_month].dropna(subset=["D03"])
    d03_vals = ref_data["D03"].values

    try:
        gamma, alpha, beta = _fit_log_logistic_lmom(d03_vals)
        p = _ll3_cdf(d03_vals, gamma, alpha, beta)
        spei_vals = _normal_ppf(p)
    except Exception as e:
        logger.warning("Log-logistic fit failed (%s), falling back to z-score", e)
        spei_vals = (d03_vals - d03_vals.mean()) / d03_vals.std()

    result = ref_data[["year"]].copy()
    result["spei03"] = spei_vals
    result = result.sort_values("year").reset_index(drop=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(CACHE, index=False)
    logger.info("SPEI03 computed for %d years, saved to %s", len(result), CACHE)
    return result


def load_turkey_region() -> "xr.Dataset":
    """
    Compatibility shim for drought.py: returns an xarray Dataset with a
    'spei03' variable and a time coordinate, mimicking the SPEIbase format.
    Computes from ERA5 monthly data if not cached.
    """
    import xarray as xr

    if not CACHE.exists():
        compute_spei03_from_era5()

    df = pd.read_csv(CACHE)
    times = pd.to_datetime(df["year"].astype(str) + "-08-01")
    ds = xr.Dataset(
        {"spei03": xr.DataArray(df["spei03"].values, coords={"time": times}, dims=["time"])},
        coords={"time": times, "lat": 41.0, "lon": 35.0},
    )
    return ds
