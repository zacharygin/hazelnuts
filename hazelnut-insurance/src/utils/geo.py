"""Helpers for geographic weighting and nearest-grid-cell extraction."""
from __future__ import annotations

import yaml
import numpy as np
import xarray as xr
from pathlib import Path

_CONFIG_DIR = Path(__file__).parents[2] / "config"


def load_provinces() -> list[dict]:
    with open(_CONFIG_DIR / "locations.yaml") as f:
        return yaml.safe_load(f)["provinces"]


def province_weights() -> dict[str, float]:
    return {p["name"]: p["weight"] for p in load_provinces()}


def extract_province_series(ds: xr.Dataset, var: str) -> dict[str, xr.DataArray]:
    """
    For each province, extract the nearest ERA5 grid cell time series.

    Handles ERA5 new API format where time dim is 'valid_time' instead of 'time'.
    Returns dict mapping province name → DataArray with a 'valid_time' or 'time' dim.
    """
    provinces = load_provinces()
    # Normalise time dimension name: rename valid_time → time if needed
    if "valid_time" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    result = {}
    for p in provinces:
        cell = ds[var].sel(
            latitude=p["lat"],
            longitude=p["lon"],
            method="nearest",
        )
        result[p["name"]] = cell
    return result


def production_weighted_mean(series_by_province: dict[str, np.ndarray]) -> np.ndarray:
    """
    Given a dict of province → 1-D array (same length), return the
    production-weighted mean across provinces.
    """
    weights = province_weights()
    total_weight = sum(weights[k] for k in series_by_province)
    out = None
    for name, arr in series_by_province.items():
        w = weights[name] / total_weight
        out = w * arr if out is None else out + w * arr
    return out
