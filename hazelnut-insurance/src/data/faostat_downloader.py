"""
FAOSTAT hazelnut production downloader for Turkey.

Downloads QCL dataset (Crops and Livestock), item 221 (Hazelnuts, in shell),
element 5510 (Production), country Turkey.

No API key required. Uses bulk CSV download as primary method.

Usage:
    python -m src.data.faostat_downloader
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "faostat"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / "turkey_hazelnut_production.csv"

# FAOSTAT bulk download URL for QCL (Crops and livestock products)
FAOSTAT_QCL_URL = (
    "https://fenixservices.fao.org/faostat/static/bulkdownloads/Production_Crops_Livestock_E_All_Data_(Normalized).zip"
)

# Identifiers
# Note: item code 221 is "Almonds, in shell" in current FAOSTAT QCL.
# "Hazelnuts, in shell" is item code 225. Verified against bulk CSV 2025-05.
HAZELNUT_ITEM_CODE = 225        # Hazelnuts, in shell
PRODUCTION_ELEMENT_CODE = 5510  # Production (metric tons)
TURKEY_COUNTRY_CODE = 223       # Türkiye (FAO area code)


def download_and_parse(force: bool = False) -> pd.DataFrame:
    """
    Download FAOSTAT QCL bulk CSV, extract Turkey hazelnut production, and save.
    Returns cleaned DataFrame with columns [year, production_mt].
    """
    if OUT_PATH.exists() and not force:
        logger.info("Cache hit: %s", OUT_PATH)
        return pd.read_csv(OUT_PATH)

    logger.info("Downloading FAOSTAT QCL bulk CSV (~200MB) ...")
    try:
        import zipfile

        resp = requests.get(FAOSTAT_QCL_URL, stream=True, timeout=120)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Find the normalized CSV inside the zip
            csv_names = [n for n in zf.namelist() if n.endswith(".csv") and "Normalized" in n]
            if not csv_names:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            csv_name = csv_names[0]
            logger.info("Reading %s from zip ...", csv_name)
            with zf.open(csv_name) as f:
                # FAOSTAT bulk CSVs are UTF-8; try that first then fall back to latin-1
                raw_bytes = f.read()
                for enc in ("utf-8-sig", "utf-8", "latin-1"):
                    try:
                        df_raw = pd.read_csv(
                            io.BytesIO(raw_bytes), encoding=enc, low_memory=False
                        )
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue

    except Exception as exc:
        logger.warning("Bulk download failed (%s). Trying faostat package ...", exc)
        df_raw = _download_via_package()

    df = _filter_turkey_hazelnut(df_raw)
    df.to_csv(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), OUT_PATH)
    return df


def _download_via_package() -> pd.DataFrame:
    """Fallback: use the faostat Python package."""
    try:
        import faostat
    except ImportError:
        raise ImportError("pip install faostat")

    df = faostat.get_data_df(
        "QCL",
        pars={
            "area": [str(TURKEY_COUNTRY_CODE)],
            "item": [str(HAZELNUT_ITEM_CODE)],
            "element": [str(PRODUCTION_ELEMENT_CODE)],
        },
    )
    return df


def _filter_turkey_hazelnut(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filter FAOSTAT bulk CSV to Turkey hazelnut production.
    Handles the normalized format (one row per area/item/element/year).

    The FAOSTAT bulk CSV has both 'Area Code' (FAO code, Turkey=223) and
    'Area Code (M49)' (UN M49 code, Turkey=792). We use the string Area column
    as the primary filter to avoid mis-matching codes across column variants.
    """
    # Use string name columns as primary filter (more robust than code lookup)
    area_name_col = next(
        (c for c in df_raw.columns if c.lower() == "area"), None
    ) or next(
        (c for c in df_raw.columns if "area" in c.lower() and "code" not in c.lower()), None
    )
    item_name_col = next(
        (c for c in df_raw.columns if c.lower() == "item"), None
    ) or next(
        (c for c in df_raw.columns if "item" in c.lower() and "code" not in c.lower()), None
    )
    elem_name_col = next(
        (c for c in df_raw.columns if c.lower() == "element"), None
    ) or next(
        (c for c in df_raw.columns if "element" in c.lower() and "code" not in c.lower()), None
    )
    year_col = next((c for c in df_raw.columns if c.lower() == "year"), None)
    value_col = next((c for c in df_raw.columns if c.lower() == "value"), None)
    unit_col = next((c for c in df_raw.columns if c.lower() == "unit"), None)

    if not all([area_name_col, item_name_col, elem_name_col, year_col, value_col]):
        logger.warning("Normalized format columns not found; trying wide format")
        return _parse_wide_format(df_raw)

    # Match Turkey under its several possible name variants in FAOSTAT:
    # "Turkey" (pre-2022), "Türkiye" (post-2022 rename), and the latin-1
    # mojibake "TÃ¼rkiye" that appears when a UTF-8 file is misread.
    turkey_pattern = r"Turkey|T[uüÃ][¼]?rkiye"
    mask = (
        df_raw[area_name_col].str.contains(turkey_pattern, na=False, case=False, regex=True)
        & df_raw[item_name_col].str.contains("Hazelnut", na=False, case=False)
        & df_raw[elem_name_col].str.contains("Production", na=False, case=False)
    )

    if mask.sum() == 0:
        logger.warning("String filter found 0 rows; falling back to wide format")
        return _parse_wide_format(df_raw)

    df = df_raw[mask][[year_col, value_col]].copy()
    df.columns = ["year", "production_mt"]
    df["year"] = df["year"].astype(int)
    df["production_mt"] = pd.to_numeric(df["production_mt"], errors="coerce")
    df = df.dropna(subset=["production_mt"]).sort_values("year").reset_index(drop=True)

    # Log the unit so we can catch any future unit changes
    if unit_col is not None:
        units = df_raw[mask][unit_col].unique()
        logger.info("FAOSTAT production unit: %s", units)

    return df


def _parse_wide_format(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Handle wide-format FAOSTAT export (years as columns)."""
    area_col = next(c for c in df_raw.columns if "area" in c.lower())
    item_col = next(c for c in df_raw.columns if "item" in c.lower())
    elem_col = next(c for c in df_raw.columns if "element" in c.lower())

    mask = (
        df_raw[area_col].str.contains("Turkey|Türkiye", na=False, case=False)
        & df_raw[item_col].str.contains("Hazelnut", na=False, case=False)
        & df_raw[elem_col].str.contains("Production", na=False, case=False)
    )
    row = df_raw[mask]
    year_cols = [c for c in df_raw.columns if c.startswith("Y") and c[1:].isdigit()]
    values = row[year_cols].iloc[0]
    df = pd.DataFrame({
        "year": [int(c[1:]) for c in year_cols],
        "production_mt": pd.to_numeric(values.values, errors="coerce"),
    }).dropna(subset=["production_mt"]).sort_values("year").reset_index(drop=True)
    return df


def load() -> pd.DataFrame:
    """Load cached FAOSTAT data, applying production overrides from config."""
    import yaml

    df = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else download_and_parse()

    config_path = Path(__file__).parents[2] / "config" / "trigger_params.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    overrides = cfg.get("production", {}).get("production_overrides", {})
    for year, value in overrides.items():
        year = int(year)
        if year in df["year"].values:
            df.loc[df["year"] == year, "production_mt"] = value
        else:
            df = pd.concat(
                [df, pd.DataFrame({"year": [year], "production_mt": [value]})],
                ignore_index=True,
            ).sort_values("year").reset_index(drop=True)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = download_and_parse(force=False)
    print(df.tail(20).to_string(index=False))
    print(f"\nTotal rows: {len(df)}")
    print(f"Year range: {df['year'].min()} – {df['year'].max()}")
