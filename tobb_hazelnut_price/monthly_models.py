"""
Monthly regressions on Giresun exchange USD price.

y  = log USD return of Giresun inshell avg price (more consistent
     measurement point than multi-exchange TOBB VWAP)

M1: ret_usd ~ shortfall_monthly + PC1 + PC3
    Combines crop-year production shortfall with commodity/FX factors.
    Shortfall is the trailing-5yr deviation carried monthly via crop-year
    calendar (Aug-Dec of year Y and Jan-Jul of Y+1 share year Y's shortfall).

M2: M1 + tmo_share_lag1
    Adds lagged TMO government-buyer share as a price-floor signal.
    TMO buys large volumes in weak markets; high share may predict recovery.

M3: deseasonalized M1
    Subtracts calendar-month mean return before fitting M1 specification.
    Tests whether seasonality drives the shortfall/factor relationship.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features import (ols, print_model, load_giresun_monthly,
                      build_shortfall_monthly, fetch_yfinance_monthly, fit_pca)

PROJ_ROOT    = Path(__file__).parent.parent
N_COMPONENTS = 8


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    """
    Merge Giresun monthly USD returns with crop-year shortfall and PC scores.
    Returns a single DataFrame ready for regression.
    """
    gir = load_giresun_monthly()
    sf  = build_shortfall_monthly()

    gir['ret_usd']       = np.log(gir['avg_usd_kg']).diff()
    gir['tmo_share_lag1'] = gir['tmo_share'].shift(1)
    gir = gir.dropna(subset=['ret_usd']).reset_index(drop=True)

    feats, ret_cols = fetch_yfinance_monthly()
    feats = feats.dropna(subset=ret_cols).reset_index(drop=True)
    pc_df, _        = fit_pca(feats, ret_cols, n_components=N_COMPONENTS)

    df = (gir.merge(sf,    on='month', how='left')
             .merge(pc_df, on='month', how='inner'))
    df = df.dropna(subset=['ret_usd', 'shortfall', 'PC1', 'PC3']).reset_index(drop=True)

    # Deseasonalized return: subtract each calendar-month's mean
    df['cal_month']  = df['month'].str[5:7].astype(int)
    month_means      = df.groupby('cal_month')['ret_usd'].transform('mean')
    df['ret_usd_ds'] = df['ret_usd'] - month_means

    return df


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = build_dataset()

    ones = np.ones(len(df))

    # M1 — shortfall + PC1 + PC3
    m1 = ols(df['ret_usd'].values,
             np.c_[ones,
                   df['shortfall'].values,
                   df['PC1'].values,
                   df['PC3'].values],
             ['intercept', 'shortfall', 'PC1', 'PC3'],
             'M1 — shortfall + PC1 + PC3')

    # M2 — add lagged TMO share; drop rows where lag is NaN
    d2   = df.dropna(subset=['tmo_share_lag1'])
    m2 = ols(d2['ret_usd'].values,
             np.c_[np.ones(len(d2)),
                   d2['shortfall'].values,
                   d2['PC1'].values,
                   d2['PC3'].values,
                   d2['tmo_share_lag1'].values],
             ['intercept', 'shortfall', 'PC1', 'PC3', 'tmo_share_lag1'],
             'M2 — M1 + TMO share lag')

    # M3 — M1 specification on deseasonalized returns
    m3 = ols(df['ret_usd_ds'].values,
             np.c_[ones,
                   df['shortfall'].values,
                   df['PC1'].values,
                   df['PC3'].values],
             ['intercept', 'shortfall', 'PC1', 'PC3'],
             'M3 — deseasonalized, shortfall + PC1 + PC3')

    return {'M1': m1, 'M2': m2, 'M3': m3}


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')

    df     = build_dataset()
    models = run(df)

    print(f"Monthly dataset: n={len(df)}, {df['month'].iloc[0]} – {df['month'].iloc[-1]}")
    print(f"Giresun USD NaN filled: {df['avg_usd_kg'].isna().sum()} rows")
    for m in models.values():
        print_model(m)
