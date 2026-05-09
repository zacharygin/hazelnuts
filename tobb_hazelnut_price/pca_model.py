"""
Monthly PCA factor regression for hazelnut USD price returns.

y  = log USD return of TOBB all-exchange monthly VWAP
X  = PCA factors from 16 commodity/equity/FX features (yfinance)

Key results (n≈207, 2005-2026):
  PC1  commodity/risk-on cycle  R²=0.227  p<0.001
  PC3  USD strength (DXY)       R²=0.112  p<0.001
  PC1+PC3 combined              R²=0.338  adj-R²=0.332

Caveat: R² collapses to ~0.016 on 2005-2020 alone — signal is
concentrated in the post-2021 TRY-collapse period.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features import (ols, print_model, load_tobb_monthly,
                      fetch_yfinance_monthly, fit_pca)

PROJ_ROOT = Path(__file__).parent.parent
RAW       = PROJ_ROOT / 'data' / 'raw'

N_COMPONENTS = 8


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    """TOBB monthly VWAP in USD merged with PCA scores from 16 yfinance features."""
    tobb = load_tobb_monthly()

    # Convert TL → USD via monthly TRY=X (TRY=X = USD per TRY → invert for TRY per USD)
    import yfinance as yf
    raw_fx = yf.download('TRY=X', start='2000-01-01', auto_adjust=True, progress=False)
    close_fx = raw_fx['Close']
    if hasattr(close_fx, 'squeeze'):
        close_fx = close_fx.squeeze()
    fx = close_fx.resample('ME').mean()
    fx.index = fx.index.strftime('%Y-%m')
    fx_df = fx.reset_index()
    fx_df.columns = ['month', 'tryusd']
    # TRY=X gives TRY per USD directly (e.g. 45 in 2026); no inversion needed
    tobb = tobb.merge(fx_df, on='month', how='left')
    tobb['vwap_usd'] = tobb['vwap_try'] / tobb['tryusd']
    tobb['ret_usd']  = np.log(tobb['vwap_usd']).diff()

    # Null out cross-gap returns: only keep return if prev month was consecutive
    tobb['prev_month'] = (pd.to_datetime(tobb['month']) - pd.DateOffset(months=1)
                          ).dt.strftime('%Y-%m')
    tobb.loc[tobb['prev_month'] != tobb['month'].shift(1), 'ret_usd'] = np.nan
    tobb = tobb.dropna(subset=['ret_usd']).reset_index(drop=True)

    feats, ret_cols = fetch_yfinance_monthly()

    # Drop rows with any NaN in ret_cols (e.g. first month of shorter-history tickers)
    feats = feats.dropna(subset=ret_cols).reset_index(drop=True)

    pc_df, pca = fit_pca(feats, ret_cols, n_components=N_COMPONENTS)

    df = tobb.merge(pc_df, on='month', how='inner')
    return df, ret_cols, pca


def run_pc_regressions(df: pd.DataFrame) -> dict:
    """Bivariate OLS for each PC; combined PC1+PC3."""
    y    = df['ret_usd'].values
    ones = np.ones(len(y))
    results = {}

    for i in range(N_COMPONENTS):
        col = f'PC{i+1}'
        if col not in df.columns:
            continue
        m = ols(y, np.c_[ones, df[col].values],
                ['intercept', col], f'{col} only')
        results[col] = m

    m13 = ols(y, np.c_[ones, df['PC1'].values, df['PC3'].values],
              ['intercept', 'PC1', 'PC3'], 'PC1 + PC3')
    results['PC1+PC3'] = m13
    return results


def run() -> dict:
    df, ret_cols, pca = build_dataset()
    models = run_pc_regressions(df)
    return {'df': df, 'pca': pca, 'ret_cols': ret_cols, 'models': models}


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')

    out = run()
    df, pca = out['df'], out['pca']

    print(f"n={len(df)}  ({df['month'].iloc[0]} – {df['month'].iloc[-1]})")

    print('\n--- Variance explained ---')
    for i, v in enumerate(pca.explained_variance_ratio_ * 100):
        print(f'  PC{i+1}: {v:.1f}%')

    print('\n--- PC regressions ---')
    rows = []
    for label, m in out['models'].items():
        rows.append({'model': label, 'n': m['n'], 'R2': round(m['r2'], 3),
                     'adj-R2': round(m['r2_adj'], 3), 'AIC': round(m['aic'], 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n--- Best model: PC1 + PC3 ---')
    print_model(out['models']['PC1+PC3'])
