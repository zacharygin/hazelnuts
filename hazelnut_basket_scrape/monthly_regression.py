"""
Monthly hazelnut price regression — no production data.

y  = monthly log return of TOBB VWAP in USD/kg
X  = monthly log returns of commodity futures, equities, FX signals

Covers ~2005-2026 with a known gap in 2017-2022 (TOBB data missing).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).parent
PROJ_ROOT = ROOT.parent

YFINANCE_TICKERS = {
    'cocoa_fut':  'CC=F',
    'corn_fut':   'ZC=F',
    'soy_fut':    'ZS=F',
    'wheat_fut':  'ZW=F',
    'sugar_fut':  'SB=F',
    'coffee_fut': 'KC=F',
    'gold_fut':   'GC=F',
    'oil_fut':    'CL=F',
    'dxy':        'DX-Y.NYB',
    'spy':        'SPY',
    'vix':        '^VIX',
    'tryusd':     'TRY=X',
    'nestle':     'NESN.SW',
    'barry':      'BARN.SW',
    'hershey':    'HSY',
    'mdlz':       'MDLZ',
    'bunge':      'BG',
    'tur':        'TUR',
}


# ---------------------------------------------------------------------------
# TOBB monthly price series
# ---------------------------------------------------------------------------

def build_monthly_tobb() -> pd.DataFrame:
    tobb = pd.read_csv(ROOT / 'data/processed/hazelnut_combined.csv', parse_dates=['date'])
    tobb = tobb[tobb['source'] == 'tobb'].copy()
    tobb['ym'] = tobb['date'].dt.to_period('M')

    def vwap(g):
        mask = g['volume_kg'].notna() & (g['volume_kg'] > 0)
        if mask.sum() > 0:
            return np.average(g.loc[mask, 'avg_price_tlkg'], weights=g.loc[mask, 'volume_kg'])
        return g['avg_price_tlkg'].mean()

    monthly = (
        tobb.groupby('ym')
        .apply(lambda g: pd.Series({'vwap_try': vwap(g), 'n_days': g['date'].nunique()}),
               include_groups=False)
        .reset_index()
    )
    monthly['month'] = monthly['ym'].dt.strftime('%Y-%m')
    return monthly[['month', 'vwap_try', 'n_days']].sort_values('month').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Monthly features
# ---------------------------------------------------------------------------

def fetch_monthly_features(start: str = '2004-01-01') -> pd.DataFrame:
    tickers = list(YFINANCE_TICKERS.values())
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    daily = raw['Close'].copy()
    daily.columns.name = None
    inv = {v: k for k, v in YFINANCE_TICKERS.items()}
    daily = daily.rename(columns=inv)

    # Monthly average
    daily.index = pd.to_datetime(daily.index)
    monthly = daily.resample('ME').mean()
    monthly.index = monthly.index.strftime('%Y-%m')
    monthly.index.name = 'month'
    return monthly.reset_index()


def load_existing_monthly() -> pd.DataFrame:
    df = pd.read_csv(PROJ_ROOT / 'data/raw/basket/equity_basket_monthly.csv')
    df = df.rename(columns={'cocoa_usd': 'cocoa_old', 'dba_usd': 'dba'})
    return df


# ---------------------------------------------------------------------------
# Build regression dataset
# ---------------------------------------------------------------------------

def build_monthly_dataset() -> pd.DataFrame:
    tobb  = build_monthly_tobb()
    feats = fetch_monthly_features()

    # merge on month string
    df = tobb.merge(feats, on='month', how='inner')

    # USD price: need TRY/USD monthly rate
    # tryusd column = TRY per USD (from TRY=X = USD per TRY in yfinance)
    # TRY=X gives USD per TRY → invert for TRY per USD
    df['tryusd_rate'] = 1.0 / df['tryusd']          # TRY per USD
    df['vwap_usd']    = df['vwap_try'] / df['tryusd_rate']   # USD per kg

    df = df.sort_values('month').reset_index(drop=True)

    # Log returns (month-over-month)
    price_cols = ['vwap_try', 'vwap_usd'] + [c for c in feats.columns if c != 'month']
    for col in price_cols:
        if col in df.columns:
            df[f'ret_{col}'] = np.log(df[col]).diff()

    # Drop first row (NaN from diff) and rows where price return is NaN
    df = df.dropna(subset=['ret_vwap_usd']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Regularized regression
# ---------------------------------------------------------------------------

def run_lasso_ridge_monthly(df: pd.DataFrame,
                             exclude: tuple = ('ret_vwap_try', 'ret_tryusd', 'ret_vwap_usd'),
                             min_coverage: float = 0.7,
                             cv: int = 5) -> dict:
    # feature columns: all ret_ columns except y and TL price
    feat_cols = [c for c in df.columns
                 if c.startswith('ret_') and c not in exclude]

    # drop features with too many NaN
    n_total = len(df)
    feat_cols = [c for c in feat_cols
                 if df[c].notna().mean() >= min_coverage]

    sub = df[['month', 'ret_vwap_usd'] + feat_cols].dropna()
    y     = sub['ret_vwap_usd'].values
    X_raw = sub[feat_cols].values

    print(f'n={len(sub)} months, {len(feat_cols)} features')
    print(f'Date range: {sub["month"].iloc[0]} – {sub["month"].iloc[-1]}')

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Lasso for selection
    lasso = LassoCV(cv=cv, max_iter=20000, random_state=42)
    lasso.fit(X, y)
    survivors = [feat_cols[i] for i, c in enumerate(lasso.coef_) if abs(c) > 1e-8]

    print(f'\nLasso alpha={lasso.alpha_:.4f}, survivors ({len(survivors)}): {survivors}')

    if not survivors:
        return {'survivors': [], 'note': 'Lasso zeroed all features'}

    X_surv = sub[survivors].values
    scaler2 = StandardScaler()
    X_s = scaler2.fit_transform(X_surv)

    ridge = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100), cv=cv, scoring='r2')
    ridge.fit(X_s, y)

    yhat   = ridge.predict(X_s)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_is  = 1 - ss_res / ss_tot
    n, k   = len(y), len(survivors) + 1
    r2_adj = 1 - (1 - r2_is) * (n - 1) / (n - k)

    coefs = pd.Series(ridge.coef_, index=survivors).sort_values(key=abs, ascending=False)

    return {
        'sub':          sub,
        'survivors':    survivors,
        'coefs':        coefs,
        'lasso_alpha':  lasso.alpha_,
        'ridge_alpha':  ridge.alpha_,
        'n':            len(y),
        'r2_insample':  r2_is,
        'r2_adj':       r2_adj,
        'r2_cv':        ridge.best_score_,
        'feat_cols':    feat_cols,
        'all_lasso_coefs': pd.Series(lasso.coef_, index=feat_cols).sort_values(key=abs, ascending=False),
    }


def run_ridge_monthly(df: pd.DataFrame,
                      exclude: tuple = ('ret_vwap_try', 'ret_tryusd', 'ret_vwap_usd'),
                      min_coverage: float = 0.7,
                      cv: int = 5) -> dict:
    feat_cols = [c for c in df.columns
                 if c.startswith('ret_') and c not in exclude]
    feat_cols = [c for c in feat_cols if df[c].notna().mean() >= min_coverage]

    sub = df[['month', 'ret_vwap_usd'] + feat_cols].dropna()
    y     = sub['ret_vwap_usd'].values
    X_raw = sub[feat_cols].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    ridge = RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000), cv=cv, scoring='r2')
    ridge.fit(X, y)

    yhat   = ridge.predict(X)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_is  = 1 - ss_res / ss_tot

    coefs = pd.Series(ridge.coef_, index=feat_cols).sort_values(key=abs, ascending=False)

    return {
        'sub':         sub,
        'coefs':       coefs,
        'alpha':       ridge.alpha_,
        'n':           len(y),
        'r2_insample': r2_is,
        'r2_cv':       ridge.best_score_,
        'feat_cols':   feat_cols,
    }


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')

    print('Building monthly dataset...')
    df = build_monthly_dataset()
    print(f'Total months with TOBB + features: {len(df)}')

    print('\n=== Ridge (all features) ===')
    ridge_res = run_ridge_monthly(df)
    print(f'n={ridge_res["n"]}, alpha={ridge_res["alpha"]}, '
          f'R2_IS={ridge_res["r2_insample"]:.3f}, R2_CV={ridge_res["r2_cv"]:.3f}')
    print(ridge_res['coefs'].round(4).to_string())

    print('\n=== Lasso -> Ridge ===')
    lr = run_lasso_ridge_monthly(df)
    if lr.get('survivors'):
        print(f'R2_IS={lr["r2_insample"]:.3f}, adj-R2={lr["r2_adj"]:.3f}, R2_CV={lr["r2_cv"]:.3f}')
        print(lr['coefs'].round(4).to_string())
