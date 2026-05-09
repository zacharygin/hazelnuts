"""
Download and aggregate financial/commodity features to annual harvest-year averages.

Harvest year N = Aug N through Oct N (same window as price series).
All features returned as prices; caller computes returns.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT      = Path(__file__).parent
PROJ_ROOT = ROOT.parent

# --- tickers to fetch from yfinance ---
YFINANCE_TICKERS = {
    'cocoa_fut':   'CC=F',
    'corn_fut':    'ZC=F',
    'soy_fut':     'ZS=F',
    'wheat_fut':   'ZW=F',
    'sugar_fut':   'SB=F',
    'coffee_fut':  'KC=F',
    'gold_fut':    'GC=F',
    'oil_fut':     'CL=F',
    'dxy':         'DX-Y.NYB',
    'spy':         'SPY',
    'gsg':         'GSG',
    'vix':         '^VIX',
}


def _harvest_avg(daily: pd.DataFrame, tickers: dict[str, str]) -> pd.DataFrame:
    """Aggregate daily close prices to harvest-year (Aug–Oct) averages."""
    daily = daily.copy()
    daily.index = pd.to_datetime(daily.index)
    daily['year']  = daily.index.year
    daily['month'] = daily.index.month
    harvest = daily[daily['month'].between(8, 10)]

    rows = []
    for col in daily.columns:
        if col in ('year', 'month'):
            continue
        agg = harvest.groupby('year')[col].mean()
        agg.name = col
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, axis=1).reset_index()
    out.columns.name = None
    return out


def fetch_yfinance(start: str = '2004-01-01') -> pd.DataFrame:
    tickers = list(YFINANCE_TICKERS.values())
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = raw['Close'].copy()
    close.columns.name = None

    # rename columns to friendly names
    inv = {v: k for k, v in YFINANCE_TICKERS.items()}
    close = close.rename(columns=inv)

    return _harvest_avg(close, YFINANCE_TICKERS)


def load_existing_equity() -> pd.DataFrame:
    """Load the existing annual equity basket (harvest-year averages already computed)."""
    path = PROJ_ROOT / 'data/raw/basket/equity_basket_prices.csv'
    df = pd.read_csv(path)
    df = df.rename(columns={'harvest_year': 'year'})
    df['year'] = df['year'].astype(int)
    return df


def build_feature_store(start: str = '2004-01-01') -> pd.DataFrame:
    """
    Combine existing equity basket + yfinance-fetched features.
    Returns wide DataFrame indexed by year (harvest-year) with all price levels.
    """
    equity = load_existing_equity()
    yf_data = fetch_yfinance(start=start)

    # Drop cocoa_usd from equity — replaced by yfinance cocoa_fut
    equity = equity.drop(columns=['cocoa_usd'], errors='ignore')

    merged = equity.merge(yf_data, on='year', how='outer').sort_values('year').reset_index(drop=True)
    return merged


def compute_returns(store: pd.DataFrame, skip_cols: tuple = ('year',)) -> pd.DataFrame:
    """
    Compute log returns for all numeric columns.
    Returns DataFrame of annual log returns, dropping the first (NaN) row.
    """
    cols = [c for c in store.columns if c not in skip_cols]
    out = store[['year']].copy()
    for col in cols:
        out[f'ret_{col}'] = np.log(store[col]).diff()
    return out.dropna(subset=['ret_' + cols[0]], how='all').reset_index(drop=True)


if __name__ == '__main__':
    store = build_feature_store()
    print('Feature store shape:', store.shape)
    print('Years:', store['year'].min(), '–', store['year'].max())
    print('\nColumn coverage (n non-null):')
    print(store.set_index('year').count().sort_values())
