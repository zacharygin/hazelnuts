"""
Annual hazelnut price regression.

y  = log USD return of harvest-season (Aug-Oct) VWAP price
X  = trailing-5yr production shortfall (% from trend, no look-ahead)

M1: ret_usd ~ shortfall_trail              R²≈0.37, n≈18
M2: ret_usd ~ shortfall_trail + ret_lag1   adds price-persistence check

Data sources:
  - Prices:    TOBB scraped harvest-season VWAP, all exchanges
  - Gap-fill:  giresun_spot_prices_cropyear.csv for 2020-2022
  - FX:        data/raw/fx/tryusd_annual.csv (TRY per USD, year-end close)
  - Shortfall: data/raw/hazelnut_35yr_master.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features import ols, print_model

PROJ_ROOT = Path(__file__).parent.parent
RAW       = PROJ_ROOT / 'data' / 'raw'
TOBB_CSV  = PROJ_ROOT / 'hazelnut_basket_scrape' / 'data' / 'processed' / 'hazelnut_combined.csv'

HARVEST_MONTHS = (8, 9, 10)


def build_price_series() -> pd.DataFrame:
    """TOBB harvest-season VWAP in TL/kg, gap-filled for 2020-2022."""
    tobb = pd.read_csv(TOBB_CSV, parse_dates=['date'])
    tobb = tobb[tobb['source'] == 'tobb'].copy()
    tobb['year']  = tobb['date'].dt.year
    tobb['month'] = tobb['date'].dt.month
    harvest = tobb[tobb['month'].isin(HARVEST_MONTHS)]

    def vwap(g):
        mask = g['volume_kg'].notna() & (g['volume_kg'] > 0)
        if mask.sum() > 0:
            return np.average(g.loc[mask, 'avg_price_tlkg'],
                              weights=g.loc[mask, 'volume_kg'])
        return g['avg_price_tlkg'].mean()

    scraped = (
        harvest.groupby('year')
        .apply(lambda g: pd.Series({'vwap_try': vwap(g),
                                    'n_days': int(g['date'].nunique())}),
               include_groups=False)
        .reset_index()
    )
    scraped['source_flag'] = 'tobb_scraped'

    cy_path = RAW / 'giresun_spot_prices_cropyear.csv'
    if cy_path.exists():
        cy = pd.read_csv(cy_path).rename(
            columns={'crop_year': 'year', 'vwap_try_kg_inshell': 'vwap_try'})
        gap_years = [y for y in (2020, 2021, 2022)
                     if y not in scraped['year'].values]
        if gap_years:
            fill = cy[cy['year'].isin(gap_years)][['year', 'vwap_try']].copy()
            fill['source_flag'] = 'giresun_cropyear'
            fill['n_days'] = np.nan
            scraped = pd.concat([scraped, fill], ignore_index=True)

    fx = pd.read_csv(RAW / 'fx' / 'tryusd_annual.csv')
    df = scraped.merge(fx[['year', 'tryusd_close']], on='year', how='left')
    df['vwap_usd'] = df['vwap_try'] * df['tryusd_close']
    return df.sort_values('year').reset_index(drop=True)


def build_dataset() -> pd.DataFrame:
    """Merge price series with shortfall; compute log returns."""
    prices = build_price_series()
    master = (pd.read_csv(RAW / 'hazelnut_35yr_master.csv')
                .rename(columns={'Unnamed: 0': 'year'}))
    master['year'] = master['year'].astype(int)
    master = master.sort_values('year')
    master['trend5']    = master['prod_mt'].rolling(5, min_periods=3).mean().shift(1)
    master['shortfall'] = (master['prod_mt'] - master['trend5']) / master['trend5'] * 100

    df = prices.merge(master[['year', 'shortfall']], on='year', how='inner')
    df = df.sort_values('year')
    df['log_usd']  = np.log(df['vwap_usd'])
    df['ret_usd']  = df['log_usd'].diff()
    df['ret_lag1'] = df['ret_usd'].shift(1)
    return df.dropna(subset=['ret_usd', 'shortfall']).reset_index(drop=True)


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = build_dataset()

    y   = df['ret_usd'].values
    one = np.ones(len(df))

    m1 = ols(y, np.c_[one, df['shortfall'].values],
             ['intercept', 'shortfall'], 'M1 — shortfall only')

    d2 = df.dropna(subset=['ret_lag1'])
    m2 = ols(d2['ret_usd'].values,
             np.c_[np.ones(len(d2)), d2['shortfall'].values, d2['ret_lag1'].values],
             ['intercept', 'shortfall', 'ret_lag1'], 'M2 — shortfall + lag')

    return {'M1': m1, 'M2': m2}


if __name__ == '__main__':
    df = build_dataset()
    print(f"Annual dataset: n={len(df)}, years {df['year'].min()}-{df['year'].max()}")
    models = run(df)
    for m in models.values():
        print_model(m)
