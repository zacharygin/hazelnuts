"""
Frost degree-hours vs harvest-season hazelnut price.

Frost happens Feb-April. Price is measured Aug-Oct harvest-season VWAP (TOBB).
Question: does a damaging frost year produce a higher harvest-season USD price?

Thresholds tested: p75 (~45 DH) and p90 (~129 DH) of 85-year frost distribution.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'tobb_hazelnut_price'))
from annual_model import build_price_series, build_dataset
from features import ols, print_model

PROJ_ROOT = Path(__file__).parent.parent
FROST_CSV = PROJ_ROOT / 'data' / 'raw' / 'era5_frost_monthly.csv'

# Thresholds from 85-year distribution
P75 = 44.8   # ~75th percentile
P90 = 128.8  # ~90th percentile


def build_frost_price() -> pd.DataFrame:
    """
    Merge TOBB harvest-season VWAP with frost degree-hours.
    Frost year Y → price measured Aug-Oct of year Y.
    Returns annual dataframe with ret_usd, frost_dh, sub-period DH columns.
    """
    df    = build_dataset()          # ret_usd, shortfall, n=18
    frost = pd.read_csv(FROST_CSV)   # year, feb_dh, emarch_dh, lmarch_dh, april_dh, frost_dh

    merged = df.merge(frost, on='year', how='inner')
    return merged.sort_values('year').reset_index(drop=True)


def run(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = build_frost_price()

    y    = df['ret_usd'].values
    ones = np.ones(len(df))

    # M_frost: ret_usd ~ frost_dh
    m_frost = ols(y, np.c_[ones, df['frost_dh'].values],
                  ['intercept', 'frost_dh'], 'ret_usd ~ frost_dh')

    # M_shortfall: ret_usd ~ shortfall (baseline from annual_model)
    m_sf = ols(y, np.c_[ones, df['shortfall'].values],
               ['intercept', 'shortfall'], 'ret_usd ~ shortfall')

    # M_both: ret_usd ~ frost_dh + shortfall
    m_both = ols(y, np.c_[ones, df['frost_dh'].values, df['shortfall'].values],
                 ['intercept', 'frost_dh', 'shortfall'], 'ret_usd ~ frost_dh + shortfall')

    # Threshold split: mean ret_usd above/below p75 and p90
    splits = {}
    for label, thresh in [('p75', P75), ('p90', P90)]:
        hi = df[df['frost_dh'] >= thresh]
        lo = df[df['frost_dh'] <  thresh]
        splits[label] = {
            'threshold':  thresh,
            'n_hi':       len(hi),
            'n_lo':       len(lo),
            'mean_ret_hi': hi['ret_usd'].mean(),
            'mean_ret_lo': lo['ret_usd'].mean(),
            'years_hi':   sorted(hi['year'].tolist()),
        }

    return {
        'df':          df,
        'm_frost':     m_frost,
        'm_shortfall': m_sf,
        'm_both':      m_both,
        'splits':      splits,
    }


if __name__ == '__main__':
    out = run()
    df  = out['df']

    print(f"Dataset: n={len(df)}, years {df['year'].min()}-{df['year'].max()}")
    print(f"Frost DH range: {df['frost_dh'].min():.1f} – {df['frost_dh'].max():.1f}")
    print()

    print('--- Year-by-year ---')
    tbl = df[['year', 'frost_dh', 'ret_usd', 'shortfall']].copy()
    tbl['frost_flag'] = tbl['frost_dh'].apply(
        lambda x: '*** p90' if x >= P90 else ('*  p75' if x >= P75 else ''))
    print(tbl.to_string(index=False))
    print()

    print('--- Model comparison ---')
    for m in [out['m_frost'], out['m_shortfall'], out['m_both']]:
        print_model(m)

    print()
    print('--- Threshold split ---')
    for label, s in out['splits'].items():
        diff = s['mean_ret_hi'] - s['mean_ret_lo']
        print(f"  {label} (>{s['threshold']:.0f} DH):  "
              f"n_hi={s['n_hi']}  mean_ret_hi={s['mean_ret_hi']:+.3f}  "
              f"mean_ret_lo={s['mean_ret_lo']:+.3f}  diff={diff:+.3f}")
        print(f"    frost years: {s['years_hi']}")
