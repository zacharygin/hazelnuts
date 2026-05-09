"""Shared data loading for all price models in this directory."""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import t as t_dist

RAW   = Path(__file__).parent.parent / 'data' / 'raw'
TOBB  = Path(__file__).parent.parent / 'hazelnut_basket_scrape' / 'data' / 'processed' / 'hazelnut_combined.csv'

YF_TICKERS = {
    'cocoa':'CC=F','corn':'ZC=F','soy':'ZS=F','wheat':'ZW=F',
    'sugar':'SB=F','coffee':'KC=F','gold':'GC=F','oil':'CL=F',
    'dxy':'DX-Y.NYB','spy':'SPY','vix':'^VIX',
    'nestle':'NESN.SW','barry':'BARN.SW','hershey':'HSY','mdlz':'MDLZ','bunge':'BG',
}


def ols(y, X, names, label):
    """Numpy OLS. Returns dict: params, tvalues, pvalues, r2, r2_adj, aic, fitted, resid."""
    n, k     = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted   = X @ beta
    resid    = y - fitted
    ss_res   = resid @ resid
    ss_tot   = ((y - y.mean()) ** 2).sum()
    r2       = 1 - ss_res / ss_tot
    r2_adj   = 1 - (1 - r2) * (n - 1) / (n - k)
    s2       = ss_res / (n - k)
    se       = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    tvals    = beta / se
    pvals    = 2 * t_dist.sf(np.abs(tvals), df=n - k)
    aic      = n * np.log(ss_res / n) + 2 * k
    return dict(label=label, n=n,
                params=pd.Series(beta, index=names),
                tvalues=pd.Series(tvals, index=names),
                pvalues=pd.Series(pvals, index=names),
                r2=r2, r2_adj=r2_adj, aic=aic,
                fitted=fitted, resid=resid)


def print_model(m):
    print(f"\n{'='*55}\n  {m['label']}")
    print(f"  n={m['n']}  R²={m['r2']:.3f}  adj-R²={m['r2_adj']:.3f}  AIC={m['aic']:.1f}")
    print(f"{'='*55}")
    print(pd.DataFrame({'coef':m['params'],'t':m['tvalues'],'p':m['pvalues']}).round(4).to_string())


def load_giresun_monthly() -> pd.DataFrame:
    """
    Giresun exchange monthly prices (2001-2025).
    Missing USD values filled via yfinance TRY=X monthly average (TRY per 1 USD).
    Returns: month, avg_try_kg, avg_usd_kg, tmo_qty_kg, total_qty_kg, tmo_share.
    """
    g = (pd.read_csv(RAW / 'giresun_spot_prices_monthly.csv')
           .query("period == 'full'").copy())
    g['month'] = g['year'].astype(str) + '-' + g['month'].astype(str).str.zfill(2)
    g = g.rename(columns={'avg_try_kg_inshell':'avg_try_kg',
                           'avg_usd_kg_inshell':'avg_usd_kg'})

    if g['avg_usd_kg'].isna().any():
        raw = yf.download('TRY=X', start='2000-01-01', auto_adjust=True, progress=False)
        close = raw['Close']
        if hasattr(close, 'squeeze'):
            close = close.squeeze()   # DataFrame → Series if single ticker
        fx = close.resample('ME').mean()
        fx.index = fx.index.strftime('%Y-%m')
        mask = g['avg_usd_kg'].isna() & g['avg_try_kg'].notna()
        for idx in g[mask].index:
            mo = g.loc[idx, 'month']
            if mo in fx.index:
                g.loc[idx, 'avg_usd_kg'] = g.loc[idx, 'avg_try_kg'] / float(fx.loc[mo])

    g['tmo_share'] = g['tmo_qty_kg'] / g['total_qty_kg']
    return (g[['month','avg_try_kg','avg_usd_kg','tmo_qty_kg','total_qty_kg','tmo_share']]
              .sort_values('month').reset_index(drop=True))


def build_shortfall_monthly() -> pd.DataFrame:
    """
    Annual trailing-5yr shortfall carried to monthly frequency.
    Crop-year logic: Aug-Dec of year Y and Jan-Jul of Y+1 share year Y's shortfall.
    Returns: month, shortfall.
    """
    m = (pd.read_csv(RAW / 'hazelnut_35yr_master.csv')
           .rename(columns={'Unnamed: 0':'year'}))
    m['year'] = m['year'].astype(int)
    m = m.sort_values('year')
    m['trend5']    = m['prod_mt'].rolling(5, min_periods=3).mean().shift(1)
    m['shortfall'] = (m['prod_mt'] - m['trend5']) / m['trend5'] * 100

    rows = []
    for _, r in m.iterrows():
        cy, sf = int(r['year']), r['shortfall']
        for mo in list(range(8, 13)) + list(range(1, 8)):
            yr = cy if mo >= 8 else cy + 1
            rows.append({'month': f'{yr}-{mo:02d}', 'shortfall': sf})
    return pd.DataFrame(rows)


def fetch_yfinance_monthly(start='2000-01-01') -> tuple[pd.DataFrame, list]:
    """
    16 financial/commodity features, monthly average, log-differenced.
    Downloads tickers one-at-a-time to avoid bulk rate-limit failures.
    Returns: (df[month + ret_cols], ret_col_names).
    """
    frames = {}
    for name, ticker in YF_TICKERS.items():
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            continue
        close = raw['Close']
        if hasattr(close, 'iloc') and close.ndim > 1:
            close = close.iloc[:, 0]
        monthly = close.resample('ME').mean()
        monthly.index = monthly.index.strftime('%Y-%m')
        frames[name] = monthly

    if not frames:
        return pd.DataFrame(columns=['month']), []

    combined = pd.DataFrame(frames)
    combined.index.name = 'month'
    combined = combined.reset_index()

    ret_cols = []
    for c in YF_TICKERS:
        if c in combined.columns:
            combined[f'ret_{c}'] = np.log(combined[c]).diff()
            ret_cols.append(f'ret_{c}')

    keep = ['month'] + ret_cols
    result = combined[[c for c in keep if c in combined.columns]].copy()
    result = result[result[ret_cols].notna().any(axis=1)].reset_index(drop=True)
    return result, ret_cols


def fit_pca(feat_df, ret_cols, n_components=8):
    """PCA on standardised features. Returns (df[month + PC1..n], fitted PCA object)."""
    X   = StandardScaler().fit_transform(feat_df[ret_cols].values)
    pca = PCA(n_components=n_components).fit(X)
    sc  = pca.transform(X)
    out = feat_df[['month']].copy()
    for i in range(n_components):
        out[f'PC{i+1}'] = sc[:, i]
    return out, pca


def load_tobb_monthly() -> pd.DataFrame:
    """
    All-exchange TOBB monthly VWAP from scraped data.
    Returns: month, vwap_try, n_days.
    """
    tobb = pd.read_csv(TOBB, parse_dates=['date'])
    tobb = tobb[tobb['source'] == 'tobb'].copy()
    tobb['ym'] = tobb['date'].dt.to_period('M')

    def vwap(g):
        mask = g['volume_kg'].notna() & (g['volume_kg'] > 0)
        if mask.sum() > 0:
            return np.average(g.loc[mask,'avg_price_tlkg'], weights=g.loc[mask,'volume_kg'])
        return g['avg_price_tlkg'].mean()

    out = (tobb.groupby('ym')
               .apply(lambda g: pd.Series({'vwap_try': vwap(g),
                                           'n_days': g['date'].nunique()}),
                      include_groups=False)
               .reset_index())
    out['month'] = out['ym'].dt.strftime('%Y-%m')
    return out[['month','vwap_try','n_days']].sort_values('month').reset_index(drop=True)
