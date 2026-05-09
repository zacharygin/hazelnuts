"""
Build TOBB-based annual price series and run price ~ shortfall regressions.

Data flow:
  1. Scraped TOBB CSV → harvest-season (Aug–Oct) VWAP, all exchanges, volume-weighted
  2. Gap years 2020–2022 filled from crop-year file (TOBB data, prior collection)
  3. Convert TL → USD via annual FX close
  4. Merge with production shortfall from 35yr master
  5. OLS: ret_usd ~ shortfall_trail  (+lag variant)
"""

from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent
PROJ_ROOT = ROOT.parent


@dataclass
class OLSResult:
    """Minimal OLS result container (numpy-only, no statsmodels)."""
    name:        str
    n:           int
    params:      pd.Series
    tvalues:     pd.Series
    pvalues:     pd.Series
    rsquared:    float
    rsquared_adj:float
    aic:         float
    resid:       np.ndarray
    fitted:      np.ndarray

    def summary_line(self) -> str:
        return (f'{self.name}  n={self.n}  R²={self.rsquared:.3f}  '
                f'adj-R²={self.rsquared_adj:.3f}  AIC={self.aic:.1f}')


def _ols(y: np.ndarray, X: np.ndarray, feature_names: list[str], name: str) -> OLSResult:
    """OLS via normal equations with intercept assumed to be first column of X."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted  = X @ beta
    resid   = y - fitted
    ss_res  = resid @ resid
    ss_tot  = ((y - y.mean()) ** 2).sum()
    r2      = 1 - ss_res / ss_tot
    r2_adj  = 1 - (1 - r2) * (n - 1) / (n - k)
    s2      = ss_res / (n - k)
    cov     = s2 * np.linalg.inv(X.T @ X)
    se      = np.sqrt(np.diag(cov))
    tvals   = beta / se
    from scipy.stats import t as t_dist
    pvals   = 2 * t_dist.sf(np.abs(tvals), df=n - k)
    aic     = n * np.log(ss_res / n) + 2 * k
    return OLSResult(
        name=name, n=n,
        params   =pd.Series(beta,  index=feature_names),
        tvalues  =pd.Series(tvals, index=feature_names),
        pvalues  =pd.Series(pvals, index=feature_names),
        rsquared=r2, rsquared_adj=r2_adj, aic=aic,
        resid=resid, fitted=fitted,
    )


def build_price_series(harvest_months: tuple = (8, 9, 10)) -> pd.DataFrame:
    tobb = pd.read_csv(ROOT / 'data/processed/hazelnut_combined.csv', parse_dates=['date'])
    tobb = tobb[tobb['source'] == 'tobb'].copy()
    tobb['year']  = tobb['date'].dt.year
    tobb['month'] = tobb['date'].dt.month

    harvest = tobb[tobb['month'].isin(harvest_months)]

    def vwap(g):
        mask = g['volume_kg'].notna() & (g['volume_kg'] > 0)
        if mask.sum() > 0:
            return np.average(g.loc[mask, 'avg_price_tlkg'], weights=g.loc[mask, 'volume_kg'])
        return g['avg_price_tlkg'].mean()

    scraped = (
        harvest.groupby('year')
        .apply(lambda g: pd.Series({
            'vwap_try': vwap(g),
            'n_days':   int(g['date'].nunique()),
        }), include_groups=False)
        .reset_index()
    )
    scraped['source_flag'] = 'tobb_scraped'

    # Fill 2020–2022 gap from crop-year file
    cy_path = PROJ_ROOT / 'data/raw/giresun_spot_prices_cropyear.csv'
    if cy_path.exists():
        cy = pd.read_csv(cy_path).rename(
            columns={'crop_year': 'year', 'vwap_try_kg_inshell': 'vwap_try'})
        gap_years = [y for y in (2020, 2021, 2022) if y not in scraped['year'].values]
        if gap_years:
            fallback = cy[cy['year'].isin(gap_years)][['year', 'vwap_try']].copy()
            fallback['source_flag'] = 'tobb_cropyear'
            fallback['n_days'] = np.nan
            scraped = pd.concat([scraped, fallback], ignore_index=True)

    fx = pd.read_csv(PROJ_ROOT / 'data/raw/fx/tryusd_annual.csv')
    df = scraped.merge(fx[['year', 'tryusd_close']], on='year', how='left')
    df['vwap_usd'] = df['vwap_try'] * df['tryusd_close']
    return df.sort_values('year').reset_index(drop=True)


def build_regression_dataset(price_df: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(PROJ_ROOT / 'data/raw/hazelnut_35yr_master.csv')
    master = master.rename(columns={'Unnamed: 0': 'year'})
    master['year'] = master['year'].astype(int)
    master = master.sort_values('year')

    master['trend_trail5']    = master['prod_mt'].rolling(5, min_periods=3).mean().shift(1)
    master['shortfall_trail'] = (
        (master['prod_mt'] - master['trend_trail5']) / master['trend_trail5'] * 100
    )

    df = price_df.merge(
        master[['year', 'prod_mt', 'shortfall_trail', 'frost_dh']],
        on='year', how='inner'
    )
    df = df.sort_values('year')
    df['log_price_usd'] = np.log(df['vwap_usd'])
    df['ret_usd']       = df['log_price_usd'].diff()
    df['ret_usd_lag1']  = df['ret_usd'].shift(1)
    return df.dropna(subset=['ret_usd', 'shortfall_trail']).reset_index(drop=True)


def run_models(df: pd.DataFrame) -> dict:
    y = df['ret_usd'].values

    # M1: ret_usd ~ shortfall_trail
    X1 = np.column_stack([np.ones(len(df)), df['shortfall_trail'].values])
    m1 = _ols(y, X1, ['intercept', 'shortfall_trail'], 'M1_shortfall')

    # M2: ret_usd ~ shortfall_trail + ret_usd_lag1
    d2 = df.dropna(subset=['ret_usd_lag1'])
    y2 = d2['ret_usd'].values
    X2 = np.column_stack([np.ones(len(d2)), d2['shortfall_trail'].values, d2['ret_usd_lag1'].values])
    m2 = _ols(y2, X2, ['intercept', 'shortfall_trail', 'ret_usd_lag1'], 'M2_shortfall_lag')

    return {'M1_shortfall': m1, 'M2_shortfall_lag': m2}


# ---------------------------------------------------------------------------
# Regularized regression (Ridge, Lasso→Ridge) — sklearn-based
# ---------------------------------------------------------------------------

def build_feature_matrix(reg_df: pd.DataFrame, min_obs: int = 14) -> pd.DataFrame:
    """
    Join regression dataset with full feature store, compute log returns,
    drop features with fewer than min_obs non-NaN values.
    Returns a DataFrame with 'year', 'ret_usd', 'shortfall_trail', and all
    ret_<feature> columns that pass the coverage filter.
    """
    from fetch_features import build_feature_store, compute_returns

    store   = build_feature_store()
    ret_all = compute_returns(store)

    merged = reg_df[['year', 'ret_usd', 'shortfall_trail', 'frost_dh']].merge(
        ret_all, on='year', how='inner'
    )

    # Drop feature columns with too few obs
    feat_cols = [c for c in merged.columns
                 if c.startswith('ret_') and c not in ('ret_usd',)]
    keep = [c for c in feat_cols if merged[c].notna().sum() >= min_obs]
    return merged[['year', 'ret_usd', 'shortfall_trail', 'frost_dh'] + keep]


def run_ridge(feature_df: pd.DataFrame, alphas: tuple = (0.01, 0.1, 1, 10, 100, 1000)) -> dict:
    """RidgeCV on all available features (shortfall + everything else)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in feature_df.columns
                 if c not in ('year', 'ret_usd')]
    sub = feature_df[['ret_usd'] + feat_cols].dropna()

    y = sub['ret_usd'].values
    X_raw = sub[feat_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = RidgeCV(alphas=alphas, cv=5, scoring='r2')
    model.fit(X, y)

    coefs = pd.Series(model.coef_, index=feat_cols).sort_values(key=abs, ascending=False)
    yhat  = model.predict(X)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_is  = 1 - ss_res / ss_tot
    cv_r2  = model.best_score_  # mean CV R²

    return {
        'model':      model,
        'scaler':     scaler,
        'features':   feat_cols,
        'coefs':      coefs,
        'alpha':      model.alpha_,
        'n':          len(y),
        'r2_insample':r2_is,
        'r2_cv':      cv_r2,
    }


def run_lasso_ridge(feature_df: pd.DataFrame, cv: int = 5) -> dict:
    """LassoCV for feature selection, then RidgeCV refit on survivors."""
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.preprocessing import StandardScaler

    feat_cols = [c for c in feature_df.columns
                 if c not in ('year', 'ret_usd')]
    sub = feature_df[['ret_usd'] + feat_cols].dropna()

    y     = sub['ret_usd'].values
    X_raw = sub[feat_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    lasso = LassoCV(cv=cv, max_iter=10000, random_state=42)
    lasso.fit(X, y)
    survivors = [feat_cols[i] for i, c in enumerate(lasso.coef_) if c != 0]

    if not survivors:
        return {'survivors': [], 'note': 'Lasso zeroed all features'}

    X_surv = sub[survivors].values
    scaler2 = StandardScaler()
    X_surv_s = scaler2.fit_transform(X_surv)

    ridge = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100, 1000), cv=cv, scoring='r2')
    ridge.fit(X_surv_s, y)

    yhat  = ridge.predict(X_surv_s)
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_is  = 1 - ss_res / ss_tot

    coefs = pd.Series(ridge.coef_, index=survivors).sort_values(key=abs, ascending=False)

    return {
        'model':        ridge,
        'scaler':       scaler2,
        'lasso_alpha':  lasso.alpha_,
        'survivors':    survivors,
        'coefs':        coefs,
        'ridge_alpha':  ridge.alpha_,
        'n':            len(y),
        'r2_insample':  r2_is,
        'r2_cv':        ridge.best_score_,
    }


if __name__ == '__main__':
    price  = build_price_series()
    df     = build_regression_dataset(price)
    models = run_models(df)
    for name, res in models.items():
        print(f'\n=== {res.summary_line()} ===')
        tbl = pd.DataFrame({'coef': res.params, 't': res.tvalues, 'p': res.pvalues}).round(4)
        print(tbl.to_string())
