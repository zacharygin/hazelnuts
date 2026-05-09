"""
PCA factor regression for monthly hazelnut USD price returns.

No production data. Extracts latent factors from financial/commodity
feature universe, identifies which explain hazelnut price moves.

Key result: PC1 (commodity/risk-on) + PC3 (USD strength) explain
~33% of monthly hazelnut USD returns, n=207, both p<0.001.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from monthly_regression import build_monthly_dataset
from tobb_price_regression import _ols

ROOT = Path(__file__).parent

EXCLUDE_FEATURES = {'ret_vwap_try', 'ret_tryusd', 'ret_vwap_usd', 'ret_tur'}
MIN_COVERAGE     = 0.70
N_COMPONENTS     = 8


def build_feature_matrix(min_coverage: float = MIN_COVERAGE) -> tuple[pd.DataFrame, list[str]]:
    """Return (sub, feat_cols) — aligned monthly dataset with feature columns."""
    df = build_monthly_dataset()
    feat_cols = [
        c for c in df.columns
        if c.startswith('ret_')
        and c not in EXCLUDE_FEATURES
        and df[c].notna().mean() >= min_coverage
    ]
    sub = df[['month', 'ret_vwap_usd'] + feat_cols].dropna()
    return sub, feat_cols


def run_pca(sub: pd.DataFrame, feat_cols: list[str],
            n_components: int = N_COMPONENTS) -> dict:
    """Fit PCA on features, return components, loadings, and scores."""
    X_raw = sub[feat_cols].values
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_raw)

    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(X_s)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feat_cols,
        columns=[f'PC{i+1}' for i in range(n_components)],
    )

    return {
        'pca':      pca,
        'scaler':   scaler,
        'scores':   scores,
        'loadings': loadings,
        'var_explained': pca.explained_variance_ratio_ * 100,
    }


def run_pc_regressions(sub: pd.DataFrame, pca_out: dict,
                       max_pcs: int = 6) -> pd.DataFrame:
    """
    OLS of ret_vwap_usd ~ each PC individually, then sequentially added.
    Returns DataFrame of results.
    """
    y     = sub['ret_vwap_usd'].values
    ones  = np.ones(len(y))
    scores = pca_out['scores']
    rows   = []

    # Bivariate: each PC alone
    for i in range(max_pcs):
        m = _ols(y, np.c_[ones, scores[:, i]],
                 ['intercept', f'PC{i+1}'], f'PC{i+1}')
        rows.append({
            'model':   f'PC{i+1} only',
            'n':       m.n,
            'R2':      round(m.rsquared, 3),
            'adj_R2':  round(m.rsquared_adj, 3),
            'AIC':     round(m.aic, 1),
            'PC_t':    round(m.tvalues[f'PC{i+1}'], 2),
            'PC_p':    round(m.pvalues[f'PC{i+1}'], 3),
        })

    # Best combination found: PC1 + PC3
    m13 = _ols(y, np.c_[ones, scores[:, 0], scores[:, 2]],
               ['intercept', 'PC1', 'PC3'], 'PC1+PC3')
    rows.append({
        'model':  'PC1 + PC3',
        'n':      m13.n,
        'R2':     round(m13.rsquared, 3),
        'adj_R2': round(m13.rsquared_adj, 3),
        'AIC':    round(m13.aic, 1),
        'PC_t':   None,
        'PC_p':   None,
    })

    return pd.DataFrame(rows), m13


def run_all() -> dict:
    sub, feat_cols = build_feature_matrix()
    pca_out        = run_pca(sub, feat_cols)
    results, best  = run_pc_regressions(sub, pca_out)

    y      = sub['ret_vwap_usd'].values
    scores = pca_out['scores']
    ones   = np.ones(len(y))
    best_full = _ols(y, np.c_[ones, scores[:, 0], scores[:, 2]],
                     ['intercept', 'PC1', 'PC3'], 'PC1+PC3')

    return {
        'sub':        sub,
        'feat_cols':  feat_cols,
        'pca_out':    pca_out,
        'results':    results,
        'best_model': best_full,
    }


if __name__ == '__main__':
    out = run_all()
    pca_out = out['pca_out']
    sub     = out['sub']

    print(f"n={len(sub)}  ({sub['month'].iloc[0]} – {sub['month'].iloc[-1]})")
    print(f"Features: {out['feat_cols']}\n")

    print('--- Variance explained by component ---')
    for i, v in enumerate(pca_out['var_explained']):
        print(f'  PC{i+1}: {v:.1f}%')

    print('\n--- PC loadings (top 5 per component) ---')
    for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
        ld = pca_out['loadings'][pc].sort_values(key=abs, ascending=False)
        top = ', '.join([f'{k}({v:+.2f})' for k, v in ld.head(5).items()])
        print(f'  {pc}: {top}')

    print('\n--- Regression results ---')
    print(out['results'].to_string(index=False))

    m = out['best_model']
    print(f'\n--- Best model: PC1 + PC3 ---')
    print(pd.DataFrame({
        'coef': m.params, 't': m.tvalues, 'p': m.pvalues
    }).round(3).to_string())
