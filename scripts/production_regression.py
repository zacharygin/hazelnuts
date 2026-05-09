#!/usr/bin/env python3
"""
Turkey hazelnut production regression — relaxed L1→L2.

Target  : YoY log-return of FAOSTAT Turkey hazelnut production (annual)
Features: ERA5 frost degree-hours (5-phase) + ERA5 precipitation
          + lagged production return (on/off year proxy)
          + optional equity/FX/commodity annual returns (--with-equity)

Stage 1 : LassoCV  — feature selection
Stage 2 : RidgeCV  — unbiased refit on survivors

Usage:
    python scripts/production_regression.py
    python scripts/production_regression.py --with-equity
    python scripts/production_regression.py --save-figs --no-show
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

ROOT       = Path(__file__).parent.parent
DATA       = ROOT / 'data' / 'raw'
FIGDIR     = ROOT / 'notebooks' / 'figures'
FROST_CSV  = DATA / 'era5_frost_monthly.csv'
PRECIP_CSV = DATA / 'era5_precip_monthly.csv'
FAOSTAT_CSV = DATA / 'faostat' / 'turkey_hazelnut_production.csv'

EQUITY_TICKERS = {
    'tur':    'TUR',      'tryusd': 'TRYUSD=X', 'eurusd': 'EURUSD=X',
    'bist':   'XU100.IS', 'barry':  'BARN.SW',  'lindt':  'LISN.SW',
    'hershey':'HSY',      'mdlz':   'MDLZ',     'ulker':  'ULKER.IS',
    'adm':    'ADM',      'cocoa':  'CC=F',      'dba':    'DBA',
    'sp500':  '^GSPC',    'vix':    '^VIX',
}


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_production() -> pd.Series:
    df = pd.read_csv(FAOSTAT_CSV, index_col='year')['production_mt'].dropna()
    df.index = pd.to_datetime(df.index.astype(str), format='%Y')
    return df.rename('production_mt')


def load_weather() -> pd.DataFrame:
    parts = []
    if FROST_CSV.exists():
        f = pd.read_csv(FROST_CSV, index_col='year')
        f.index = pd.to_datetime(f.index.astype(str), format='%Y')
        parts.append(f)
        print(f'  Frost DH    : {len(f)} years ({f.index.year.min()}–{f.index.year.max()})')
    else:
        print('  WARNING: frost CSV not found')
    if PRECIP_CSV.exists():
        p = pd.read_csv(PRECIP_CSV, index_col='year')
        p.index = pd.to_datetime(p.index.astype(str), format='%Y')
        parts.append(p)
        print(f'  Precip      : {len(p)} years ({p.index.year.min()}–{p.index.year.max()})')
    else:
        print('  WARNING: precip CSV not found')
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def download_equity_annual() -> pd.DataFrame:
    print('Downloading equity data (monthly → annual)...')
    monthly = {}
    for name, ticker in EQUITY_TICKERS.items():
        try:
            print(f'  {name:10s} ({ticker}) ...', end=' ', flush=True)
            raw = yf.download(ticker, start='1999-01-01', end='2024-12-31',
                              interval='1mo', progress=False, auto_adjust=True)
            if raw.empty:
                print('no data'); continue
            close = raw['Close'].squeeze()
            close.index = close.index.to_period('M').to_timestamp()
            monthly[name] = close
            print(f'{close.notna().sum()} months')
        except Exception as e:
            print(f'ERROR — {e}')
    if not monthly:
        return pd.DataFrame()
    df = pd.DataFrame(monthly)
    log_m = np.log(df / df.shift(1))
    annual = log_m.resample('YS').sum(min_count=6)
    annual.index = annual.index.to_period('Y').to_timestamp()
    cnt = df.resample('YS').count()
    cnt.index = cnt.index.to_period('Y').to_timestamp()
    annual[cnt < 10] = np.nan
    return annual


# ── Regression ────────────────────────────────────────────────────────────────
def relaxed_lasso_ridge(X, y, names, n):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = min(5, n // 6)

    print(f'\nStage 1: LassoCV  (cv={cv})...')
    lasso = LassoCV(alphas=np.logspace(-4, 2, 300), cv=cv, max_iter=50000, n_jobs=-1)
    lasso.fit(Xs, y)
    mask = np.abs(lasso.coef_) > 1e-6
    sel  = [names[i] for i in range(len(names)) if mask[i]]
    print(f'  α={lasso.alpha_:.5f}   selected {mask.sum()}/{len(names)}: {sel}')

    if mask.sum() == 0:
        print('  WARNING: all zeroed — relaxing to 25th-pct α')
        lasso.alpha_ = lasso.alphas_[len(lasso.alphas_) // 4]
        lasso.coef_  = Lasso(alpha=lasso.alpha_, max_iter=50000).fit(Xs, y).coef_
        mask = np.abs(lasso.coef_) > 1e-6
        sel  = [names[i] for i in range(len(names)) if mask[i]]
        print(f'  Relaxed α={lasso.alpha_:.5f}   selected {mask.sum()}/{len(names)}: {sel}')

    print(f'\nStage 2: RidgeCV on {mask.sum()} features  (cv={cv})...')
    Xs_sel = Xs[:, mask]
    ridge  = RidgeCV(alphas=np.logspace(-3, 4, 200), cv=cv)
    ridge.fit(Xs_sel, y)
    yhat   = ridge.predict(Xs_sel)
    r2_is  = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)
    cv_r2  = cross_val_score(ridge, Xs_sel, y, cv=cv, scoring='r2').mean()
    rmse   = np.sqrt(np.mean((y - yhat)**2))
    print(f'  R²(in-sample)={r2_is:.3f}   CV R²={cv_r2:.3f}   RMSE={rmse:.4f}')

    coef_df = pd.DataFrame({
        'feature': sel, 'ridge_coef': ridge.coef_, 'lasso_coef': lasso.coef_[mask]
    }).sort_values('ridge_coef', key=abs, ascending=False)
    print('\nCoefficients:\n', coef_df.to_string(index=False))

    return dict(lasso=lasso, ridge=ridge, scaler=scaler, mask=mask, sel=sel,
                coef_df=coef_df, yhat=yhat, r2_is=r2_is, cv_r2=cv_r2, rmse=rmse,
                names=names, Xs=Xs)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(res, y, years, prod_raw, save, show):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = []

    # 1. Feature correlations
    corrs = pd.Series({f: np.corrcoef(res['Xs'][:, i], y)[0, 1]
                       for i, f in enumerate(res['names'])}).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, len(corrs) * 0.3)))
    ax.barh(corrs.index, corrs.values,
            color=['#e74c3c' if v < 0 else '#2ecc71' for v in corrs], edgecolor='none')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Pearson r  (vs annual production log-return)')
    ax.set_title('Feature correlations — Turkey hazelnut production')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_correlations.png', dpi=150)
    figs.append(fig)

    # 2. Lasso path
    mse_path = np.mean(res['lasso'].mse_path_, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(res['lasso'].alphas_, mse_path, color='steelblue', lw=1.5)
    ax.axvline(res['lasso'].alpha_, color='red', ls='--', lw=1.2,
               label=f"α={res['lasso'].alpha_:.5f}")
    ax.set_xlabel('Alpha'); ax.set_ylabel('CV MSE')
    ax.set_title('Lasso regularisation path'); ax.legend()
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_lasso_path.png', dpi=150)
    figs.append(fig)

    # 3. Ridge coefficients
    coef_df = res['coef_df']
    fig, ax = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.55)))
    ax.barh(coef_df['feature'], coef_df['ridge_coef'],
            color=['#e74c3c' if v < 0 else '#2ecc71' for v in coef_df['ridge_coef']],
            edgecolor='none')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Ridge coefficient (std. units)')
    ax.set_title(f"Ridge coefs — R²={res['r2_is']:.3f}  CV R²={res['cv_r2']:.3f}")
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_ridge_coefs.png', dpi=150)
    figs.append(fig)

    # 4. Actual vs predicted (levels + returns)
    prod_aligned = prod_raw.reindex(years)
    pred_levels  = prod_raw.shift(1).reindex(years) * np.exp(res['yhat'])
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.bar(years.year, prod_aligned / 1e3, color='#bdc3c7', label='Actual (kt)')
    ax1.plot(years.year, pred_levels / 1e3, 'r--', lw=1.5, label='Predicted (kt)')
    ax1.set_ylabel('Production (000 t)'); ax1.legend(fontsize=9)
    ax1.set_title('Turkey hazelnut production: actual vs model')

    ax2.plot(years.year, y, color='#2c3e50', lw=1.2, marker='o', ms=3, label='Actual return')
    ax2.plot(years.year, res['yhat'], 'r--', lw=1.0, label='Predicted')
    ax2.axhline(0, color='grey', lw=0.5); ax2.set_ylabel('Log return')
    ax2.set_xlabel('Year'); ax2.legend(fontsize=9)

    ax3.scatter(res['yhat'], y, alpha=0.5, s=20, color='steelblue', edgecolors='none')
    mn, mx = min(y.min(), res['yhat'].min()), max(y.max(), res['yhat'].max())
    ax3.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
    ax3.set_title(f"R²={res['r2_is']:.3f}")
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_fit.png', dpi=150)
    figs.append(fig)

    # 5. Residuals
    resid = y - res['yhat']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(years.year, resid,
            color=['#e74c3c' if r < 0 else '#2ecc71' for r in resid])
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_title('Residuals by year'); ax1.set_ylabel('Actual − Predicted log return')
    ax2.hist(resid, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='red', ls='--', lw=0.8)
    ax2.set_title(f'Residuals  σ={resid.std():.3f}')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_residuals.png', dpi=150)
    figs.append(fig)

    if show: plt.show()
    for f in figs: plt.close(f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-equity', action='store_true',
                    help='Add annual equity/FX returns as features')
    ap.add_argument('--save-figs',   action='store_true')
    ap.add_argument('--no-show',     action='store_true')
    args = ap.parse_args()

    print('Loading FAOSTAT production...')
    prod = load_production()
    print(f'  {len(prod)} years ({prod.index.year.min()}–{prod.index.year.max()})')

    print('Loading ERA5 weather features...')
    weather = load_weather()

    equity = download_equity_annual() if args.with_equity else pd.DataFrame()

    print('Building feature matrix...')
    prod_ret_lag = np.log(prod / prod.shift(1)).shift(1).rename('prod_ret_lag1')
    parts = [prod_ret_lag, weather]
    if not equity.empty:
        parts.append(equity)
    X_df = pd.concat(parts, axis=1)

    prod_ret = np.log(prod / prod.shift(1)).rename('prod_ret')
    combined = pd.concat([prod_ret, X_df], axis=1).dropna(subset=['prod_ret'])

    thresh   = int(0.5 * len(combined))
    n_before = combined.shape[1] - 1
    combined = combined.dropna(axis=1, thresh=thresh)
    dropped  = n_before - (combined.shape[1] - 1)
    if dropped: print(f'  Dropped {dropped} features with >50% missing')

    feat_cols = [c for c in combined.columns if c != 'prod_ret']
    for col in feat_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].mean())
    combined = combined[combined['prod_ret'].notna()]

    print(f'\nSample: {len(combined)} years  '
          f'({combined.index.year.min()}–{combined.index.year.max()})  '
          f'{combined.shape[1]-1} features')

    y    = combined['prod_ret'].values
    cols = [c for c in combined.columns if c != 'prod_ret']
    X    = combined[cols].values

    res = relaxed_lasso_ridge(X, y, cols, n=len(y))

    print('\nRendering plots...')
    plot_all(res, y, combined.index, prod, save=args.save_figs, show=not args.no_show)
    print('Done — close plot windows to exit.' if not args.no_show else 'Done.')

    print('\n' + '=' * 60)
    print('PRODUCTION REGRESSION SUMMARY')
    print('=' * 60)
    print(f'Target       : FAOSTAT Turkey hazelnut production (YoY log-return)')
    print(f'Observations : {len(y)} years')
    print(f'Features     : {len(cols)} candidate → {res["mask"].sum()} selected')
    print(f'In-sample R² : {res["r2_is"]:.4f}')
    print(f'CV R²        : {res["cv_r2"]:.4f}')
    print(f'RMSE         : {res["rmse"]:.4f}  (log-return units)')
    print('=' * 60)


if __name__ == '__main__':
    main()
