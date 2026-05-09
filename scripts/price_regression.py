#!/usr/bin/env python3
"""
Giresun hazelnut price regression — relaxed L1→L2.

Target  : monthly log-return of Giresun in-shell USD/kg spot price
Features: equity/FX/commodity monthly log-returns
          + month fixed-effects
          + optional Google Trends (--trends)
          + optional NLP news features from scrape_news.py output (--news)

Stage 1 : LassoCV  — feature selection
Stage 2 : RidgeCV  — unbiased refit on survivors

Usage:
    python scripts/price_regression.py
    python scripts/price_regression.py --trends
    python scripts/price_regression.py --news
    python scripts/price_regression.py --save-figs --no-show
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

ROOT     = Path(__file__).parent.parent
DATA     = ROOT / 'data' / 'raw'
FIGDIR   = ROOT / 'notebooks' / 'figures'
NEWS_CSV = DATA / 'news' / 'news_features_monthly.csv'

TICKERS = {
    # Turkey / EM / FX
    'tur':     'TUR',        # iShares MSCI Turkey ETF
    'tryusd':  'TRYUSD=X',   # TRY/USD
    'eurusd':  'EURUSD=X',   # EUR/USD
    'eem':     'EEM',         # EM broad equity
    'dxy':     'DX-Y.NYB',   # USD Index
    'bist':    'XU100.IS',   # BIST-100
    # Confectionery / buyers
    'barry':   'BARN.SW',    # Barry Callebaut
    'lindt':   'LISN.SW',    # Lindt
    'hershey': 'HSY',         # Hershey
    'mdlz':    'MDLZ',       # Mondelēz
    'ulker':   'ULKER.IS',   # Ülker
    'adm':     'ADM',         # ADM
    # Soft commodities
    'cocoa':   'CC=F',
    'coffee':  'KC=F',
    'sugar':   'SB=F',
    'dba':     'DBA',
    'moo':     'MOO',
    # Energy
    'crude':   'USO',
    'natgas':  'UNG',
    # Macro
    'sp500':   '^GSPC',
    'stoxx':   '^STOXX50E',
    'vix':     '^VIX',
    'gold':    'GLD',
}

START = '2000-01-01'
END   = '2024-12-31'


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_giresun() -> pd.Series:
    df = pd.read_csv(DATA / 'giresun_spot_prices_monthly.csv')
    df = df[df['period'] == 'full'].copy()
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )
    s = df.set_index('date')['avg_usd_kg_inshell'].sort_index()
    return s[~s.index.duplicated(keep='first')].rename('haz_usd')


def download_tickers() -> pd.DataFrame:
    print('Downloading market data...')
    frames = {}
    for name, ticker in TICKERS.items():
        try:
            print(f'  {name:10s} ({ticker}) ...', end=' ', flush=True)
            raw = yf.download(ticker, start=START, end=END,
                              interval='1mo', progress=False, auto_adjust=True)
            if raw.empty:
                print('no data'); continue
            close = raw['Close'].squeeze()
            close.index = close.index.to_period('M').to_timestamp()
            frames[name] = close
            print(f'{close.notna().sum()} months '
                  f'({close.index[close.notna()].min().year}–'
                  f'{close.index[close.notna()].max().year})')
        except Exception as e:
            print(f'ERROR — {e}')
    return pd.DataFrame(frames)


def load_google_trends() -> pd.DataFrame:
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print('pytrends not installed: pip install pytrends')
        return pd.DataFrame()
    print('Fetching Google Trends...')
    pt = TrendReq(hl='en-US', tz=360)
    results = {}
    for col, (kw, geo) in {
        'trends_nutella':  ('Nutella',       ''),
        'trends_hazelnut': ('hazelnut',       ''),
        'trends_findik':   ('fındık fiyatı', 'TR'),
    }.items():
        try:
            pt.build_payload([kw], timeframe=f'{START[:7]} {END[:7]}', geo=geo)
            df = pt.interest_over_time()
            if df.empty: continue
            results[col] = df[kw].resample('MS').mean()
            print(f'  {col}: {len(results[col])} months')
        except Exception as e:
            print(f'  {col}: ERROR — {e}')
    return pd.DataFrame(results) if results else pd.DataFrame()


def load_news_features() -> pd.DataFrame:
    if not NEWS_CSV.exists():
        print(f'  News features not found at {NEWS_CSV}')
        print('  Run: python scripts/scrape_news.py')
        return pd.DataFrame()
    df = pd.read_csv(NEWS_CSV, index_col=0, parse_dates=True)
    print(f'  News features: {len(df)} months  ({df.index.year.min()}–{df.index.year.max()})')
    return df


# ── Feature matrix ────────────────────────────────────────────────────────────
def build_features(prices, trends, news):
    log_rets = np.log(prices / prices.shift(1)).iloc[1:]

    # Month dummies (Jan = baseline)
    m_dummies = pd.get_dummies(log_rets.index.month, prefix='m', drop_first=True)
    m_dummies.index = log_rets.index

    parts = [log_rets, m_dummies]

    if not trends.empty:
        t_ret = np.log((trends + 1) / (trends.shift(1) + 1))
        parts.append(t_ret.reindex(log_rets.index))

    if not news.empty:
        parts.append(news.reindex(log_rets.index))

    return pd.concat(parts, axis=1)


# ── Regression ────────────────────────────────────────────────────────────────
def relaxed_lasso_ridge(X, y, names):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print('\nStage 1: LassoCV (cv=5)...')
    lasso = LassoCV(alphas=np.logspace(-4, 2, 300), cv=5, max_iter=50000, n_jobs=-1)
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

    print(f'\nStage 2: RidgeCV on {mask.sum()} features...')
    Xs_sel = Xs[:, mask]
    ridge  = RidgeCV(alphas=np.logspace(-3, 4, 200), cv=5)
    ridge.fit(Xs_sel, y)
    yhat   = ridge.predict(Xs_sel)
    r2_is  = 1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2)
    cv_r2  = cross_val_score(ridge, Xs_sel, y, cv=5, scoring='r2').mean()
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
def plot_all(res, y, dates, save, show):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = []

    # 1. Feature correlations
    corrs = pd.Series({f: np.corrcoef(res['Xs'][:, i], y)[0, 1]
                       for i, f in enumerate(res['names'])}).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, len(corrs) * 0.28)))
    ax.barh(corrs.index, corrs.values,
            color=['#e74c3c' if v < 0 else '#2ecc71' for v in corrs], edgecolor='none')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Pearson r  (vs monthly price log-return)')
    ax.set_title('Feature correlations — Giresun USD/kg price')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'price_correlations.png', dpi=150)
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
    if save: fig.savefig(FIGDIR / 'price_lasso_path.png', dpi=150)
    figs.append(fig)

    # 3. Ridge coefficients
    coef_df = res['coef_df']
    fig, ax = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.5)))
    ax.barh(coef_df['feature'], coef_df['ridge_coef'],
            color=['#e74c3c' if v < 0 else '#2ecc71' for v in coef_df['ridge_coef']],
            edgecolor='none')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Ridge coefficient (std. units)')
    ax.set_title(f"Ridge coefs — R²={res['r2_is']:.3f}  CV R²={res['cv_r2']:.3f}")
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'price_ridge_coefs.png', dpi=150)
    figs.append(fig)

    # 4. Actual vs predicted
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.plot(dates, y, color='#2c3e50', lw=1.2, label='Actual')
    ax1.plot(dates, res['yhat'], color='#e74c3c', lw=1.0, ls='--', alpha=0.85, label='Predicted')
    ax1.axhline(0, color='grey', lw=0.5); ax1.set_ylabel('Log return')
    ax1.set_title('Giresun price: actual vs basket-predicted'); ax1.legend(fontsize=9)

    ax2.plot(dates, np.cumsum(y), color='#2c3e50', lw=1.2, label='Actual cumulative')
    ax2.plot(dates, np.cumsum(res['yhat']), color='#e74c3c', lw=1.0, ls='--', label='Predicted')
    ax2.set_ylabel('Cumulative log return'); ax2.legend(fontsize=9)

    ax3.scatter(res['yhat'], y, alpha=0.4, s=18, color='steelblue', edgecolors='none')
    mn, mx = min(y.min(), res['yhat'].min()), max(y.max(), res['yhat'].max())
    ax3.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
    ax3.set_title(f"R²={res['r2_is']:.3f}")
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'price_fit.png', dpi=150)
    figs.append(fig)

    # 5. Residuals
    resid = y - res['yhat']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(dates, resid, color='#7f8c8d', lw=0.8)
    ax1.axhline(0, color='red', ls='--', lw=0.8)
    ax1.set_title('Residuals over time  (= basis risk)')
    ax2.hist(resid, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='red', ls='--', lw=0.8)
    ax2.set_title(f'Residuals  σ={resid.std():.4f}')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'price_residuals.png', dpi=150)
    figs.append(fig)

    if show: plt.show()
    for f in figs: plt.close(f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trends',    action='store_true', help='Add Google Trends features')
    ap.add_argument('--news',      action='store_true', help='Add NLP news features from scrape_news.py')
    ap.add_argument('--save-figs', action='store_true')
    ap.add_argument('--no-show',   action='store_true')
    args = ap.parse_args()

    print('Loading Giresun spot prices...')
    haz = load_giresun()
    print(f'  {haz.notna().sum()} months ({haz.index.min().year}–{haz.index.max().year})')

    prices = download_tickers()

    trends = load_google_trends() if args.trends else pd.DataFrame()

    news = pd.DataFrame()
    if args.news:
        print('Loading NLP news features...')
        news = load_news_features()

    print('Building feature matrix...')
    X_df = build_features(prices, trends, news)

    haz_ret  = np.log(haz / haz.shift(1)).rename('haz_ret')
    combined = pd.concat([haz_ret, X_df], axis=1).dropna(subset=['haz_ret'])

    thresh   = int(0.5 * len(combined))
    n_before = combined.shape[1] - 1
    combined = combined.dropna(axis=1, thresh=thresh)
    dropped  = n_before - (combined.shape[1] - 1)
    if dropped: print(f'  Dropped {dropped} features with >50% missing')

    feat_cols = [c for c in combined.columns if c != 'haz_ret']
    combined[feat_cols] = combined[feat_cols].ffill(limit=6)
    for col in feat_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].mean())
    combined = combined[combined['haz_ret'].notna()]

    print(f'\nSample: {len(combined)} months  '
          f'({combined.index.min().date()} – {combined.index.max().date()})  '
          f'{combined.shape[1]-1} features')

    y    = combined['haz_ret'].values
    cols = [c for c in combined.columns if c != 'haz_ret']
    X    = combined[cols].values

    res = relaxed_lasso_ridge(X, y, cols)

    print('\nRendering plots...')
    plot_all(res, y, combined.index, save=args.save_figs, show=not args.no_show)
    print('Done — close plot windows to exit.' if not args.no_show else 'Done.')

    print('\n' + '=' * 60)
    print('PRICE REGRESSION SUMMARY')
    print('=' * 60)
    print(f'Target       : Giresun USD/kg in-shell (monthly log-return)')
    print(f'Observations : {len(y)} months')
    print(f'Features     : {len(cols)} candidate → {res["mask"].sum()} selected')
    print(f'In-sample R² : {res["r2_is"]:.4f}')
    print(f'CV R²        : {res["cv_r2"]:.4f}')
    print(f'RMSE         : {res["rmse"]:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
