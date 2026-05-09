#!/usr/bin/env python3
"""
Hazelnut production regression — relaxed L1→L2 feature selection.

Target  : YoY log-return of Turkey hazelnut production (FAOSTAT, annual)
Features: ERA5 frost degree-hours (5-phase) + ERA5 precipitation
          + lagged production return (on/off year proxy)
          + annual equity/FX/commodity log-returns (calendar year)

Stage 1 : LassoCV  — selects non-zero features
Stage 2 : RidgeCV  — refits on survivors, removes lasso shrinkage bias

Usage:
    python scripts/basket_regression.py                   # basket only (equity/FX/commodity)
    python scripts/basket_regression.py --with-weather    # add ERA5 frost+precip features
    python scripts/basket_regression.py --save-figs       # save to notebooks/figures/
    python scripts/basket_regression.py --no-show         # suppress plt.show() (CI mode)
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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA      = ROOT / 'data' / 'raw'
FIGDIR    = ROOT / 'notebooks' / 'figures'
FROST_CSV = DATA / 'era5_frost_monthly.csv'
PRECIP_CSV = DATA / 'era5_precip_monthly.csv'
FAOSTAT_CSV = DATA / 'faostat' / 'turkey_hazelnut_production.csv'

# ── Equity ticker universe (aggregated to annual returns) ─────────────────────
TICKERS = {
    # Turkey / EM / FX
    'tur':    'TUR',        # iShares MSCI Turkey ETF
    'tryusd': 'TRYUSD=X',   # TRY/USD
    'eurusd': 'EURUSD=X',   # EUR/USD
    'eem':    'EEM',         # EM broad equity
    'bist':   'XU100.IS',   # BIST-100
    # Confectionery / hazelnut buyers
    'barry':  'BARN.SW',    # Barry Callebaut
    'lindt':  'LISN.SW',    # Lindt
    'hershey':'HSY',         # Hershey
    'mdlz':   'MDLZ',       # Mondelēz
    'ulker':  'ULKER.IS',   # Ülker
    'adm':    'ADM',         # Archer-Daniels-Midland
    # Soft commodities
    'cocoa':  'CC=F',        # Cocoa futures
    'coffee': 'KC=F',        # Coffee futures
    'sugar':  'SB=F',        # Sugar futures
    'dba':    'DBA',         # DB Agriculture ETF
    # Macro
    'sp500':  '^GSPC',
    'vix':    '^VIX',
    'gold':   'GLD',
}

EQUITY_START = '1999-01-01'   # one extra year to compute first annual return
EQUITY_END   = '2024-12-31'


# ── Load FAOSTAT production ───────────────────────────────────────────────────
def load_production() -> pd.Series:
    df = pd.read_csv(FAOSTAT_CSV, index_col='year')['production_mt'].dropna()
    df.index = pd.to_datetime(df.index.astype(str), format='%Y')
    return df.rename('production_mt')


# ── Load ERA5 weather features (annual) ──────────────────────────────────────
def load_weather_features() -> pd.DataFrame:
    frames = {}

    if FROST_CSV.exists():
        frost = pd.read_csv(FROST_CSV, index_col='year')
        frost.index = pd.to_datetime(frost.index.astype(str), format='%Y')
        frames['frost'] = frost
        print(f'  Frost DH: {len(frost)} years  '
              f'({frost.index.year.min()}–{frost.index.year.max()})')
    else:
        print('  WARNING: frost CSV not found')

    if PRECIP_CSV.exists():
        precip = pd.read_csv(PRECIP_CSV, index_col='year')
        precip.index = pd.to_datetime(precip.index.astype(str), format='%Y')
        frames['precip'] = precip
        print(f'  Precipitation: {len(precip)} years  '
              f'({precip.index.year.min()}–{precip.index.year.max()})')
    else:
        print('  WARNING: precip CSV not found')

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1)


# ── Download equity data and aggregate to annual returns ─────────────────────
def download_tickers_annual() -> pd.DataFrame:
    print('Downloading equity/FX/commodity data (monthly → annual)...')
    monthly_frames = {}

    for name, ticker in TICKERS.items():
        try:
            print(f'  {name:10s} ({ticker}) ...', end=' ', flush=True)
            raw = yf.download(ticker, start=EQUITY_START, end=EQUITY_END,
                              interval='1mo', progress=False, auto_adjust=True)
            if raw.empty:
                print('no data')
                continue
            close = raw['Close'].squeeze()
            close.index = close.index.to_period('M').to_timestamp()
            monthly_frames[name] = close
            print(f'{close.notna().sum()} months '
                  f'({close.index[close.notna()].min().year}–'
                  f'{close.index[close.notna()].max().year})')
        except Exception as e:
            print(f'ERROR — {e}')

    if not monthly_frames:
        return pd.DataFrame()

    monthly = pd.DataFrame(monthly_frames)

    # Calendar-year log return: sum of monthly log-returns within each year
    log_monthly = np.log(monthly / monthly.shift(1))
    annual = log_monthly.resample('YS').sum(min_count=6)  # need ≥6 months
    annual.index = annual.index.to_period('Y').to_timestamp()

    # Flag years with <10 valid months as NaN
    monthly_count = monthly.resample('YS').count()
    monthly_count.index = monthly_count.index.to_period('Y').to_timestamp()
    annual[monthly_count < 10] = np.nan

    print(f'  Annual equity returns: {len(annual)} years  '
          f'({annual.index.year.min()}–{annual.index.year.max()})')
    return annual


# ── Build feature matrix ──────────────────────────────────────────────────────
def build_features(prod: pd.Series,
                   weather: pd.DataFrame,
                   equity: pd.DataFrame,
                   include_weather: bool = False) -> pd.DataFrame:
    parts = []

    # Lagged production log-return (on/off year proxy) — always included
    prod_ret = np.log(prod / prod.shift(1)).rename('prod_ret_lag1')
    parts.append(prod_ret.shift(1))

    # Weather: ERA5 frost DH + precip — Component A triggers, excluded from basket by default
    if include_weather and not weather.empty:
        parts.append(weather)

    # Equity/FX/commodity annual returns — the tradeable basket (Component B)
    if not equity.empty:
        parts.append(equity)

    return pd.concat(parts, axis=1)


# ── Relaxed L1 → L2 regression ───────────────────────────────────────────────
def relaxed_lasso_ridge(X: np.ndarray, y: np.ndarray,
                        feature_names: list[str], n_obs: int) -> dict:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cv_folds = min(5, n_obs // 6)

    print(f'\nStage 1: LassoCV feature selection  (cv={cv_folds})...')
    alphas = np.logspace(-4, 2, 300)
    lasso  = LassoCV(alphas=alphas, cv=cv_folds, max_iter=50000, n_jobs=-1)
    lasso.fit(Xs, y)

    selected_mask  = np.abs(lasso.coef_) > 1e-6
    selected_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    n_selected     = selected_mask.sum()
    print(f'  Alpha: {lasso.alpha_:.5f}   Features selected: {n_selected}/{len(feature_names)}')
    print(f'  Selected: {selected_names}')

    if n_selected == 0:
        print('  WARNING: Lasso zeroed all features — relaxing to 25th-percentile alpha.')
        lasso.alpha_ = lasso.alphas_[len(lasso.alphas_) // 4]
        lasso.coef_  = Lasso(alpha=lasso.alpha_, max_iter=50000).fit(Xs, y).coef_
        selected_mask  = np.abs(lasso.coef_) > 1e-6
        selected_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        n_selected     = selected_mask.sum()
        print(f'  Relaxed alpha: {lasso.alpha_:.5f}   Features selected: {n_selected}/{len(feature_names)}')

    print(f'\nStage 2: RidgeCV on {n_selected} selected features  (cv={cv_folds})...')
    Xs_sel       = Xs[:, selected_mask]
    ridge_alphas = np.logspace(-3, 4, 200)
    ridge        = RidgeCV(alphas=ridge_alphas, cv=cv_folds)
    ridge.fit(Xs_sel, y)

    y_hat  = ridge.predict(Xs_sel)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_is  = 1 - ss_res / ss_tot
    cv_r2  = cross_val_score(ridge, Xs_sel, y, cv=cv_folds, scoring='r2').mean()
    rmse   = np.sqrt(np.mean((y - y_hat) ** 2))

    print(f'  Ridge alpha:    {ridge.alpha_:.4f}')
    print(f'  In-sample R²:   {r2_is:.3f}')
    print(f'  CV R² ({cv_folds}-fold):  {cv_r2:.3f}')
    print(f'  RMSE:           {rmse:.4f}  (log-return units)')

    coef_df = pd.DataFrame({
        'feature':    selected_names,
        'ridge_coef': ridge.coef_,
        'lasso_coef': lasso.coef_[selected_mask],
    }).sort_values('ridge_coef', key=abs, ascending=False)

    print('\nCoefficients (standardised units):')
    print(coef_df.to_string(index=False))

    return {
        'lasso': lasso, 'ridge': ridge, 'scaler': scaler,
        'selected_mask': selected_mask, 'selected_names': selected_names,
        'coef_df': coef_df, 'y_hat': y_hat,
        'r2_is': r2_is, 'cv_r2': cv_r2, 'rmse': rmse,
        'feature_names': feature_names, 'Xs': Xs,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(result: dict, y: np.ndarray, years: pd.DatetimeIndex,
             prod_raw: pd.Series, save: bool, show: bool) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)

    coef_df      = result['coef_df']
    y_hat        = result['y_hat']
    all_features = result['feature_names']
    Xs           = result['Xs']
    lasso        = result['lasso']
    r2_is        = result['r2_is']
    cv_r2        = result['cv_r2']

    figs = []

    # ── 1. Feature correlations with production return ────────────────────────
    corrs = pd.Series(
        {f: np.corrcoef(Xs[:, i], y)[0, 1] for i, f in enumerate(all_features)},
        name='pearson_r'
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, len(corrs) * 0.3)))
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in corrs]
    ax.barh(corrs.index, corrs.values, color=colors, edgecolor='none')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Pearson r  (vs annual production log-return)')
    ax.set_title('Feature correlations with Turkey hazelnut production')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_correlations.png', dpi=150)
    figs.append(fig)

    # ── 2. Lasso regularisation path ─────────────────────────────────────────
    mse_path = np.mean(lasso.mse_path_, axis=1)
    fig, ax  = plt.subplots(figsize=(8, 4))
    ax.semilogx(lasso.alphas_, mse_path, color='steelblue', lw=1.5)
    ax.axvline(lasso.alpha_, color='red', ls='--', lw=1.2,
               label=f'Selected α={lasso.alpha_:.5f}')
    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel('MSE (CV)')
    ax.set_title('Lasso regularisation path')
    ax.legend()
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_lasso_path.png', dpi=150)
    figs.append(fig)

    # ── 3. Ridge coefficients ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(4, len(coef_df) * 0.55)))
    colors  = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_df['ridge_coef']]
    ax.barh(coef_df['feature'], coef_df['ridge_coef'], color=colors, edgecolor='none')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Ridge coefficient  (std. units)')
    ax.set_title(f'Ridge coefficients on Lasso-selected features\n'
                 f'In-sample R²={r2_is:.3f}   CV R²={cv_r2:.3f}')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_ridge_coefs.png', dpi=150)
    figs.append(fig)

    # ── 4. Actual vs predicted production (MT levels) ────────────────────────
    # Back-compute predicted production levels from log-return predictions
    prod_aligned = prod_raw.reindex(years)
    prod_prev    = prod_raw.shift(1).reindex(years)
    pred_levels  = prod_prev * np.exp(y_hat)

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[3, 1])

    ax_lev = fig.add_subplot(gs[0, 0])
    ax_ret = fig.add_subplot(gs[1, 0], sharex=ax_lev)
    ax_sc  = fig.add_subplot(gs[:, 1])

    ax_lev.bar(years.year, prod_aligned / 1e3, color='#bdc3c7', label='Actual (kt)')
    ax_lev.plot(years.year, pred_levels / 1e3, 'r--', lw=1.5, label='Predicted (kt)')
    ax_lev.set_ylabel('Production (000 t)')
    ax_lev.set_title('Turkey hazelnut production: actual vs model')
    ax_lev.legend(fontsize=9)

    ax_ret.plot(years.year, y,     color='#2c3e50', lw=1.2, marker='o', ms=3, label='Actual YoY return')
    ax_ret.plot(years.year, y_hat, color='#e74c3c', lw=1.0, ls='--', label='Predicted')
    ax_ret.axhline(0, color='grey', lw=0.5)
    ax_ret.set_ylabel('Log return')
    ax_ret.set_xlabel('Year')
    ax_ret.legend(fontsize=9)

    ax_sc.scatter(y_hat, y, alpha=0.5, s=20, color='steelblue', edgecolors='none')
    mn, mx = min(y.min(), y_hat.min()), max(y.max(), y_hat.max())
    ax_sc.plot([mn, mx], [mn, mx], 'k--', lw=0.8)
    ax_sc.set_xlabel('Predicted log return')
    ax_sc.set_ylabel('Actual log return')
    ax_sc.set_title(f'R²={r2_is:.3f}')

    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_fit.png', dpi=150)
    figs.append(fig)

    # ── 5. Residuals ──────────────────────────────────────────────────────────
    resid = y - y_hat
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(years.year, resid, color=['#e74c3c' if r < 0 else '#2ecc71' for r in resid])
    ax1.axhline(0, color='black', lw=0.8)
    ax1.set_title('Residuals by year')
    ax1.set_ylabel('Actual − Predicted log return')
    ax1.set_xlabel('Year')
    ax2.hist(resid, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='red', ls='--', lw=0.8)
    ax2.set_title(f'Residual distribution  (σ={resid.std():.3f})')
    ax2.set_xlabel('Residual')
    plt.tight_layout()
    if save: fig.savefig(FIGDIR / 'prod_residuals.png', dpi=150)
    figs.append(fig)

    if show:
        plt.show()
    for f in figs:
        plt.close(f)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-weather', action='store_true',
                        help='Include ERA5 frost/precip features (Component A triggers — excluded by default)')
    parser.add_argument('--save-figs', action='store_true')
    parser.add_argument('--no-show',   action='store_true')
    args = parser.parse_args()

    show = not args.no_show

    # ── Target ────────────────────────────────────────────────────────────────
    print('Loading FAOSTAT production...')
    prod = load_production()
    print(f'  {len(prod)} years  ({prod.index.year.min()}–{prod.index.year.max()})')

    # ── Weather features ──────────────────────────────────────────────────────
    print('Loading ERA5 weather features...')
    weather = load_weather_features()

    # ── Equity annual returns ─────────────────────────────────────────────────
    equity = download_tickers_annual()

    # ── Build feature matrix and target ──────────────────────────────────────
    print('Building feature matrix...')
    X_df    = build_features(prod, weather, equity)
    prod_ret = np.log(prod / prod.shift(1)).rename('prod_ret')

    combined = pd.concat([prod_ret, X_df], axis=1).dropna(subset=['prod_ret'])

    # Drop columns with >50% missing
    thresh   = int(0.5 * len(combined))
    n_before = combined.shape[1] - 1
    combined = combined.dropna(axis=1, thresh=thresh)
    n_after  = combined.shape[1] - 1
    if n_before != n_after:
        print(f'  Dropped {n_before - n_after} features with >50% missing')

    # Mean-impute remaining gaps (equity pre-2000, some weather gaps)
    feat_cols = [c for c in combined.columns if c != 'prod_ret']
    for col in feat_cols:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].mean())

    combined = combined[combined['prod_ret'].notna()]

    print(f'\nRegression sample: {len(combined)} years, '
          f'{combined.shape[1]-1} features')
    print(f'Date range: {combined.index.year.min()} – {combined.index.year.max()}')

    y    = combined['prod_ret'].values
    cols = [c for c in combined.columns if c != 'prod_ret']
    X    = combined[cols].values

    # ── Relaxed Lasso → Ridge ─────────────────────────────────────────────────
    result = relaxed_lasso_ridge(X, y, cols, n_obs=len(y))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nRendering plots...')
    plot_all(result, y, combined.index, prod, save=args.save_figs, show=show)
    print('Done — close plot windows to exit.' if show else 'Done.')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('PRODUCTION REGRESSION SUMMARY')
    print('=' * 60)
    print(f'Target            : FAOSTAT Turkey hazelnut production (YoY log-return)')
    print(f'Observations      : {len(y)} years')
    print(f'Candidate features: {len(cols)}')
    print(f'Selected (Lasso)  : {result["selected_mask"].sum()}')
    print(f'In-sample R²      : {result["r2_is"]:.4f}')
    print(f'CV R²             : {result["cv_r2"]:.4f}')
    print(f'RMSE              : {result["rmse"]:.4f}  (log-return units)')
    print()
    print('Top features by Ridge |coefficient|:')
    print(result['coef_df'].to_string(index=False))
    print('=' * 60)


if __name__ == '__main__':
    main()
