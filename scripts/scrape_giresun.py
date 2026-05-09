"""
Scrape Giresun Commodity Exchange hazelnut spot prices.

Two source pages:
  1. monthly_sales_corporations.php?sezonyil=YYYY
     Per-crop-year breakdown by buyer type (Open Market / Fiskobirlik / TMO / Total).
     Columns: Qty (kg), Value (TL), Price (TL).  Includes Ağustos and Ağustos_S.
     Price is volume-weighted average TL/kg. Used as primary source.

  2. sales_prices.php (page=1..7, 4-year blocks)
     Provides USD prices and FX rates directly. Used to enrich with USD columns.

Month name notes:
  - Page encoding: UTF-8 bytes served as cp1252 mojibake;
    fix: s.encode('cp1252').decode('utf-8')
  - "Ağustos"   = August, start of crop year  (month 8, period='open')
  - "Ağustos_S" = Ağustos Sonu (end of August / closing period, month 8, period='close')
  - Both are included in crop-year averages

Crop year convention:
  Crop year t = Ağustos(t) + Sep(t) … Jul(t+1) + Ağustos_S(t+1)
  i.e., all records for sezonyil=t belong to crop year t.

Output files:
  data/raw/giresun_spot_prices_monthly.csv     — one row per (year, month, period)
  data/raw/giresun_spot_prices_cropyear.csv    — volume-weighted crop-year averages
  data/raw/giresun_spot_prices_annual.csv      — calendar-year averages (for legacy compat)
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / 'data' / 'raw'
BASE        = 'https://www.giresuntb.org.tr/EN'
HEADERS     = {'User-Agent': 'Mozilla/5.0 (research data collection)'}
DELAY_S     = 1.0
KERNEL_YLD  = 0.45

CROP_YEARS = list(range(2000, 2025))   # sezonyil parameter
PRICE_PAGES = range(1, 8)              # page=1..7 for sales_prices.php

# ── Month name → (month_int, period) ─────────────────────────────────────────
MONTH_MAP = {
    # Turkish
    'ocak':       (1,  'full'),
    'şubat':      (2,  'full'),
    'mart':       (3,  'full'),
    'nisan':      (4,  'full'),
    'mayıs':      (5,  'full'),
    'haziran':    (6,  'full'),
    'temmuz':     (7,  'full'),
    'ağustos':    (8,  'open'),    # start of crop year
    'ağustos_s':  (8,  'close'),   # Ağustos Sonu — end of marketing year
    'eylül':      (9,  'full'),
    'ekim':       (10, 'full'),
    'kasım':      (11, 'full'),
    'aralık':     (12, 'full'),
    # English
    'january':    (1,  'full'),
    'february':   (2,  'full'),
    'march':      (3,  'full'),
    'april':      (4,  'full'),
    'may':        (5,  'full'),
    'june':       (6,  'full'),
    'july':       (7,  'full'),
    'august':     (8,  'open'),
    'august_s':   (8,  'close'),
    'september':  (9,  'full'),
    'october':    (10, 'full'),
    'november':   (11, 'full'),
    'december':   (12, 'full'),
}


def fix_encoding(s: str) -> str:
    """Repair cp1252 mojibake (UTF-8 bytes decoded as cp1252)."""
    try:
        return s.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def parse_month(raw: str):
    """Return (month_int, period) or (None, None)."""
    cleaned = fix_encoding(raw).strip().lower()
    cleaned = re.sub(r'[^\w_]', '', cleaned)
    return MONTH_MAP.get(cleaned, (None, None))


def parse_num(s: str) -> float:
    """Parse Turkish-formatted number: '1.234,56' → 1234.56"""
    s = s.strip().replace('\xa0', '').replace(' ', '')
    if not s or s in ('-', '0,00', '0.00'):
        return 0.0
    s = s.replace('.', '').replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return float('nan')


def old_tl_to_new(value: float, year: int) -> float:
    """Convert old TL to new TL where needed (redenomination Jan 2005)."""
    if year < 2005:
        return value / 1_000_000
    if year == 2005 and value > 1_000:
        return value / 1_000_000
    return value


# ── Scraper 1: monthly_sales_corporations.php ─────────────────────────────────
def scrape_corps_year(year: int) -> list[dict]:
    """Scrape one crop year from monthly_sales_corporations.php."""
    url = f'{BASE}/monthly_sales_corporations.php?sezonyil={year}'
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    if r.status_code == 500:
        return []

    soup = BeautifulSoup(r.content, 'lxml')
    rows_out = []

    for table in soup.find_all('table'):
        data_rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            if len(cells) < 10:
                continue
            # First cell should be a month name (not a header like 'MONTH' or 'TOTAL')
            month_name = cells[0].strip()
            if month_name.upper() in ('MONTH', 'QUANTITY (KGS)', 'TOTAL', ''):
                continue
            month_num, period = parse_month(month_name)
            if month_num is None:
                print(f'  [warn] unrecognised month: {repr(month_name)} (crop year {year})')
                continue

            # Columns: Month | Open(Qty,Val,Prc) | Fisko(Qty,Val,Prc) | TMO(Qty,Val,Prc) | Total(Qty,Val,Prc)
            # Indices:   0        1   2   3           4    5   6           7   8   9          10  11  12
            try:
                open_qty   = parse_num(cells[1])
                open_val   = parse_num(cells[2])
                open_prc   = parse_num(cells[3])
                fisko_qty  = parse_num(cells[4])
                fisko_val  = parse_num(cells[5])
                fisko_prc  = parse_num(cells[6])
                tmo_qty    = parse_num(cells[7])
                tmo_val    = parse_num(cells[8])
                tmo_prc    = parse_num(cells[9])
                total_qty  = parse_num(cells[10])
                total_val  = parse_num(cells[11])
                total_prc  = parse_num(cells[12])
            except IndexError:
                continue

            # Old TL correction on values (prices were already per-kg)
            total_val_new = old_tl_to_new(total_val, year)
            total_prc_new = old_tl_to_new(total_prc, year)

            # Recompute price from value/qty for robustness (handles old TL values)
            if total_qty > 0:
                total_prc_new = total_val_new / total_qty

            data_rows.append({
                'crop_year':         year,
                'month':             month_num,
                'period':            period,
                'open_qty_kg':       open_qty,
                'open_price_tl':     old_tl_to_new(open_prc, year),
                'fisko_qty_kg':      fisko_qty,
                'fisko_price_tl':    old_tl_to_new(fisko_prc, year),
                'tmo_qty_kg':        tmo_qty,
                'tmo_price_tl':      old_tl_to_new(tmo_prc, year),
                'total_qty_kg':      total_qty,
                'total_value_tl':    total_val_new,
                'total_price_tl_kg': total_prc_new,
            })

        if data_rows:
            rows_out = data_rows
            break

    return rows_out


# ── Scraper 2: sales_prices.php (for FX rates + USD prices) ──────────────────
def scrape_usd_prices() -> pd.DataFrame:
    """Scrape all 4-year blocks from sales_prices.php to get FX + USD prices."""
    all_rows = []
    for page in PRICE_PAGES:
        url = f'{BASE}/sales_prices.php?page={page}'
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
        except Exception as e:
            print(f'  [warn] sales_prices page={page}: {e}')
            time.sleep(DELAY_S)
            continue

        soup = BeautifulSoup(r.content, 'lxml')
        current_year = None

        for table in soup.find_all('table'):
            found_data = False
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                if len(cells) < 8:
                    continue
                year_str = cells[0].strip()
                if re.match(r'^\d{4}$', year_str):
                    current_year = int(year_str)

                if current_year is None:
                    continue

                month_num, period = parse_month(cells[1])
                if month_num is None:
                    continue

                try:
                    fx       = parse_num(cells[2])
                    avg_tl   = old_tl_to_new(parse_num(cells[7]), current_year)
                    avg_usd  = parse_num(cells[8]) if len(cells) > 8 else float('nan')
                except IndexError:
                    continue

                all_rows.append({
                    'year':      current_year,
                    'month':     month_num,
                    'period':    period,
                    'fx_try_per_usd': fx,
                    'avg_try_kg_inshell': avg_tl,
                    'avg_usd_kg_inshell': avg_usd,
                })
                found_data = True

            if found_data:
                break

        time.sleep(DELAY_S)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df['avg_try_kg_shelled'] = df['avg_try_kg_inshell'] / KERNEL_YLD
    df['avg_usd_kg_shelled'] = df['avg_usd_kg_inshell'] / KERNEL_YLD
    return df.sort_values(['year','month','period']).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Part 1: scrape monthly_sales_corporations for all crop years ──────────
    print('=== Scraping monthly_sales_corporations.php ===')
    corps_rows = []
    for yr in CROP_YEARS:
        print(f'  crop year {yr} ...', end=' ')
        try:
            rows = scrape_corps_year(yr)
            print(f'{len(rows)} rows  '
                  f'(Aug: {sum(1 for r in rows if r["month"]==8 and r["period"]=="open")}  '
                  f'Aug_S: {sum(1 for r in rows if r["month"]==8 and r["period"]=="close")})')
            corps_rows.extend(rows)
        except Exception as e:
            print(f'ERROR: {e}')
        time.sleep(DELAY_S)

    corps_df = pd.DataFrame(corps_rows)

    # ── Part 2: scrape sales_prices.php for FX rates + USD prices ────────────
    print('\n=== Scraping sales_prices.php for FX / USD ===')
    usd_df = scrape_usd_prices()
    print(f'  {len(usd_df)} rows with USD prices')

    # ── Merge: enrich corps_df with FX and USD prices ─────────────────────────
    # sales_prices uses calendar year; corps uses crop_year — align on calendar year
    # For month >= 9 (Sep-Dec): calendar year = crop_year
    # For month <= 8 (Jan-Aug): calendar year = crop_year + 1
    corps_df['year'] = corps_df.apply(
        lambda r: r['crop_year'] if r['month'] >= 9 else r['crop_year'] + 1, axis=1
    )

    if not usd_df.empty:
        usd_lookup = usd_df.set_index(['year','month','period'])
        def get_usd(row):
            key = (row['year'], row['month'], row['period'])
            if key in usd_lookup.index:
                return usd_lookup.loc[key, 'avg_usd_kg_inshell']
            # fallback: use TL price + FX rate if available
            fx_key = (row['year'], row['month'], row['period'])
            if fx_key in usd_lookup.index:
                fx = usd_lookup.loc[fx_key, 'fx_try_per_usd']
                if fx > 0:
                    return row['total_price_tl_kg'] / fx
            return float('nan')

        def get_fx(row):
            key = (row['year'], row['month'], row['period'])
            if key in usd_lookup.index:
                return usd_lookup.loc[key, 'fx_try_per_usd']
            return float('nan')

        corps_df['fx_try_per_usd']    = corps_df.apply(get_fx, axis=1)
        corps_df['avg_usd_kg_inshell'] = corps_df.apply(get_usd, axis=1)
    else:
        corps_df['fx_try_per_usd']    = float('nan')
        corps_df['avg_usd_kg_inshell'] = float('nan')

    # Compute USD from TL+FX for rows missing USD (fills in August gaps)
    missing_usd = corps_df['avg_usd_kg_inshell'].isna() & corps_df['fx_try_per_usd'].notna()
    corps_df.loc[missing_usd, 'avg_usd_kg_inshell'] = (
        corps_df.loc[missing_usd, 'total_price_tl_kg'] /
        corps_df.loc[missing_usd, 'fx_try_per_usd']
    )

    corps_df['avg_try_kg_inshell'] = corps_df['total_price_tl_kg']
    corps_df['avg_usd_kg_shelled'] = corps_df['avg_usd_kg_inshell'] / KERNEL_YLD
    corps_df['avg_try_kg_shelled'] = corps_df['avg_try_kg_inshell'] / KERNEL_YLD

    # ── Save monthly file ─────────────────────────────────────────────────────
    out_cols = ['crop_year','year','month','period',
                'total_qty_kg','total_value_tl','total_price_tl_kg',
                'open_qty_kg','open_price_tl','fisko_qty_kg','fisko_price_tl',
                'tmo_qty_kg','tmo_price_tl',
                'fx_try_per_usd','avg_try_kg_inshell','avg_usd_kg_inshell',
                'avg_try_kg_shelled','avg_usd_kg_shelled']
    monthly_out = corps_df[out_cols].sort_values(['crop_year','month','period'])
    monthly_out.to_csv(DATA_DIR / 'giresun_spot_prices_monthly.csv', index=False)
    print(f'\nSaved monthly: {len(monthly_out)} rows')

    aug_rows = monthly_out[monthly_out['month'] == 8]
    print(f'  August rows: {len(aug_rows)} ({aug_rows["period"].value_counts().to_dict()})')

    # ── Crop-year averages (volume-weighted) ─────────────────────────────────
    # Exclude Ağustos_S from the average if also have Ağustos (avoid double-counting Aug)
    # Include Ağustos_S only as end-of-season indicator
    cy_avg = []
    for yr, grp in corps_df.groupby('crop_year'):
        # Use total volume-weighted price across all months in crop year
        valid = grp.dropna(subset=['total_price_tl_kg'])
        if len(valid) == 0:
            continue
        total_qty   = valid['total_qty_kg'].sum()
        total_val   = valid['total_value_tl'].sum()
        vwap_tl     = total_val / total_qty if total_qty > 0 else float('nan')

        valid_usd = valid.dropna(subset=['avg_usd_kg_inshell'])
        vwap_usd = float('nan')
        if len(valid_usd) > 0 and valid_usd['total_qty_kg'].sum() > 0:
            usd_qty = valid_usd['total_qty_kg'].sum()
            usd_val = (valid_usd['avg_usd_kg_inshell'] * valid_usd['total_qty_kg']).sum()
            vwap_usd = usd_val / usd_qty

        cy_avg.append({
            'crop_year':             yr,
            'n_months':              len(valid),
            'total_qty_kg':          total_qty,
            'vwap_try_kg_inshell':   vwap_tl,
            'vwap_usd_kg_inshell':   vwap_usd,
            'vwap_try_kg_shelled':   vwap_tl / KERNEL_YLD,
            'vwap_usd_kg_shelled':   vwap_usd / KERNEL_YLD if not pd.isna(vwap_usd) else float('nan'),
        })

    cy_df = pd.DataFrame(cy_avg)
    cy_df.to_csv(DATA_DIR / 'giresun_spot_prices_cropyear.csv', index=False)
    print(f'Saved crop-year: {len(cy_df)} rows')

    # ── Calendar-year averages (legacy compat) ─────────────────────────────────
    ann = (corps_df.dropna(subset=['avg_usd_kg_inshell'])
           .groupby('year')
           .agg(avg_try_kg_inshell=('avg_try_kg_inshell','mean'),
                avg_usd_kg_inshell=('avg_usd_kg_inshell','mean'))
           .reset_index())
    ann['avg_try_kg_shelled'] = ann['avg_try_kg_inshell'] / KERNEL_YLD
    ann['avg_usd_kg_shelled'] = ann['avg_usd_kg_inshell'] / KERNEL_YLD
    ann.to_csv(DATA_DIR / 'giresun_spot_prices_annual.csv', index=False)
    print(f'Saved annual:   {len(ann)} rows')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\nCrop-year averages (USD/kg shelled):')
    print(cy_df[['crop_year','n_months','total_qty_kg','vwap_usd_kg_shelled']].to_string(index=False))


if __name__ == '__main__':
    main()
