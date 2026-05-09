#!/usr/bin/env python3
"""
Hazelnut news scraper + keyword NLP.

Sources (in order of reliability):
  1. Google News RSS   — Turkish queries: fındık, fındık fiyatı, fındık don, fındık hasat
  2. Google News RSS   — English queries: turkey hazelnut, hazelnut price, hazelnut frost
  3. USDA FAS website  — Turkey tree nuts GAIN report listings (HTML scrape)
  4. Haberler.com      — Turkish news aggregator search for fındık

NLP: keyword-based event detection (no ML model required).
  Outputs binary/count columns per month for use in price_regression.py.

Output:
  data/raw/news/headlines.csv           — all scraped headlines (deduplicated)
  data/raw/news/news_features_monthly.csv — monthly event feature matrix
  data/raw/news/news_features_annual.csv  — annual event feature matrix

Usage:
    python scripts/scrape_news.py                    # full scrape
    python scripts/scrape_news.py --rebuild-features # recompute features from existing headlines
    python scripts/scrape_news.py --limit 50         # cap articles per source (testing)
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
NEWS_DIR  = ROOT / 'data' / 'raw' / 'news'
HEADLINES = NEWS_DIR / 'headlines.csv'
MONTHLY   = NEWS_DIR / 'news_features_monthly.csv'
ANNUAL    = NEWS_DIR / 'news_features_annual.csv'

HEADERS = {'User-Agent': 'Mozilla/5.0 (research bot; hazelnut crop data)'}
TIMEOUT = 15

# ── Keyword event categories ───────────────────────────────────────────────────
# Granular events (used for diagnostics)
EVENTS = [
    ('frost_news',    ['don ', 'donma', 'soğuk hava', 'frost', 'freeze', 'cold damage',
                       'ilkbahar donu', 'don hasarı', 'spring frost']),
    ('harvest_news',  ['hasat', 'rekolte', 'ürün', 'harvest', 'crop estimate',
                       'production forecast', 'yield']),
    ('price_news',    ['fiyat', 'price', 'piyasa', 'market price', 'spot price',
                       'kg fiyat', 'usd/kg']),
    ('support_price', ['destek fiyat', 'taban fiyat', 'tmo fiyat', 'support price',
                       'floor price', 'fiskobirlik fiyat', 'fisko fiyat']),
    ('production_cut',['rekolte düş', 'üretim azal', 'production decline', 'crop damage',
                       'ürün kayb', 'verim düş', 'shortage', 'supply cut']),
    ('production_up', ['rekolte artı', 'üretim artı', 'bol ürün', 'record harvest',
                       'bumper crop', 'high yield', 'production increase']),
    ('export_news',   ['ihracat', 'export', 'gümrük', 'tariff', 'trade ban', 'ithalat']),
]

# ── Bullish / bearish sentiment on production ─────────────────────────────────
# Bearish: anything implying lower output (frost, damage, shortage, production cut)
BEARISH_PROD = [
    'don ', 'donma', 'soğuk hava', 'frost', 'freeze', 'cold damage', 'ilkbahar donu',
    'don hasarı', 'spring frost', 'rekolte düş', 'üretim azal', 'production decline',
    'crop damage', 'ürün kayb', 'verim düş', 'shortage', 'supply cut', 'poor harvest',
    'kuraklık', 'drought', 'hail', 'dolu', 'flood', 'sel',
]
# Bullish: anything implying higher output (bumper crop, good conditions, high yield)
BULLISH_PROD = [
    'rekolte artı', 'üretim artı', 'bol ürün', 'record harvest', 'bumper crop',
    'high yield', 'production increase', 'iyi hasat', 'güzel ürün', 'yüksek rekolte',
    'favorable', 'good conditions', 'above average', 'normal sezon', 'beklenti aşıldı',
]
# Bearish on price (excess supply, price drop, buyer pullback)
BEARISH_PRICE = [
    'fiyat düş', 'price drop', 'price decline', 'fiyat geriledi', 'düşüş', 'ucuzlad',
    'oversupply', 'arz fazlası', 'alıcı yok', 'talep azal', 'demand decline',
]
# Bullish on price (scarcity, price rise, strong demand)
BULLISH_PRICE = [
    'fiyat artı', 'price rise', 'price increase', 'fiyat yükseldi', 'artış',
    'pahalılaştı', 'shortage', 'arz açığı', 'güçlü talep', 'strong demand',
    'record price', 'rekor fiyat',
]

# ── RSS helper ────────────────────────────────────────────────────────────────
def _parse_rss_date(s: str) -> pd.Timestamp | None:
    for fmt in ('%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S GMT'):
        try:
            return pd.Timestamp(datetime.strptime(s.strip(), fmt))
        except Exception:
            pass
    try:
        return pd.Timestamp(s)
    except Exception:
        return None


def fetch_google_news_rss(queries: list[str], limit: int = 200) -> list[dict]:
    rows = []
    for q in queries:
        from urllib.parse import quote
        url = f'https://news.google.com/rss/search?q={quote(q)}&hl=tr&gl=TR&ceid=TR:tr'
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            items = root.findall('.//item')
            for item in items[:limit]:
                title = item.findtext('title', '').strip()
                pub   = _parse_rss_date(item.findtext('pubDate', ''))
                src   = item.findtext('source', 'Google News')
                if title and pub:
                    rows.append({'date': pub, 'source': src,
                                 'headline': title, 'query': q})
            log.info('  Google News RSS [%s]: %d articles', q, len(items[:limit]))
        except Exception as e:
            log.warning('  Google News RSS [%s]: ERROR — %s', q, e)
        time.sleep(1.5)
    return rows


# ── USDA FAS GAIN ─────────────────────────────────────────────────────────────
def fetch_usda_gain(limit: int = 50) -> list[dict]:
    rows = []
    url = ('https://fas.usda.gov/data/search?'
           'field_commodity_tid=All&field_region_tid=All'
           '&field_country=Turkey&type=gain_report&field_topic=All')
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning('  USDA GAIN: BeautifulSoup4 not installed (pip install beautifulsoup4) — skipping')
        return rows
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # Articles appear as <div class="views-row"> or similar — grab all links with dates
        for link in soup.select('a[href*="gain-report"]')[:limit]:
            title = link.get_text(strip=True)
            # Try to find a nearby date element
            parent = link.find_parent(['div', 'li', 'tr'])
            date_str = ''
            if parent:
                date_el = parent.find(class_=lambda c: c and 'date' in c.lower())
                if date_el:
                    date_str = date_el.get_text(strip=True)
            pub = pd.Timestamp(date_str) if date_str else None
            if title:
                rows.append({'date': pub, 'source': 'USDA FAS GAIN',
                             'headline': title, 'query': 'usda_gain_turkey'})
        log.info('  USDA GAIN: %d reports', len(rows))
    except Exception as e:
        log.warning('  USDA GAIN: ERROR — %s', e)
    return rows


# ── Haberler.com ──────────────────────────────────────────────────────────────
def fetch_haberler(query: str = 'fındık', limit: int = 100) -> list[dict]:
    rows = []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning('  Haberler.com: BeautifulSoup4 not installed — skipping')
        return rows

    from urllib.parse import quote
    url = f'https://www.haberler.com/ara/?q={quote(query)}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Haberler.com search results: h3 > a elements with dates nearby
        for article in soup.select('article, .news-item, .haber-item, h3 a')[:limit]:
            if article.name == 'a':
                title = article.get_text(strip=True)
                parent = article.find_parent(['article', 'div', 'li'])
                date_text = ''
                if parent:
                    for el in parent.find_all(True):
                        cls = ' '.join(el.get('class', []))
                        if any(k in cls.lower() for k in ['date', 'tarih', 'time', 'zaman']):
                            date_text = el.get_text(strip=True)
                            break
                pub = None
                if date_text:
                    try:
                        pub = pd.Timestamp(date_text)
                    except Exception:
                        pass
                if title and len(title) > 10:
                    rows.append({'date': pub, 'source': 'Haberler.com',
                                 'headline': title, 'query': query})
            else:
                title_el = article.find(['h3', 'h2', 'a'])
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                date_el = article.find(class_=lambda c: c and
                                       any(k in c.lower() for k in ['date', 'tarih', 'time']))
                pub = None
                if date_el:
                    try:
                        pub = pd.Timestamp(date_el.get_text(strip=True))
                    except Exception:
                        pass
                if title and len(title) > 10:
                    rows.append({'date': pub, 'source': 'Haberler.com',
                                 'headline': title, 'query': query})

        log.info('  Haberler.com [%s]: %d articles', query, len(rows))
    except Exception as e:
        log.warning('  Haberler.com [%s]: ERROR — %s', query, e)
    return rows


# ── GDELT (historical, back to 2015 via free API) ─────────────────────────────
def fetch_gdelt(query: str = 'hazelnut Turkey', limit: int = 250) -> list[dict]:
    """
    GDELT DOC 2.0 API — free, no key needed, covers 2015+.
    For pre-2015 history use the GDELT bulk download (separate process).
    """
    from urllib.parse import quote
    url = (f'https://api.gdeltproject.org/api/v2/doc/doc'
           f'?query={quote(query)}&mode=artlist&maxrecords={min(limit, 250)}'
           f'&format=json&sort=DateDesc')
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        articles = data.get('articles', [])
        for a in articles:
            pub = pd.Timestamp(a.get('seendate', '')) if a.get('seendate') else None
            rows.append({
                'date':     pub,
                'source':   a.get('domain', 'GDELT'),
                'headline': a.get('title', '').strip(),
                'query':    query,
            })
        log.info('  GDELT [%s]: %d articles', query, len(rows))
    except Exception as e:
        log.warning('  GDELT [%s]: ERROR — %s', query, e)
    return rows


# ── NLP: keyword event + sentiment classification ─────────────────────────────
def classify_headlines(df: pd.DataFrame) -> pd.DataFrame:
    text = df['headline'].str.lower().fillna('')
    # Granular event flags
    for col, keywords in EVENTS:
        df[col] = text.apply(lambda t: int(any(k in t for k in keywords)))
    # Sentiment on production and price
    df['bearish_prod']  = text.apply(lambda t: int(any(k in t for k in BEARISH_PROD)))
    df['bullish_prod']  = text.apply(lambda t: int(any(k in t for k in BULLISH_PROD)))
    df['bearish_price'] = text.apply(lambda t: int(any(k in t for k in BEARISH_PRICE)))
    df['bullish_price'] = text.apply(lambda t: int(any(k in t for k in BULLISH_PRICE)))
    return df


# ── Assign crop year ──────────────────────────────────────────────────────────
def assign_crop_year(dates: pd.Series) -> pd.Series:
    """
    Crop year for production regression: Nov(t-1) → Aug(t) maps to year t.

    This matches the production announcement calendar — the official Turkey
    hazelnut number is released ~September of year t, so all features used
    to predict it must come from before that point.

    Nov 2022 – Aug 2023  →  crop_year 2023
    Sep 2023 – Oct 2023  →  NaN  (post-harvest, pre-announcement; excluded)
    Nov 2023 – Aug 2024  →  crop_year 2024
    """
    year  = dates.dt.year
    month = dates.dt.month
    crop  = year.copy().astype(float)
    # Nov–Dec: belongs to next year's crop
    crop[month >= 11] = year[month >= 11] + 1
    # Sep–Oct: post-harvest, pre-announcement — exclude from production features
    crop[month.isin([9, 10])] = float('nan')
    return crop


# ── Aggregate to monthly / annual / crop-year features ───────────────────────
def build_feature_tables(df: pd.DataFrame):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df = df.dropna(subset=['date'])
    df['date'] = df['date'].dt.tz_localize(None)
    df['month']     = df['date'].dt.to_period('M').dt.to_timestamp()
    df['crop_year'] = assign_crop_year(df['date'])

    event_cols     = [c for c, _ in EVENTS]
    sentiment_cols = ['bearish_prod', 'bullish_prod', 'bearish_price', 'bullish_price']
    all_cols       = event_cols + sentiment_cols

    def add_net_scores(agg: pd.DataFrame) -> pd.DataFrame:
        agg['news_net_prod']  = agg['news_bullish_prod']  - agg['news_bearish_prod']
        agg['news_net_price'] = agg['news_bullish_price'] - agg['news_bearish_price']
        return agg

    # Monthly features (for price_regression.py)
    monthly = (df.groupby('month')[all_cols].sum()
                 .rename(columns={c: f'news_{c}' for c in all_cols}))
    monthly = add_net_scores(monthly)
    monthly.index.name = 'date'

    # Crop-year features (for production_regression.py — Nov(t-1)→Aug(t) window)
    crop_df = df.dropna(subset=['crop_year']).copy()
    crop_df['crop_year'] = pd.to_datetime(
        crop_df['crop_year'].astype(int).astype(str), format='%Y'
    )
    annual = (crop_df.groupby('crop_year')[all_cols].sum()
                     .rename(columns={c: f'news_{c}' for c in all_cols}))
    annual = add_net_scores(annual)
    annual.index.name = 'date'

    return monthly, annual


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild-features', action='store_true',
                    help='Skip scraping; recompute feature CSVs from existing headlines.csv')
    ap.add_argument('--limit', type=int, default=200,
                    help='Max articles per source (default 200; use 20 for testing)')
    args = ap.parse_args()

    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.rebuild_features:
        all_rows = []

        log.info('Fetching Google News RSS (Turkish queries)...')
        all_rows += fetch_google_news_rss(
            ['fındık', 'fındık fiyatı', 'fındık don', 'fındık hasat',
             'fiskobirlik', 'findik rekolte'],
            limit=args.limit
        )

        log.info('Fetching Google News RSS (English queries)...')
        all_rows += fetch_google_news_rss(
            ['turkey hazelnut', 'hazelnut price', 'hazelnut frost damage',
             'hazelnut harvest Turkey', 'Turkish hazelnut crop'],
            limit=args.limit
        )

        log.info('Fetching USDA FAS GAIN reports...')
        all_rows += fetch_usda_gain(limit=args.limit)

        log.info('Fetching GDELT (historical, 2015+)...')
        all_rows += fetch_gdelt('hazelnut Turkey', limit=args.limit)
        all_rows += fetch_gdelt('hazelnut frost Turkey', limit=args.limit)
        all_rows += fetch_gdelt('hazelnut price Turkey', limit=args.limit)

        log.info('Fetching Haberler.com...')
        all_rows += fetch_haberler('fındık', limit=args.limit)
        all_rows += fetch_haberler('fındık fiyatı', limit=args.limit)

        if not all_rows:
            log.error('No articles collected. Check network and dependencies.')
            return

        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset='headline', keep='first')
        df = classify_headlines(df)

        # Merge with existing headlines if present
        if HEADLINES.exists():
            existing = pd.read_csv(HEADLINES, parse_dates=['date'])
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset='headline', keep='first')

        df.to_csv(HEADLINES, index=False)
        log.info('\nSaved %d headlines → %s', len(df), HEADLINES)

    else:
        if not HEADLINES.exists():
            log.error('No headlines.csv found. Run without --rebuild-features first.')
            return
        df = pd.read_csv(HEADLINES, parse_dates=['date'])
        df = classify_headlines(df)
        log.info('Loaded %d headlines from %s', len(df), HEADLINES)

    # Build feature tables
    monthly, annual = build_feature_tables(df)
    monthly.to_csv(MONTHLY)
    annual.to_csv(ANNUAL)

    log.info('Monthly features: %d months → %s', len(monthly), MONTHLY)
    log.info('Annual features:  %d years  → %s', len(annual),  ANNUAL)

    # Summary
    event_cols = [f'news_{c}' for c, _ in EVENTS]
    log.info('\nEvent counts (total articles flagged):')
    for col in event_cols:
        if col in monthly.columns:
            log.info('  %-30s %d months with signal', col, (monthly[col] > 0).sum())

    log.info('\nSample (most recent 5 headlines):')
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None)
    df_dated = df.dropna(subset=['date']).sort_values('date', ascending=False)
    for _, row in df_dated.head(5).iterrows():
        flags = [c for c, _ in EVENTS if row.get(c, 0)]
        log.info('  [%s] %s  →  %s',
                 str(row['date'])[:10], row['headline'][:70], flags or 'none')


if __name__ == '__main__':
    main()
