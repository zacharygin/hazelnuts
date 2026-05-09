"""
TOBB Hazelnut Exchange Price Scraper
Source: borsa.tobb.org.tr — FINDIK KABUKLU TOMBUL (LEVANT), ana_kod=9, alt_kod=904
Coverage: 2005–present, daily, multiple regional exchanges
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import bs4
import pandas as pd

from utils import (
    RAW_DIR, empty_row, get_session, polite_sleep,
    save_processed, save_raw, setup_logging, tr_float,
)

URL_CURRENT  = 'https://borsa.tobb.org.tr/fiyat_urun3.php?alt_kod=904&ana_kod=9'
URL_HISTORIC = 'https://borsa.tobb.org.tr/fiyat_sorgu2.php?ana_kod=9&alt_kod=904'
SOURCE       = 'tobb'
RAW_SUBDIR   = RAW_DIR / 'tobb'


def scrape_current(session) -> list[dict]:
    """Scrape the current-day summary page (one row per active exchange)."""
    logging.info('TOBB — scraping current prices: %s', URL_CURRENT)
    resp = session.get(URL_CURRENT, timeout=20)
    resp.raise_for_status()
    save_raw(resp.text, RAW_SUBDIR / 'current_latest.html')

    soup  = bs4.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table')
    if not table:
        logging.warning('TOBB current: no table found')
        return []

    rows = table.find_all('tr')
    records = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cells) < 8:
            continue
        rec = empty_row()
        rec.update({
            'source':              SOURCE,
            'exchange':            cells[0],
            'date':                cells[1],
            'min_price_tlkg':      tr_float(cells[2]),
            'max_price_tlkg':      tr_float(cells[3]),
            'avg_price_tlkg':      tr_float(cells[4]),
            'volume_kg':           tr_float(cells[5]),
            'transaction_count':   tr_float(cells[6]),
            'transaction_value_tl':tr_float(cells[7]),
            'product':             'FINDIK KABUKLU TOMBUL (LEVANT)',
            'variety':             'Levant',
            'url':                 URL_CURRENT,
        })
        records.append(rec)
    logging.info('TOBB current: %d rows', len(records))
    return records


def scrape_historical(session, year_start: int = 2005, year_end: int | None = None) -> list[dict]:
    """Scrape full historical data in 2-year chunks (avoids server truncation)."""
    if year_end is None:
        year_end = date.today().year

    all_records: list[dict] = []

    # Iterate in 2-year windows
    chunk_start = year_start
    while chunk_start <= year_end:
        chunk_end = min(chunk_start + 1, year_end)
        logging.info('TOBB historical: %d–%d', chunk_start, chunk_end)

        payload = {
            'gun1': '01', 'ay1': '01', 'yil1': str(chunk_start),
            'gun2': '31', 'ay2': '12', 'yil2': str(chunk_end),
            'borsa': '', 'siralama1': 'tarih', 'siralama2': 'ASC',
            'gonder': 'Sorgula',
        }
        resp = session.post(URL_HISTORIC, data=payload, timeout=30)
        resp.raise_for_status()

        raw_path = RAW_SUBDIR / f'historical_{chunk_start}_{chunk_end}.html'
        save_raw(resp.text, raw_path)

        soup  = bs4.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table')
        if not table:
            logging.warning('  no table for %d–%d', chunk_start, chunk_end)
            chunk_start += 2
            polite_sleep(1.5)
            continue

        rows = table.find_all('tr')
        n = 0
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all('td')]
            if len(cells) < 8:
                continue
            rec = empty_row()
            rec.update({
                'source':               SOURCE,
                'exchange':             cells[0],
                'date':                 cells[1],
                'min_price_tlkg':       tr_float(cells[2]),
                'max_price_tlkg':       tr_float(cells[3]),
                'avg_price_tlkg':       tr_float(cells[4]),
                'volume_kg':            tr_float(cells[5]),
                'transaction_count':    tr_float(cells[6]),
                'transaction_value_tl': tr_float(cells[7]),
                'product':              'FINDIK KABUKLU TOMBUL (LEVANT)',
                'variety':              'Levant',
                'url':                  URL_HISTORIC,
            })
            all_records.append(rec)
            n += 1

        logging.info('  %d rows', n)
        chunk_start += 2
        polite_sleep(1.5)

    return all_records


def run(year_start: int = 2005, skip_current: bool = False) -> pd.DataFrame:
    session  = get_session()
    records: list[dict] = []

    if not skip_current:
        records += scrape_current(session)

    records += scrape_historical(session, year_start=year_start)

    df = pd.DataFrame(records)
    if df.empty:
        logging.warning('TOBB: no data scraped')
        return df

    # Normalise date
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    df = df.dropna(subset=['date'])

    # Deduplicate
    df = df.drop_duplicates(subset=['source', 'date', 'exchange', 'product'])
    df = df.sort_values(['exchange', 'date']).reset_index(drop=True)

    save_processed(df, 'tobb_hazelnut.csv')
    logging.info('TOBB: %d total rows, %d exchanges, %s–%s',
                 len(df), df['exchange'].nunique(),
                 df['date'].min().date(), df['date'].max().date())
    return df


if __name__ == '__main__':
    setup_logging()
    run()
