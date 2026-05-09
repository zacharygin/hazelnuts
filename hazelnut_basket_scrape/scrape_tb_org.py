"""
TB.org (Türkiye Barolar Birliği / commodity exchange) Free Market Price Scraper
Source: tb.org.tr/tr/findik-fiyatlari
Coverage: 2019–present, variety-level (Yağlı / Levant / Sivri), TL/kg high+low
"""

import logging

import bs4
import pandas as pd

from utils import (
    RAW_DIR, empty_row, get_session, polite_sleep,
    save_processed, save_raw, setup_logging, tr_float,
)

URL    = 'https://www.tb.org.tr/tr/findik-fiyatlari'
SOURCE = 'tb_org'


def run() -> pd.DataFrame:
    session = get_session()
    logging.info('TB.org — scraping: %s', URL)

    resp = session.get(URL, timeout=20)
    resp.raise_for_status()
    save_raw(resp.text, RAW_DIR / 'tb_org' / 'findik_fiyatlari.html')

    soup  = bs4.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table')
    if not table:
        logging.error('TB.org: no table found')
        return pd.DataFrame()

    rows = table.find_all('tr')
    logging.info('TB.org: %d raw rows', len(rows) - 1)

    records = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
        if len(cells) < 4:
            continue
        # columns: Çeşit | En Yüksek Fiyat | En Düşük Fiyat | Tarih
        rec = empty_row()
        rec.update({
            'source':         SOURCE,
            'variety':        cells[0],
            'max_price_tlkg': tr_float(cells[1]),
            'min_price_tlkg': tr_float(cells[2]),
            'date':           cells[3],
            'product':        'FINDIK SERBEST PIYASA',
            'exchange':       'Serbest Piyasa',
            'url':            URL,
        })
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        logging.warning('TB.org: no data parsed')
        return df

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.drop_duplicates(subset=['source', 'date', 'variety'])
    df = df.sort_values(['variety', 'date']).reset_index(drop=True)

    # Derived avg
    df['avg_price_tlkg'] = (df['min_price_tlkg'] + df['max_price_tlkg']) / 2

    save_processed(df, 'tb_org_hazelnut.csv')
    logging.info('TB.org: %d rows, varieties=%s, %s–%s',
                 len(df), df['variety'].unique().tolist(),
                 df['date'].min().date(), df['date'].max().date())
    return df


if __name__ == '__main__':
    setup_logging()
    run()
